# train_bc.py
# 行为克隆(BC)：用Stockfish生成 (state, best_move_idx, legal_mask) 在线训练策略网络
# 依赖：python-chess numpy torch tqdm
# pip install chess numpy torch tqdm

import os
import time
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from rl_chess_core import (
    BoardEncoder, AZ73Action,
    StockfishWrapper, StockfishConfig,
    sample_bc_batch
)

# -----------------------------
# 配置
# -----------------------------
@dataclass
class TrainConfig:
    # 资源&稳定性
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True                     # 混合精度
    grad_clip_norm: float = 1.0

    # 数据
    batch_size: int = 128                    # 3060/12G可用（如OOM可降到64）
    num_steps: int = 8_000                   # 训练步数（每步一个batch）
    eval_interval: int = 500
    save_interval: int = 1000

    # 优化器
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)

    # 模型
    policy_dim: int = 8*8*73

    # Stockfish 采样
    sf_path: str = "stockfish"               # Windows: "stockfish.exe"
    sf_skill: int = 3                        # 0..20，越高采样越慢
    sf_movetime_ms: int = 60                 # 每步思考时间（BC建议 30~80ms）
    sf_threads: int = 1
    sf_hash_mb: int = 64

    # 其他
    out_dir: str = "ckpts_bc"

CFG = TrainConfig()


# -----------------------------
# 模型（轻量 ResNet）
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = F.relu(out, inplace=True)
        return out

class PolicyValueNet(nn.Module):
    """
    输入： (B, 18, 8, 8)
    输出：
      - policy_logits: (B, 4672)
      - value: (B, 1)  （BC阶段可不用）
    """
    def __init__(self, in_ch=18, width=64, n_blocks=6, policy_dim=4672):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(n_blocks)])

        # policy head
        self.ph_conv = nn.Conv2d(width, 32, 1, bias=False)
        self.ph_bn   = nn.BatchNorm2d(32)
        self.ph_fc   = nn.Linear(32*8*8, policy_dim)

        # value head（可选，BC阶段先不使用）
        self.vh_conv = nn.Conv2d(width, 32, 1, bias=False)
        self.vh_bn   = nn.BatchNorm2d(32)
        self.vh_fc1  = nn.Linear(32*8*8, 128)
        self.vh_fc2  = nn.Linear(128, 1)

        # 小技巧：更快收敛
        nn.init.zeros_(self.ph_fc.bias)
        nn.init.zeros_(self.vh_fc2.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)

        # policy
        p = self.ph_conv(x)
        p = self.ph_bn(p)
        p = F.relu(p, inplace=True)
        p = p.view(p.size(0), -1)
        policy_logits = self.ph_fc(p)

        # value
        v = self.vh_conv(x)
        v = self.vh_bn(v)
        v = F.relu(v, inplace=True)
        v = v.view(v.size(0), -1)
        v = F.relu(self.vh_fc1(v), inplace=True)
        value = torch.tanh(self.vh_fc2(v))
        return policy_logits, value


# -----------------------------
# 工具函数
# -----------------------------
def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """
    CrossEntropy over masked logits:
      - logits: (B, C)
      - targets: (B,)  in [0, C)
      - legal_mask: (B, C) 0/1
    """
    very_neg = torch.finfo(logits.dtype).min / 4  # 避免 inf 导致的NaN，用一个很小的数
    masked_logits = torch.where(legal_mask > 0, logits, very_neg)
    return F.cross_entropy(masked_logits, targets)

def top1_match_rate(logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor) -> float:
    very_neg = torch.finfo(logits.dtype).min / 4
    masked_logits = torch.where(legal_mask > 0, logits, very_neg)
    pred = masked_logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

def save_ckpt(model, opt, scaler, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step
    }, path)

def load_ckpt_if_exists(model, opt, scaler, path):
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"[Info] Loaded checkpoint from {path} (step={ckpt.get('step')})")
        return ckpt.get("step", 0)
    return 0


# -----------------------------
# 训练主函数
# -----------------------------
def main():
    print("==> Device:", CFG.device)
    device = torch.device(CFG.device)

    # Stockfish 采样器
    sf = StockfishWrapper(StockfishConfig(
        path=CFG.sf_path,
        skill_level=CFG.sf_skill,
        threads=CFG.sf_threads,
        hash_mb=CFG.sf_hash_mb,
        movetime_ms=CFG.sf_movetime_ms
    ))

    # 模型 & 优化器
    model = PolicyValueNet(in_ch=18, width=64, n_blocks=6, policy_dim=CFG.policy_dim).to(device)
    opt = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, betas=CFG.betas)
    scaler = GradScaler(enabled=(CFG.use_amp and device.type == "cuda"))

    # 断点续训
    os.makedirs(CFG.out_dir, exist_ok=True)
    ckpt_path = os.path.join(CFG.out_dir, "bc_latest.pt")
    start_step = load_ckpt_if_exists(model, opt, scaler, ckpt_path)

    # 训练循环
    model.train()
    pbar = tqdm(range(start_step, CFG.num_steps), desc="Training(BC)")

    try:
        for step in pbar:
            # 在线采样一个 batch
            with torch.no_grad():
                X, y, M = sample_bc_batch(sf, batch_size=CFG.batch_size)  # CPU生成
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            M = M.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(scaler is not None and CFG.use_amp and device.type=="cuda")):
                logits, _ = model(X)
                loss = masked_ce_loss(logits, y, M)

            if scaler is not None:
                scaler.scale(loss).backward()
                if CFG.grad_clip_norm is not None:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if CFG.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip_norm)
                opt.step()

            # 简单的 top-1 匹配率（与Stockfish最优着一致率）
            with torch.no_grad():
                acc = top1_match_rate(logits, y, M)

            pbar.set_postfix(loss=float(loss.item()), acc=acc)

            # 评估
            if (step + 1) % CFG.eval_interval == 0:
                eval_acc = evaluate_match_rate(model, sf, device, n_batches=4, bs=CFG.batch_size//2)
                print(f"\n[Eval] step {step+1}: acc={eval_acc:.3f}")

            # 保存
            if (step + 1) % CFG.save_interval == 0 or step == CFG.num_steps - 1:
                save_ckpt(model, opt, scaler, step+1, ckpt_path)
                torch.save(model.state_dict(), os.path.join(CFG.out_dir, f"bc_step{step+1}.model"))
    finally:
        sf.close()


@torch.no_grad()
def evaluate_match_rate(model: nn.Module, sf: StockfishWrapper, device, n_batches=4, bs=64) -> float:
    model.eval()
    accs = []
    for _ in range(n_batches):
        X, y, M = sample_bc_batch(sf, batch_size=bs)
        X = X.to(device)
        y = y.to(device)
        M = M.to(device)
        logits, _ = model(X)
        acc = top1_match_rate(logits, y, M)
        accs.append(acc)
    model.train()
    return sum(accs) / len(accs) if accs else 0.0


if __name__ == "__main__":
    main()
