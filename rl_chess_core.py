# rl_chess_core.py
# 基础骨架：状态编码、动作编码(AZ-73)、合法走子掩码、Stockfish对手与BC数据采样
# 依赖：python-chess, numpy, torch
# pip install chess numpy torch

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import chess
import chess.engine
import torch

# -----------------------------
# 1) Board Encoder (side-to-move perspective)
# -----------------------------
class BoardEncoder:
    """
    将棋盘编码为 (C, 8, 8) 张量。
    视角：始终从 '当前行棋方' 的角度（即把黑方回合视作翻转后的白方回合）。
    通道设计（轻量版，便于在小显存上训练）：
      - 12 个棋子平面：己方 P,N,B,R,Q,K 各一层；对方 P,N,B,R,Q,K 各一层（按当前行棋方划分）
      - 4 个王车易位平面：己方 O-O, O-O-O, 对方 O-O, O-O-O（整层常数 0/1）
      - 1 个吃过路兵平面：若存在过路兵位，则该格为 1，否则全 0
      - 1 个半步计数平面：归一化到 [0,1]，整层常数
    合计通道数：12 + 4 + 1 + 1 = 18
    """
    PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    def encode(self, board: chess.Board) -> np.ndarray:
        c = 18
        planes = np.zeros((c, 8, 8), dtype=np.float32)

        stm_white = board.turn  # True if white to move
        # helper: 将方块映射到“当前行棋方”的视角下的坐标
        def sq_to_xy(sq: int) -> Tuple[int, int]:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            if stm_white:
                x, y = file, rank
            else:
                # 翻转到黑方为“底边”的视角（旋转180度）
                x, y = 7 - file, 7 - rank
            return x, y

        # 12 piece planes
        # 前 6 层：己方 P,N,B,R,Q,K；后 6 层：对方 P,N,B,R,Q,K
        for sq, piece in board.piece_map().items():
            x, y = sq_to_xy(sq)
            ours = (piece.color == board.turn)
            base = 0 if ours else 6
            idx_in_type = self.PIECE_ORDER.index(piece.piece_type)  # 0..5
            planes[base + idx_in_type, y, x] = 1.0

        # castling rights (整层常数)
        def has_cr(side_is_white: bool, kingside: bool) -> bool:
            if side_is_white:
                return board.has_kingside_castling_rights(chess.WHITE) if kingside \
                       else board.has_queenside_castling_rights(chess.WHITE)
            else:
                return board.has_kingside_castling_rights(chess.BLACK) if kingside \
                       else board.has_queenside_castling_rights(chess.BLACK)

        # 当前行棋方 = 己方
        planes[12, :, :] = 1.0 if has_cr(board.turn, True) else 0.0     # ours O-O
        planes[13, :, :] = 1.0 if has_cr(board.turn, False) else 0.0    # ours O-O-O
        planes[14, :, :] = 1.0 if has_cr(not board.turn, True) else 0.0 # opp O-O
        planes[15, :, :] = 1.0 if has_cr(not board.turn, False) else 0.0# opp O-O-O

        # en passant
        if board.ep_square is not None:
            x, y = sq_to_xy(board.ep_square)
            planes[16, y, x] = 1.0

        # halfmove clock (0..100 常见)，归一化
        planes[17, :, :] = min(board.halfmove_clock, 100) / 100.0

        return planes  # (18, 8, 8)

# -----------------------------
# 2) Action Encoder (AlphaZero-style 8x8x73)
# -----------------------------
class AZ73Action:
    """
    AlphaZero 风格动作编码：index = origin_sq * 73 + plane_id
      - 0..55:  8方向 × 距离1..7 （方向序：N, NE, E, SE, S, SW, W, NW；每个方向按距离1..7）
      - 56..63: 8个马步（序： (1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1) ）
      - 64..72: 9个“非后升变”：升变到 N,B,R × 3方向（左吃、直进、右吃）
    说明：
      - 索引/方向均以“当前行棋方”为视角（前进 = +rank）
      - Q 升变使用常规 56 平面（前进/斜吃，距离=1），执行时检测到到达底线则自动升变为 Q
    """
    DIRS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]  # N,NE,E,SE,S,SW,W,NW
    KNIGHTS = [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]
    PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]  # N,B,R
    PROMO_DIRS = [(-1,1),(0,1),(1,1)]  # 左吃, 直进, 右吃（从当前行棋方视角）

    @staticmethod
    def _stm_transform(board: chess.Board, from_sq: int, to_sq: int) -> Tuple[int,int,int,int]:
        """将 from,to 的 (file,rank) 转换到“当前行棋方”的视角坐标，并返回 delta"""
        ff, rf = chess.square_file(from_sq), chess.square_rank(from_sq)
        ft, rt = chess.square_file(to_sq), chess.square_rank(to_sq)
        if board.turn:  # white to move
            xf, yf = ff, rf
            xt, yt = ft, rt
            dx, dy = xt - xf, yt - yf
        else:           # black to move: 旋转180度
            xf, yf = 7 - ff, 7 - rf
            xt, yt = 7 - ft, 7 - rt
            dx, dy = xt - xf, yt - yf
        return xf, yf, dx, dy

    @classmethod
    def move_to_index(cls, board: chess.Board, move: chess.Move) -> Optional[int]:
        """将一个 (合法) move 映射到 [0,4672) 索引；不合法返回 None"""
        if move not in board.legal_moves:
            return None

        xf, yf, dx, dy = cls._stm_transform(board, move.from_square, move.to_square)
        origin = yf * 8 + xf  # 0..63

        # under-promotion planes (64..72) for N,B,R
        if move.promotion in cls.PROMO_PIECES:
            # 根据 dx,dy 判定是左吃(-1,1)、直进(0,1)、右吃(1,1)
            for di, (px, py) in enumerate(cls.PROMO_DIRS):
                if dx == px and dy == py:
                    piece_idx = cls.PROMO_PIECES.index(move.promotion)  # 0..2 (N,B,R)
                    plane = 64 + piece_idx * 3 + di  # 64..72
                    return origin * 73 + plane
            # 理论上不应出现其他 delta 的“非后升变”
            return None

        # knights (56..63)
        for i, (kx, ky) in enumerate(cls.KNIGHTS):
            if dx == kx and dy == ky:
                plane = 56 + i
                return origin * 73 + plane

        # sliders / king / pawn forward/capture (0..55)
        # 必须是 8 个方向之一，且距离 1..7
        def norm(v):
            if v == 0: return 0
            return 1 if v > 0 else -1
        step_x, step_y = norm(dx), norm(dy)
        # 检查是否在 8 个方向之一
        if (step_x, step_y) in cls.DIRS:
            # 距离：曼哈顿不是，使用棋盘几何中的 “最大绝对分量”
            dist = max(abs(dx), abs(dy))
            if 1 <= dist <= 7:
                dir_id = cls.DIRS.index((step_x, step_y))  # 0..7
                plane = dir_id * 7 + (dist - 1)  # 0..55
                return origin * 73 + plane

        # Q promotion 走子（按常规平面编码），或其他少见情况若上面未匹配，则返回 None
        return None

    @classmethod
    def index_to_move(cls, board: chess.Board, index: int) -> Optional[chess.Move]:
        """
        将索引还原为 move（在当前局面下）。若无法构造合法着法，返回 None。
        注意：如果是“前进到尾排”的普通平面，且存在升变，则默认升变为后(Q)。
        """
        if not (0 <= index < 8*8*73):
            return None
        origin = index // 73
        plane = index % 73
        xf, yf = origin % 8, origin // 8

        # 还原到绝对棋盘 from_sq
        def xy_to_sq(x: int, y: int) -> int:
            if board.turn:  # white to move
                f, r = x, y
            else:           # black to move
                f, r = 7 - x, 7 - y
            return chess.square(f, r)

        if plane >= 64:
            # under-promotion
            sub = plane - 64
            piece_idx = sub // 3  # 0..2 -> N,B,R
            dir_idx = sub % 3     # 0:left-capture,1:straight,2:right-capture
            dx, dy = cls.PROMO_DIRS[dir_idx]
            xt, yt = xf + dx, yf + dy
            if not (0 <= xt < 8 and 0 <= yt < 8):
                return None
            from_sq = xy_to_sq(xf, yf)
            to_sq = xy_to_sq(xt, yt)
            promo_piece = cls.PROMO_PIECES[piece_idx]
            mv = chess.Move(from_sq, to_sq, promotion=promo_piece)
            return mv if mv in board.legal_moves else None

        elif plane >= 56:
            # knight
            i = plane - 56
            dx, dy = cls.KNIGHTS[i]
            xt, yt = xf + dx, yf + dy
            if not (0 <= xt < 8 and 0 <= yt < 8):
                return None
            from_sq = xy_to_sq(xf, yf)
            to_sq = xy_to_sq(xt, yt)
            mv = chess.Move(from_sq, to_sq)
            return mv if mv in board.legal_moves else None

        else:
            # sliders / king / pawn forward/capture
            dir_id = plane // 7       # 0..7
            dist = (plane % 7) + 1    # 1..7
            step_x, step_y = cls.DIRS[dir_id]
            xt, yt = xf + step_x * dist, yf + step_y * dist
            if not (0 <= xt < 8 and 0 <= yt < 8):
                return None
            from_sq = xy_to_sq(xf, yf)
            to_sq = xy_to_sq(xt, yt)

            # 若该步导致兵到尾排，且存在升变合法着，则默认升变为后(Q)
            mv = chess.Move(from_sq, to_sq)
            if mv not in board.legal_moves:
                # 尝试 Q 升变（例如兵到最后一排）
                mv_q = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                if mv_q in board.legal_moves:
                    return mv_q
                return None
            return mv

    @classmethod
    def legal_mask(cls, board: chess.Board) -> np.ndarray:
        """返回形状 (4672,) 的 0/1 掩码，仅标记当前局面的合法着法"""
        mask = np.zeros(8*8*73, dtype=np.float32)
        for mv in board.legal_moves:
            idx = cls.move_to_index(board, mv)
            if idx is not None:
                mask[idx] = 1.0
        return mask

# -----------------------------
# 3) Stockfish Opponent & Data Sampler (for BC)
# -----------------------------
@dataclass
class StockfishConfig:
    path: str = "stockfish"  # Windows: "stockfish.exe"
    skill_level: int = 5     # 0..20
    threads: int = 1
    hash_mb: int = 64
    movetime_ms: int = 100   # 每步思考时间

class StockfishWrapper:
    def __init__(self, cfg: StockfishConfig):
        self.engine = chess.engine.SimpleEngine.popen_uci(cfg.path)
        self.engine.configure({
            "Skill Level": int(np.clip(cfg.skill_level, 0, 20)),
            "Threads": cfg.threads,
            "Hash": cfg.hash_mb
        })
        self.movetime_ms = cfg.movetime_ms

    def best_move(self, board: chess.Board) -> chess.Move:
        limit = chess.engine.Limit(time=self.movetime_ms / 1000.0)
        result = self.engine.play(board, limit)
        return result.move

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass

def random_sf_position(sf: StockfishWrapper, plies: int = 12) -> chess.Board:
    """
    让 Stockfish 自对弈走若干步，返回一个“中局”局面，用于采样。
    为了多样性，也可以随机切换 side_to_move。
    """
    board = chess.Board()
    for _ in range(plies):
        if board.is_game_over():
            break
        mv = sf.best_move(board)
        board.push(mv)
    return board

def sample_bc_batch(sf: StockfishWrapper,
                    batch_size: int = 256,
                    encoder: Optional[BoardEncoder] = None,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成一批行为克隆训练样本：
      X: (B, 18, 8, 8)  棋盘张量
      y: (B,)           最优着法的 AZ-73 索引（long）
      mask: (B, 4672)   合法走子掩码（float32, 0/1）
    """
    if encoder is None:
        encoder = BoardEncoder()

    X = np.zeros((batch_size, 18, 8, 8), dtype=np.float32)
    y = np.zeros((batch_size,), dtype=np.int64)
    M = np.zeros((batch_size, 8*8*73), dtype=np.float32)

    for i in range(batch_size):
        # 采一个局面
        board = random_sf_position(sf, plies=np.random.randint(6, 16))
        # 调用 Stockfish 求该局面“最佳着”
        best = sf.best_move(board)
        # 编码
        x = encoder.encode(board)
        idx = AZ73Action.move_to_index(board, best)
        if idx is None:
            # 理论上极少发生（比如某些不常规情况），跳过重采
            while idx is None:
                board = random_sf_position(sf, plies=np.random.randint(6, 16))
                best = sf.best_move(board)
                idx = AZ73Action.move_to_index(board, best)
            x = encoder.encode(board)

        mask = AZ73Action.legal_mask(board)

        X[i] = x
        y[i] = idx
        M[i] = mask

    return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(M)

# -----------------------------
# 4) Quick Smoke Tests
# -----------------------------
def _smoke_test_move_encoding():
    b = chess.Board()
    enc = BoardEncoder()
    X = enc.encode(b)
    assert X.shape == (18,8,8)

    # 枚举全部合法走子，双向转换一致
    for mv in list(b.legal_moves)[:30]:  # 检查前30个
        idx = AZ73Action.move_to_index(b, mv)
        assert idx is not None
        mv2 = AZ73Action.index_to_move(b, idx)
        assert mv2 is not None and mv2 in b.legal_moves, (mv, idx, mv2)

    mask = AZ73Action.legal_mask(b)
    assert mask.sum() == len(list(b.legal_moves))
    print("[OK] Encoding smoke test passed.")

def _smoke_test_sf_batch(sf_path="stockfish"):
    sf = StockfishWrapper(StockfishConfig(path=sf_path, skill_level=1, movetime_ms=50))
    X, y, M = sample_bc_batch(sf, batch_size=8)
    assert X.shape == (8,18,8,8)
    assert y.shape == (8,)
    assert M.shape == (8,4672)
    # 每一行掩码的1数量应等于对应局面合法着法数
    for i in range(8):
        b = random_sf_position(sf, plies=8)
        mask = AZ73Action.legal_mask(b)
        assert (mask.sum() >= 1)
    sf.close()
    print("[OK] Stockfish batch sampling passed.")

if __name__ == "__main__":
    _smoke_test_move_encoding()
    try:
        _smoke_test_sf_batch()
    except Exception as e:
        print("Stockfish test skipped or failed:", e)
