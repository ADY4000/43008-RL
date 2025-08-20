import pygame
import chess
import chess.engine

# --- 参数 ---
BOARD_SIZE = 8
SQUARE_SIZE = 80
WINDOW_SIZE = BOARD_SIZE * SQUARE_SIZE
INFO_HEIGHT = 40  # 底部信息栏高度
FPS = 30
STOCKFISH_PATH = "stockfish"  # Windows 写 "stockfish.exe"

# --- 颜色 ---
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT = (186, 202, 68)
TEXT_COLOR = (20, 20, 20)

# --- 难度设置 ---
skill_level = int(input("请输入Stockfish难度 (0-20): "))
if skill_level < 0: skill_level = 0
if skill_level > 20: skill_level = 20

# --- 初始化 ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + INFO_HEIGHT))
pygame.display.set_caption("Chess vs Stockfish")
clock = pygame.time.Clock()

# 字体：尝试加载 chess_alpha.ttf
try:
    piece_font = pygame.font.Font("Alpha.ttf", 48)  # 你下载的字体
    print("使用 Chess Alpha 字体")
    piece_symbols = {
        "P": "P", "N": "N", "B": "B", "R": "R", "Q": "Q", "K": "K",
        "p": "p", "n": "n", "b": "b", "r": "r", "q": "q", "k": "k"
    }
except:
    # 回退到 Unicode 字符
    piece_font = pygame.font.SysFont("DejaVu Sans", 48)
    print("未找到棋子字体，使用 Unicode")
    piece_symbols = {
        "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
        "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"
    }

PIECES = {p: piece_font.render(s, True, (0, 0, 0)) for p, s in piece_symbols.items()}
info_font = pygame.font.SysFont("Arial", 24)

# 棋局
board = chess.Board()

# 启动 Stockfish
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
engine.configure({"Skill Level": skill_level})

# 选中状态
selected_square = None

def draw_board():
    """绘制棋盘和棋子"""
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            color = WHITE if (row + col) % 2 == 0 else BLACK
            if selected_square == chess.square(col, 7 - row):
                color = HIGHLIGHT
            pygame.draw.rect(screen, color, rect)

            # 棋子
            square = chess.square(col, 7 - row)
            piece = board.piece_at(square)
            if piece:
                piece_symbol = piece.symbol()
                screen.blit(PIECES[piece_symbol], rect.move(15, 10))

def draw_info():
    """绘制底部信息栏"""
    pygame.draw.rect(screen, (220, 220, 220), (0, WINDOW_SIZE, WINDOW_SIZE, INFO_HEIGHT))

    if board.is_game_over():
        result = "游戏结束: " + board.result()
    else:
        turn = "白方" if board.turn else "黑方"
        result = f"轮到: {turn}"

    text = f"难度: {skill_level} | {result}"
    info_surface = info_font.render(text, True, TEXT_COLOR)
    screen.blit(info_surface, (10, WINDOW_SIZE + 10))

def get_square_from_mouse(pos):
    """鼠标坐标 → 棋盘格"""
    col = pos[0] // SQUARE_SIZE
    row = pos[1] // SQUARE_SIZE
    return chess.square(col, 7 - row)

def stockfish_move():
    """Stockfish 走棋"""
    global board
    result = engine.play(board, chess.engine.Limit(time=0.1))
    board.push(result.move)

# --- 主循环 ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and not board.is_game_over():
            square = get_square_from_mouse(event.pos)

            if selected_square is None:
                if board.piece_at(square) and board.color_at(square) == chess.WHITE:
                    selected_square = square
            else:
                move = chess.Move(selected_square, square)
                if move in board.legal_moves:
                    board.push(move)
                    if not board.is_game_over():
                        stockfish_move()
                selected_square = None

    draw_board()
    draw_info()
    pygame.display.flip()
    clock.tick(FPS)

engine.quit()
pygame.quit()
