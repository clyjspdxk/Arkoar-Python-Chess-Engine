import re
import time
from tqdm import tqdm

class ChessEngine:
    def __init__(self):
        self.board = []
        self.turn = 'w'
        self.castling = ''
        self.en_passant_square = '-'
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.depth = 3
        self.piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0,'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 0}

        # 棋子位置评估表，值表示每个位置的评估分
        self.piece_square_table = {
            'P': [ 
                [0, 0, 0, 0, 0, 0, 0, 0],
                [5, 5, 5, 5, 5, 5, 5, 5],
                [1, 1, 2, 3, 3, 2, 1, 1],
                [0.5, 0.5, 1, 2.5, 2.5, 1, 0.5, 0.5],
                [0, 0, 0, 2, 2, 0, 0, 0],
                [0.5, -0.5, -1, 0, 0, -1, -0.5, 0.5],
                [0.5, 1, 1, -2, -2, 1, 1, 0.5],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ],
            'N': [
                [-5, -4, -3, -3, -3, -3, -4, -5],
                [-4, -2, 0, 0, 0, 0, -2, -4],
                [-3, 0, 1, 1.5, 1.5, 1, 0, -3],
                [-3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3],
                [-3, 0, 1.5, 2, 2, 1.5, 0, -3],
                [-3, 0.5, 1, 1.5, 1.5, 1, 0.5, -3],
                [-4, -2, 0, 0.5, 0.5, 0, -2, -4],
                [-5, -4, -3, -3, -3, -3, -4, -5]
            ],
            'B': [
                [-2, -1, -1, -1, -1, -1, -1, -2],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0.5, 1, 1, 0.5, 0, -1],
                [-1, 0.5, 0.5, 1, 1, 0.5, 0.5, -1],
                [-1, 0, 1, 1, 1, 1, 0, -1],
                [-1, 1, 1, 1, 1, 1, 1, -1],
                [-1, 0.5, 0, 0, 0, 0, 0.5, -1],
                [-2, -1, -1, -1, -1, -1, -1, -2]
            ],
            'R': [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 1, 1, 1, 1, 1, 1, 0.5],
                [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
                [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
                [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
                [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
                [-0.5, 0, 0, 0, 0, 0, 0, -0.5],
                [0, 0, 0, 0.5, 0.5, 0, 0, 0]
            ],
            'Q': [
                [-2, -1, -1, -0.5, -0.5, -1, -1, -2],
                [-1, 0, 0, 0, 0, 0, 0, -1],
                [-1, 0, 0.5, 0.5, 0.5, 0.5, 0, -1],
                [-0.5, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5],
                [0, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5],
                [-1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, -1],
                [-1, 0, 0.5, 0, 0, 0, 0, -1],
                [-2, -1, -1, -0.5, -0.5, -1, -1, -2]
            ],
            'K': [
                [2, 3, 1, 0, 0, 1, 3, 2],
                [2, 2, 0, 0, 0, 0, 2, 2],
                [1, 1, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [-2, -2, -2, -2, -2, -2, -2, -2],
                [-3, -3, -3, -3, -3, -3, -3, -3],
                [-4, -4, -4, -4, -4, -4, -4, -4]
            ]
        }

    def ask_for_fen(self):
        """询问用户输入FEN字符串"""
        while True:
            fen = input("请输入FEN字符串: ")
            if self.is_valid_fen(fen):
                self.set_fen(fen)
                break
            else:
                print("无效的FEN字符串，请重新输入。")

    def is_valid_fen(self, fen):
        """验证FEN字符串的合法性"""
        # 简单的正则表达式验证FEN格式
        pattern = r'^([rnbqkpRNBQKP1-8]+\/){7}[rnbqkpRNBQKP1-8]+\s[bw]\s(-|[KQkq]{1,4})\s(-|[a-h][36])\s\d+\s\d+$'
        return re.match(pattern, fen) is not None

    def set_fen(self, fen):
        """根据FEN设置棋局"""
        parts = fen.split()
        rows = parts[0].split('/')
        self.board = []
        for row in rows:
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend(['.'] * int(char))
                else:
                    board_row.append(char)
            self.board.append(board_row)
        self.turn = parts[1]
        self.castling = parts[2]
        self.en_passant_square = parts[3]
        self.halfmove_clock = int(parts[4])
        self.fullmove_number = int(parts[5])

    def print_board(self):
        """打印当前棋盘状态"""
        for row in self.board:
            print(' '.join(row))
        print(f"当前回合: {'白棋' if self.turn == 'w' else '黑棋'}")

    def get_piece_value(self, piece, position):
        """返回棋子的价值，包括位置评估（正值为白棋，负值为黑棋）"""
        value = self.piece_values[piece.upper()]
        row, col = position
        if piece.isupper():
            value += self.piece_square_table[piece][row][col]  # 白棋位置评估
        else:
            value -= self.piece_square_table[piece.upper()][7 - row][col]  # 黑棋位置评估
        return value if piece.isupper() else -value

    def evaluate_position(self):
        """返回当前棋局的评估分数"""
        score = 0
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece != '.':
                    score += self.get_piece_value(piece, (i, j))
        return score

    def is_in_check(self, turn):
        """检测指定玩家是否处于被将军状态"""
        king_pos = None
        opponent_moves = []

        # 找到国王的位置
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if (turn == "white" and piece == "K") or (
                    turn == "black" and piece == "k"
                ):
                    king_pos = (i, j)
                    break
            if king_pos:
                break

        if not king_pos:
            return False  # 理论上不应该出现找不到国王的情况

        # 生成对手的所有可能走法
        opponent_turn = "black" if turn == "white" else "white"
        opponent_moves = self.generate_all_moves(opponent_turn)

        # 检查是否有走法可以攻击国王位置
        for move in opponent_moves:
            _, end = move
            if end == king_pos:
                return True  # 被将军
        return False  # 没有将军
        
        # 检查对方棋子是否能够攻击到王的位置
        opponent_moves = self.generate_all_moves('b' if turn == 'w' else 'w')
        for move in opponent_moves:
            if move[1] == king_pos:
                return True
        return False

    def is_checkmate(self):
        """检测是否将杀"""
        if not self.is_in_check(self.turn):
            return False
        all_moves = self.generate_all_moves(self.turn)
        for move in all_moves:
            self.make_move(move)
            if not self.is_in_check(self.turn):
                self.undo_move(move)
                return False
            self.undo_move(move)
        return True

    def generate_all_moves(self, turn):
        """生成当前回合所有棋子的合法走法"""
        moves = []
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if (turn == 'w' and piece.isupper()) or (turn == 'b' and piece.islower()):
                    moves.extend(self.generate_legal_moves_for_piece((i, j), piece))
        return moves

    def generate_legal_moves_for_piece(self, position, piece):
        """生成指定棋子的所有合法走法（包括兵、王、后、车、象、马的走法）"""
        moves = []
        i, j = position
        if piece.upper() == 'P':  # 兵的走法
            direction = -1 if piece.isupper() else 1
            if 0 <= i + direction < 8 and self.board[i + direction][j] == '.':
                moves.append(((i, j), (i + direction, j)))
            if i == 2 or i==8 and self.board[i + 2 * direction][j] == ".":
                moves.append(((i, j), (i + 2 * direction, j))) 
            # 吃子
            if j > 0 and self.board[i + direction][j - 1] != '.' and \
                    ((piece.isupper() and self.board[i + direction][j - 1].islower()) or
                     (piece.islower() and self.board[i + direction][j - 1].isupper())):
                moves.append(((i, j), (i + direction, j - 1)))
            if j < 7 and self.board[i + direction][j + 1] != '.' and \
                    ((piece.isupper() and self.board[i + direction][j + 1].islower()) or
                     (piece.islower() and self.board[i + direction][j + 1].isupper())):
                moves.append(((i, j), (i + direction, j + 1)))
        elif piece.upper() == 'N':  # 马的走法
            knight_moves = [
                (i - 2, j - 1), (i - 2, j + 1), (i + 2, j - 1), (i + 2, j + 1),
                (i - 1, j - 2), (i - 1, j + 2), (i + 1, j - 2), (i + 1, j + 2)
            ]
            for ni, nj in knight_moves:
                if 0 <= ni < 8 and 0 <= nj < 8 and (self.board[ni][nj] == '.' or
                                                     (piece.isupper() and self.board[ni][nj].islower()) or
                                                     (piece.islower() and self.board[ni][nj].isupper())):
                    moves.append(((i, j), (ni, nj)))
        elif piece.upper() == 'B':  # 象的走法
            moves += self.generate_sliding_moves((i, j), [(1, 1), (1, -1), (-1, 1), (-1, -1)])
        elif piece.upper() == 'R':  # 车的走法
            moves += self.generate_sliding_moves((i, j), [(1, 0), (-1, 0), (0, 1), (0, -1)])
        elif piece.upper() == 'Q':  # 后的走法
            moves += self.generate_sliding_moves((i, j), [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)])
        elif piece.upper() == 'K':  # 王的走法
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 8 and 0 <= nj < 8 and (self.board[ni][nj] == '.' or
                                                        (piece.isupper() and self.board[ni][nj].islower()) or
                                                        (piece.islower() and self.board[ni][nj].isupper())):
                        moves.append(((i, j), (ni, nj)))
        return moves

    def generate_sliding_moves(self, position, directions):
        """生成滑动棋子的所有走法（象、车、后）"""
        i, j = position
        moves = []
        for di, dj in directions:
            ni, nj = i + di, j + dj
            while 0 <= ni < 8 and 0 <= nj < 8:
                if self.board[ni][nj] == '.':
                    moves.append(((i, j), (ni, nj)))
                elif (self.board[i][j].isupper() and self.board[ni][nj].islower()) or \
                     (self.board[i][j].islower() and self.board[ni][nj].isupper()):
                    moves.append(((i, j), (ni, nj)))
                    break
                else:
                    break
                ni += di
                nj += dj
        return moves

    def make_move(self, move):
        """执行走法"""
        start, end = move
        piece = self.board[start[0]][start[1]]
        self.board[end[0]][end[1]] = piece
        self.board[start[0]][start[1]] = '.'

    def undo_move(self, move):
        """撤销走法"""
        start, end = move
        self.board[start[0]][start[1]] = self.board[end[0]][end[1]]
        self.board[end[0]][end[1]] = '.'

    def alpha_beta_search(self, depth, alpha, beta, maximizing_player):
        """α-β剪枝搜索算法"""
        if depth == 0 or self.is_checkmate():
            return self.evaluate_position()
        
        all_moves = self.generate_all_moves(self.turn)
        if maximizing_player:
            max_eval = float('-inf')
            for move in all_moves:
                self.make_move(move)
                eval = self.alpha_beta_search(depth - 1, alpha, beta, False)
                self.undo_move(move)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in all_moves:
                self.make_move(move)
                eval = self.alpha_beta_search(depth - 1, alpha, beta, True)
                self.undo_move(move)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move(self):
        """选择最佳走法"""
        best_move = None
        best_score = float('-inf') if self.turn == 'w' else float('inf')
        all_moves = self.generate_all_moves(self.turn)
        progress_bar = tqdm(total=len(all_moves), desc="搜索进度")
    
    # 增加变量用于记录杀棋
        checkmate_move = None

        for move in all_moves:
            self.make_move(move)
        
        # 检查对方是否无合法走法（杀棋情况）
            if self.is_checkmate():
                self.undo_move(move)
                checkmate_move = move
                break  # 如果有一步杀棋，直接退出循环

        # 正常的 alpha-beta 搜索
            score = self.alpha_beta_search(self.depth, float('-inf'), float('inf'), self.turn == 'b')
            self.undo_move(move)
        
        # 更新最佳走法
            adjusted_score = score + 3980 if self.turn == 'b' else score - 3980
            if (self.turn == 'w' and score > best_score) or (self.turn == 'b' and score < best_score):
                best_score = adjusted_score
                best_move = move
        
            progress_bar.update(1)
    
        progress_bar.close()

    # 如果有杀棋走法，优先执行杀棋
        if checkmate_move:
            best_score = M1
            best_move = checkmate_move
            return checkmate_move

        print(f"局面评估分数: {best_score}")
        return best_move

def convert_move_to_uci(move):
    """将棋子的走法转换为标准的UCI格式"""
    start, end = move
    start_file = chr(start[1] + ord('a'))
    start_rank = str(8 - start[0])
    end_file = chr(end[1] + ord('a'))
    end_rank = str(8 - end[0])
    return f"{start_file}{start_rank}{end_file}{end_rank}"

# 主函数
def main():
    engine = ChessEngine()
    engine.ask_for_fen()  # 询问用户输入FEN
    engine.print_board()  # 打印当前棋盘
    best_move = engine.get_best_move()  # 获取最佳走法
    uci_move = convert_move_to_uci(best_move)  # 转换为UCI格式
    print(f"最佳走法: {uci_move}")

if __name__ == '__main__':
    main()

#rnbqk2r/pppp1ppp/5n2/2b1p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4