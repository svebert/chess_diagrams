import chess
import random
from typing import Dict

def random_board_from_material(white: Dict[str, int], black: Dict[str, int]) -> chess.Board:
    """
    Generate a random chess board from given material composition.

    Args:
        white (dict): Mapping of white piece counts, e.g. {"K":1, "Q":1, "R":0, "B":0, "N":0, "P":0}
        black (dict): Mapping of black piece counts, same format.

    Returns:
        chess.Board: A chess board object with pieces randomly placed.
    """
    # Copy all available squares and shuffle randomly
    all_squares = list(chess.SQUARES)
    random.shuffle(all_squares)
    board = chess.Board(None)  # start with empty board

    piece_map = {
        "K": chess.KING,
        "Q": chess.QUEEN,
        "R": chess.ROOK,
        "B": chess.BISHOP,
        "N": chess.KNIGHT,
        "P": chess.PAWN,
    }

    # Place white pieces
    for piece_type, count in white.items():
        for _ in range(count):
            if not all_squares:
                break
            square = all_squares.pop()
            board.set_piece_at(square, chess.Piece(piece_map[piece_type], chess.WHITE))

    # Place black pieces
    for piece_type, count in black.items():
        for _ in range(count):
            if not all_squares:
                break
            square = all_squares.pop()
            board.set_piece_at(square, chess.Piece(piece_map[piece_type], chess.BLACK))

    return board


def is_position_legal(board: chess.Board, no_promotion: bool = True) -> bool:
    """
    Check if a random position is legal according to chess rules.

    Args:
        board (chess.Board): The board to check.
        no_promotion (bool): If True, bishop color diversity is enforced for each side.

    Returns:
        bool: True if the position is legal, False otherwise.
    """
    # Basic validity check
    if not board.is_valid():
        return False

    # 1️⃣ Pawn column check: no more than 4 pawns in any file per color
    for color in [chess.WHITE, chess.BLACK]:
        pawns_by_file = [0] * 8
        for square in board.pieces(chess.PAWN, color):
            file_idx = chess.square_file(square)
            pawns_by_file[file_idx] += 1
        if any(count >= 5 for count in pawns_by_file):
            return False

    # 2️⃣ Bishop color check (only if no_promotion=True)
    if no_promotion:
        for color in [chess.WHITE, chess.BLACK]:
            bishops = list(board.pieces(chess.BISHOP, color))
            if len(bishops) >= 2:
                # compute colors of squares: light=True, dark=False
                colors = [((chess.square_file(b) + chess.square_rank(b)) % 2 == 0) for b in bishops]
                # if all bishops on same color → invalid
                if all(colors) or not any(colors):
                    return False

    return True


# Example usage
if __name__ == "__main__":
    white = {"K": 1, "Q": 1, "R": 0, "B": 0, "N": 0, "P": 0}
    black = {"K": 1, "Q": 0, "R": 2, "B": 0, "N": 0, "P": 0}

    trials = 1000
    legal = 0

    for _ in range(trials):
        b = random_board_from_material(white, black)
        if is_position_legal(b):
            legal += 1

    print(f"Out of {trials} random boards, {legal} were legal ({legal / trials:.4%})")
