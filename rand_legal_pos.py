import chess
from chess import STATUS_VALID, STATUS_PAWNS_ON_BACKRANK
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


# --------------------------------------------------------
# 🔧 Erweiterte Board-Klasse
# --------------------------------------------------------
class CustomBoard(chess.Board):
    """Erweiterte Board-Klasse, die Bauern auf der Grundreihe erlaubt."""

    def is_valid_no_promotion(self) -> bool:
        """
        Prüft die Stellung auf interne Konsistenz, ignoriert aber den
        STATUS_PAWNS_ON_BACKRANK-Fehler (Bauern auf der 1./8. Reihe).
        """
        status = self.status()

        # STATUS_VALID == 0 → alles ok
        if status == STATUS_VALID:
            return True

        # Ignoriere nur den Backrank-Fehler
        if status & ~STATUS_PAWNS_ON_BACKRANK == STATUS_VALID:
            return True

        return False


# --------------------------------------------------------
# 🧠 Hauptfunktion: is_position_legal()
# --------------------------------------------------------
def is_position_legal(board: chess.Board, no_promotion: bool = True) -> bool:
    """
    Prüft, ob eine gegebene Stellung schachregelkonform ist.

    Args:
        board (chess.Board): Die zu prüfende Stellung.
        no_promotion (bool): Wenn True, wird STATUS_PAWNS_ON_BACKRANK ignoriert
                             und keine Promotions angenommen.

    Returns:
        bool: True, wenn die Stellung gültig ist, sonst False.
    """

    # 🔹 1️⃣ Grundcheck (mit/ohne Promotionregel)
    if no_promotion:
        if not isinstance(board, CustomBoard):
            board = CustomBoard(board.fen())
        if not board.is_valid_no_promotion():
            return False
    else:
        if not board.is_valid():
            return False

    # 🔹 2️⃣ Pawn-File-Check – pro Spalte max. 5 (bzw. 6 ohne Promotion) Bauern
    max_pawn_per_file = 6 if no_promotion else 5
    for color in [chess.WHITE, chess.BLACK]:
        pawns_by_file = [0] * 8
        for sq in board.pieces(chess.PAWN, color):
            pawns_by_file[chess.square_file(sq)] += 1
        if any(count > max_pawn_per_file for count in pawns_by_file):
            return False

    # 🔹 3️⃣ Bishop-Farbregel:
    #     - Immer prüfen, wenn no_promotion=True
    #     - Sonst nur, wenn alle 8 Bauern der Farbe noch vorhanden sind
    for color in [chess.WHITE, chess.BLACK]:
        bishops = list(board.pieces(chess.BISHOP, color))
        num_pawns = len(board.pieces(chess.PAWN, color))

        # Prüfen, ob Läuferregel gilt
        check_bishop_colors = no_promotion or num_pawns == 8
        if check_bishop_colors and len(bishops) >= 2:
            # True = hell, False = dunkel
            colors = [
                (chess.square_file(b) + chess.square_rank(b)) % 2 == 0
                for b in bishops
            ]
            # alle gleichfarbig → ungültig
            if all(colors) or not any(colors):
                return False

    # 🔹 4️⃣ Alles bestanden
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
