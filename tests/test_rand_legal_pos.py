import chess
from rand_legal_pos import random_board_from_material, is_position_legal


# -------------------------------------------------------
# 🧩 Struktur-Test für Random-Funktion
# -------------------------------------------------------
def test_random_board_from_material_structure():
    white = {"K": 1, "Q": 1, "R": 0, "B": 0, "N": 0, "P": 0}
    black = {"K": 1, "Q": 0, "R": 2, "B": 0, "N": 0, "P": 0}
    board = random_board_from_material(white, black)
    assert isinstance(board, chess.Board)
    assert len(list(board.piece_map().values())) == sum(white.values()) + sum(black.values())


# -------------------------------------------------------
# 🧩 Pawn-File-Regel
# -------------------------------------------------------
def test_pawn_file_rule_no_promotion_true():
    """
    Bei no_promotion=True dürfen max. 6 Bauern pro Spalte stehen.
    """
    board = chess.Board(None)
    pawn_squares = [chess.A2, chess.A3, chess.A4, chess.A5, chess.A6, chess.A7]
    for sq in pawn_squares:
        board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    assert not is_position_legal(board, no_promotion=True), \
        "Mehr als 6 Bauern in einer Spalte sollten bei no_promotion=True illegal sein."


def test_pawn_file_rule_with_promotion_allowed():
    """
    Bei no_promotion=False dürfen max. 5 Bauern pro Spalte stehen.
    """
    board = chess.Board(None)
    pawn_squares = [chess.A2, chess.A3, chess.A4, chess.A5, chess.A6]
    for sq in pawn_squares:
        board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    assert not is_position_legal(board, no_promotion=False), \
        "Mehr als 5 Bauern in einer Spalte sollten bei regulärem Schach illegal sein."


# -------------------------------------------------------
# 🧩 Pawn-Rank-Regel (1. und 8. Reihe)
# -------------------------------------------------------
def test_pawn_on_backrank_no_promotion_false():
    """
    Bei no_promotion=False darf kein Bauer auf der 1. oder 8. Reihe stehen.
    """
    board = chess.Board(None)
    board.set_piece_at(chess.A8, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    assert not is_position_legal(board, no_promotion=False), \
        "Bauer auf der 8. Reihe sollte im echten Schach illegal sein."


def test_pawn_on_backrank_no_promotion_true():
    """
    Bei no_promotion=True wird STATUS_PAWNS_ON_BACKRANK ignoriert.
    """
    board = chess.Board(None)
    board.set_piece_at(chess.H1, chess.Piece(chess.PAWN, chess.BLACK))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    assert is_position_legal(board, no_promotion=True), \
        "Bauer auf der 1./8. Reihe darf bei no_promotion=True toleriert werden."


# -------------------------------------------------------
# 🧩 Bishop-Farbenregel
# -------------------------------------------------------
def test_bishop_light_color_rule_no_promotion_true():
    """
    Zwei Läufer gleicher Farbe sind illegal, wenn keine Promotion erlaubt ist.
    """
    board = chess.Board(None)
    board.set_piece_at(chess.B3, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.F5, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    assert not is_position_legal(board, no_promotion=True)


def test_bishop_dark_color_rule_no_promotion_true():
    """
    Zwei schwarze Läufer gleicher Farbe → illegal bei no_promotion=True.
    """
    board = chess.Board(None)
    board.set_piece_at(chess.B4, chess.Piece(chess.BISHOP, chess.BLACK))
    board.set_piece_at(chess.F6, chess.Piece(chess.BISHOP, chess.BLACK))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    assert not is_position_legal(board, no_promotion=True)


def test_bishop_rule_with_all_pawns_present():
    """
    Selbst bei no_promotion=False gilt die Läuferregel, wenn noch 8 Bauern vorhanden sind.
    """
    board = chess.Board(None)
    for file in range(8):
        board.set_piece_at(chess.square(file, 1), chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.C1, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.E3, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    assert not is_position_legal(board, no_promotion=False), \
        "Mit 8 Bauern muss die Läufer-Farbregel gelten, auch wenn Promotion erlaubt ist."


def test_bishop_rule_with_missing_pawn():
    """
    Bei no_promotion=False und <8 Bauern darf die Läufer-Farbregel verletzt werden.
    """
    board = chess.Board(None)
    for file in range(7):  # Nur 7 Bauern
        board.set_piece_at(chess.square(file, 1), chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.C1, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.E3, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    assert is_position_legal(board, no_promotion=False), \
        "Mit weniger als 8 Bauern darf es gleiche Läuferfarben geben (Promotion möglich)."


# -------------------------------------------------------
# 🧩 Minimal ungültige Stellungen
# -------------------------------------------------------
def test_simple_invalid_missing_king():
    board = chess.Board(None)
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    assert not is_position_legal(board, no_promotion=False)
