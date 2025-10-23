import chess
from rand_legal_pos import random_board_from_material, is_position_legal

def test_random_board_from_material_structure():
    white = {"K": 1, "Q": 1, "R": 0, "B": 0, "N": 0, "P": 0}
    black = {"K": 1, "Q": 0, "R": 2, "B": 0, "N": 0, "P": 0}
    board = random_board_from_material(white, black)
    assert isinstance(board, chess.Board)
    assert len(list(board.piece_map().values())) == sum(white.values()) + sum(black.values())

def test_pawn_file_rule():
    board = chess.Board(None)
    # Place 5 white pawns on file a
    for rank in range(5):
        board.set_piece_at(chess.square(0, rank), chess.Piece(chess.PAWN, chess.WHITE))
    assert not is_position_legal(board)

def test_bishop_light_color_rule_no_promotion_true():
    board = chess.Board(None)
    # Two bishops on same color (both light squares)
    board.set_piece_at(chess.B3, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.F5, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    assert not is_position_legal(board, no_promotion=True)

def test_bishop_dark_color_rule_no_promotion_true():
    board = chess.Board(None)
    # Two bishops on same color (both light squares)
    board.set_piece_at(chess.B4, chess.Piece(chess.BISHOP, chess.BLACK))
    board.set_piece_at(chess.F6, chess.Piece(chess.BISHOP, chess.BLACK))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    assert not is_position_legal(board, no_promotion=True)

def test_bishop_color_rule_no_promotion_false():
    board = chess.Board(None)
    # Two bishops on same color, but no_promotion=False should ignore
    board.set_piece_at(chess.B3, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.F5, chess.Piece(chess.BISHOP, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    assert is_position_legal(board, no_promotion=False)
