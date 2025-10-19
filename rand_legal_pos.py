import chess
import random
def random_board_from_material(white, black):
    """ Erzeugt ein zufälliges Schachbrett 
    (chess.Board) aus einer gegebenen 
    Materialkombination. white / black sind Dicts 
    mit Anzahl pro Figurtyp, z. B. {"K":1, "Q":1, 
    "R":0, "B":0, "N":0, "P":0} """
    # Kopiere alle freien Felder
    all_squares = list(chess.SQUARES) 
    random.shuffle(all_squares)
    board = chess.Board(None) # leeres Brett 
    pieces_placed = 0
    # Helper: Figurtypen-Map
    piece_map = { "K": chess.KING, "Q": chess.QUEEN, 
        "R": chess.ROOK, "B": chess.BISHOP, "N": 
        chess.KNIGHT, "P": chess.PAWN
    }
    # Erst Weiß platzieren
    for piece_type, count in white.items():
        for _ in range(count):
            if not all_squares:
                break
            square = all_squares.pop() 
            board.set_piece_at(square, 
            chess.Piece(piece_map[piece_type], 
            chess.WHITE))
            pieces_placed += 1
    # Dann Schwarz platzieren
    for piece_type, count in black.items(): 
        for _ in range(count):
            if not all_squares: 
                break 
            square =  all_squares.pop() 
            board.set_piece_at(square, 
            chess.Piece(piece_map[piece_type], 
            chess.BLACK)) 
            pieces_placed += 1
    return board
def is_position_legal(board):
    """ 
    Prüft, ob eine zufällig erzeugte Stellung legal 
    ist (laut Schachregeln erreichbar).
    """
    return board.is_valid()
# Beispielnutzung
if __name__ == "__main__": 
    white = {"K": 1, "Q": 1, 
    "R": 0, "B": 0, "N": 0, "P": 0} 
    black = {"K": 1, 
    "Q": 0, "R": 2, "B": 0, "N": 0, "P": 0}
    trials =1000
    legal = 0 
    for _ in range(trials):
        b = random_board_from_material(white, black) 
        if is_position_legal(b):
            legal += 1
    print(f"Von {trials} zufälligen Stellungen waren {legal} legal ({legal/trials:.4%})")

