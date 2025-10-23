import math
import pytest
from material_classes import count_diagrams


def test_basic_case_distinct():
    """
    Two distinct white pieces (K, Q) and two distinct black pieces (K, Q).
    No identical pieces, so no divisor.
    Expected: math.perm(64, 4)
    """
    white = {"K": 1, "Q": 1}
    black = {"K": 1, "Q": 1}
    result = count_diagrams(white, black)
    expected = math.perm(64, 4)
    assert result == expected


def test_identical_within_one_color():
    """
    White has 2 rooks (identical), black has 1 king.
    Total of 4 pieces; identical rooks lead to division by 2!.
    Expected: P(64, 4) / 2!
    """
    white = {"K": 1, "R": 2}
    black = {"K": 1}
    result = count_diagrams(white, black)
    expected = math.perm(64, 4) // math.factorial(2)
    assert result == expected


def test_multiple_identical_groups():
    """
    White: 2 rooks (identical), Black: 2 knights (identical).
    Two groups of identical pieces => divide by 2! * 2!.
    Expected: P(64, 6) / (2! * 2!)
    """
    white = {"K": 1, "R": 2}
    black = {"K": 1, "N": 2}
    result = count_diagrams(white, black)
    expected = math.perm(64, 6) // (math.factorial(2) * math.factorial(2))
    assert result == expected


def test_large_numbers_positive():
    """
    Large configuration (8 pawns per side).
    Should produce a positive integer without overflow.
    """
    white = {"K": 1, "P": 8}
    black = {"K": 1, "P": 8}
    result = count_diagrams(white, black)
    assert isinstance(result, int)
    assert result > 0


def test_extremely_large_input_completes():
    """
    Maximal realistic material case (standard limits).
    The computation should complete and return a positive integer.
    """
    white = {"K": 1, "Q": 1, "R": 2, "B": 2, "N": 2, "P": 8}
    black = {"K": 1, "Q": 1, "R": 2, "B": 2, "N": 2, "P": 8}
    result = count_diagrams(white, black)
    assert isinstance(result, int)
    assert result > 0


def test_known_value_small_case():
    """
    Simple sanity check: placing 3 distinct pieces on 64 squares.
    Expected P(64, 3) = 64 * 63 * 62 = 249,984
    """
    white = {"K": 1, "Q": 1}
    black = {"K": 1}
    result = count_diagrams(white, black)
    assert result == 64 * 63 * 62  # 249,984


def test_large_consistency_with_math_perm():
    """
    For a large total number of distinct pieces (e.g. 10),
    count_diagrams should exactly match math.perm(64, 10)
    when there are no identical pieces.
    """
    white = {"K": 1, "Q": 1, "R": 2, "B": 2, "N": 2, "P": 2}  # total = 10
    black = {}
    total = sum(white.values())
    result = count_diagrams(white, black)
    expected = math.perm(64, total)
    assert result == expected
