import math
import pytest
import os
import pandas as pd
import tempfile
from material_classes import count_diagrams, generate_material_classes, main

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
    when there are NO identical pieces (all counts <= 1).
    """
    white = {"K": 1, "Q": 1, "R": 1, "B": 1, "N": 1}
    black = {"K": 1, "Q": 1, "R": 1, "B": 1, "N": 1}
    total = sum(white.values()) + sum(black.values())  # = 10
    result = count_diagrams(white, black)
    expected = math.perm(64, total)
    assert result == expected


def test_default_limits_structure():
    """
    The function should return a list of tuples (white_dict, black_dict).
    Each side must contain standard chess piece keys.
    """
    classes = generate_material_classes(max_classes=1)
    assert isinstance(classes, list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in classes)

    w, b = classes[0]
    expected_keys = {"K", "Q", "R", "B", "N", "P"}
    assert expected_keys.issubset(w.keys())
    assert expected_keys.issubset(b.keys())


def test_respects_piece_limits():
    """
    Each generated material class must not exceed the per-piece limits
    (e.g., no more than 8 pawns, 2 rooks, etc.).
    """
    limits = {"K": 1, "Q": 1, "R": 2, "B": 2, "N": 2, "P": 8}
    classes = generate_material_classes(limits, max_classes=10000)
    for w, b in classes:
        for piece, max_count in limits.items():
            assert 0 <= w[piece] <= max_count
            assert 0 <= b[piece] <= max_count


def test_total_piece_count_constraint():
    """
    Total number of pieces (white + black) must not exceed 32.
    Each side must not exceed 16 pieces.
    """
    classes = generate_material_classes(max_classes=10000)
    for w, b in classes:
        total_w = sum(w.values())
        total_b = sum(b.values())
        assert total_w <= 16
        assert total_b <= 16
        assert total_w + total_b <= 32


def test_custom_limits_small_case():
    """
    For smaller custom limits, number of generated classes should shrink drastically.
    """
    small_limits = {"K": 1, "Q": 1, "R": 1, "B": 0, "N": 0, "P": 1}
    classes = generate_material_classes(small_limits)
    # possible combinations: Q(0–1), R(0–1), P(0–1) per side => 2×2×2=8 per side -> 64 total
    assert len(classes) == 64
    for w, b in classes:
        assert sum(w.values()) + sum(b.values()) <= 32


def test_main_creates_output_files(tmp_path):
    """
    Integration test for main():
    Ensures that CSV and Parquet files are created correctly and contain data.
    """
    csv_path = tmp_path / "classes_test.csv"
    parquet_path = tmp_path / "classes_test.parquet"

    # Run main() with a small dataset for performance
    main(output_csv=str(csv_path), output_parquet=str(parquet_path), max_classes=5)

    # ✅ Check that files exist
    assert csv_path.exists(), "CSV file was not created"
    assert parquet_path.exists(), "Parquet file was not created"

    # ✅ Check that CSV and Parquet can be read and are not empty
    df_csv = pd.read_csv(csv_path)
    df_parquet = pd.read_parquet(parquet_path)

    assert not df_csv.empty, "CSV file is empty"
    assert not df_parquet.empty, "Parquet file is empty"

    # ✅ Check for expected columns
    expected_columns = {"id", "white", "black", "total_pieces", "diagrams"}
    assert expected_columns.issubset(df_csv.columns), "CSV missing expected columns"
    assert expected_columns.issubset(df_parquet.columns), "Parquet missing expected columns"

    # ✅ Check consistent row count
    assert len(df_csv) == len(df_parquet), "Row count mismatch between CSV and Parquet"
