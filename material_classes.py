import argparse
import itertools
import pandas as pd
import math
import numpy as np
from typing import Dict, Tuple, List, Optional


# -------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------

def factorial_division(n: int, k: int) -> int:
    """Compute n! / (n - k)! efficiently using math.perm for performance."""
    return math.perm(n, k)


def count_diagrams(white: Dict[str, int], black: Dict[str, int]) -> int:
    """
    Calculate number of distinct diagrams possible for given material setup.
    """
    total = sum(white.values()) + sum(black.values())

    # Mehrfach vorkommende Figuren erfassen
    identical_counts = [
        count for color in (white, black) for count in color.values() if count > 1
    ]

    numerator = factorial_division(64, total)
    denominator = int(np.prod([math.factorial(c) for c in identical_counts])) if identical_counts else 1

    return numerator // denominator


# -------------------------------------------------------
# Materialkombinationen erzeugen
# -------------------------------------------------------

def generate_material_classes(
    limits: Optional[Dict[str, int]] = None,
    max_classes: Optional[int] = None,
    allow_promotions: bool = False,
) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
    """
    Generate all distinct material classes (white/black piece combinations).

    Parameters
    ----------
    limits : dict of str -> int, optional
        Maximum allowed pieces per type. Defaults to standard chess piece limits.
    max_classes : int, optional
        If provided, stops generation after this many material class combinations.
    allow_promotions : bool, optional
        If True, expands limits for promoted pieces.

    Returns
    -------
    list of tuple(dict, dict)
        A list of (white_materials, black_materials) tuples.
    """

    # Standardlimits
    if limits is None:
        limits = {"K": 1, "Q": 1, "R": 2, "B": 2, "N": 2, "P": 8}

    # Promotions erlauben → theoretisch bis zu 9 Damen/Türme/Läufer/Springer möglich
    if allow_promotions:
        limits = {"K": 1, "Q": 9, "R": 10, "B": 10, "N": 10, "P": 8}

    def all_side_materials() -> List[Dict[str, int]]:
        """Generate all valid material configurations for one side."""
        side_classes = []
        for q in range(limits["Q"] + 1):
            for r in range(limits["R"] + 1):
                for b in range(limits["B"] + 1):
                    for n in range(limits["N"] + 1):
                        for p in range(limits["P"] + 1):
                            side_classes.append(
                                {"K": 1, "Q": q, "R": r, "B": b, "N": n, "P": p}
                            )
        return side_classes

    white_materials = all_side_materials()
    black_materials = white_materials.copy()

    classes = []
    for w, b in itertools.product(white_materials, black_materials):
        total = sum(w.values()) + sum(b.values())
        if total <= 32:  # Grundbegrenzung
            classes.append((w, b))
            if max_classes is not None and len(classes) >= max_classes:
                break

    return classes


# -------------------------------------------------------
# Hauptfunktion
# -------------------------------------------------------

def main(
    output_csv: str = "material_classes_diagrams.csv",
    output_parquet: str = "material_classes_diagrams.parquet",
    max_classes: Optional[int] = None,
):
    """
    Generate and export material class diagrams to a single CSV and Parquet file.
    Both variants (no promotion / with promotion) are combined with a flag column.
    """

    all_records = []

    # ---- 1️⃣ Ohne Promotion ----
    print("⏳ Generating material classes (no promotion)...")
    classes_no_promo = generate_material_classes(max_classes=max_classes, allow_promotions=False)
    print(f"→ {len(classes_no_promo):,} material classes (no promotion)")

    for i, (w, b) in enumerate(classes_no_promo, start=1):
        total = sum(w.values()) + sum(b.values())
        diagrams = count_diagrams(w, b)
        all_records.append(
            {
                "id": i,
                "white": str(w),
                "black": str(b),
                "total_pieces": total,
                "diagrams": diagrams,
                "no_promotion": True,
            }
        )
        if i % 500 == 0:
            print(f"{i:,}/{len(classes_no_promo):,} no-promotion classes done …")

    # ---- 2️⃣ Mit Promotion ----
    print("\n⏳ Generating material classes (with promotion)...")
    classes_promo = generate_material_classes(max_classes=max_classes, allow_promotions=True)
    print(f"→ {len(classes_promo):,} material classes (with promotion)")

    for i, (w, b) in enumerate(classes_promo, start=1):
        total = sum(w.values()) + sum(b.values())
        diagrams = count_diagrams(w, b)
        all_records.append(
            {
                "id": i,
                "white": str(w),
                "black": str(b),
                "total_pieces": total,
                "diagrams": diagrams,
                "no_promotion": False,
            }
        )
        if i % 500 == 0:
            print(f"{i:,}/{len(classes_promo):,} promotion classes done …")

    # ---- 3️⃣ Zusammenführen & Speichern ----
    df = pd.DataFrame(all_records)
    df["diagrams"] = df["diagrams"].astype(str)

    df = df.sort_values(by=["no_promotion", "total_pieces", "id"]).reset_index(drop=True)

    df.to_csv(output_csv, index=False)
    df.to_parquet(output_parquet, index=False)

    print(f"\n✅ Combined files saved:")
    print(f"   CSV → {output_csv}")
    print(f"   PARQUET → {output_parquet}")
    

# -------------------------------------------------------
# CLI Entry
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chess material class diagrams.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="dist/material_classes_diagrams.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        default="dist/material_classes_diagrams.parquet",
        help="Path to output Parquet file.",
    )
    parser.add_argument(
        "--max_classes",
        type=int,
        default=None,
        help="Limit number of material classes to generate (for testing/performance).",
    )
    args = parser.parse_args()
    main(output_csv=args.output_csv, output_parquet=args.output_parquet, max_classes=args.max_classes)
