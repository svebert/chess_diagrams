import argparse
import itertools
import pandas as pd
import math
import numpy as np
from typing import Dict, Tuple, List, Optional


def factorial_division(n: int, k: int) -> int:
    """
    Compute n! / (n - k)! efficiently using math.perm for performance.

    Parameters
    ----------
    n : int
        Total number of elements.
    k : int
        Number of selected elements.

    Returns
    -------
    int
        The result of n! / (n - k)!.

    Notes
    -----
    Uses math.perm (Python >= 3.8), which is implemented in C and much faster
    than a manual loop or NumPy product.
    """
    return math.perm(n, k)


def count_diagrams(white: Dict[str, int], black: Dict[str, int]) -> int:
    """
    Calculate the number of possible board diagrams for a given material setup.

    Parameters
    ----------
    white : dict of str -> int
        Dictionary representing the count of each piece type for White.
    black : dict of str -> int
        Dictionary representing the count of each piece type for Black.

    Returns
    -------
    int
        Number of distinct diagrams possible for the given material composition.
    """
    total = sum(white.values()) + sum(black.values())

    # Sammle alle mehrfach vorkommenden Figuren
    identical_counts = [
        count for color in (white, black) for count in color.values() if count > 1
    ]

    numerator = factorial_division(64, total)

    if identical_counts:
        # Nutzung von NumPy für effiziente Produktbildung auch bei großen Listen
        denominator = int(np.prod([math.factorial(c) for c in identical_counts]))
    else:
        denominator = 1

    return numerator // denominator
    
from typing import Dict, List, Optional, Tuple
import itertools


def generate_material_classes(
    limits: Optional[Dict[str, int]] = None,
    max_classes: Optional[int] = None,
) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
    """
    Generate all distinct material classes (white/black piece combinations).

    Parameters
    ----------
    limits : dict of str -> int, optional
        Maximum allowed pieces per type. Defaults to standard chess piece limits.
    max_classes : int, optional
        If provided, stops generation after this many material class combinations.
        Useful for testing or performance-limited environments.

    Returns
    -------
    list of tuple(dict, dict)
        A list of (white_materials, black_materials) tuples.
    """
    if limits is None:
        limits = {"K": 1, "Q": 1, "R": 2, "B": 2, "N": 2, "P": 8}

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
        # Limit total pieces on board
        if total <= 32 and len(w.values()) <=16 and len(b.values()) <= 16:
            classes.append((w, b))
            # Stop early if max_classes reached
            if max_classes is not None and len(classes) >= max_classes:
                break

    return classes


def main(
    output_csv: str = "material_classes_diagrams.csv",
    output_parquet: str = "material_classes_diagrams.parquet",
    max_classes: Optional[int] = None,
):
    """
    Generate and export material class diagrams to CSV and Parquet.

    Parameters
    ----------
    output_csv : str
        Path to output CSV file.
    output_parquet : str
        Path to output Parquet file.
    max_classes : int, optional
        Limit number of classes for testing/performance.
    """
    classes = generate_material_classes(max_classes=max_classes)
    print(f"→ {len(classes):,} material classes generated.")

    records = []
    for i, (w, b) in enumerate(classes, start=1):
        total = sum(w.values()) + sum(b.values())
        diagrams = count_diagrams(w, b)
        records.append(
            {
                "id": i,
                "white": str(w),
                "black": str(b),
                "total_pieces": total,
                "diagrams": diagrams,
            }
        )
        if i % 500 == 0:
            print(f"{i:,}/{len(classes):,} classes done …")

    df = pd.DataFrame(records)
    df["diagrams"] = df["diagrams"].astype(str)

    df.to_csv(output_csv, index=False)
    df.to_parquet(output_parquet, index=False)

    print(f"\n✅ Files saved to:\n   CSV → {output_csv}\n   Parquet → {output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chess material class diagrams.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="material_classes_diagrams.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        default="material_classes_diagrams.parquet",
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
