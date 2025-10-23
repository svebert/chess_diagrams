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


def generate_material_classes(
    limits: Optional[Dict[str, int]] = None
) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
    """
    Generate all distinct material classes (white/black piece combinations).

    Parameters
    ----------
    limits : dict of str -> int, optional
        Maximum allowed pieces per type. Defaults to standard chess piece limits.

    Returns
    -------
    list of tuple(dict, dict)
        A list of (white_materials, black_materials) tuples.
    """
    if limits is None:
        limits = {"K": 1, "Q": 1, "R": 2, "B": 2, "N": 2, "P": 8}

    def all_side_materials() -> List[Dict[str, int]]:
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
        if total <= 32 and w.values() <= 16 and b.values() <= 16:
            classes.append((w, b))
    return classes


def main(
    output_csv: str = "material_classes_diagrams.csv",
    output_parquet: str = "material_classes_diagrams.parquet",
) -> None:
    """
    Main routine: Generate material classes and store the number of possible diagrams.

    Parameters
    ----------
    output_csv : str, optional
        File path to save CSV output.
    output_parquet : str, optional
        File path to save Parquet output.

    Returns
    -------
    None
    """
    classes = generate_material_classes()
    print(f"→ {len(classes):,} Materialklassen erzeugt.")

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
            print(f"{i:,}/{len(classes):,} Klassen fertig …")

    df = pd.DataFrame(records)
    df["diagrams"] = df["diagrams"].astype(str)
    df.to_csv(output_csv, index=False)
    df.to_parquet(output_parquet, index=False)

    print(f"\n✅ Fertig gespeichert:\n   CSV → {output_csv}\n   Parquet → {output_parquet}")


if __name__ == "__main__":
    main()
