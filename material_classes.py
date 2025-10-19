import itertools
import pandas as pd
import math


def factorial_division(n, k):
    """Hilfsfunktion: n! / (n-k)! effizient berechnen."""
    result = 1
    for i in range(n - k + 1, n + 1):
        result *= i
    return result


def count_positions(white, black):
    """Berechnet Anzahl möglicher Feldbelegungen für gegebene Materialklasse."""
    total = sum(white.values()) + sum(black.values())
    identical_counts = []
    for color in (white, black):
        for piece, count in color.items():
            if count > 1:
                identical_counts.append(count)

    numerator = factorial_division(64, total)
    denominator = math.prod(math.factorial(c) for c in identical_counts) if identical_counts else 1
    return numerator // denominator


def generate_material_classes():
    """Materialklassen ohne Promotionen."""
    limits = {"K": 1, "Q": 1, "R": 2, "B": 2, "N": 2, "P": 8}

    def all_side_materials():
        side_classes = []
        for q in range(limits["Q"] + 1):
            for r in range(limits["R"] + 1):
                for b in range(limits["B"] + 1):
                    for n in range(limits["N"] + 1):
                        for p in range(limits["P"] + 1):
                            side_classes.append({"K": 1, "Q": q, "R": r, "B": b, "N": n, "P": p})
        return side_classes

    white_materials = all_side_materials()
    black_materials = all_side_materials()

    classes = []
    for w, b in itertools.product(white_materials, black_materials):
        total = sum(w.values()) + sum(b.values())
        if total <= 32:
            classes.append((w, b))
    return classes


def main(output_csv="material_classes_positions.csv", output_parquet="material_classes_positions.parquet"):
    classes = generate_material_classes()
    print(f"→ {len(classes):,} Materialklassen erzeugt.")

    records = []
    for i, (w, b) in enumerate(classes, start=1):
        total = sum(w.values()) + sum(b.values())
        positions = count_positions(w, b)
        records.append({
            "id": i,
            "white": str(w),
            "black": str(b),
            "total_pieces": total,
            "positions": positions
        })
        if i % 500 == 0:
            print(f"{i:,}/{len(classes):,} Klassen fertig …")

    df = pd.DataFrame(records)
    # Große Zahlen als String konvertieren
    df["positions"] = df["positions"].astype(str)
    df.to_csv(output_csv, index=False)
    df.to_parquet(output_parquet, index=False)

    print(f"\n✅ Fertig gespeichert:\n   CSV → {output_csv}\n   Parquet → {output_parquet}")


if __name__ == "__main__":
    main()
