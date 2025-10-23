# simulate_legality.py
import os
import time
import argparse
import logging
import pandas as pd
from tqdm import tqdm

from rand_legal_pos import random_board_from_material, is_position_legal
from material_classes import generate_material_classes, count_diagrams

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# === Config ===
INITIAL_SAMPLES = [1000, 2000]
MAX_SAMPLE = 128_000
REL_STD_THRESHOLD = 0.05  # stop when relative std < 5%


def test_legality_for_class(white_material: dict, black_material: dict, sample_size: int, no_promotion: bool = True):
    """
    Estimate the fraction of legal diagrams and its standard error.
    """
    legal = 0
    for _ in range(sample_size):
        try:
            board = random_board_from_material(white_material, black_material)
            if is_position_legal(board, no_promotion=no_promotion):
                legal += 1
        except Exception:
            continue

    ratio = legal / sample_size if sample_size > 0 else 0.0
    std_error = (ratio * (1 - ratio) / sample_size) ** 0.5 if sample_size > 0 else 0.0
    return ratio, std_error


def process_material_range(df: pd.DataFrame, start: int, end: int, output_file: str):
    """
    Process a range of material classes sequentially and store legality ratios with standard errors.
    """
    results = []
    slice_df = df.iloc[start:end]
    for _, row in tqdm(slice_df.iterrows(), total=len(slice_df), desc=f"Range {start}-{end}"):
        class_id = int(row["id"])
        white_material = eval(row["white"])
        black_material = eval(row["black"])

        class_rows = []

        # Adaptive sampling loop
        sample_size = INITIAL_SAMPLES[0]
        while True:
            ratio, std_error = test_legality_for_class(white_material, black_material, sample_size)
            rel_std = std_error / ratio if ratio > 0 else 1.0

            logging.info(f"Class {class_id}, sample {sample_size}: {ratio:.4e} legal, rel_std={rel_std:.4f}")

            class_rows.append({
                "id": class_id,
                "sample_size": sample_size,
                "legal_ratio": ratio,
                "std_error": std_error,
                "rel_std": rel_std
            })

            if rel_std < REL_STD_THRESHOLD or sample_size >= MAX_SAMPLE:
                break

            sample_size = min(sample_size * 2, MAX_SAMPLE)
            if any(r["sample_size"] == sample_size for r in class_rows):
                break  # prevent infinite loop if MAX_SAMPLE reached

        results.extend(class_rows)

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    out_df.to_parquet(output_file, index=False)
    logging.info(f"Saved {len(out_df)} rows to {output_file}")


def main():
    """
    CLI entry point for legality simulation.
    """
    parser = argparse.ArgumentParser(description="Simulate legality ratios for chess material diagrams.")
    parser.add_argument("--start-id", type=int, default=0, help="Start index (inclusive).")
    parser.add_argument("--end-id", type=int, help="End index (exclusive). Default=all.")
    parser.add_argument("--input", type=str, default="material_classes.parquet", help="Material parquet input.")
    parser.add_argument("--output", type=str, default="legality_results.parquet", help="Output parquet file.")
    parser.add_argument("--no-promotion", action="store_true", help="Apply stricter bishop/pawn legality rules.")
    args = parser.parse_args()

    logging.info(f"Loading material classes from {args.input}")
    df = pd.read_parquet(args.input)
    df = df.sort_values("diagrams", ascending=False).reset_index(drop=True)
    total = len(df)
    logging.info(f"{total} classes loaded.")

    start = args.start_id
    end = args.end_id if args.end_id is not None else total
    if start < 0 or end > total or start >= end:
        raise ValueError("Invalid start/end range.")

    logging.info(f"Processing range {start}..{end} (total {end-start} classes)")
    process_material_range(df, start, end, args.output)


if __name__ == "__main__":
    main()
