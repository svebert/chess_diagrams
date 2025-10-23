# simulate_legality.py
import os
import time
import argparse
import logging
import pandas as pd
from tqdm import tqdm

from rand_legal_pos import random_board_from_material
from material_classes import generate_material_classes, count_diagrams

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# Defaults / thresholds
INITIAL_SAMPLES = [1000, 2000]
MAX_SAMPLE = 128_000
REL_THRESHOLD = 0.10
ABS_THRESHOLD = 1e-6


def test_legality_for_class(white_material, black_material, sample_size: int) -> float:
    """Return fraction of generated random boards that are legal (board.is_valid())."""
    legal = 0
    for _ in range(sample_size):
        try:
            board = random_board_from_material(white_material, black_material)
            if board.is_valid():
                legal += 1
        except Exception:
            continue
    return legal / sample_size if sample_size > 0 else 0.0


def process_material_range(df: pd.DataFrame, start: int, end: int, output_file: str):
    """Process a range of material classes sequentially and write results to parquet."""
    results = []
    slice_df = df.iloc[start:end]
    for _, row in tqdm(slice_df.iterrows(), total=len(slice_df), desc=f"Range {start}-{end}"):
        class_id = int(row["id"])
        white_material = eval(row["white"])
        black_material = eval(row["black"])

        # initial samples
        class_rows = []
        for sample_size in INITIAL_SAMPLES:
            start_t = time.time()
            ratio = test_legality_for_class(white_material, black_material, sample_size)
            duration = time.time() - start_t
            logging.info(f"Class {class_id}, sample {sample_size}: {ratio:.4e} legal ({duration:.1f}s)")
            class_rows.append({"id": class_id, "sample_size": sample_size, "legal_ratio": ratio})

        # adaptive extra sample if necessary
        if len(class_rows) >= 2:
            last_two = sorted(class_rows, key=lambda r: r["sample_size"], reverse=True)[:2]
            ratio_max = last_two[0]["legal_ratio"]
            ratio_second = last_two[1]["legal_ratio"]
            rel_diff = abs(ratio_max - ratio_second) / max(abs(ratio_second), 1e-12)
            abs_diff = abs(ratio_max - ratio_second)
            if rel_diff > REL_THRESHOLD and abs_diff > ABS_THRESHOLD:
                next_sample = min(int(last_two[0]["sample_size"] * 2), MAX_SAMPLE)
                ratio = test_legality_for_class(white_material, black_material, next_sample)
                class_rows.append({"id": class_id, "sample_size": next_sample, "legal_ratio": ratio})

        results.extend(class_rows)

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    out_df.to_parquet(output_file, index=False)
    logging.info(f"Saved {len(out_df)} rows to {output_file}")


def merge_parquets(input_dir: str, output_file: str):
    """Merge all parquet files from a directory into one file."""
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".parquet")])
    if not files:
        logging.warning(f"No parquet files found in {input_dir}")
        return
    dfs = [pd.read_parquet(os.path.join(input_dir, f)) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(output_file, index=False)
    logging.info(f"Merged {len(files)} parquet files into {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Simulate legality ratios for chess material diagrams.")
    parser.add_argument("--start-id", type=int, default=0, help="Start index (inclusive).")
    parser.add_argument("--end-id", type=int, help="End index (exclusive). Default=all.")
    parser.add_argument("--input", type=str, default="material_classes_diagrams.parquet", help="Material parquet input.")
    parser.add_argument("--output", type=str, default="legality_results.parquet", help="Output parquet file.")
    parser.add_argument("--merge-dir", type=str, help="If set, merge all parquets from this dir into --output.")
    args = parser.parse_args()

    if args.merge_dir:
        merge_parquets(args.merge_dir, args.output)
        return

    logging.info(f"Loading material classes from {args.input}")
    df = pd.read_parquet(args.input)
    df = df.sort_values("diagram_count", ascending=False).reset_index(drop=True)
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
