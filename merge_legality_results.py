# merge_legality_results.py
import os
import argparse
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

def merge_parquets(input_dir: str, output_file: str):
    """
    Merge all parquet files from a directory (recursively) into a single file.

    Args:
        input_dir (str): Directory containing partial parquet files.
        output_file (str): Path to output combined parquet.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' not found.")
    
    # Recursively collect all .parquet files
    files = []
    for root, _, filenames in os.walk(input_dir):
        for f in filenames:
            if f.endswith(".parquet"):
                files.append(os.path.join(root, f))
    
    if not files:
        logging.warning(f"No parquet files found in {input_dir}")
        return
    
    logging.info(f"Merging {len(files)} parquet files from {input_dir} ...")
    dfs = [pd.read_parquet(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    combined.to_parquet(output_file, index=False)
    logging.info(f"✅ Merged {len(files)} files into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge partial legality result parquets.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with partial parquet files.")
    parser.add_argument("--output", type=str, default="dist/legality_results.parquet", help="Output parquet path.")
    args = parser.parse_args()
    
    merge_parquets(args.input_dir, args.output)
