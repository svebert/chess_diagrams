import argparse
import os
import pandas as pd
import subprocess
import multiprocessing as mp


def run_worker(worker_index, total_workers, input_file, out_dir, max_classes):
    df = pd.read_parquet(input_file)

    if max_classes is not None:
        df = df.head(max_classes)

    total = len(df)
    chunk = (total + total_workers - 1) // total_workers  # ceil

    start = worker_index * chunk
    end = min(start + chunk, total)

    if start >= end:
        print(f"[Worker {worker_index}] nothing to do")
        return

    out_file = os.path.join(out_dir, f"results_{worker_index}.parquet")

    print(f"[Worker {worker_index}] Processing {start} → {end}  -> {out_file}")

    cmd = [
        "python", "simulate_legality.py",
        "--input", input_file,
        "--start-id", str(start),
        "--end-id", str(end),
        "--output", out_file,
    ]

    subprocess.run(cmd, check=True)


def merge(out_dir, output_file):
    files = sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.endswith(".parquet")
    )

    print(f"Merging {len(files)} files...")

    dfs = [pd.read_parquet(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_parquet(output_file, index=False)

    print(f"✅ Done. Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="material_classes.parquet")
    parser.add_argument("--output", default="legality_results.parquet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-classes", type=int, default=None)
    args = parser.parse_args()

    out_dir = "legality_parts"
    os.makedirs(out_dir, exist_ok=True)

    pool = mp.Pool(args.workers)

    jobs = [
        pool.apply_async(run_worker, (i, args.workers, args.input, out_dir, args.max_classes))
        for i in range(args.workers)
    ]

    for j in jobs:
        j.wait()

    pool.close()
    pool.join()

    merge(out_dir, args.output)


if __name__ == "__main__":
    main()
