# run_parallel_legality.py
import argparse
import os
import multiprocessing as mp
import pandas as pd
import subprocess


def worker(worker_index, num_workers, input_file, out_dir, max_classes):
    df = pd.read_parquet(input_file)

    if max_classes is not None:
        df = df.head(max_classes)

    total = len(df)
    chunk = (total + num_workers - 1) // num_workers  # evenly divide

    start = worker_index * chunk
    end = min(start + chunk, total)
    if start >= end:
        return  # no work for this worker

    out_file = os.path.join(out_dir, f"legality_{worker_index}.parquet")

    print(f"[Worker {worker_index}] Processing rows {start}:{end} -> {out_file}")

    cmd = [
        "python", "simulate_legality.py",
        "--input", input_file,
        "--start-id", str(start),
        "--end-id", str(end),
        "--output", out_file
    ]
    subprocess.run(cmd, check=True)


def merge(out_dir, final_output):
    parts = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".parquet")])
    print(f"Merging {len(parts)} files...")

    dfs = [pd.read_parquet(p) for p in parts]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_parquet(final_output, index=False)
    print(f"✅ Saved merged results to {final_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="legality_results.parquet")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-classes", type=int, default=None)
    args = parser.parse_args()

    out_dir = "legality_parts"
    os.makedirs(out_dir, exist_ok=True)

    pool = mp.Pool(args.workers)

    jobs = [
        pool.apply_async(worker, (w, args.workers, args.input, out_dir, args.max_classes))
        for w in range(args.workers)
    ]

    for j in jobs:
        j.wait()

    pool.close()
    pool.join()

    merge(out_dir, args.output)


if __name__ == "__main__":
    main()
