# run_parallel_legality.py
import argparse
import os
import multiprocessing as mp
import pandas as pd
import subprocess

def run_shard(shard_index, total_shards, input_file, out_dir, max_classes):
    df = pd.read_parquet(input_file)

    # Apply max_classes if provided
    if max_classes is not None:
        df = df.head(max_classes)

    total = len(df)
    per = (total + total_shards - 1) // total_shards  # ceil division

    start = shard_index * per
    end = min(start + per, total)

    out_file = os.path.join(out_dir, f"results_{shard_index}.parquet")

    print(f"[Shard {shard_index}] Processing classes {start}-{end} → {out_file}")

    cmd = [
        "python", "simulate_legality.py",
        "--input", input_file,
        "--start-id", str(start),
        "--end-id", str(end),
        "--output", out_file
    ]
    subprocess.run(cmd, check=True)


def merge_results(out_dir, final_output):
    parts = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".parquet")]
    print(f"Merging {len(parts)} shards...")

    dfs = [pd.read_parquet(p) for p in parts]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_parquet(final_output, index=False)

    print(f"✅ Saved merged results to: {final_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="material_classes.parquet")
    parser.add_argument("--output", default="legality_results.parquet")
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-classes", type=int, default=None,
                        help="Limit number of classes processed (useful for testing)")
    args = parser.parse_args()

    out_dir = "legality_parts"
    os.makedirs(out_dir, exist_ok=True)

    workers = args.workers or args.shards
    pool = mp.Pool(workers)

    jobs = []
    for shard in range(args.shards):
        jobs.append(pool.apply_async(
            run_shard,
            (shard, args.shards, args.input, out_dir, args.max_classes)
        ))

    for j in jobs:
        j.wait()

    pool.close()
    pool.join()

    merge_results(out_dir, args.output)


if __name__ == "__main__":
    main()
