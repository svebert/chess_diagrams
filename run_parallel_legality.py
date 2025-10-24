import argparse
import os
import multiprocessing as mp
import pandas as pd

from simulate_legality import process_material_range


def worker(args):
    df, start, end, output_file = args
    try:
        process_material_range(df, start, end, output_file)
        return True, output_file
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-classes", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if args.max_classes is not None:
        df = df.head(args.max_classes)

    total = len(df)
    batch = (total + args.workers - 1) // args.workers

    out_dir = "legality_parts"
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for i in range(args.workers):
        start = i * batch
        end = min(start + batch, total)
        if start >= end: 
            continue
        out_file = os.path.join(out_dir, f"part_{i}.parquet")
        jobs.append((df, start, end, out_file))

    with mp.Pool(args.workers) as pool:
        results = pool.map(worker, jobs)

    # Merge
    dfs = []
    for ok, val in results:
        if ok:
            dfs.append(pd.read_parquet(val))
        else:
            print("❗ Worker failed:", val)

    final = pd.concat(dfs, ignore_index=True)
    final.to_parquet(args.output, index=False)
    print(f"✅ Saved merged results to {args.output}")


if __name__ == "__main__":
    main()
