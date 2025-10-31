import argparse
import os
import multiprocessing as mp
import pandas as pd
import time
from datetime import datetime

from simulate_legality import process_material_range


def worker(args):
    df, start, end, output_file, wid = args
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏁 Worker {wid} started: rows {start}-{end}")
        process_material_range(df, start, end, output_file)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Worker {wid} finished: wrote {output_file}")
        return True, output_file
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Worker {wid} failed: {e}")
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
        jobs.append((df, start, end, out_file, i))

    print(f"\n🧩 Starting {len(jobs)} workers with {args.workers} parallel processes …\n")

    start_time = time.time()
    completed = 0
    total_jobs = len(jobs)

    # --- Fortschritt via apply_async ---
    def log_result(result):
        nonlocal completed
        ok, val = result
        completed += 1
        if ok:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🟢 Done ({completed}/{total_jobs}) → {val}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔴 Error ({completed}/{total_jobs}) → {val}")

    with mp.Pool(args.workers) as pool:
        results = [pool.apply_async(worker, (job,), callback=log_result) for job in jobs]

        # Monitoring Loop
        while any(not r.ready() for r in results):
            running = sum(1 for r in results if not r.ready())
            done = completed
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ {done}/{total_jobs} done, {running} still running …", end="\r")
            time.sleep(5)

        pool.close()
        pool.join()

    print("\n🧮 All workers finished. Merging results …")

    # Merge
    dfs = []
    for res in results:
        ok, val = res.get()
        if ok:
            dfs.append(pd.read_parquet(val))
        else:
            print("❗ Worker failed:", val)

    final = pd.concat(dfs, ignore_index=True)
    final.to_parquet(args.output, index=False)

    elapsed = time.time() - start_time
    print(f"\n✅ Saved merged results to {args.output}")
    print(f"⏱️  Total time: {elapsed/60:.2f} minutes\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
