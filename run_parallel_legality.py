def worker(worker_index, num_workers, input_file, out_dir, max_classes):
    try:
        print(f"[Worker {worker_index}] Loading dataframe from {input_file}")
        df = pd.read_parquet(input_file)
    except Exception as e:
        print(f"❌ [Worker {worker_index}] FAILED to open input file: {e}")
        return

    try:
        if max_classes is not None:
            df = df.head(max_classes)

        total = len(df)
        if total == 0:
            print(f"⚠️ [Worker {worker_index}] No rows to process (total=0).")
            return

        chunk = (total + num_workers - 1) // num_workers
        start = worker_index * chunk
        end = min(start + chunk, total)
        if start >= end:
            print(f"⚠️ [Worker {worker_index}] No assigned range.")
            return

        out_file = os.path.join(out_dir, f"legality_{worker_index}.parquet")
        print(f"[Worker {worker_index}] Processing rows {start}:{end} -> {out_file}")

        cmd = [
            "python", "simulate_legality.py",
            "--input", input_file,
            "--start-id", str(start),
            "--end-id", str(end),
            "--output", out_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ [Worker {worker_index}] simulate_legality.py FAILED")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
        else:
            print(f"✅ [Worker {worker_index}] Finished successfully.")

    except Exception as e:
        print(f"❌ [Worker {worker_index}] ERROR: {e}")
