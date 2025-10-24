#!/usr/bin/env python3
"""
Compute legality estimates for chess material classes.

Inputs:
 - material_classes_diagrams.parquet
      Columns: id, white, black, diagrams
 - legality_results.parquet
      Columns: id, sample_size, legal_ratio

Outputs:
 - legality_analysis.parquet (one row per class, see columns below)
 - Printed global statistics
"""

import pandas as pd
import numpy as np
import math
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MATERIAL_FILE = "material_classes_diagrams.parquet"
RESULT_FILE = "legality_results.parquet"
OUTPUT_FILE = "legality_analysis.parquet"


def main():

    if not os.path.exists(MATERIAL_FILE):
        raise FileNotFoundError(MATERIAL_FILE)
    if not os.path.exists(RESULT_FILE):
        raise FileNotFoundError(RESULT_FILE)

    df_mat = pd.read_parquet(MATERIAL_FILE)
    df_res = pd.read_parquet(RESULT_FILE)

    logger.info(f"Material classes: {len(df_mat):,}")
    logger.info(f"Legality result rows: {len(df_res):,}")

    # Pick best sample per class
    df_best = (
        df_res.sort_values("sample_size", ascending=False)
              .groupby("id")
              .agg(
                  best_legal_ratio=("legal_ratio", "first"),
                  best_sample_size=("sample_size", "first"),
                  legal_ratio_std=("legal_ratio", "std"),
              )
              .reset_index()
    )

    # Merge onto material classes
    df = df_mat.merge(df_best, on="id", how="left")

    # Fill missing ratios (classes without samples)
    df["best_legal_ratio"] = df["best_legal_ratio"].fillna(1.0)
    df["best_sample_size"] = df["best_sample_size"].fillna(0).astype(int)
    df["legal_ratio_std"] = df["legal_ratio_std"].fillna(0.0)

    # Convert diagrams to float for calculation
    diagrams = df["diagrams"].astype(float).to_numpy()
    ratios = df["best_legal_ratio"].to_numpy()

    estimated_legal = diagrams * ratios

    # Store counts as string to avoid precision loss
    df["diagram_count_str"] = df["diagrams"].astype(str)
    df["estimated_legal_count_str"] = estimated_legal.astype(str)

    # Compute global sums using high-precision summation
    total_diagrams = math.fsum(diagrams)
    total_estimated_legal = math.fsum(estimated_legal)

    global_factor = total_estimated_legal / total_diagrams
    mean_legal_ratio = df["best_legal_ratio"].mean()

    # Save result parquet
    out_cols = [
        "id", "white", "black",
        "best_legal_ratio", "best_sample_size", "legal_ratio_std",
        "diagram_count_str", "estimated_legal_count_str"
    ]
    df[out_cols].to_parquet(OUTPUT_FILE, index=False)

    logger.info(f"Saved: {OUTPUT_FILE}")
    logger.info("")
    logger.info(f"Total diagrams:          {total_diagrams:.3e}")
    logger.info(f"Total legal diagrams:    {total_estimated_legal:.3e}")
    logger.info(f"Global legal factor:     {global_factor:.6e}")
    logger.info(f"Mean per-class factor:   {mean_legal_ratio:.6e}")
    logger.info("✅ Done.\n")


if __name__ == "__main__":
    main()
