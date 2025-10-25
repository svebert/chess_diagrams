#!/usr/bin/env python3
"""
Compute legality estimates for chess material classes.

Inputs:
 - material_classes_diagrams.parquet
      Columns: id, white, black, diagrams
 - legality_results.parquet
      Columns: id, sample_size, legal_ratio, std_error, rel_std

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

    # Pick best sample per class (largest sample_size)
    df_best = (
        df_res.sort_values("sample_size", ascending=False)
              .groupby("id")
              .agg(
                  best_legal_ratio=("legal_ratio", "first"),
                  best_sample_size=("sample_size", "first"),
                  best_std_error=("std_error", "first"),
                  best_rel_std=("rel_std", "first"),
              )
              .reset_index()
    )

    # Merge onto material classes
    df = df_mat.merge(df_best, on="id", how="left")

    # Fill missing ratios (classes without samples)
    df["best_legal_ratio"] = df["best_legal_ratio"].fillna(1.0)
    df["best_sample_size"] = df["best_sample_size"].fillna(0).astype(int)

    # Falls std_error fehlt, aus rel_std ableiten
    df["best_std_error"] = df["best_std_error"].fillna(
        df["best_rel_std"] * df["best_legal_ratio"]
    )
    df["best_std_error"] = df["best_std_error"].fillna(0.0)

    # Arrays für Berechnung
    diagrams = df["diagrams"].astype(float).to_numpy()
    ratios = df["best_legal_ratio"].to_numpy()
    sigma_r = df["best_std_error"].to_numpy()

    estimated_legal = diagrams * ratios

    # Fehlerfortpflanzung
    variance_sum = np.sum((diagrams * sigma_r) ** 2)
    estimated_legal_std = math.sqrt(variance_sum)

    # Strings für Speicherung
    df["diagram_count_str"] = df["diagrams"].astype(str)
    df["estimated_legal_count_str"] = estimated_legal.astype(str)

    # Globale Summen (präzise)
    total_diagrams = math.fsum(diagrams)
    total_estimated_legal = math.fsum(estimated_legal)

    global_factor = total_estimated_legal / total_diagrams
    mean_legal_ratio = df["best_legal_ratio"].mean()

    # Relative Gesamtunsicherheit
    rel_uncertainty = (
        estimated_legal_std / total_estimated_legal
        if total_estimated_legal > 0
        else float("nan")
    )

    # Sample size stats
    total_sample_size = df["best_sample_size"].sum()
    mean_sample_size = df["best_sample_size"].mean()
    max_sample_size = df["best_sample_size"].max()
    min_sample_size = df["best_sample_size"].min()
    median_sample_size = df["best_sample_size"].median()

    # Save result parquet
    out_cols = [
        "id", "white", "black",
        "best_legal_ratio", "best_sample_size", "best_std_error", "best_rel_std",
        "diagram_count_str", "estimated_legal_count_str"
    ]
    df[out_cols].to_parquet(OUTPUT_FILE, index=False)

    # Logging final stats
    logger.info(f"Saved: {OUTPUT_FILE}")
    logger.info("")
    logger.info(f"Total diagrams:          {total_diagrams:.3e}")
    logger.info(
        f"Total legal diagrams:    {total_estimated_legal:.3e} "
        f"± {estimated_legal_std:.3e} ({rel_uncertainty:.2%})"
    )
    logger.info(f"Global legal factor:     {global_factor:.6e}")
    logger.info(f"Mean per-class factor:   {mean_legal_ratio:.6e}")
    logger.info("")
    logger.info(f"Total sample sizes:      {total_sample_size:,}")
    logger.info(f"Mean sample size:        {mean_sample_size:.2f}")
    logger.info(f"Median sample size:      {median_sample_size}")
    logger.info(f"Min sample size:         {min_sample_size}")
    logger.info(f"Max sample size:         {max_sample_size}")
    logger.info("✅ Done.\n")


if __name__ == "__main__":
    main()
