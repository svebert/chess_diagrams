#!/usr/bin/env python3
"""
plot.py

Generates various plots for chess material classes and legality factors:

1. Histogram of number of pieces per material class.
2. Positions per material class + weighted by valid_ratio (only valid_ratio < 1).
3. Max sample_size per class.
4. Valid ratio per class (log y-axis, sorted by positions) with average lines.
5. Positions by number of pieces (binned, log y-axis, sum of positions).
6. Valid ratio by number of pieces (binned, log y-axis, weighted mean by positions).

Optimized for large DataFrames.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------
# Input / Output filenames
# ---------------------------
MATERIAL_FILE = "material_classes_positions.parquet"
ANALYSIS_FILE = "legality_analysis.parquet"
RESULT_FILE = "legality_results.parquet"

OUT_HIST_NUM_PIECES = "material_class_histogram.png"
OUT_POS_SORTED = "positions_vs_weighted_sorted_filtered.png"
OUT_MAX_SAMPLE = "max_sample_per_class.png"
OUT_VALID_RATIO_SORTED = "valid_ratio_sorted.png"
OUT_POS_BY_PIECES = "positions_by_num_pieces.png"
OUT_VALID_RATIO_BY_PIECES = "valid_ratio_by_num_pieces.png"

# ---------------------------
# Main
# ---------------------------
def main():
    # ---------------------------
    # Load data
    # ---------------------------
    logger.info("Loading material classes...")
    df_mat = pd.read_parquet(MATERIAL_FILE)
    logger.info(f"{len(df_mat):,} material classes loaded.")

    logger.info("Loading analysis results...")
    df_analysis = pd.read_parquet(ANALYSIS_FILE)
    logger.info(f"{len(df_analysis):,} rows loaded.")

    logger.info("Loading max sample_size results...")
    df_res = pd.read_parquet(RESULT_FILE)
    logger.info(f"{len(df_res):,} rows loaded.")

    # ---------------------------
    # 1. Histogram of number of pieces
    # ---------------------------
    df_mat['num_pieces'] = df_mat['white'].apply(lambda x: sum(eval(x).values())) + \
                           df_mat['black'].apply(lambda x: sum(eval(x).values()))
    plt.figure(figsize=(10,6))
    sns.histplot(df_mat['num_pieces'], bins=range(2,33), kde=False, color='skyblue')
    plt.xlabel("Number of Pieces")
    plt.ylabel("Number of Material Classes")
    plt.title("Distribution of Material Classes by Number of Pieces")
    plt.tight_layout()
    plt.savefig(OUT_HIST_NUM_PIECES, dpi=150)
    logger.info(f"Histogram saved: {OUT_HIST_NUM_PIECES}")
    plt.close()

    # ---------------------------
    # 2. Positions vs weighted (sorted, points, only valid_ratio < 1)
    # ---------------------------
    df_plot = df_analysis.copy()
    df_plot['positions_float'] = df_plot['positions'].astype(float)
    df_plot['weighted_positions_float'] = df_plot['weighted_estimated_legal_str'].astype(float)

    # Filter only valid_ratio < 1
    df_plot = df_plot[df_plot['weighted_positions_float'] < df_plot['positions_float']]

    # Sort by positions
    df_plot = df_plot.sort_values('positions_float').reset_index(drop=True)

    plt.figure(figsize=(12,6))
    plt.scatter(range(len(df_plot)), df_plot['positions_float'], label='Positions', alpha=0.7, s=10)
    plt.scatter(range(len(df_plot)), df_plot['weighted_positions_float'], label='Positions * valid_ratio', alpha=0.7, s=10, color='orange')
    plt.xlabel("Material Class (sorted by positions)")
    plt.ylabel("Number of Positions")
    plt.title("Number of Positions per Material Class (calculated)")
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(OUT_POS_SORTED, dpi=150)
    logger.info(f"Positions plot saved: {OUT_POS_SORTED}")
    plt.close()

    # ---------------------------
    # 3. Max sample_size per class
    # ---------------------------
    df_max_sample = df_res.groupby('id')['sample_size'].max().reset_index()
    plt.figure(figsize=(12,6))
    plt.scatter(df_max_sample['id'], df_max_sample['sample_size'], s=2)
    plt.xlabel("Material Class ID")
    plt.ylabel("Max Sample Size")
    plt.title("Max Sample Size per Material Class")
    plt.tight_layout()
    plt.savefig(OUT_MAX_SAMPLE, dpi=150)
    logger.info(f"Max sample size plot saved: {OUT_MAX_SAMPLE}")
    plt.close()

    # ---------------------------
    # 4. Valid ratio per class (log y-axis) sorted by positions
    # ---------------------------
    df_valid_ratio = df_plot.copy()
    df_valid_ratio['valid_ratio'] = df_valid_ratio['weighted_positions_float'] / df_valid_ratio['positions_float']

    avg_ratio = df_valid_ratio['valid_ratio'].mean()
    weighted_avg_ratio = np.average(df_valid_ratio['valid_ratio'], weights=df_valid_ratio['positions_float'])

    plt.figure(figsize=(12,6))
    plt.scatter(range(len(df_valid_ratio)), df_valid_ratio['valid_ratio'], s=10, alpha=0.7)
    plt.axhline(avg_ratio, color='red', linestyle='--', label='Average valid_ratio')
    plt.axhline(weighted_avg_ratio, color='green', linestyle='--', label='Weighted avg valid_ratio')
    plt.xlabel("Material Class (sorted by positions)")
    plt.ylabel("Valid Ratio")
    plt.title("Valid Ratio per Material Class")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_VALID_RATIO_SORTED, dpi=150)
    logger.info(f"Valid ratio plot saved: {OUT_VALID_RATIO_SORTED}")
    plt.close()

    # ---------------------------
    # 5. Positions by number of pieces (binned, sum)
    # ---------------------------
    df_bin = df_plot.copy()
    df_bin['num_pieces'] = df_mat['num_pieces']

    positions_sum = df_bin.groupby('num_pieces')['positions_float'].sum()
    weighted_positions_sum = df_bin.groupby('num_pieces')['weighted_positions_float'].sum()

    plt.figure(figsize=(10,6))
    plt.bar(positions_sum.index - 0.2, positions_sum.values, width=0.4, label='Positions', color='skyblue')
    plt.bar(weighted_positions_sum.index + 0.2, weighted_positions_sum.values, width=0.4, label='Positions * valid_ratio', color='orange')
    plt.xlabel("Number of Pieces")
    plt.ylabel("Number of Positions")
    plt.title("Positions per Number of Pieces (calculated)")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_POS_BY_PIECES, dpi=150)
    logger.info(f"Positions by pieces plot saved: {OUT_POS_BY_PIECES}")
    plt.close()

    # ---------------------------
    # 6. Valid ratio by number of pieces (binned, weighted mean)
    # ---------------------------
    df_bin['valid_ratio'] = df_bin['weighted_positions_float'] / df_bin['positions_float']

    def weighted_mean(group):
        return np.average(group['valid_ratio'], weights=group['positions_float'])

    valid_ratio_bin = df_bin.groupby('num_pieces').apply(weighted_mean)

    avg_ratio_overall = valid_ratio_bin.mean()
    weighted_avg_ratio_overall = np.average(valid_ratio_bin, weights=positions_sum.values)

    plt.figure(figsize=(10,6))
    plt.bar(valid_ratio_bin.index - 0.2, valid_ratio_bin.values, width=0.4, color='orange', label='Weighted valid_ratio')
    plt.axhline(avg_ratio_overall, color='red', linestyle='--', label='Average valid_ratio')
    plt.axhline(weighted_avg_ratio_overall, color='green', linestyle='--', label='Weighted avg valid_ratio')
    plt.xlabel("Number of Pieces")
    plt.ylabel("Valid Ratio")
    plt.title("Valid Ratio per Number of Pieces (weighted)")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_VALID_RATIO_BY_PIECES, dpi=150)
    logger.info(f"Valid ratio by pieces plot saved: {OUT_VALID_RATIO_BY_PIECES}")
    plt.close()


if __name__ == "__main__":
    main()
