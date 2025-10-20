#!/usr/bin/env python3
"""
plot.py

Erzeugt verschiedene Plots zu den Materialklassen und Legalitätsfaktoren.

Plots:
1. Histogram of number of pieces per material class.
2. Positions per material class (sorted, points, only calculated classes).
3. Maximal sample size per class.
4. Valid ratio per class (sorted, points, only calculated classes).
5. Sum of positions per number of pieces (binned, log y-axis, only calculated).
6. Weighted valid ratio per number of pieces (binned, log y-axis, only calculated).

All plots are optimized for large DataFrames. Only classes with valid_ratio < 1
(i.e., already calculated) are plotted where applicable.

Author: Sven
Date: YYYY-MM-DD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Input / Output files ---
MATERIAL_FILE = "material_classes_positions.parquet"
ANALYSIS_FILE = "legality_analysis.parquet"
RESULT_FILE = "legality_results.parquet"

OUT_HIST_NUM_PIECES = "material_class_histogram.png"
OUT_POS_SORTED = "positions_per_class.png"
OUT_MAX_SAMPLE = "max_sample_per_class.png"
OUT_VALID_RATIO_SORTED = "valid_ratio_per_class.png"
OUT_POS_PER_NUM_PIECES = "positions_per_num_pieces.png"
OUT_VALID_RATIO_PER_NUM_PIECES = "valid_ratio_per_num_pieces.png"

# ---------------------------
# Plot functions
# ---------------------------
def plot_hist_num_pieces(df_mat):
    df_mat['num_pieces'] = df_mat['white'].apply(lambda x: sum(eval(x).values())) + \
                           df_mat['black'].apply(lambda x: sum(eval(x).values()))
    plt.figure(figsize=(10,6))
    sns.histplot(df_mat['num_pieces'], bins=range(2,33), kde=False, color='skyblue')
    plt.xlabel("Number of Pieces")
    plt.ylabel("Number of Material Classes")
    plt.title("Distribution of Material Classes by Number of Pieces")
    plt.grid(True, axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_HIST_NUM_PIECES, dpi=150)
    logger.info(f"Histogram saved: {OUT_HIST_NUM_PIECES}")
    plt.close()

def plot_positions_sorted(df_analysis):
    """
    Plot positions per material class, sorted by positions.
    Only include calculated classes (valid_ratio < 1).
    Logarithmic y-axis. No y-axis limits.
    """
    df_plot = df_analysis.copy()
    # Positions as float
    df_plot['positions_float'] = df_plot['positions'].astype(float)
    df_plot['weighted_positions_float'] = df_plot['weighted_estimated_legal_str'].astype(float)
    
    # Only include calculated classes
    df_plot = df_plot[df_plot['weighted_positions_float'] != df_plot['positions_float']]
    
    # Sort by positions
    df_plot = df_plot.sort_values('positions_float').reset_index(drop=True)

    plt.figure(figsize=(12,6))
    plt.scatter(range(len(df_plot)), df_plot['positions_float'], label=r'$\mathrm{Positions}$', alpha=0.7, s=10, color='blue')
    plt.scatter(range(len(df_plot)), df_plot['weighted_positions_float'], label=r'$\mathrm{Positions \cdot valid\_ratio}$', alpha=0.7, s=10, color='orange')
    plt.xlabel("Material class (sorted by positions)")
    plt.ylabel("Number of positions")
    plt.title("Number of positions per material class (calculated)")
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_POS_SORTED, dpi=150)
    logger.info(f"Positions plot saved: {OUT_POS_SORTED}")
    plt.close()


def plot_max_sample(df_res):
    df_max_sample = df_res.groupby('id')['sample_size'].max().reset_index()
    plt.figure(figsize=(12,6))
    plt.scatter(df_max_sample['id'], df_max_sample['sample_size'], s=2)
    plt.xlabel("Material Class ID")
    plt.ylabel("Max Sample Size")
    plt.title("Max Sample Size per Material Class")
    plt.grid(True, axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_MAX_SAMPLE, dpi=150)
    logger.info(f"Max sample size plot saved: {OUT_MAX_SAMPLE}")
    plt.close()


def plot_valid_ratio_sorted(df_analysis):
    df_plot = df_analysis.copy()
    df_plot['positions_float'] = df_plot['positions'].astype(float)
    df_plot['weighted_positions_float'] = df_plot['weighted_estimated_legal_str'].astype(float)

    # Only plot entries with valid_ratio < 1
    df_plot = df_plot[df_plot['weighted_positions_float'] < df_plot['positions_float']]

    # Sort by positions
    df_plot = df_plot.sort_values('positions_float').reset_index(drop=True)

    # Compute valid_ratio
    df_plot['valid_ratio'] = df_plot['weighted_positions_float'] / df_plot['positions_float']

    avg_ratio = df_plot['valid_ratio'].mean()
    weighted_avg_ratio = np.average(df_plot['valid_ratio'], weights=df_plot['positions_float'])

    plt.figure(figsize=(12,6))
    plt.scatter(range(len(df_plot)), df_plot['valid_ratio'], s=10, alpha=0.7)
    plt.axhline(avg_ratio, color='red', linestyle='--', label=r'$\bar{r}$')
    plt.axhline(weighted_avg_ratio, color='green', linestyle='--', label=r'$\bar{r}_w$')
    plt.xlabel("Material Class (sorted by positions)")
    plt.ylabel("Valid Ratio")
    plt.title("Valid Ratio per Material Class")
    plt.yscale('log')
    plt.ylim(1e-4, 1e0)
    plt.yticks([1e-3,1e-2,1e-1,1e0], ['10$^{-3}$','10$^{-2}$','10$^{-1}$','10$^{0}$'])
    plt.grid(True, axis='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_VALID_RATIO_SORTED, dpi=150)
    logger.info(f"Valid ratio plot saved: {OUT_VALID_RATIO_SORTED}")
    plt.close()


def plot_positions_per_num_pieces(df_analysis):
    """
    Plot total positions per number of pieces, binned.
    Include both raw positions and positions scaled by valid_ratio.
    Logarithmic y-axis. 
    """
    df_plot = df_analysis.copy()
    df_plot['positions_float'] = df_plot['positions'].astype(float)
    df_plot['weighted_positions_float'] = df_plot['weighted_estimated_legal_str'].astype(float)
    df_plot['num_pieces'] = df_plot['white'].apply(lambda x: sum(eval(x).values())) + \
                             df_plot['black'].apply(lambda x: sum(eval(x).values()))
    
    # Only consider classes where valid_ratio < 1 for weighted
    df_weighted = df_plot[df_plot['weighted_positions_float'] != df_plot['positions_float']]
    
    # Aggregate per number of pieces
    agg_positions = df_plot.groupby('num_pieces')['positions_float'].sum()
    agg_weighted = df_weighted.groupby('num_pieces')['weighted_positions_float'].sum()
    
    plt.figure(figsize=(10,6))
    plt.scatter(agg_positions.index, agg_positions.values, label=r'$\mathrm{Positions}$', color='blue', s=40)
    plt.scatter(agg_weighted.index, agg_weighted.values, label=r'$\mathrm{Positions \cdot valid\_ratio}$', color='orange', s=40)
    plt.xlabel("Number of pieces")
    plt.ylabel("Number of positions")
    plt.title("Total positions per number of pieces")
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_POS_PER_NUM_PIECES, dpi=150)
    logger.info(f"Positions per number of pieces plot saved: {OUT_POS_PER_NUM_PIECES}")
    plt.close()


def plot_valid_ratio_per_num_pieces(df_analysis, df_mat):
    """Plot weighted valid_ratio per number of pieces (log y-axis)"""
    df_plot = df_analysis.copy()
    df_plot['positions_float'] = df_plot['positions'].astype(float)
    df_plot['weighted_positions_float'] = df_plot['weighted_estimated_legal_str'].astype(float)

    # Only entries with valid_ratio < 1
    df_plot = df_plot[df_plot['weighted_positions_float'] < df_plot['positions_float']]

    df_plot['valid_ratio'] = df_plot['weighted_positions_float'] / df_plot['positions_float']

    # Add number of pieces
    df_mat['num_pieces'] = df_mat['white'].apply(lambda x: sum(eval(x).values())) + \
                           df_mat['black'].apply(lambda x: sum(eval(x).values()))
    df_plot = df_plot.merge(df_mat[['id','num_pieces']], on='id', how='left')

    # Group by number of pieces, weighted mean of valid_ratio
    grouped = df_plot.groupby('num_pieces').apply(
        lambda g: np.average(g['valid_ratio'], weights=g['positions_float'])
    ).reset_index(name='weighted_valid_ratio')

    avg_ratio = df_plot['valid_ratio'].mean()
    weighted_avg_ratio = np.average(df_plot['valid_ratio'], weights=df_plot['positions_float'])

    plt.figure(figsize=(10,6))
    plt.scatter(grouped['num_pieces'], grouped['weighted_valid_ratio'], s=50, color='orange', alpha=0.7)
    plt.axhline(avg_ratio, color='red', linestyle='--', label=r'$\bar{r}$')
    plt.axhline(weighted_avg_ratio, color='green', linestyle='--', label=r'$\bar{r}_w$')
    plt.xlabel("Number of Pieces")
    plt.ylabel("Weighted Valid Ratio")
    plt.title("Weighted Valid Ratio per Number of Pieces")
    plt.yscale('log')
    plt.ylim(1e-4, 1e0)
    plt.yticks([1e-3,1e-2,1e-1,1e0], ['10$^{-3}$','10$^{-2}$','10$^{-1}$','10$^{0}$'])
    plt.grid(True, axis='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_VALID_RATIO_PER_NUM_PIECES, dpi=150)
    logger.info(f"Weighted valid ratio per number of pieces plot saved: {OUT_VALID_RATIO_PER_NUM_PIECES}")
    plt.close()


# ---------------------------
# Main function
# ---------------------------
def main():
    # Load data
    logger.info("Loading material classes...")
    df_mat = pd.read_parquet(MATERIAL_FILE)
    logger.info(f"{len(df_mat):,} material classes loaded.")

    logger.info("Loading analysis data...")
    df_analysis = pd.read_parquet(ANALYSIS_FILE)
    logger.info(f"{len(df_analysis):,} rows loaded.")

    logger.info("Loading results data...")
    df_res = pd.read_parquet(RESULT_FILE)
    logger.info(f"{len(df_res):,} rows loaded.")

    # Call plots
    plot_hist_num_pieces(df_mat)
    plot_positions_sorted(df_analysis)
    plot_max_sample(df_res)
    plot_valid_ratio_sorted(df_analysis)
    plot_positions_per_num_pieces(df_analysis, df_mat)
    plot_valid_ratio_per_num_pieces(df_analysis, df_mat)


if __name__ == "__main__":
    main()
