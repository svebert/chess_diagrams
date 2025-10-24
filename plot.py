#!/usr/bin/env python3
"""
plot.py

Generates several plots for legality factor and material classes.

Plots:
1. Histogram of number of pieces per material class.
2. Diagrams per material class (sorted, points, only calculated classes).
3. Maximal sample size per class.
4. Legal ratio per class (sorted, points, only calculated classes).
5. Sum of diagrams per number of pieces (binned, log y-axis, only calculated).
6. Weighted legal ratio per number of pieces (binned, log y-axis, only calculated).

Author: Sven
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Input / Output files ---
MATERIAL_FILE = "material_classes_diagrams.parquet"
ANALYSIS_FILE = "legality_analysis.parquet"
RESULT_FILE = "legality_results.parquet"

OUT_HIST_NUM_PIECES = "material_class_histogram.png"
OUT_DIAG_SORTED = "diagrams_per_class.png"
OUT_MAX_SAMPLE = "max_sample_per_class.png"
OUT_LEGAL_RATIO_SORTED = "legal_ratio_per_class.png"
OUT_DIAG_PER_NUM_PIECES = "diagrams_per_num_pieces.png"
OUT_LEGAL_RATIO_PER_NUM_PIECES = "legal_ratio_per_num_pieces.png"


# ---------------------------
# Plot functions
# ---------------------------
def plot_hist_num_pieces(df_mat: pd.DataFrame):
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


def plot_diagrams_sorted(df_analysis: pd.DataFrame):
    """Plot diagrams per material class, sorted by total diagrams. Log y-axis."""
    df_plot = df_analysis.copy()
    df_plot['diagrams'] = df_plot['diagram_count_str'].apply(int)
    df_plot['estimated_legal_diagrams'] = df_plot['estimated_legal_count_str'].apply(int)

    # Only calculated classes (legal_ratio < 1)
    df_plot = df_plot[df_plot['estimated_legal_diagrams'] != df_plot['diagrams']]

    df_plot = df_plot.sort_values('diagrams').reset_index(drop=True)

    plt.figure(figsize=(12,6))
    plt.scatter(range(len(df_plot)), df_plot['diagrams'], label='Diagrams', alpha=0.7, s=10, color='blue')
    plt.scatter(range(len(df_plot)), df_plot['estimated_legal_diagrams'], label='Diagrams * legal_ratio', alpha=0.7, s=10, color='orange')
    plt.xlabel("Material class (sorted by diagrams)")
    plt.ylabel("Number of diagrams")
    plt.title("Number of diagrams per material class (calculated)")
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIAG_SORTED, dpi=150)
    logger.info(f"Diagrams plot saved: {OUT_DIAG_SORTED}")
    plt.close()


def plot_max_sample(df_res: pd.DataFrame):
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


def plot_legal_ratio_sorted(df_analysis: pd.DataFrame):
    df_plot = df_analysis.copy()
    df_plot['diagrams'] = df_plot['diagram_count_str'].apply(int)
    df_plot['estimated_legal_diagrams'] = df_plot['estimated_legal_count_str'].apply(int)
    df_plot['legal_ratio'] = df_plot['estimated_legal_diagrams'] / df_plot['diagrams']

    df_plot = df_plot.sort_values('diagrams').reset_index(drop=True)

    avg_ratio = df_plot['legal_ratio'].mean()
    weighted_avg_ratio = np.average(df_plot['legal_ratio'], weights=df_plot['diagrams'])

    plt.figure(figsize=(12,6))
    plt.scatter(range(len(df_plot)), df_plot['legal_ratio'], s=10, alpha=0.7)
    plt.axhline(avg_ratio, color='red', linestyle='--', label=r'$\bar{r}$')
    plt.axhline(weighted_avg_ratio, color='green', linestyle='--', label=r'$\bar{r}_w$')
    plt.xlabel("Material Class (sorted by diagrams)")
    plt.ylabel("Legal Ratio")
    plt.title("Legal Ratio per Material Class")
    plt.yscale('log')
    plt.ylim(1e-4, 1e0)
    plt.yticks([1e-3,1e-2,1e-1,1e0], ['10$^{-3}$','10$^{-2}$','10$^{-1}$','10$^{0}$'])
    plt.grid(True, axis='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_LEGAL_RATIO_SORTED, dpi=150)
    logger.info(f"Legal ratio plot saved: {OUT_LEGAL_RATIO_SORTED}")
    plt.close()


def plot_diagrams_per_num_pieces(df_analysis: pd.DataFrame):
    """Aggregate diagrams per number of pieces."""
    df_plot = df_analysis.copy()
    df_plot['diagrams'] = df_plot['diagram_count_str'].apply(int)
    df_plot['estimated_legal_diagrams'] = df_plot['estimated_legal_count_str'].apply(int)
    df_plot['num_pieces'] = df_plot['white'].apply(lambda x: sum(eval(x).values())) + \
                            df_plot['black'].apply(lambda x: sum(eval(x).values()))

    # Only calculated classes
    df_weighted = df_plot[df_plot['estimated_legal_diagrams'] != df_plot['diagrams']]

    agg_diagrams = df_plot.groupby('num_pieces')['diagrams'].sum()
    agg_weighted = df_weighted.groupby('num_pieces')['estimated_legal_diagrams'].sum()

    plt.figure(figsize=(10,6))
    plt.scatter(agg_diagrams.index, agg_diagrams.values, label='Diagrams', color='blue', s=40)
    plt.scatter(agg_weighted.index, agg_weighted.values, label='Diagrams * legal_ratio', color='orange', s=40)
    plt.xlabel("Number of Pieces")
    plt.ylabel("Number of Diagrams")
    plt.title("Total Diagrams per Number of Pieces")
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIAG_PER_NUM_PIECES, dpi=150)
    logger.info(f"Diagrams per number of pieces plot saved: {OUT_DIAG_PER_NUM_PIECES}")
    plt.close()


def plot_legal_ratio_per_num_pieces(df_analysis: pd.DataFrame):
    """Weighted legal_ratio per number of pieces (log y-axis)."""
    df_plot = df_analysis.copy()
    df_plot['diagrams'] = df_plot['diagram_count_str'].apply(int)
    df_plot['estimated_legal_diagrams'] = df_plot['estimated_legal_count_str'].apply(int)
    df_plot['legal_ratio'] = df_plot['estimated_legal_diagrams'] / df_plot['diagrams']
    df_plot['num_pieces'] = df_plot['white'].apply(lambda x: sum(eval(x).values())) + \
                            df_plot['black'].apply(lambda x: sum(eval(x).values()))

    # Only calculated classes
    df_plot = df_plot[df_plot['estimated_legal_diagrams'] != df_plot['diagrams']]

    grouped = df_plot.groupby('num_pieces').apply(
        lambda g: np.average(g['legal_ratio'], weights=g['diagrams'])
    ).reset_index(name='weighted_legal_ratio')

    avg_ratio = df_plot['legal_ratio'].mean()
    weighted_avg_ratio = np.average(df_plot['legal_ratio'], weights=df_plot['diagrams'])

    plt.figure(figsize=(10,6))
    plt.scatter(grouped['num_pieces'], grouped['weighted_legal_ratio'], s=50, color='orange', alpha=0.7)
    plt.axhline(avg_ratio, color='red', linestyle='--', label=r'$\bar{r}$')
    plt.axhline(weighted_avg_ratio, color='green', linestyle='--', label=r'$\bar{r}_w$')
    plt.xlabel("Number of Pieces")
    plt.ylabel("Weighted Legal Ratio")
    plt.title("Weighted Legal Ratio per Number of Pieces")
    plt.yscale('log')
    plt.ylim(1e-4, 1

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
    plot_diagrams_sorted(df_analysis)  
    plot_max_sample(df_res)  
    plot_legal_ratio_sorted(df_analysis)  
    plot_diagrams_per_num_pieces(df_analysis)  
    plot_legal_ratio_per_num_pieces(df_analysis, df_mat)

if __name__ == "__main__":
    main()
