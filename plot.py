#!/usr/bin/env python3
"""
plot.py

Erzeugt verschiedene Plots zu den Materialklassen und Legalitätsfaktoren:
1. Histogramm der Anzahl Figuren pro Materialklasse.
2. Positionen pro Materialklasse + gewichtet durch valid_ratio.
3. Maximal sample_size pro Klasse.

Optimiert für große DataFrames.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Dateien ---
MATERIAL_FILE = "material_classes_positions.parquet"
ANALYSIS_FILE = "legality_analysis.parquet"
RESULT_FILE = "legality_results.parquet"

def main():
    # --- Daten laden ---
    logger.info("Lade Materialklassen...")
    df_mat = pd.read_parquet(MATERIAL_FILE)
    logger.info(f"{len(df_mat):,} Materialklassen geladen.")

    logger.info("Lade Analyse-Ergebnisse...")
    df_analysis = pd.read_parquet(ANALYSIS_FILE)
    logger.info(f"{len(df_analysis):,} Zeilen geladen.")

    logger.info("Lade Ergebnisse für max sample_size...")
    df_res = pd.read_parquet(RESULT_FILE)
    logger.info(f"{len(df_res):,} Zeilen geladen.")

    # --- Histogramm Anzahl Figuren pro Klasse ---
    df_mat['num_pieces'] = df_mat['white'].apply(lambda x: sum(eval(x).values())) + \
                           df_mat['black'].apply(lambda x: sum(eval(x).values()))
    plt.figure(figsize=(10,6))
    sns.histplot(df_mat['num_pieces'], bins=range(2,33), kde=False, color='skyblue')
    plt.xlabel("Anzahl Figuren")
    plt.ylabel("Anzahl Materialklassen")
    plt.title("Verteilung der Materialklassen nach Figurenanzahl")
    plt.tight_layout()
    plt.savefig("material_class_histogram.png", dpi=150)
    logger.info("Histogramm der Materialklassen gespeichert: material_class_histogram.png")
    plt.close()

    # --- Positions vs weighted (sortiert, Punkte) ---
    df_plot = df_analysis.copy()
    df_plot['positions_float'] = df_plot['positions'].astype(float)
    df_plot['weighted_positions_float'] = df_plot['weighted_estimated_legal_str'].astype(float)

    # Sortieren nach Positionsanzahl
    df_plot = df_plot.sort_values('positions_float').reset_index(drop=True)

    plt.figure(figsize=(12,6))
    plt.scatter(range(len(df_plot)), df_plot['positions_float'], label='Positions', alpha=0.7, s=10)
    plt.scatter(range(len(df_plot)), df_plot['weighted_positions_float'], label='Positions * valid_factor', alpha=0.7, s=10)
    plt.xlabel("Materialklasse (sortiert nach Positionsanzahl)")
    plt.ylabel("Anzahl Stellungen")
    plt.title("Anzahl Positionen pro Materialklasse (sortiert)")
    plt.legend()
    plt.yscale('log')  # optional logarithmisch, wenn gewünscht
    plt.tight_layout()
    plt.savefig("positions_vs_weighted_sorted.png", dpi=150)
    logger.info("Positions-Plot gespeichert: positions_vs_weighted_sorted.png")
    plt.close()

    # --- Max sample_size pro Klasse ---
    df_max_sample = df_res.groupby('id')['sample_size'].max().reset_index()
    plt.figure(figsize=(12,6))
    plt.scatter(df_max_sample['id'], df_max_sample['sample_size'], s=2)
    plt.xlabel("Materialklasse ID")
    plt.ylabel("Maximal berechnetes Sample")
    plt.title("Maximal berechnetes Sample pro Materialklasse")
    plt.tight_layout()
    plt.savefig("max_sample_per_class.png", dpi=150)
    logger.info("Max Sample Size-Plot gespeichert: max_sample_per_class.png")
    plt.close()

if __name__ == "__main__":
    main()
