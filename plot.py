#!/usr/bin/env python3
"""
plot.py

Erstellt Plots basierend auf:
- material_classes_positions.parquet
- legality_analysis.parquet
- legality_results.parquet (für max sample_size)
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # kein GUI, direkt PNG speichern
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import numpy as np

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Dateien ---
MATERIAL_FILE = "material_classes_positions.parquet"
ANALYSIS_FILE = "legality_analysis.parquet"
RESULT_FILE = "legality_results.parquet"

def main():
    # --- Materialklassen laden ---
    logger.info("Lade Materialklassen...")
    df_mat = pd.read_parquet(MATERIAL_FILE)
    logger.info(f"{len(df_mat):,} Materialklassen geladen.")

    # --- Analyse laden ---
    logger.info("Lade Analyse-Ergebnisse...")
    df_analysis = pd.read_parquet(ANALYSIS_FILE)
    logger.info(f"{len(df_analysis):,} Zeilen geladen.")

    # --- Max sample_size laden ---
    logger.info("Lade Ergebnisse für max sample_size...")
    df_res = pd.read_parquet(RESULT_FILE)
    max_sample_df = df_res.sort_values("sample_size", ascending=False).drop_duplicates("id")

    # --- Plot 1: Verteilung der Materialklassen nach Figurenanzahl ---
    logger.info("Erstelle Histogramm der Materialklassen nach Figurenzahl...")
    df_mat['num_pieces'] = df_mat['white'].apply(lambda x: len(eval(x))) + df_mat['black'].apply(lambda x: len(eval(x)))
    plt.figure(figsize=(10,5))
    plt.hist(df_mat['num_pieces'], bins=range(2,33), color='skyblue', edgecolor='black')
    plt.xlabel("Anzahl Figuren (2-32)")
    plt.ylabel("Anzahl Materialklassen")
    plt.title("Verteilung der Materialklassen nach Figurenanzahl")
    plt.tight_layout()
    plt.savefig("material_class_distribution.png", dpi=150)
    plt.close()
    logger.info("Histogramm gespeichert: material_class_distribution.png")

    # --- Plot 2: Positions vs weighted ---
    logger.info("Erstelle Positions vs weighted Positions Plot...")
    df_plot = df_analysis.copy()
    df_plot['positions_float'] = df_plot['positions'].astype(float)
    df_plot['weighted_positions_float'] = df_plot['positions_float'] * df_plot['weighted_mean_ratio']

    # Optionales Downsampling, falls zu groß
    sample_step = max(1, len(df_plot)//5000)  # max 5000 Punkte für Plot
    x_vals = df_plot['id'].iloc[::sample_step]
    y_positions = df_plot['positions_float'].iloc[::sample_step]
    y_weighted = df_plot['weighted_positions_float'].iloc[::sample_step]

    plt.figure(figsize=(15,5))
    plt.plot(x_vals, y_positions, color='blue', lw=1, label='Positions')
    plt.plot(x_vals, y_weighted, color='red', lw=1, label='Positions * valid_factor')
    plt.xlabel("Materialklasse ID")
    plt.ylabel("Anzahl Positionen")
    plt.title("Positionen pro Materialklasse vs gewichtete Positionen")
    plt.legend()
    plt.tight_layout()
    plt.savefig("positions_vs_weighted.png", dpi=150)
    plt.close()
    logger.info("Positions-Plot gespeichert: positions_vs_weighted.png")

    # --- Plot 3: Max sample_size pro Klasse ---
    logger.info("Erstelle Max sample_size pro Klasse Plot...")
    x_vals = max_sample_df['id']
    y_vals = max_sample_df['sample_size']

    plt.figure(figsize=(15,5))
    plt.plot(x_vals, y_vals, color='green', lw=1)
    plt.xlabel("Materialklasse ID")
    plt.ylabel("Max sample_size")
    plt.title("Maximal berechnete Samplegröße pro Materialklasse")
    plt.tight_layout()
    plt.savefig("max_sample_size.png", dpi=150)
    plt.close()
    logger.info("Max sample_size-Plot gespeichert: max_sample_size.png")

    logger.info("✅ Alle Plots erstellt und gespeichert.")

if __name__ == "__main__":
    main()
