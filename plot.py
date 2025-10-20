#!/usr/bin/env python3
"""
plot_analysis.py

Erstellt Plots für die Materialklassen-Analyse:
1. Histogramm: Anzahl Figuren vs. Anzahl Materialklassen
2. Positionen pro Klasse: original vs. weighted_estimated_legal
3. Max sample_size pro Klasse
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Dateien ---
MATERIAL_FILE = "material_classes_positions.parquet"
RESULT_FILE = "legality_results.parquet"
ANALYSIS_FILE = "legality_analysis.parquet"

# --- Lade Daten ---
print("Lade Materialklassen...")
df_mat = pd.read_parquet(MATERIAL_FILE)
print(f"{len(df_mat):,} Materialklassen geladen.")

print("Lade Analyse-Ergebnisse...")
df_analysis = pd.read_parquet(ANALYSIS_FILE)
print(f"{len(df_analysis):,} Zeilen geladen.")

print("Lade Ergebnisse für max sample_size...")
df_res = pd.read_parquet(RESULT_FILE)
df_max_sample = df_res.groupby("id")["sample_size"].max().reset_index()

# --- 1) Histogramm Anzahl Figuren ---
def count_figures(mat_str):
    """Summiert alle Figuren in einem Material-Dict String"""
    d = eval(mat_str)
    return sum(d.values())

df_mat["num_pieces"] = df_mat["white"].apply(count_figures) + df_mat["black"].apply(count_figures)

plt.figure(figsize=(10,5))
sns.histplot(df_mat["num_pieces"], bins=range(2, 34), color="skyblue", edgecolor="black")
plt.xlabel("Anzahl Figuren (2-32)")
plt.ylabel("Anzahl Materialklassen")
plt.title("Verteilung der Materialklassen nach Figurenzahl")
plt.grid(axis="y", alpha=0.75)
plt.tight_layout()
plt.savefig("material_classes_histogram.png")
plt.close()
print("Histogramm gespeichert: material_classes_histogram.png")

# --- 2) Positionen pro Klasse ---
plt.figure(figsize=(14,6))
x = df_analysis["id"]
y_positions = df_analysis["positions"].astype(float)
y_weighted = df_analysis["weighted_estimated_legal_str"].apply(lambda s: float(s))

plt.bar(x, y_positions, color="skyblue", label="Positions")
plt.bar(x, y_weighted, color="salmon", alpha=0.6, label="Positions * valid_factor")
plt.xlabel("Materialklasse ID")
plt.ylabel("Anzahl Positionen")
plt.title("Positionen pro Materialklasse (original vs. weighted)")
plt.legend()
plt.tight_layout()
plt.savefig("positions_vs_weighted.png")
plt.close()
print("Positions-Plot gespeichert: positions_vs_weighted.png")

# --- 3) Max sample_size pro Klasse ---
plt.figure(figsize=(14,6))
plt.bar(df_max_sample["id"], df_max_sample["sample_size"], color="lightgreen")
plt.xlabel("Materialklasse ID")
plt.ylabel("Max Sample Size")
plt.title("Maximale Sample Size pro Materialklasse")
plt.tight_layout()
plt.savefig("max_sample_size_per_class.png")
plt.close()
print("Max sample_size-Plot gespeichert: max_sample_size_per_class.png")

print("✅ Alle Plots erstellt.")
