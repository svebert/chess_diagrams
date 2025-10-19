#!/usr/bin/env python3
"""
analyze.py

Lädt:
 - material_classes_positions.parquet (Spalten: id, white, black, positions)
 - legality_results.parquet (Spalten: id, sample_size, valid_ratio)

Berechnet:
 - Für jede Klasse: valid_ratio des größten Samples
 - weighted_estimated_legal = positions * valid_ratio
 - Summen (Fast Sum) mit Ladebalken
 - speichert Ergebnisse
"""
from decimal import Decimal, InvalidOperation
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Dateien ---
MATERIAL_FILE = "material_classes_positions.parquet"
RESULT_FILE = "legality_results.parquet"
OUTPUT_FILE = "legality_analysis.parquet"


def safe_to_decimal(x):
    """Konvertiert String/Nummer zu Decimal. Wenn fehlerhaft -> Decimal(1)."""
    if pd.isna(x):
        return Decimal(1)
    try:
        return Decimal(str(x))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(1)


def get_largest_sample_ratio(group):
    """Gibt valid_ratio des größten sample_size in der Gruppe zurück."""
    if len(group) == 0:
        return 1.0
    group_sorted = group.sort_values("sample_size", ascending=False)
    return float(group_sorted.iloc[0]["valid_ratio"])


def load_material_classes(filename):
    if not os.path.exists(filename):
        logger.error(f"Materialdatei nicht gefunden: {filename}")
        raise FileNotFoundError
    df = pd.read_parquet(filename)
    logger.info(f"{len(df):,} Materialklassen geladen.")
    return df


def load_results(filename):
    if not os.path.exists(filename):
        logger.warning(f"Resultatdatei nicht gefunden: {filename} — alle Faktoren = 1.0")
        return pd.DataFrame(columns=["id", "sample_size", "valid_ratio"])
    df = pd.read_parquet(filename)
    logger.info(f"{len(df):,} Ergebnis-Zeilen geladen.")
    return df


def compute_ratios(df_mat, df_res):
    tqdm.pandas()
    grouped = df_res.groupby("id") if len(df_res) > 0 else {}

    stats_rows = []
    ids = df_mat["id"].tolist()
    logger.info("Hole largest-sample valid_ratio pro Klasse...")
    for cid in tqdm(ids, desc="Largest sample ratios", ncols=100):
        if len(df_res) == 0 or cid not in grouped.groups:
            stats_rows.append((cid, 1.0))
        else:
            grp = grouped.get_group(cid)
            largest_ratio = get_largest_sample_ratio(grp)
            stats_rows.append((cid, largest_ratio))

    stats_df = pd.DataFrame(stats_rows, columns=["id", "valid_ratio_largest"])
    df_final = df_mat.merge(stats_df, on="id", how="left")
    df_final["valid_ratio_largest"] = df_final["valid_ratio_largest"].fillna(1.0)
    return df_final


def compute_weighted_estimates(df_final):
    logger.info("Konvertiere 'positions' zu Decimal für Referenz und float für Fast Sum ...")
    positions_decimal = [safe_to_decimal(x) for x in tqdm(df_final["positions"], desc="Positions -> Decimal", ncols=100)]
    df_final["positions_decimal"] = [str(x) for x in positions_decimal]

    positions_float = np.array([float(x) for x in positions_decimal], dtype=np.float64)
    valid_ratio_float = df_final["valid_ratio_largest"].to_numpy(dtype=np.float64)
    weighted_estimated_float = positions_float * valid_ratio_float

    df_final["weighted_estimated_legal_str"] = [str(Decimal(str(x))) for x in weighted_estimated_float]

    total_positions = np.sum(positions_float, dtype=np.float64)
    total_estimated_legal = np.sum(weighted_estimated_float, dtype=np.float64)

    logger.info(f"Gesamt regelunabhängig (positions) ≈ {total_positions:.3e}")
    logger.info(f"Gesamt geschätzte legale Stellungen ≈ {total_estimated_legal:.3e}")
    logger.info(f"Verhältnis legal / theoretisch ≈ {total_estimated_legal/total_positions:.6e}")
    return df_final


def save_results(df_final):
    out_cols = list(df_final.columns)
    df_final[out_cols].to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Analyse vollständig gespeichert: {OUTPUT_FILE}")
    logger.info("✅ Fertig.")


def main():
    df_mat = load_material_classes(MATERIAL_FILE)
    df_res = load_results(RESULT_FILE)
    df_final = compute_ratios(df_mat, df_res)
    df_final = compute_weighted_estimates(df_final)
    save_results(df_final)


if __name__ == "__main__":
    main()
