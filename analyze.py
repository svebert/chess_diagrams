#!/usr/bin/env python3
"""
analyze.py

Lädt:
 - material_classes_positions.parquet (Spalten: id, white_material, black_material, positions)
 - legality_results.parquet (Spalten: id, sample_size, valid_ratio)

Berechnet:
 - gewichtete Mittel der valid_ratio (Gewichte = sample_size)
 - relative Abweichung pro Klasse
 - für jede Klasse: weighted_estimated_legal = positions * weighted_mean
 - Summen (Fast Sum) mit Ladebalken
 - speichert Ergebnisse inkl. instabiler Klassen
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
UNSTABLE_FILE = "unstable_classes.parquet"

# --- Hilfsfunktionen ---
def safe_to_decimal(x):
    """Konvertiert String/Nummer zu Decimal. Wenn fehlerhaft -> Decimal(1)."""
    if pd.isna(x):
        return Decimal(1)
    try:
        return Decimal(str(x))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(1)

def weighted_stats_from_group(ratios, weights):
    """Berechnet gewichtetes Mittel und relative StdDev."""
    ratios = np.array(ratios, dtype=float)
    weights = np.array(weights, dtype=float)
    mask = ~np.isnan(ratios) & (weights > 0)
    if not mask.any():
        return 1.0, 0.0
    r = ratios[mask]
    w = weights[mask]
    wm = np.average(r, weights=w)
    wvar = np.average((r - wm) ** 2, weights=w)
    rel_std = np.sqrt(wvar) / wm if wm > 0 else 0.0
    return float(wm), float(rel_std)

# --- Hauptprogramm ---
def main():
    tqdm.pandas()
    
    logger.info("Lade Materialklassen...")
    if not os.path.exists(MATERIAL_FILE):
        logger.error(f"Materialdatei nicht gefunden: {MATERIAL_FILE}")
        return
    df_mat = pd.read_parquet(MATERIAL_FILE)
    logger.info(f"{len(df_mat):,} Materialklassen geladen.")

    logger.info("Lade Simulationsergebnisse (legality_results)...")
    if not os.path.exists(RESULT_FILE):
        logger.warning(f"Resultatdatei nicht gefunden: {RESULT_FILE} — alle Faktoren = 1.0")
        df_res = pd.DataFrame(columns=["id", "sample_size", "valid_ratio"])
    else:
        df_res = pd.read_parquet(RESULT_FILE)
        logger.info(f"{len(df_res):,} Ergebnis-Zeilen geladen.")

    # --- Merge vorbereiten ---
    grouped = df_res.groupby("id") if len(df_res) > 0 else {}

    stats_rows = []
    ids = df_mat["id"].tolist()
    logger.info("Berechne gewichtete Mittel pro Klasse...")
    for cid in tqdm(ids, desc="Weighted stats", ncols=100):
        if len(df_res) == 0 or cid not in grouped.groups:
            stats_rows.append((cid, 1.0, 0.0, 0))
        else:
            grp = grouped.get_group(cid)
            ratios = pd.to_numeric(grp["valid_ratio"], errors="coerce").fillna(1.0).values
            weights = pd.to_numeric(grp["sample_size"], errors="coerce").fillna(0).values
            wm, relstd = weighted_stats_from_group(ratios, weights)
            total_w = int(np.nansum(weights))
            stats_rows.append((cid, wm, relstd, total_w))

    stats_df = pd.DataFrame(stats_rows, columns=["id", "weighted_mean_ratio", "relative_stddev", "total_weight"])
    df_final = df_mat.merge(stats_df, on="id", how="left")

    df_final["weighted_mean_ratio"] = df_final["weighted_mean_ratio"].fillna(1.0)
    df_final["relative_stddev"] = df_final["relative_stddev"].fillna(0.0)

    # --- Fast Sum Berechnung ---
    logger.info("Konvertiere 'positions' zu Decimal für Referenz und float für Fast Sum ...")
    positions_decimal = [safe_to_decimal(x) for x in tqdm(df_final["positions"], desc="Positions -> Decimal", ncols=100)]
    df_final["positions_decimal"] = [str(x) for x in positions_decimal]

    positions_float = np.array([float(x) for x in positions_decimal], dtype=np.float64)
    valid_ratio_float = df_final["weighted_mean_ratio"].to_numpy(dtype=np.float64)
    weighted_estimated_float = positions_float * valid_ratio_float

    df_final["weighted_estimated_legal_str"] = [str(Decimal(str(x))) for x in weighted_estimated_float]

    total_positions = np.sum(positions_float, dtype=np.float64)
    total_estimated_legal = np.sum(weighted_estimated_float, dtype=np.float64)

    logger.info(f"Gesamt regelunabhängig (positions) ≈ {total_positions:.3e}")
    logger.info(f"Gesamt geschätzte legale Stellungen ≈ {total_estimated_legal:.3e}")
    logger.info(f"Verhältnis legal / theoretisch ≈ {total_estimated_legal/total_positions:.6e}")

    # --- Instabile Klassen ---
    unstable = df_final[df_final["relative_stddev"] > 0.25]
    logger.info(f"Instabile Klassen (relative_stddev > 0.25): {len(unstable)}")
    if len(unstable) > 0:
        unstable.head(200).to_parquet(UNSTABLE_FILE, index=False)
        logger.info(f"Beispiel instabiler Klassen gespeichert: {UNSTABLE_FILE}")

    # --- Speichere Analyse ---
    out_cols = list(df_mat.columns) + ["weighted_mean_ratio", "relative_stddev",
                                      "positions_decimal", "weighted_estimated_legal_str", "total_weight"]
    df_final[out_cols].to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Analyse vollständig gespeichert: {OUTPUT_FILE}")
    logger.info("✅ Fertig.")

if __name__ == "__main__":
    main()
