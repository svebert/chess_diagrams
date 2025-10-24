#!/usr/bin/env python3
"""
Analyze legality estimation across material classes.

Inputs:
 - material_classes_diagrams.parquet
      Columns: id, white, black, diagrams
 - legality_results.parquet
      Columns: id, sample_size, legal_ratio

Process:
 - For each class, select the legal_ratio from the largest sample_size.
 - Compute estimated number of legal positions: diagrams * legal_ratio.
 - Compute global totals and save final merged dataset.

Output:
 - legality_analysis.parquet
"""

from decimal import Decimal, InvalidOperation
from typing import Union, Iterable
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

# File paths
MATERIAL_FILE = "material_classes_diagrams.parquet"
RESULT_FILE = "legality_results.parquet"
OUTPUT_FILE = "legality_analysis.parquet"


def safe_to_decimal(x: Union[str, float, int, None]) -> Decimal:
    """Safely convert input to Decimal. Falls back to 1 if invalid."""
    if pd.isna(x):
        return Decimal(1)
    try:
        return Decimal(str(x))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(1)


def get_largest_sample_ratio(group: pd.DataFrame) -> float:
    """
    Select the legal_ratio associated with the largest sample_size in the group.
    """
    if len(group) == 0:
        return 1.0
    row = group.sort_values("sample_size", ascending=False).iloc[0]
    return float(row["legal_ratio"])


def load_material_classes(filename: str) -> pd.DataFrame:
    """Load material class data."""
    if not os.path.exists(filename):
        logger.error(f"Material file not found: {filename}")
        raise FileNotFoundError(filename)
    df = pd.read_parquet(filename)
    logger.info(f"Loaded {len(df):,} material classes.")
    return df


def load_results(filename: str) -> pd.DataFrame:
    """Load legality result table, return empty table if missing."""
    if not os.path.exists(filename):
        logger.warning(f"Results file not found: {filename} — all ratios will default to 1.0.")
        return pd.DataFrame(columns=["id", "sample_size", "legal_ratio"])
    df = pd.read_parquet(filename)
    logger.info(f"Loaded {len(df):,} legality result rows.")
    return df


def compute_ratios(df_mat: pd.DataFrame, df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Attach largest-sample legal_ratio to each material class.
    """
    tqdm.pandas()
    grouped = df_res.groupby("id") if len(df_res) > 0 else {}

    logger.info("Determining largest-sample legal ratios per class...")
    result_list = []
    for cid in tqdm(df_mat["id"], desc="Ratios", ncols=100, ascii=True):
        if len(df_res) == 0 or cid not in grouped.groups:
            ratio = 1.0
        else:
            ratio = get_largest_sample_ratio(grouped.get_group(cid))
        result_list.append((cid, ratio))

    ratios_df = pd.DataFrame(result_list, columns=["id", "legal_ratio_largest"])
    merged = df_mat.merge(ratios_df, on="id", how="left")
    merged["legal_ratio_largest"].fillna(1.0, inplace=True)
    return merged


def compute_weighted_estimates(df_final: pd.DataFrame) -> pd.DataFrame:
    """
    Compute estimated number of legal diagrams per class, and global totals.
    """
    logger.info("Converting diagrams to Decimal (reference-safe) and float (fast sum)...")

    diagrams_decimal: Iterable[Decimal] = [
        safe_to_decimal(x) for x in tqdm(df_final["diagrams"], desc="diagrams→Decimal", ncols=100, ascii=True)
    ]
    df_final["diagrams_decimal"] = [str(x) for x in diagrams_decimal]

    diagrams_float = np.array([float(x) for x in diagrams_decimal], dtype=np.float64)
    legal_ratio = df_final["legal_ratio_largest"].to_numpy(dtype=np.float64)
    weighted = diagrams_float * legal_ratio

    df_final["estimated_legal_diagrams_str"] = [str(Decimal(str(v))) for v in weighted]

    total_theoretical = np.sum(diagrams_float, dtype=np.float64)
    total_estimated = np.sum(weighted, dtype=np.float64)

    logger.info(f"Total theoretical diagrams ≈ {total_theoretical:.3e}")
    logger.info(f"Total estimated legal diagrams ≈ {total_estimated:.3e}")
    logger.info(f"Fraction legal ≈ {total_estimated / total_theoretical:.6e}")

    return df_final


def save_results(df_final: pd.DataFrame) -> None:
    """Save full analysis table."""
    df_final.to_parquet(OUTPUT_FILE, index=False)
    logger.info(f"Saved analysis: {OUTPUT_FILE}")
    logger.info("✅ Done.")


def main() -> None:
    df_mat = load_material_classes(MATERIAL_FILE)
    df_res = load_results(RESULT_FILE)
    df_final = compute_ratios(df_mat, df_res)
    df_final = compute_weighted_estimates(df_final)
    save_results(df_final)


if __name__ == "__main__":
    main()
