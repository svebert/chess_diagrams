import os
import time
import pandas as pd
import logging
from tqdm import tqdm
from rand_legal_pos import random_board_from_material

# === Logging konfigurieren ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# === Parameter ===
MATERIAL_FILE = "material_classes_positions.parquet"
RESULT_FILE = "legality_results.parquet"
INITIAL_SAMPLES = [1000, 2000]
MAX_SAMPLE = 128_000
REL_THRESHOLD = 0.10  # relative Abweichung für adaptive Samplegröße
ABS_THRESHOLD = 1e-6  # absolute Abweichung unterhalb dieser Zahl gilt als stabil


def test_legality_for_class(white_material, black_material, sample_size: int) -> float:
    """Generiert sample_size Zufallsstellungen und prüft, ob sie legal sind."""
    valid = 0
    for _ in range(sample_size):
        try:
            board = random_board_from_material(white_material, black_material)
            if board.is_valid():
                valid += 1
        except Exception:
            continue
    return valid / sample_size if sample_size > 0 else 0.0


def main():
    # === Materialklassen laden und nach positions sortieren ===
    df = pd.read_parquet(MATERIAL_FILE)
    df["positions"] = df["positions"].apply(lambda x: float(x))
    df = df.sort_values("positions", ascending=False).reset_index(drop=True)
    logging.info(f"{len(df)} Materialklassen geladen und nach 'positions' sortiert.")

    # === Ergebnisse vorbereiten oder laden ===
    if os.path.exists(RESULT_FILE):
        results = pd.read_parquet(RESULT_FILE)
        logging.info(f"Bestehende Ergebnisse geladen ({len(results)} Zeilen).")
    else:
        results = pd.DataFrame(columns=["id", "sample_size", "valid_ratio"])
        logging.info("Keine bestehenden Ergebnisse gefunden, starte neu.")

    # === Hauptschleife über Materialklassen ===
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Klassen"):
        class_id = row["id"]
        white_material = eval(row["white"])
        black_material = eval(row["black"])

        # Vorhandene Ergebnisse für die Klasse
        class_results = results.loc[results["id"] == class_id].copy()
        sample_sizes_done = sorted([int(s) for s in class_results["sample_size"].tolist()])

        # Startwerte für die Schleife
        if len(sample_sizes_done) < 2:
            # Fehlende INITIAL_SAMPLES berechnen
            pending_samples = [s for s in INITIAL_SAMPLES if s not in sample_sizes_done]
        else:
            # Prüfen auf Differenz zwischen größtem und zweitgrößtem Sample
            last_two = class_results.sort_values("sample_size", ascending=False).head(2)
            ratio_max = last_two.iloc[0]["valid_ratio"]
            ratio_second = last_two.iloc[1]["valid_ratio"]
            rel_diff = abs(ratio_max - ratio_second) / max(abs(ratio_second), 1e-12)
            abs_diff = abs(ratio_max - ratio_second)

            if rel_diff > REL_THRESHOLD and abs_diff > ABS_THRESHOLD:
                next_sample = min(int(last_two.iloc[0]["sample_size"] * 2), MAX_SAMPLE)
                pending_samples = [next_sample]
            else:
                pending_samples = []

        for sample_size in pending_samples:
            sample_size = int(sample_size)  # Sicherheit: int für range()
            start = time.time()
            ratio = test_legality_for_class(white_material, black_material, sample_size)
            duration = time.time() - start

            logging.info(
                f"→ Klasse {class_id}, Sample={sample_size}: "
                f"{ratio:.4e} gültig ({duration:.1f}s)"
            )

            # Ergebnis speichern
            new_row = pd.DataFrame([{
                "id": class_id,
                "sample_size": sample_size,
                "valid_ratio": ratio
            }])
            results = pd.concat([results, new_row], ignore_index=True)
            results.to_parquet(RESULT_FILE, index=False)

        if i % 1000 == 0:
            logging.info(f"→ Fortschritt: {i}/{len(df)} Klassen verarbeitet ...")

    logging.info("Simulation abgeschlossen ✅")
    logging.info(f"Ergebnisse gespeichert in: {RESULT_FILE}")


if __name__ == "__main__":
    main()
