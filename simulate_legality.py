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
SAMPLE_SIZES = [100, 1000, 10000]


def test_legality_for_class(white_material, black_material, sample_size):
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
    # === Materialklassen laden ===
    df = pd.read_parquet(MATERIAL_FILE)
    logging.info(f"{len(df)} Materialklassen aus {MATERIAL_FILE} geladen.")

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

        # Prüfen, ob alle Samplegrößen schon vorhanden sind
        existing_samples = set(results.loc[results["id"] == class_id, "sample_size"])
        pending_samples = [s for s in SAMPLE_SIZES if s not in existing_samples]

        if not pending_samples:
            continue  # alles schon berechnet

        if i % 1000 == 0:
            logging.info(f"→ Fortschritt: {i}/{len(df)} Klassen verarbeitet ...")

        logging.info(
            f"Klasse {class_id}: {white_material} vs {black_material} | "
            f"noch ausstehend: {pending_samples}"
        )

        for sample_size in pending_samples:
            start = time.time()
            ratio = test_legality_for_class(white_material, black_material, sample_size)
            duration = time.time() - start

            logging.info(
                f"→ Klasse {class_id}, {sample_size} Tests: "
                f"{ratio:.4f} gültig ({duration:.1f}s)"
            )

            # Ergebnis speichern
            new_row = pd.DataFrame(
                [{"id": class_id, "sample_size": sample_size, "valid_ratio": ratio}]
            )
            results = pd.concat([results, new_row], ignore_index=True)

            # Nach jedem Schritt abspeichern
            results.to_parquet(RESULT_FILE, index=False)

    logging.info("Simulation abgeschlossen ✅")
    logging.info(f"Ergebnisse gespeichert in: {RESULT_FILE}")


if __name__ == "__main__":
    main()
