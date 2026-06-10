"""Simple entrypoint that runs data preparation (Phase 2) and prints a summary."""
from __future__ import annotations

import json
import logging
import sys

from shared.data_preparation import prepare_phase2

logger = logging.getLogger("run_phase2")


def main() -> None:
    logger.info("Debut de la Phase 2 - Preparation des donnees...")
    try:
        prepared = prepare_phase2()
    except Exception:
        logger.exception("Phase 2 - Echec de la preparation des donnees.")
        sys.exit(1)

    summary = {
        "status": "completed",
        "raw_shape": prepared.metadata["raw_shape"],
        "cleaned_shape": prepared.metadata["shape_after_missing_drop"],
        "train_shape": prepared.metadata["train_shape"],
        "validation_shape": prepared.metadata["validation_shape"],
        "test_shape": prepared.metadata["test_shape"],
        "num_clients": len(prepared.client_splits),
        "metadata_path": "artifacts/data/metadata/preprocessing_metadata.json",
    }
    print(json.dumps(summary, indent=2))
    logger.info("Phase 2 terminee. Shape finale : %s", prepared.metadata["shape_after_missing_drop"])


if __name__ == "__main__":
    main()
