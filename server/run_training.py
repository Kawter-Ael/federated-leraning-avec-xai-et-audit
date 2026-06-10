"""Thin entrypoint for launching the federated training server."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from server.federated_trainer import run_distributed_training
from shared.run_phase2 import main as run_phase2_main

logger = logging.getLogger("server.run_training")


def main() -> None:
    server_address = os.environ.get("FL_SERVER_ADDRESS", "0.0.0.0:8080")
    override_json = os.environ.get("CONFIG_OVERRIDE_JSON")
    config_override = json.loads(override_json) if override_json else None
    force_exit = os.environ.get("FL_FORCE_EXIT_ON_COMPLETE", "0") == "1"
    metadata_path = Path("artifacts/data/metadata/preprocessing_metadata.json")
    if not metadata_path.exists() and not config_override:
        logger.info("Artefacts Phase 2 absents et aucun override. Lancement de la Phase 2...")
        try:
            run_phase2_main()
        except Exception:
            logger.exception("Phase 2 - Echec.")
            sys.exit(1)

    logger.info("Demarrage du serveur Flower sur %s (force_exit=%s).", server_address, force_exit)
    try:
        result = run_distributed_training(
            server_address=server_address,
            config_override=config_override,
            force_exit_on_complete=force_exit,
        )
        print(json.dumps(result, indent=2))
    except Exception:
        logger.exception("Entrainement federate - Echec.")
        sys.exit(1)


if __name__ == "__main__":
    main()
