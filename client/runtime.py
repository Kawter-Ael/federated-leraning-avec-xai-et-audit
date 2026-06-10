from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from flwr.client import start_numpy_client

from client.federated_client import FederatedClient
from shared.federated_data import load_project_config
from shared.utils import apply_config_override

logger = logging.getLogger("client.runtime")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed Flower client runtime.")
    parser.add_argument(
        "--client-id", default=os.environ.get("CLIENT_ID", "client_user")
    )
    parser.add_argument(
        "--server-address", default=os.environ.get("FL_SERVER_ADDRESS", "server:8080")
    )
    parser.add_argument(
        "--artifact-root", default=os.environ.get("CLIENT_ARTIFACT_ROOT", "artifacts")
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("CLIENT_CONNECT_RETRIES", "20")),
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=float(os.environ.get("CLIENT_CONNECT_DELAY", "3")),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(
        os.environ.get(
            "CLIENT_CONFIG_PATH", Path(args.artifact_root) / "effective_config.json"
        )
    )
    config = load_project_config(
        config_path if config_path.exists() else "config/project-config.json"
    )
    override_json = os.environ.get("CONFIG_OVERRIDE_JSON")
    if override_json:
        apply_config_override(config, json.loads(override_json))
    user_context = {
        "client_id": args.client_id,
        "username": args.client_id,
        "artifact_root": args.artifact_root,
        "dataset_path": config["data"]["raw_dataset_path"],
        "global_artifact_root": os.environ.get("GLOBAL_ARTIFACT_ROOT", "artifacts"),
        "config_override": json.loads(override_json) if override_json else None,
    }

    try:
        client = FederatedClient(
            client_id=args.client_id, config=config, user_context=user_context
        )
    except Exception as exc:
        error_path = Path(args.artifact_root) / "client_prepare_error.json"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(
            json.dumps(
                {
                    "phase": "prepare_data",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "client_id": args.client_id,
                }
            ),
            encoding="utf-8",
        )
        logger.exception("Client prepare_data failed for %s", args.client_id)
        raise

    attempts = 0
    while True:
        attempts += 1
        try:
            start_numpy_client(
                server_address=args.server_address,
                client=client,
                insecure=True,
            )
            break
        except (ConnectionRefusedError, OSError):
            logger.warning(
                "Attempt %d/%d failed (server %s).",
                attempts,
                args.max_retries,
                args.server_address,
            )
            if attempts >= args.max_retries:
                raise
            time.sleep(args.retry_delay)


if __name__ == "__main__":
    main()
