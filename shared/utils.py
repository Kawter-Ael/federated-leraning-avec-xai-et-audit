"""Shared utility functions for the federated learning project."""

from __future__ import annotations

import json
import logging
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_pipeline_logger = logging.getLogger("pipeline.trace")


def utc_now() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> Any:
    """Load and parse a JSON file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_free_port() -> int:
    """Find and return a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def apply_config_override(target: dict[str, Any], override: dict[str, Any]) -> None:
    """Deep-merge *override* into *target* in place."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            apply_config_override(target[key], value)
        else:
            target[key] = value


def contains_forbidden_keys(obj: Any, forbidden_keys: set[str]) -> list[str]:
    """Recursively search *obj* for any forbidden dict key or scalar string value."""
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in forbidden_keys:
                found.append(key)
            found.extend(contains_forbidden_keys(value, forbidden_keys))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(contains_forbidden_keys(item, forbidden_keys))
    elif isinstance(obj, tuple):
        for item in obj:
            found.extend(contains_forbidden_keys(item, forbidden_keys))
    elif isinstance(obj, set):
        for item in obj:
            found.extend(contains_forbidden_keys(item, forbidden_keys))
    elif isinstance(obj, str) and obj in forbidden_keys:
        found.append(obj)
    return sorted(set(found))


class PipelineTracer:
    def __init__(self, artifact_root: str | Path, run_id: str = "") -> None:
        self._root = Path(artifact_root)
        self._run_id = run_id
        self._entries: list[dict[str, Any]] = []
        self._log_path = self._root / "pipeline_trace.jsonl"
        self._elapsed_cache: list[tuple[str, str, float]] = []
        self._written_count = 0

    def _write_new_entries(self) -> None:
        if self._written_count >= len(self._entries):
            return
        self._root.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as fh:
            for entry in self._entries[self._written_count :]:
                fh.write(json.dumps(entry) + "\n")
        self._written_count = len(self._entries)

    def mark(self, phase: str, event: str, detail: str = "") -> None:
        now = time.monotonic()
        entry = {
            "monotonic": now,
            "utc": utc_now(),
            "phase": phase,
            "event": event,
        }
        if detail:
            entry["detail"] = detail
        self._entries.append(entry)
        _pipeline_logger.info("[%s] %s %s", phase, event, detail)
        self._write_new_entries()

    def flush(self) -> Path:
        self._elapsed_cache.extend(self._compute_elapsed())
        self._write_new_entries()
        return self._log_path

    def _compute_elapsed(self) -> list[tuple[str, str, float]]:
        starts = {
            e["phase"]: e["monotonic"] for e in self._entries if e["event"] == "start"
        }
        result: list[tuple[str, str, float]] = []
        for e in self._entries:
            if e["event"] == "end" and e["phase"] in starts:
                result.append(
                    (
                        e["phase"],
                        e.get("detail", ""),
                        e["monotonic"] - starts[e["phase"]],
                    )
                )
        return result

    @property
    def elapsed_pairs(self) -> list[tuple[str, str, float]]:
        return self._elapsed_cache + self._compute_elapsed()

    def summary(self) -> dict[str, float]:
        return {phase: round(elapsed, 3) for phase, _, elapsed in self.elapsed_pairs}
