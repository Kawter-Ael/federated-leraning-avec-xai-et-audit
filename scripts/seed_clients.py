"""Seed pre-provisioned client accounts into MongoDB.

Run once after `docker compose up mongodb` or before first portal launch.
Idempotent — skips existing accounts.

Usage:
    python -m scripts.seed_clients
    MONGODB_URI=mongodb://localhost:27017 python -m scripts.seed_clients

Default credentials are intentionally simple for the academic prototype.
Change passwords in production via the ENSAJ_CLIENT_PASSWORDS env var or
by editing the SEED_CLIENTS list below.
"""
from __future__ import annotations

import os
import sys

# Allow running as: python -m scripts.seed_clients from project root
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from shared.auth_store import create_client_user, get_users_collection

# ---------------------------------------------------------------------------
# Client roster — edit here or override via ENSAJ_CLIENT_PASSWORDS env var
# Format for env var: "client_1:Pass1!,client_2:Pass2!" (comma-separated)
# ---------------------------------------------------------------------------
_DEFAULT_CLIENTS: list[tuple[str, str]] = [
    ("client_1", "Client1Pass!"),
    ("kawter", "kawter"),
    ("client_3", "Client3Pass!"),
    ("client_4", "Client4Pass!"),
    ("client_5", "Client5Pass!"),
]


def _load_client_roster() -> list[tuple[str, str]]:
    raw = os.environ.get("ENSAJ_CLIENT_PASSWORDS", "").strip()
    if not raw:
        return list(_DEFAULT_CLIENTS)
    roster: list[tuple[str, str]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" not in entry:
            print(f"[seed] WARNING: skipping malformed entry '{entry}' (expected user:pass)")
            continue
        username, password = entry.split(":", 1)
        roster.append((username.strip(), password.strip()))
    return roster


def seed_clients() -> None:
    roster = _load_client_roster()
    existing = {
        doc["username"]
        for doc in get_users_collection().find({}, {"username": 1})
    }

    created = 0
    skipped = 0
    for username, password in roster:
        normalized = username.strip().lower()
        if normalized in existing:
            print(f"[seed] SKIP  {normalized!r} — already exists")
            skipped += 1
            continue
        ok, message = create_client_user(username, password)
        if ok:
            print(f"[seed] OK    {normalized!r}")
            created += 1
        else:
            print(f"[seed] ERROR {normalized!r}: {message}")

    print(f"\n[seed] Done — {created} created, {skipped} skipped.")


if __name__ == "__main__":
    seed_clients()
