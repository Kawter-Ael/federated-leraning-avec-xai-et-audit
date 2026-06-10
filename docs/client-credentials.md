# Client Credentials

Pre-provisioned accounts seeded by `scripts/seed_clients.py`.

> **Change these passwords** before any non-demo deployment.

## Default accounts

| Username   | Password       | Role   |
|------------|---------------|--------|
| `client_1` | `Client1Pass!` | client |
| `client_2` | `Client2Pass!` | client |
| `client_3` | `Client3Pass!` | client |
| `client_4` | `Client4Pass!` | client |
| `client_5` | `Client5Pass!` | client |

## How to seed

```bash
# With Docker Compose running (MongoDB must be healthy):
docker compose exec client-app python -m scripts.seed_clients

# Locally (MongoDB on localhost:27017):
python -m scripts.seed_clients

# Custom passwords via env var:
ENSAJ_CLIENT_PASSWORDS="alice:AlicePass!,bob:BobPass!" python -m scripts.seed_clients
```

## How to add a new client

```bash
# Re-run seed with expanded ENSAJ_CLIENT_PASSWORDS (existing accounts are skipped):
ENSAJ_CLIENT_PASSWORDS="client_1:Client1Pass!,...,client_6:NewPass!" python -m scripts.seed_clients
```

## What each client gets on login

1. A dedicated session with their own run workspace (`artifacts/user_runs/<run_id>/`).
2. Their own FL client Docker container (`ensaj-fl-client`) spawned automatically when they start training.
3. The container connects to the shared FL server over `fl-net` and trains on their uploaded dataset.
4. After training, results (model, SHAP, audit) are stored in their run workspace.

## Self-signup is disabled

The "Create account" form has been removed from the portal. Client accounts can only be created via `scripts/seed_clients.py` or by a system administrator with MongoDB access. This ensures institutional control over FL participants.
