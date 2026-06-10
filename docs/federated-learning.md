# Federated Learning

## Overview

The project uses a Flower server as the central coordinator and authenticated users as logical federated clients. Each user prepares local data, trains locally, and sends safe updates to the server.

## Design / Methodology

The global model is a PyTorch binary classifier for the diabetes use case. The server applies `FedAvg` over client model updates and persists the global model and round metrics. Clients do not transmit raw rows or per-instance explainability traces.

## Implementation Summary

Each client:

- loads its own dataset
- rebuilds local splits using the shared preprocessing schema
- receives global weights from the Flower server
- trains locally on CPU
- computes metrics, SHAP summary and rule summary
- sends only model updates and safe summaries

The server:

- aggregates model parameters
- selects a global threshold
- stores round-level metrics
- generates global explainability and audit artifacts after training

## Outputs / Artifacts

- `artifacts/models/global_model.pt`
- `artifacts/metrics/federated_training_metrics.json`

## Multi-client Docker mode (soutenance)

En mode soutenance/production Streamlit, chaque client FL est lancé dans un container Docker séparé avec l'image `ensaj-fl-client`. Chaque container reçoit sa propre partition de données et communique avec le serveur Flower via le réseau Docker `ensaj_fl-net`. Le serveur agrège les poids des clients avec FedAvg, pondéré par le nombre réel d'exemples.

### Preuve d'exécution validée

**Run ID:** `principal-20260506220641-2ff86b85`

| Élément | Valeur |
|---|---|
| Containers spawned | `ensaj-fl-client_1-client_1`, `ensaj-fl-client_2-client_2` |
| Image | `ensaj-fl-client` |
| Réseau | `ensaj_fl-net` |
| `observed_client_ids` | `["client_1", "client_2"]` |
| `num_examples` | client_1 = 269, client_2 = 268, total = 537 |
| `num_clients_participated` | 2 (rounds 1 et 2) |
| Pipeline complet | server start → FL rounds → XAI → audit |

### Commande de vérification pendant training

```bash
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
# Attendu :
# ensaj-fl-client_2-client_2   ensaj-fl-client   Up N seconds
# ensaj-fl-client_1-client_1   ensaj-fl-client   Up N seconds
```

## Limitations

- this remains an academic prototype
- the default deployment is single-host Docker, even though the FL flow is networked through Flower
