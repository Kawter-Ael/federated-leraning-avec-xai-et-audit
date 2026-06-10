#!/bin/bash
# Orchestrator: run Phase 2, partition data, then launch Docker FL training.
set -e

NUM_CLIENTS="${NUM_CLIENTS:-3}"

echo "=== Step 1: Run Phase 2 (data preparation) ==="
python -m shared.run_phase2

echo "=== Step 2: Partition data for ${NUM_CLIENTS} clients ==="
python scripts/partition_data.py --num-clients "$NUM_CLIENTS"

echo "=== Step 3: Start Docker FL training ==="
CLIENT_SERVICES=""
for i in $(seq 1 "$NUM_CLIENTS"); do
    CLIENT_SERVICES="$CLIENT_SERVICES fl-client-$i"
done
docker compose --profile fl up server $CLIENT_SERVICES
