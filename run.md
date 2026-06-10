## Setup & Run (docker_per_user mode)

### 1. Configure environment
```bash
cp .env.example .env
```
Edit `.env` — set absolute host paths:
```
FL_ARTIFACT_ROOT_HOST=url de projet/artifacts
FL_DATA_ROOT_HOST=url de projet/data
FL_CONFIG_ROOT_HOST=url de projet /config
```

### 2. Build all images (including fl-client)
```bash
docker compose --profile fl build
```

### 3. Start services
```bash
docker compose up -d
```

Starts: `mongodb`, `dashboard`, and `client-app`.
No static server is used — the client-app spawns a server dynamically for each training run.

If you encounter an error (failed to copy: httpReadSeeker: failed open: failed to do request:):
```bash
docker pull mongo:7
docker compose up -d
```

### 4. Seed client accounts (one-time)
```bash
docker compose exec client-app python -m scripts.seed_clients
```
Creates `client_1`…`client_5` in MongoDB (default passwords in `docs/client-credentials.md`).

### 5. Use the portal
| Service | URL |
|---|---|
| Client portal (FL training) | http://localhost:8502 |
| Dashboard (aggregate results) | http://localhost:8501 |

Login → upload dataset → configure → Train.
Each training run spawns a dedicated `ensaj-fl-client` container automatically.

---

---

## Scénario de démonstration multi-client (soutenance)

> **Recommandé : `num_clients=2`** — rapide (~3 min) et stable. `num_clients=3` possible si le temps le permet.

### Étapes

1. Ouvrir Chrome → http://localhost:8502
2. Login `client_1` / `Client1Pass!`
3. Onglet **Dataset** → uploader `data/diabetes.csv`
4. Configurer : `num_clients=2`, `rounds=2`, `epochs=3`, target=`Outcome`, positive class=`1`
5. Cliquer **Validate and prepare run** → noter le Run ID affiché
6. Phase 2 → attendre complétion
7. Onglet **Training** → cliquer **Start training**
8. **Pendant l'entraînement**, dans un terminal :
   ```bash
   docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
   ```
   Résultat attendu :
   ```
   ensaj-fl-client_2-client_2   ensaj-fl-client   Up N seconds
   ensaj-fl-client_1-client_1   ensaj-fl-client   Up N seconds
   ensaj-client-app             ...               Up ...
   ```
9. Attendre **Done!** (progressbar 100%)
10. Ouvrir http://localhost:8501 → sélectionner le Run ID
11. Montrer `observed_client_ids = ["client_1", "client_2"]`
12. Montrer `per_client_metrics` : client_1 (269 ex.), client_2 (268 ex.)
13. Montrer FedAvg aggregation : `total_examples = 537`
14. Montrer SHAP global agrégé + règles des deux clients
15. Montrer audit : status `passed` ou `warning`

---

## Questions/Réponses jury

**Q : Est-ce que chaque client est vraiment dans un container Docker ?**
> Oui. Dans la version de soutenance, quand on lance `num_clients=2` depuis Chrome, le système crée deux containers Docker séparés avec l'image `ensaj-fl-client`. Chaque container a son propre volume de données et ses propres logs. Nous avons validé cela avec `docker ps` pendant le training.

**Q : Quelle preuve avez-vous ?**
> Sur le run `principal-20260506220641-2ff86b85`, `docker ps` a montré `ensaj-fl-client_1-client_1` et `ensaj-fl-client_2-client_2` simultanément. Le dashboard affiche `observed_client_ids = ["client_1", "client_2"]`, avec `per_client_metrics` pour les deux clients (269 + 268 = 537 exemples).

**Q : Est-ce que ce sont des subprocess ?**
> Non pour le mode soutenance Docker. Le mode `docker_per_user` utilise de vrais containers Docker par client. Les tests unitaires peuvent utiliser des subprocess pour rester rapides, mais le scénario de démonstration utilise exclusivement `docker_per_user` avec un container par client.

**Q : Pourquoi les client_ids sont client_1, client_2 et non sim_client_X ?**
> Le premier client (index 0) prend l'identifiant de l'utilisateur connecté (`client_1`). Les clients supplémentaires reçoivent des IDs `client_2`, `client_3`, etc. — nommage cohérent avec les comptes MongoDB seedés.

---

### Note Darija

F soutenance, multi-client maشي ghir simulation. Ila khdmina `num_clients=2`, système kaytl3 jوج containers Docker b image `ensaj-fl-client`. Kola client 3ndo partition dyalo, logs dyalo, metrics dyalo. Server Flower kayjme3 weights dyalhom b FedAvg 3la 7sab `num_examples` الحقيقي.

---

### Optional: static server mode (legacy docker mode)
```bash
docker compose --profile fl up
```
Starts server + fl-client-1/2/3 in addition to the default services.

### Rebuild after code changes
```bash
docker compose --profile fl build && docker compose up -d
```
