# Infra — Airflow + Prometheus + Grafana

Orchestration du pipeline ML et monitoring pour **Pluie prévue**.

## Démarrage

```bash
cd infra
docker compose up -d --build
```

| Service | URL | Login |
|---------|-----|-------|
| Airflow | http://localhost:8080 | admin / admin |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Metrics exporter | http://localhost:9101/metrics | — |

## DAG `pluie_pretraitement`

1. `preparer_features` — lance `pretraiter.py` (GeoTIFF, indices, SST → CSV)
2. `verifier_features` — exige les deux CSV et écrit `preprocessed/handoff.json`
3. `envoyer_vers_entrainement` — déclenche `pluie_prevue_pipeline` avec le chemin du CSV

Les deux DAG partagent le disque du projet. Le modèle est entraîné sur le CSV CHIRPS. Le CSV IMERG est produit et vérifié, mais les deux périodes ne se recouvrent pas, donc il n'entre pas dans l'entraînement.

## DAG `pluie_prevue_pipeline`

1. `check_data_ready` — vérifie le CSV reçu (ou `features_chirps_sahel.csv` si le run est manuel)
2. `train_lead1` / `train_lead3` — lance `train_model.py` (lourd en RAM)
3. `publish_metrics` — écrit `infra/monitoring/ml_metrics.json` pour Grafana

Déclenchement **manuel** (pas de schedule) depuis l’UI Airflow. Lancer `pluie_pretraitement` pour enchaîner les deux.

## API + Prometheus

```bash
# depuis la racine du projet
.venv/bin/pip install prometheus-fastapi-instrumentator
.venv/bin/python api.py --host 0.0.0.0 --port 8000
```

Prometheus scrape `host.docker.internal:8000/metrics`.

## Arrêt

```bash
cd infra && docker compose down
```
