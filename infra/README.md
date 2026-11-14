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
| API | http://localhost:8000/api/health | — |

## DAG `pluie_pretraitement` (ETL)

Déposer les fichiers dans `donnees_pluie_prevue/telecharger/in`. L'entrée peut contenir n'importe quoi : seuls les CSV sont transformés.

1. `nettoyer_csv` — remet les colonnes dans l'ordre du contrat, écrit `out/features_modele.csv`, puis charge la table Postgres `features`
2. `verifier_csv` — refuse le fichier si les 30 colonnes du modèle ne sont pas là
3. `envoyer_vers_entrainement` — passe ce CSV au DAG du modèle

Le modèle n'est pas entraîné ici. Il ne reçoit pas les GeoTIFF.

## DAG `pluie_prevue_pipeline`

1. `check_data_ready` — le fichier reçu doit être `out/features_modele.csv`
2. `train_lead1` / `train_lead3` — entraînement uniquement sur ce CSV
3. `publish_metrics` — RMSE, MAE, R², biais pour Grafana et Prometheus
4. `verifier_api` — attend le JSON `GET /api/afrique`

Déclenchement **manuel** (pas de schedule) depuis l’UI Airflow. Lancer `pluie_pretraitement` pour enchaîner les deux.

## API + Prometheus

Le service `api` du compose reste démarré. Il lit la table Postgres `features` (chargée par l'ETL), recharge quand un nouveau chargement est enregistré, et sert le JSON. Les modèles restent des fichiers dans `out/`. `GET /api/afrique` renvoie la région, les prévisions et les alertes pour l'Afrique de l'Ouest.

Prometheus scrape `api:8000/metrics`. L'exporter `9101` publie les KPI d'entraînement (`pluie_model_rmse`, `pluie_model_mae`, `pluie_model_r2`, `pluie_model_bias`).

## Arrêt

```bash
cd infra && docker compose down
```
