"""DAG prétraitement — fichiers bruts → CSV de features → DAG d'entraînement.

Le passage des données se fait par le disque partagé (/opt/project) et par
la conf du run déclenché. Ce DAG n'entraîne pas le modèle.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

PROJECT = Path("/opt/project")
PRE = PROJECT / "donnees_pluie_prevue" / "preprocessed"
PYTHON = sys.executable

CHIRPS_CSV = PRE / "features_chirps_sahel.csv"
IMERG_CSV = PRE / "features_imerg_sahel.csv"
HANDOFF = PRE / "handoff.json"

DEFAULT_ARGS = {
    "owner": "pluie-prevue",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def verifier_features() -> None:
    """Échoue si le prétraitement n'a pas produit les tables attendues."""
    missing = [p for p in (CHIRPS_CSV, IMERG_CSV) if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "Tables absentes ou vides : " + ", ".join(str(p) for p in missing)
        )

    payload = {
        "ready_at": datetime.utcnow().isoformat() + "Z",
        "features_csv": str(CHIRPS_CSV),
        "features_imerg_csv": str(IMERG_CSV),
        "chirps_bytes": CHIRPS_CSV.stat().st_size,
        "imerg_bytes": IMERG_CSV.stat().st_size,
    }
    HANDOFF.write_text(json.dumps(payload, indent=2))
    print(
        f"OK CHIRPS {payload['chirps_bytes'] / 1e6:.1f} Mo, "
        f"IMERG {payload['imerg_bytes'] / 1e6:.1f} Mo → {HANDOFF}"
    )


with DAG(
    dag_id="pluie_pretraitement",
    description="GeoTIFF / indices / SST → CSV de features, puis déclenche l'entraînement",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=["nasa-hack", "pluie", "pretraitement"],
) as dag:
    # Réécrit les mêmes fichiers : un retry ne duplique pas les lignes.
    preparer = BashOperator(
        task_id="preparer_features",
        bash_command=(
            f"cd {PROJECT} && {PYTHON} pretraiter.py --out {PRE}"
        ),
        execution_timeout=timedelta(hours=2),
    )

    verifier = PythonOperator(
        task_id="verifier_features",
        python_callable=verifier_features,
    )

    # Le modèle n'est entraîné que sur le CSV CHIRPS (périodes CHIRPS et IMERG disjointes).
    envoyer = TriggerDagRunOperator(
        task_id="envoyer_vers_entrainement",
        trigger_dag_id="pluie_prevue_pipeline",
        conf={
            "features_csv": str(CHIRPS_CSV),
            "features_imerg_csv": str(IMERG_CSV),
            "handoff": str(HANDOFF),
        },
        wait_for_completion=False,
    )

    preparer >> verifier >> envoyer
