"""DAG ETL — telecharger/in (n'importe quoi) → CSV du modèle.

Ce DAG ne lance pas le modèle. Il ne lit pas les GeoTIFF.
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
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pretraiter import verifier_table  # noqa: E402

ENTREE = PROJECT / "donnees_pluie_prevue" / "telecharger" / "in"
OUT = PROJECT / "donnees_pluie_prevue" / "out"
PYTHON = sys.executable
CSV = OUT / "features_modele.csv"
HANDOFF = OUT / "handoff.json"

DEFAULT_ARGS = {
    "owner": "pluie-prevue",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def verifier_features() -> None:
    info = verifier_table(CSV)
    payload = {
        "ready_at": datetime.utcnow().isoformat() + "Z",
        "entree": str(ENTREE),
        "features_csv": str(CSV),
        "columns": info["columns"],
        "bytes": info["bytes"],
    }
    HANDOFF.write_text(json.dumps(payload, indent=2))
    print(f"OK CSV modèle {info['bytes'] / 1e6:.1f} Mo, {info['columns']} colonnes")


with DAG(
    dag_id="pluie_pretraitement",
    description="telecharger/in → CSV nettoyé au contrat du modèle",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=["nasa-hack", "pluie", "etl"],
) as dag:
    preparer = BashOperator(
        task_id="nettoyer_csv",
        bash_command=(
            f"cd {PROJECT} && {PYTHON} etl.py --entree {ENTREE} --sortie {CSV}"
        ),
        execution_timeout=timedelta(hours=1),
    )

    verifier = PythonOperator(
        task_id="verifier_csv",
        python_callable=verifier_features,
    )

    envoyer = TriggerDagRunOperator(
        task_id="envoyer_vers_entrainement",
        trigger_dag_id="pluie_prevue_pipeline",
        conf={"features_csv": str(CSV), "handoff": str(HANDOFF)},
        wait_for_completion=False,
    )

    preparer >> verifier >> envoyer
