"""DAG pluie prévue — contrôle données → entraînement → publication métriques."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT = Path("/opt/project")
PRE = PROJECT / "donnees_pluie_prevue" / "preprocessed"
MONITORING = Path("/opt/monitoring")
PYTHON = sys.executable

DEFAULT_ARGS = {
    "owner": "pluie-prevue",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def check_data_ready(**context) -> None:
    dag_run = context.get("dag_run")
    conf = dict(dag_run.conf or {}) if dag_run is not None else {}
    csv_path = Path(conf.get("features_csv") or (PRE / "features_chirps_sahel.csv"))
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV manquant : {csv_path}")
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    source = "DAG pluie_pretraitement" if conf.get("features_csv") else "fichier déjà présent"
    print(f"OK features CSV ({size_mb:.1f} Mo) via {source} : {csv_path}")
    if conf.get("handoff"):
        print(f"Handoff : {conf['handoff']}")


def publish_metrics() -> None:
    MONITORING.mkdir(parents=True, exist_ok=True)
    models: dict = {}

    for target in ("rain_next1_mm", "rain_next3_mm"):
        per_target = MONITORING / f"metrics_{target}.json"
        if per_target.exists():
            models[target] = json.loads(per_target.read_text())

    legacy = PRE / "metrics.json"
    if legacy.exists():
        block = json.loads(legacy.read_text())
        target = block.get("target", "unknown")
        models.setdefault(target, block)
        (MONITORING / f"metrics_{target}.json").write_text(
            json.dumps(block, indent=2, ensure_ascii=False)
        )

    payload = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "models": models,
    }
    out = MONITORING / "ml_metrics.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Publié {out} ({len(models)} modèles)")


with DAG(
    dag_id="pluie_prevue_pipeline",
    description="Prétraitement prêt → XGBoost lead 1/3 → métriques Grafana",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,  # déclenchement manuel (hackathon)
    catchup=False,
    tags=["nasa-hack", "pluie", "xgboost"],
) as dag:
    t_check = PythonOperator(
        task_id="check_data_ready",
        python_callable=check_data_ready,
    )

    # Entraînement : écrit aussi une copie dans /opt/monitoring
    train_cmd = (
        f'cd {PROJECT} && {PYTHON} train_model.py --target {{{{ params.target }}}} '
        f'--out {PRE} && '
        f'cp {PRE}/metrics.json {MONITORING}/metrics_{{{{ params.target }}}}.json'
    )

    t_train_1 = BashOperator(
        task_id="train_lead1",
        bash_command=train_cmd,
        params={"target": "rain_next1_mm"},
    )

    t_train_3 = BashOperator(
        task_id="train_lead3",
        bash_command=train_cmd,
        params={"target": "rain_next3_mm"},
    )

    t_publish = PythonOperator(
        task_id="publish_metrics",
        python_callable=publish_metrics,
    )

    t_check >> [t_train_1, t_train_3] >> t_publish
