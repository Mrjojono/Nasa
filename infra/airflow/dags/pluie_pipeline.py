"""DAG pluie prévue — out/ → entraînement → métriques → API REST."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT = Path("/opt/project")
OUT = PROJECT / "donnees_pluie_prevue" / "out"
MONITORING = Path("/opt/monitoring")
PYTHON = sys.executable
API_BASE = "http://api:8000"

DEFAULT_ARGS = {
    "owner": "pluie-prevue",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def check_data_ready(**context) -> None:
    dag_run = context.get("dag_run")
    conf = dict(dag_run.conf or {}) if dag_run is not None else {}
    csv_path = Path(conf.get("features_csv") or (OUT / "features_modele.csv"))
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
        per_target = OUT / f"metrics_{target}.json"
        published = MONITORING / f"metrics_{target}.json"
        if per_target.exists():
            block = json.loads(per_target.read_text())
            models[target] = block
            published.write_text(json.dumps(block, indent=2, ensure_ascii=False))

    payload = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "models": models,
    }
    out = MONITORING / "ml_metrics.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Publié {out} ({len(models)} modèles)")
    if len(models) < 2:
        raise RuntimeError("Métriques incomplètes : les deux cibles doivent être entraînées")


def attendre_api() -> None:
    """Le conteneur API tourne déjà ; on attend qu'il serve le contrat JSON."""
    health_url = f"{API_BASE}/api/health"
    last_error = "aucun essai"
    for _ in range(30):
        try:
            with urllib.request.urlopen(health_url, timeout=120) as response:
                health = json.loads(response.read().decode())
            if health.get("status") != "ok" or set(health.get("models_loaded") or []) < {1, 3}:
                last_error = f"health={health}"
                time.sleep(2)
                continue
            with urllib.request.urlopen(f"{API_BASE}/api/afrique", timeout=120) as response:
                afrique = json.loads(response.read().decode())
            if afrique.get("resource") != "afrique":
                raise RuntimeError("la réponse /api/afrique n'est pas le JSON Afrique")
            if "region" not in afrique or "forecasts" not in afrique:
                raise RuntimeError(f"JSON Afrique incomplet : {list(afrique)}")
            print(
                f"API JSON Afrique prête, prévisions : {sorted(afrique['forecasts'])}"
            )
            return
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"API pas prête ({API_BASE}) : {last_error}")


with DAG(
    dag_id="pluie_prevue_pipeline",
    description="out/ → XGBoost lead 1/3 → métriques Grafana → API REST",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["nasa-hack", "pluie", "xgboost"],
) as dag:
    t_check = PythonOperator(
        task_id="check_data_ready",
        python_callable=check_data_ready,
    )

    train_cmd = (
        f"cd {PROJECT} && {PYTHON} train_model.py "
        f"--target {{{{ params.target }}}} "
        f"--csv {OUT}/features_modele.csv "
        f"--out {OUT}"
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

    t_api = PythonOperator(
        task_id="verifier_api",
        python_callable=attendre_api,
    )

    t_check >> [t_train_1, t_train_3] >> t_publish >> t_api
