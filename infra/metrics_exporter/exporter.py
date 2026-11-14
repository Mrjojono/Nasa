#!/usr/bin/env python3
"""Expose les métriques ML (JSON) et l'état des artefacts en Prometheus."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from prometheus_client import Gauge, Info, start_http_server

MONITORING = Path(os.environ.get("MONITORING_DIR", "/monitoring"))
PREPROCESSED = Path(
    os.environ.get(
        "PREPROCESSED_DIR",
        "/opt/project/donnees_pluie_prevue/out",
    )
)

RMSE = Gauge("pluie_model_rmse", "RMSE du modèle", ["target"])
MAE = Gauge("pluie_model_mae", "MAE du modèle", ["target"])
R2 = Gauge("pluie_model_r2", "R² du modèle", ["target"])
BIAS = Gauge("pluie_model_bias", "Biais (pred - réel)", ["target"])
N_TEST = Gauge("pluie_model_n_test", "Taille du jeu de test", ["target"])
N_TRAIN = Gauge("pluie_model_n_train", "Taille du jeu d'entraînement", ["target"])
ARTIFACT = Gauge(
    "pluie_artifact_present",
    "1 si le fichier artefact existe",
    ["name"],
)
LAST_REFRESH = Gauge(
    "pluie_metrics_last_refresh_unixtime",
    "Dernier rechargement des métriques (unix)",
)
META = Info("pluie_pipeline", "Métadonnées pipeline pluie prévue")


def _load_one(path: Path) -> None:
    data = json.loads(path.read_text())
    target = data.get("target") or path.stem.replace("metrics_", "")
    metrics = data.get("metrics") or {}
    if "rmse" in metrics:
        RMSE.labels(target=target).set(float(metrics["rmse"]))
    if "mae" in metrics:
        MAE.labels(target=target).set(float(metrics["mae"]))
    if "r2" in metrics:
        R2.labels(target=target).set(float(metrics["r2"]))
    if "bias" in metrics:
        BIAS.labels(target=target).set(float(metrics["bias"]))
    if "n_test" in metrics:
        N_TEST.labels(target=target).set(float(metrics["n_test"]))
    if "n_train" in data:
        N_TRAIN.labels(target=target).set(float(data["n_train"]))


def refresh() -> None:
    sources = []
    combined = MONITORING / "ml_metrics.json"
    if combined.exists():
        payload = json.loads(combined.read_text())
        # format: {"models": {"rain_next1_mm": {...}, ...}} ou un seul bloc
        if "models" in payload:
            for target, block in payload["models"].items():
                block = dict(block)
                block.setdefault("target", target)
                tmp = MONITORING / f".tmp_{target}.json"
                tmp.write_text(json.dumps(block))
                _load_one(tmp)
                tmp.unlink(missing_ok=True)
                sources.append(target)
        else:
            _load_one(combined)
            sources.append(payload.get("target", "unknown"))

    for path in sorted(MONITORING.glob("metrics_*.json")):
        _load_one(path)
        sources.append(path.stem)

    # fallback : metrics.json produit par train_model.py
    legacy = PREPROCESSED / "metrics.json"
    if legacy.exists() and not sources:
        _load_one(legacy)
        sources.append("legacy")

    for name, path in {
        "features_csv": PREPROCESSED / "features_chirps_sahel.csv",
        "model_lead1": PREPROCESSED / "xgb_rain_next1_mm.joblib",
        "model_lead3": PREPROCESSED / "xgb_rain_next3_mm.joblib",
        "metrics_json": PREPROCESSED / "metrics.json",
    }.items():
        ARTIFACT.labels(name=name).set(1.0 if path.exists() else 0.0)

    META.info(
        {
            "dataset": "chirps_sahel",
            "sources": ",".join(sources) or "none",
        }
    )
    LAST_REFRESH.set(time.time())


def main() -> None:
    port = int(os.environ.get("EXPORTER_PORT", "9101"))
    start_http_server(port)
    print(f"metrics-exporter écoute sur :{port}", flush=True)
    while True:
        try:
            refresh()
        except Exception as exc:  # noqa: BLE001 — garde le scrape vivant
            print(f"refresh error: {exc}", flush=True)
        time.sleep(int(os.environ.get("REFRESH_SECONDS", "30")))


if __name__ == "__main__":
    main()
