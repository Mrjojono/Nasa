#!/usr/bin/env python3
"""API de prévision de pluie mensuelle — XGBoost.

Contrat frontend (mêmes formes que contract/mock) :
  GET /api/meta
  GET /api/frames?layer=rain_mm|anomaly_mm|anomaly_pct|anomaly_z
  GET /api/forecast?lead=1|3
  GET /api/point?lat=&lon=
  GET /api/alerts
  GET /api/signals
  GET /api/metrics
  GET /api/afrique

Maille unique :
  GET /api/predict?lat=&lon=&year=&month=&lead=1|3

Usage :
    .venv/bin/python api.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chemins import out_dir
from contract_live import CONTRACT_VERSION, Catalog
from base_pluie import database_url, etat_chargement, lire_features

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:  # optionnel hors stack monitoring
    Instrumentator = None  # type: ignore[misc, assignment]

PRE = out_dir()
MODELE_CSV = PRE / "features_modele.csv"
CSV = MODELE_CSV if MODELE_CSV.exists() else PRE / "features_chirps_sahel.csv"

FEATURES = [
    "year", "month", "lat", "lon",
    "rain_mm", "clim_mm", "anom_mm", "anom_z", "pct_clim",
    "rain_3m_mm", "anom_3m_mm",
    "rain_lag1_mm", "rain_lag3_mm", "rain_lag6_mm", "rain_lag12_mm",
    "oni", "roni", "soi",
    "nino12_anom", "nino3_anom", "nino4_anom", "nino34_anom", "atl3_anom",
    "oni_lag3", "oni_lag6",
    "month_sin", "month_cos",
]

TARGET_BY_LEAD = {1: "rain_next1_mm", 3: "rain_next3_mm"}
MODEL_BY_LEAD = {
    1: PRE / "xgb_rain_next1_mm.joblib",
    3: PRE / "xgb_rain_next3_mm.joblib",
}

app = FastAPI(title="Pluie prévue — API XGBoost", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

if Instrumentator is not None:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# état chargé au démarrage, rechargé si out/ change (nouvel entraînement)
_df: pd.DataFrame | None = None
_models: dict[int, object] = {}
_lat_unique: np.ndarray = np.array([])
_lon_unique: np.ndarray = np.array([])
_catalog: Catalog | None = None
_sig: tuple | None = None
_lock = threading.Lock()


def _table_csv() -> Path:
    if MODELE_CSV.exists() and MODELE_CSV.stat().st_size > 0:
        return MODELE_CSV
    return PRE / "features_chirps_sahel.csv"


def _signature() -> tuple:
    modeles = tuple(
        (str(path), path.stat().st_mtime_ns if path.exists() else 0)
        for path in MODEL_BY_LEAD.values()
    )
    if database_url():
        return ("postgres", etat_chargement(), modeles)
    table = _table_csv()
    return (
        "csv",
        (str(table), table.stat().st_mtime_ns if table.exists() else 0),
        modeles,
    )


def _load_state() -> None:
    global _df, _models, _lat_unique, _lon_unique, _catalog
    if database_url():
        frame = lire_features()
        if frame.empty:
            raise FileNotFoundError("table features vide")
    else:
        table = _table_csv()
        if not table.exists():
            raise FileNotFoundError(table)
        frame = pd.read_csv(table, parse_dates=["date"])
    models: dict[int, object] = {}
    for lead, path in MODEL_BY_LEAD.items():
        if path.exists():
            models[lead] = joblib.load(path)
    _df = frame
    _models = models
    _lat_unique = np.sort(frame["lat"].unique())
    _lon_unique = np.sort(frame["lon"].unique())
    _catalog = Catalog(frame, models)


def _maybe_reload() -> None:
    global _sig
    sig = _signature()
    if sig == _sig:
        return
    with _lock:
        sig = _signature()
        if sig == _sig:
            return
        if database_url():
            if etat_chargement()[1] == 0:
                return
        else:
            table = _table_csv()
            if not table.exists() or table.stat().st_size == 0:
                return
        _load_state()
        _sig = _signature()


@app.middleware("http")
async def _reload_artifacts(request, call_next):
    try:
        _maybe_reload()
    except Exception as exc:  # le serveur reste up, /api/health dira waiting
        print(f"rechargement impossible : {exc}", flush=True)
    return await call_next(request)


@app.get("/api/health")
def health():
    ready = _df is not None and {1, 3} <= set(_models)
    body = {
        "status": "ok" if ready else "waiting",
        "models_loaded": sorted(_models.keys()),
        "source": "postgres" if database_url() else "csv",
        "out": str(PRE),
    }
    if not ready:
        return JSONResponse(body, status_code=503)
    return body


@app.get("/api/predict")
def predict(
    lat: float = Query(..., description="Latitude de la maille"),
    lon: float = Query(..., description="Longitude de la maille"),
    year: int = Query(..., ge=1900, le=2100, description="Année du mois courant"),
    month: int = Query(..., ge=1, le=12, description="Mois courant"),
    lead: int = Query(1, ge=1, le=3, description="Échelle en mois (1 ou 3)"),
) -> dict:
    if _df is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    if lead not in _models:
        raise HTTPException(status_code=400, detail=f"lead non supporté : {lead}")

    target = TARGET_BY_LEAD[lead]
    model = _models[lead]

    # snap aux centres de maille les plus proches (grille régulière)
    lat_snap = float(_lat_unique[np.argmin(np.abs(_lat_unique - lat))])
    lon_snap = float(_lon_unique[np.argmin(np.abs(_lon_unique - lon))])

    # lookup exact sur (lat, lon, year, month)
    mask = (
        (_df["lat"] == lat_snap)
        & (_df["lon"] == lon_snap)
        & (_df["year"] == year)
        & (_df["month"] == month)
    )
    rows = _df[mask]
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Aucune donnée pour lat={lat} (snappé {lat_snap}), "
                f"lon={lon} (snappé {lon_snap}), {year:04d}-{month:02d}"
            ),
        )

    row = rows.iloc[0]
    X = row[FEATURES].to_numpy(dtype="float32", copy=True).reshape(1, -1)
    # remplacer NaN par la médiane du jeu d'entraînement (0 si tout NaN)
    X = np.where(np.isfinite(X), X, 0.0)

    pred = float(model.predict(X)[0])
    return {
        "target": target,
        "lead_months": lead,
        "lat": round(float(row["lat"]), 3),
        "lon": round(float(row["lon"]), 3),
        "year": int(row["year"]),
        "month": int(row["month"]),
        "date": str(row["date"])[:10],
        "rain_mm": None if pd.isna(row["rain_mm"]) else float(row["rain_mm"]),
        "predicted_rain_next_mm": round(pred, 2),
        "clim_mm": None if pd.isna(row["clim_mm"]) else float(row["clim_mm"]),
        "anom_mm": None if pd.isna(row["anom_mm"]) else float(row["anom_mm"]),
    }


def _ready() -> Catalog:
    if _catalog is None:
        raise HTTPException(status_code=503, detail="Données non chargées")
    return _catalog


@app.get("/api/meta")
def meta() -> dict:
    return _ready().meta


@app.get("/api/meta/layers")
def layers() -> list:
    return _ready().meta["layers"]


@app.get("/api/frames")
def frames(layer: str = Query(..., description="rain_mm|anomaly_mm|anomaly_pct|anomaly_z")) -> dict:
    catalog = _ready()
    if layer not in catalog.frames:
        raise HTTPException(status_code=400, detail=f"layer inconnu : {layer}")
    return catalog.frames[layer]


@app.get("/api/forecast")
def forecast(lead: int = Query(..., description="1 ou 3")) -> dict:
    catalog = _ready()
    if lead not in catalog.forecasts:
        raise HTTPException(status_code=400, detail=f"lead non supporté : {lead}")
    return catalog.forecasts[lead]


@app.get("/api/point")
def point(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
) -> dict:
    return _ready().point(lat, lon)


@app.get("/api/afrique")
def afrique() -> dict:
    """Prévision JSON pour l'Afrique de l'Ouest, produite à partir du CSV du modèle."""
    catalog = _ready()
    return {
        "resource": "afrique",
        "contract_version": CONTRACT_VERSION,
        "region": catalog.meta["region"],
        "time": catalog.meta["time"],
        "forecasts": {str(lead): catalog.forecasts[lead] for lead in sorted(catalog.forecasts)},
        "alerts": catalog.alerts,
    }


@app.get("/api/alerts")
def alerts() -> dict:
    return _ready().alerts


def _ocean_series() -> list[dict]:
    path = PRE / "signaux_ocean.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    series = payload.get("series")
    return series if isinstance(series, list) else []


@app.get("/api/signals")
def signals() -> dict:
    return _ready().signals(_ocean_series())


@app.get("/api/metrics")
def model_metrics() -> dict:
    return _ready().metrics()


def main() -> int:
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    # 0.0.0.0 si Prometheus scrape depuis Docker (host.docker.internal)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
