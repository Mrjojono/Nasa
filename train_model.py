#!/usr/bin/env python3
"""Entraînement d'un modèle XGBoost de prévision de pluie mensuelle.

Source : donnees_pluie_prevue/out/features_chirps_sahel.csv
Cibles : rain_next1_mm (pluie mois+1) et rain_next3_mm (pluie mois+3)
Sortie : donnees_pluie_prevue/out/xgb_<cible>.joblib
         donnees_pluie_prevue/out/metrics_<cible>.json

Usage :
    .venv/bin/python train_model.py
    .venv/bin/python train_model.py --target rain_next1_mm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from chemins import out_dir

PRE = out_dir()

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

ID_COLS = ["date", "year", "month", "lat", "lon"]
TARGETS = ["rain_next1_mm", "rain_next3_mm"]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def prepare(df: pd.DataFrame, target: str):
    """Retire les lignes sans cible et remplit les NaN features."""
    d = df.dropna(subset=[target]).copy()

    # split temporel : pas de shuffle (évite la fuite temporelle)
    train = d[d["date"] < "1984-01-01"].copy()
    test = d[d["date"] >= "1984-01-01"].copy()

    X_train = train[FEATURES].copy()
    y_train = train[target].values
    X_test = test[FEATURES].copy()
    y_test = test[target].values

    # XGBoost gère les NaN nativement, mais on remplit pour robustesse
    fill = X_train.median(numeric_only=True)
    X_train = X_train.fillna(fill)
    X_test = X_test.fillna(fill)

    meta_train = train[ID_COLS].copy()
    meta_test = test[ID_COLS].copy()

    return X_train, y_train, X_test, y_test, meta_train, meta_test


def train(X_train, y_train) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    bias = float(np.mean(pred - y_test))
    return {"rmse": rmse, "mae": mae, "r2": r2, "bias": bias, "n_test": int(len(y_test))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="rain_next1_mm",
                    choices=TARGETS, help="cible à prédire")
    ap.add_argument("--csv", type=Path,
                    default=PRE / "features_modele.csv",
                    help="CSV nettoyé par l'ETL")
    ap.add_argument("--out", type=Path, default=PRE,
                    help="dossier de sortie pour le modèle et les métriques")
    args = ap.parse_args()

    target = args.target
    out = args.out
    csv_path = args.csv
    out.mkdir(parents=True, exist_ok=True)

    print(f"Chargement : {csv_path}")
    df = load_data(csv_path)
    print(f"  {len(df):,} lignes, {df['date'].nunique()} mois, "
          f"{df.groupby(['lat','lon']).ngroups:,} mailles")

    print(f"\nPréparation (cible={target})…")
    X_train, y_train, X_test, y_test, meta_train, meta_test = prepare(df, target)
    print(f"  train : {len(X_train):,} lignes | test : {len(X_test):,} lignes")

    print(f"\nEntraînement XGBoost…")
    model = train(X_train, y_train)

    print(f"\nÉvaluation sur le test (1984-01 → 1985-06)…")
    metrics = evaluate(model, X_test, y_test)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k} = {v:.3f}")
        else:
            print(f"  {k} = {v}")

    # importance des features
    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nTop 10 features :")
    print(imp.head(10).to_string(index=False))

    # sauvegarde
    model_path = out / f"xgb_{target}.joblib"
    joblib.dump(model, model_path)
    print(f"\nModèle sauvegardé : {model_path}")

    metrics_out = {
        "status": "ok",
        "model": "xgboost",
        "target": target,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
        "metrics": metrics,
        "feature_importance": imp.to_dict("records"),
    }
    encoded = json.dumps(metrics_out, indent=2, ensure_ascii=False)
    # Fichier par cible : les deux entraînements Airflow n'écrasent pas le même JSON.
    metrics_path = out / f"metrics_{target}.json"
    metrics_path.write_text(encoded)
    (out / "metrics.json").write_text(encoded)
    print(f"Métriques sauvegardées : {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
