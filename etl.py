#!/usr/bin/env python3
"""ETL du premier DAG.

Entrée : n'importe quels fichiers déposés dans ``telecharger/in``.
Seul le CSV est transformé. Le reste est signalé, pas lu par le modèle.

Sortie : un seul CSV, colonnes dans l'ordre du contrat du modèle
(``features_modele.csv``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from chemins import entree_brute, out_dir
from pretraiter import COLUMNS, ecrire_contrat, verifier_table
from base_pluie import charger_csv, database_url

# Noms fréquents → colonne du contrat. La colonne canonique est toujours acceptée.
ALIAS = {
    "date": ("date", "time", "datetime", "date_mois"),
    "year": ("year", "annee", "yr"),
    "month": ("month", "mois"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "longitude", "long"),
    "rain_mm": ("rain_mm", "rain", "pluie", "pluie_mm", "precip", "precipitation"),
    "clim_mm": ("clim_mm", "climatologie", "climatology", "normal_mm"),
    "anom_mm": ("anom_mm", "anomalie", "anomaly_mm"),
    "anom_z": ("anom_z", "anomalie_z", "zscore"),
    "pct_clim": ("pct_clim", "pct_normale", "percent_of_normal"),
    "rain_3m_mm": ("rain_3m_mm", "pluie_3m"),
    "anom_3m_mm": ("anom_3m_mm",),
    "rain_lag1_mm": ("rain_lag1_mm", "lag1"),
    "rain_lag3_mm": ("rain_lag3_mm", "lag3"),
    "rain_lag6_mm": ("rain_lag6_mm", "lag6"),
    "rain_lag12_mm": ("rain_lag12_mm", "lag12"),
    "oni": ("oni",),
    "roni": ("roni",),
    "soi": ("soi",),
    "nino12_anom": ("nino12_anom",),
    "nino3_anom": ("nino3_anom",),
    "nino4_anom": ("nino4_anom",),
    "nino34_anom": ("nino34_anom",),
    "atl3_anom": ("atl3_anom",),
    "oni_lag3": ("oni_lag3",),
    "oni_lag6": ("oni_lag6",),
    "month_sin": ("month_sin",),
    "month_cos": ("month_cos",),
    "rain_next1_mm": ("rain_next1_mm", "cible_1", "target_1"),
    "rain_next3_mm": ("rain_next3_mm", "cible_3", "target_3"),
}

OBLIGATOIRES = ("lat", "lon", "rain_mm")


def _cle(nom: str) -> str:
    return (
        str(nom).strip().lower()
        .replace("é", "e").replace("è", "e")
        .replace(" ", "_").replace("-", "_")
    )


def _renommer(df: pd.DataFrame) -> pd.DataFrame:
    lu = {_cle(c): c for c in df.columns}
    renames = {}
    for canon, noms in ALIAS.items():
        for nom in noms:
            src = lu.get(_cle(nom))
            if src is not None and src not in renames:
                renames[src] = canon
                break
    return df.rename(columns=renames)


def _numerique(df: pd.DataFrame) -> None:
    for col in COLUMNS:
        if col == "date" or col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _dates(df: pd.DataFrame) -> None:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    manque = df["date"].isna()
    if "year" in df.columns and "month" in df.columns and manque.any():
        construit = pd.to_datetime(
            dict(year=df.loc[manque, "year"], month=df.loc[manque, "month"], day=1),
            errors="coerce",
        )
        df.loc[manque, "date"] = construit
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    angle = 2.0 * math.pi * (df["month"].to_numpy(dtype="float64") - 1.0) / 12.0
    if "month_sin" not in df.columns or df["month_sin"].isna().all():
        df["month_sin"] = np.sin(angle)
    if "month_cos" not in df.columns or df["month_cos"].isna().all():
        df["month_cos"] = np.cos(angle)


def nettoyer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remet une table quelconque dans l'ordre des colonnes du modèle."""
    brut = len(df)
    df = _renommer(df)
    manquantes = [c for c in OBLIGATOIRES if c not in df.columns]
    if manquantes or ("date" not in df.columns and not {"year", "month"} <= set(df.columns)):
        raise ValueError(
            "Colonnes indispensables absentes : "
            + ", ".join(manquantes or ["date ou year+month"])
        )
    _numerique(df)
    _dates(df)
    if "rain_mm" in df.columns:
        df.loc[df["rain_mm"] < 0, "rain_mm"] = pd.NA
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[list(COLUMNS)]
    avant = len(df)
    df = df.dropna(subset=["date", "lat", "lon", "rain_mm"])
    df = df.drop_duplicates(subset=["date", "lat", "lon"], keep="last")
    df = df.sort_values(["date", "lat", "lon"]).reset_index(drop=True)
    rapport = {
        "lignes_lues": brut,
        "lignes_apres_nettoyage": len(df),
        "lignes_ecartees": avant - len(df),
        "colonnes": list(COLUMNS),
    }
    return df, rapport


def lister(entree: Path) -> tuple[list[Path], list[Path]]:
    csvs, autres = [], []
    if not entree.is_dir():
        return csvs, autres
    for path in sorted(entree.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() == ".csv":
            csvs.append(path)
        elif path.suffix.lower() != ".md":
            autres.append(path)
    return csvs, autres


def transformer(entree: Path, sortie: Path) -> dict:
    csvs, autres = lister(entree)
    if not csvs:
        raise FileNotFoundError(
            f"Aucun CSV dans {entree}. Déposez la table à nettoyer "
            "(l'entrée peut contenir autre chose : elle est ignorée)."
        )
    tables = []
    sources = []
    for path in csvs:
        table = pd.read_csv(path)
        if table.empty:
            continue
        tables.append(table)
        sources.append(str(path))
    if not tables:
        raise ValueError(f"CSV vides dans {entree}")
    df, stats = nettoyer(pd.concat(tables, ignore_index=True))
    if df.empty:
        raise ValueError("Aucune ligne exploitable après nettoyage")
    sortie.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(sortie, index=False)
    verifier_table(sortie)
    ecrire_contrat(sortie.parent)
    stats.update({
        "entree": str(entree),
        "sortie": str(sortie),
        "csv_lus": sources,
        "ignores": [str(p) for p in autres],
    })
    (sortie.parent / "rapport_etl.txt").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entree", type=Path, default=entree_brute())
    ap.add_argument("--sortie", type=Path, default=out_dir() / "features_modele.csv")
    args = ap.parse_args()
    try:
        stats = transformer(args.entree, args.sortie)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"ETL OK {stats['lignes_apres_nettoyage']} lignes, "
        f"{len(stats['colonnes'])} colonnes → {stats['sortie']}"
    )
    if stats["ignores"]:
        print(f"Ignorés (pas du CSV) : {len(stats['ignores'])}")
    if not database_url():
        print("PLUIE_DATABASE_URL absent : le CSV n'est pas chargé dans Postgres")
        return 0
    try:
        n = charger_csv(args.sortie)
    except Exception as exc:  # connexion ou schéma
        print(f"Chargement Postgres impossible : {exc}", file=sys.stderr)
        return 1
    print(f"Postgres : {n} lignes dans features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
