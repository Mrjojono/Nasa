"""Charge et relit la table nettoyée dans Postgres.

La base `pluie` est créée au démarrage de Docker. L'ETL y copie le CSV
du modèle. L'API relit cette table.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd

from pretraiter import COLUMNS

ENTIERS = {"year", "month"}


def database_url() -> str:
    return os.environ.get("PLUIE_DATABASE_URL", "").strip()


def _connect():
    import psycopg2

    url = database_url()
    if not url:
        raise RuntimeError("PLUIE_DATABASE_URL manquant")
    return psycopg2.connect(url)


def etat_chargement() -> tuple[int, int]:
    """Dernier chargement : (id, nombre de lignes). (0, 0) si la table est vide."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, lignes FROM chargement ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
    if row is None:
        return 0, 0
    return int(row[0]), int(row[1])


def lire_features() -> pd.DataFrame:
    colonnes = ", ".join(COLUMNS)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {colonnes} FROM features ORDER BY date, lat, lon")
            rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def charger_csv(csv_path: Path) -> int:
    """Remplace le contenu de `features` par le CSV nettoyé."""
    frame = pd.read_csv(csv_path)
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"Colonnes inattendues dans {csv_path.name}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ENTIERS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, header=False, na_rep="\\N")
    buffer.seek(0)
    colonnes = ", ".join(COLUMNS)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE features")
            cur.copy_expert(
                f"COPY features ({colonnes}) FROM STDIN WITH (FORMAT csv, NULL '\\N')",
                buffer,
            )
            cur.execute(
                "INSERT INTO chargement (lignes) VALUES (%s)",
                (len(frame),),
            )
        conn.commit()
    return len(frame)
