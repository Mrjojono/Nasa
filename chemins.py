"""Emplacements du jeu « pluie prévue ».

L'ETL lit ``donnees_pluie_prevue/telecharger/in`` (CSV, le reste est ignoré).
Il écrit le CSV du modèle dans ``donnees_pluie_prevue/out``.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "donnees_pluie_prevue"


def entree_brute() -> Path:
    env = os.environ.get("PLUIE_IN")
    if env:
        return Path(env)
    return DATA / "telecharger" / "in"


def out_dir() -> Path:
    env = os.environ.get("PLUIE_OUT")
    if env:
        return Path(env)
    candidate = DATA / "out"
    if candidate.is_dir():
        return candidate
    return DATA / "preprocessed"
