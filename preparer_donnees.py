#!/usr/bin/env python3
"""Télécharge les sources du modèle, puis lance le prétraitement.

Les fichiers restent sur la machine. Ils ne sont pas versionnés.

Fenêtre par défaut : janvier 1981 → juin 1985, celle servie par l'API.
CHIRPS est limité à 2 téléchargements à la fois : le serveur renvoie 429 au-delà.

    python preparer_donnees.py
    python preparer_donnees.py --seulement-telecharger
    python preparer_donnees.py --debut 1981-01 --fin 1985-06 --entrainer

IMERG et MODIS ne sont pas téléchargés : le modèle en ligne n'en a pas besoin,
et IMERG demande un jeton NASA Earthdata.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "donnees_pluie_prevue"
CHIRPS = DATA / "00_docs" / "01_chirps_v3_mensuel"
CPC = DATA / "03_indices_cpc"
ERSST = DATA / "02_ersst_v6" / "sst.mnmean.nc"

CHIRPS_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/{name}"
ERSST_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v6/sst.mnmean.nc"
CPC_URLS = {
    "oni.ascii.txt": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "RONI.ascii.txt": "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt",
    "soi": "https://www.cpc.ncep.noaa.gov/data/indices/soi",
    "ersst5.nino.mth.91-20.ascii": "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii",
}


def mois(debut: str, fin: str) -> list[tuple[int, int]]:
    y0, m0 = (int(part) for part in debut.split("-"))
    y1, m1 = (int(part) for part in fin.split("-"))
    out: list[tuple[int, int]] = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def telecharger(url: str, dest: Path, minimum: int) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size >= minimum:
        return f"deja la  {dest.name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    requete = urllib.request.Request(url, headers={"User-Agent": "pluie-prevue"})
    try:
        with urllib.request.urlopen(requete, timeout=120) as reponse, tmp.open("wb") as sortie:
            while True:
                bloc = reponse.read(1024 * 256)
                if not bloc:
                    break
                sortie.write(bloc)
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        return f"ECHEC   {dest.name}  {exc}"
    if tmp.stat().st_size < minimum:
        tmp.unlink(missing_ok=True)
        return f"ECHEC   {dest.name}  fichier trop petit"
    tmp.replace(dest)
    return f"ok      {dest.name}  {dest.stat().st_size} octets"


def chirps(periode: list[tuple[int, int]]) -> int:
    echecs = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futurs = []
        for annee, mois_ in periode:
            nom = f"chirps-v3.0.{annee}.{mois_:02d}.tif"
            futurs.append(pool.submit(telecharger, CHIRPS_URL.format(name=nom), CHIRPS / nom, 1_000_000))
        for futur in as_completed(futurs):
            ligne = futur.result()
            print(ligne, flush=True)
            if ligne.startswith("ECHEC"):
                echecs += 1
    return echecs


def indices() -> int:
    echecs = 0
    for nom, url in CPC_URLS.items():
        ligne = telecharger(url, CPC / nom, 100)
        print(ligne, flush=True)
        if ligne.startswith("ECHEC"):
            echecs += 1
    ligne = telecharger(ERSST_URL, ERSST, 1_000_000)
    print(ligne, flush=True)
    if ligne.startswith("ECHEC"):
        echecs += 1
    return echecs


def lancer(script: str) -> int:
    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        python = Path(sys.executable)
    print(f"\n→ {python} {script}", flush=True)
    return subprocess.call([str(python), str(ROOT / script)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--debut", default="1981-01")
    ap.add_argument("--fin", default="1985-06")
    ap.add_argument("--seulement-telecharger", action="store_true")
    ap.add_argument("--entrainer", action="store_true", help="lance aussi train_model.py")
    args = ap.parse_args()

    periode = mois(args.debut, args.fin)
    print(f"{len(periode)} mois CHIRPS, de {args.debut} à {args.fin}", flush=True)
    echecs = chirps(periode) + indices()
    if echecs:
        print(f"\n{echecs} téléchargement(s) en échec. Relance la même commande : les fichiers complets sont sautés.")
        return 1
    if args.seulement_telecharger:
        return 0
    code = lancer("pretraiter.py")
    if code != 0:
        return code
    if args.entrainer:
        return lancer("train_model.py")
    print("\nTable écrite dans donnees_pluie_prevue/out/. Pour les modèles : python preparer_donnees.py --entrainer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
