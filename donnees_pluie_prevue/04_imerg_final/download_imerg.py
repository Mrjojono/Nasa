#!/usr/bin/env python3
"""
Téléchargement de GPM IMERG Final mensuel V07 (GPM_3IMERGM, version 07).

Prérequis (compte Earthdata gratuit) :
  1. Créer un compte sur https://urs.earthdata.nasa.gov
  2. Autoriser l'application "NASA GESDISC DATA ARCHIVE" :
     -> https://disc.gsfc.nasa.gov/earthdata-login  (bouton "Authorize")
  3. Installer les dépendances :
        uv pip install earthaccess
     ou  pip install --user earthaccess

Identifiants : earthaccess les lit dans ~/.netrc ou dans les variables
d'environnement EARTHDATA_USERNAME / EARTHDATA_PASSWORD.

Usage :
    python3 download_imerg.py                # tout depuis 2000-06
    python3 download_imerg.py 2020-01 2026-10  # plage précise
"""
import sys
import earthaccess

DEST = "."


def main() -> int:
    earthaccess.login(strategy="netrc")  # ou strategy="environment"

    if len(sys.argv) == 3:
        start, end = sys.argv[1], sys.argv[2]
    else:
        start, end = "2000-06-01", "2026-10-31"

    results = earthaccess.search_data(
        short_name="GPM_3IMERGM",
        version="07",
        temporal=(start, end),
    )
    print(f"{len(results)} granules trouvés ({start} -> {end})")

    files = earthaccess.download(results, local_path=DEST)
    print(f"{len(files)} fichiers téléchargés dans {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
