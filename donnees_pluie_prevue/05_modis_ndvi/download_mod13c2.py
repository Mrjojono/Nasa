#!/usr/bin/env python3
"""
Téléchargement du NDVI MODIS MOD13C2 v061 (composite mensuel, 0.05°, 2000+).

Mêmes prérequis que pour IMERG (compte Earthdata + autorisation LP DAAC) :
  https://urs.earthdata.nasa.gov
  https://appeears.earthdatacloud.nasa.gov/  (ou l'application LPCLOUD)

Installation : pip install --user earthaccess

Usage :
    python3 download_mod13c2.py --test        # 1 seul fichier pour tester
    python3 download_mod13c2.py               # tout depuis 2000-02
    python3 download_mod13c2.py 2020-01 2026-10
"""
import sys
import earthaccess

DEST = "."


def main() -> int:
    earthaccess.login(strategy="netrc")

    args = sys.argv[1:]
    if args and args[0] == "--test":
        start, end, limit = "2000-02-01", "2000-12-31", 1
    elif len(args) == 2:
        start, end, limit = args[0], args[1], None
    else:
        start, end, limit = "2000-02-01", "2026-10-31", None

    results = earthaccess.search_data(
        short_name="MOD13C2",
        version="061",
        temporal=(start, end),
    )
    if limit:
        results = results[:limit]
    print(f"{len(results)} granules trouvés ({start} -> {end})")

    files = earthaccess.download(results, local_path=DEST)
    print(f"{len(files)} fichiers téléchargés dans {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
