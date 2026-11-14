# Données « pluie prévue » — inventaire du téléchargement

Dossier créé le 14/11/2026.

Authentification NASA : token Earthdata (compte `joankekeli`), valable jusqu'au
**13/01/2027**, conservé dans `00_docs/edl_token.txt` (chmod 600). Il s'utilise
en en-tête `Authorization: Bearer <token>` — plus simple qu'un mot de passe
et suffisant pour GES DISC comme pour LP DAAC.

```
donnees_pluie_prevue/
├── 00_docs/                    README CHIRPS + pages de référence
│   └── 01_chirps_v3_mensuel/   550 GeoTIFF attendus (1981-01 → 2026-10)
├── 02_ersst_v6/                sst.mnmean.nc (ERSST v6 mensuel)
├── 03_indices_cpc/             RONI, ONI, indices Niño, SOI, tableau ONI v6
├── 04_imerg_final/             granules HDF5 + script earthaccess
├── 05_modis_ndvi/              script earthaccess (compte requis)
├── telecharger/in/             CSV à nettoyer (le reste est ignoré par l'ETL)
├── out/                        CSV du modèle, cubes historiques et modèles
└── preprocessed -> out         ancien chemin, conservé comme lien
```

## 1. CHIRPS v3.0 mensuel — `01_chirps_v3_mensuel/`

- Source : `https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/`
- Contenu : `chirps-v3.0.AAAA.MM.tif`, 550 fichiers, 1981-01 à 2026-10.
- Environ 23 Mo par fichier (~12 Go au total). GeoTIFF global 0,05°, mm/mois.
- Liste exacte des fichiers attendus : `.filelist.txt`.
- Journal du téléchargement : `.download.log` (`OK <fichier> <octets>` par succès).
- Reprise automatique : `.fetch_one.sh` ignore les fichiers déjà complets.

## 2. ERSST v6 — `02_ersst_v6/`

- `sst.mnmean.nc` : SST mensuelle NOAA ERSST v6, grille 2°, 1854 → présent,
  NetCDF4/HDF5, ~90 Mo. Source : `https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v6/sst.mnmean.nc`

  Ce fichier unique contient **tous** les mois, donc il couvre l'Atlantique,
  le Golfe de Guinée, Niño 3.4 et la Méditerranée sans télécharger les
  2 126 fichiers mensuels NCEI (`.../v6/access/ersst.v6.AAAAMM.nc`, redondants).

## 3. Indices El Niño (NOAA CPC) — `03_indices_cpc/`

| Fichier | Contenu |
|---|---|
| `RONI.ascii.txt` | Relative Oceanic Niño Index, 1950 → ASO 2026 (indice officiel actuel) |
| `oni.ascii.txt` | ONI classique, 1950 → ASO 2026 (encore mis à jour) |
| `oni_v6_table.html` | Tableau officiel ONI v6 (page CPC) |
| `ersst5.nino.mth.91-20.ascii` | SST mensuelles Niño 1+2, 3, 3.4, 4 (base 1991-2020) |
| `sstoi.indices` | Anomalies SST par région Niño |
| `soi` | Southern Oscillation Index |

## 4. IMERG Final mensuel V07 — `04_imerg_final/`

- 333 granules (`3B-MO.MS.MRG.3IMERG.AAAAMM01-...V07B.HDF5`), 1998-01 → 2025-09,
  0,1° mondial, mm/mois, ~18 Mo par fichier (~6 Go).
- Téléchargement : `.urls.txt` (liste CMR) + `.fetch_one.sh` (curl + Bearer token),
  journal dans `.download.log`.
- Script Python équivalent : `download_imerg.py`. Attention, `earthaccess`
  attend un mot de passe EDL ; avec un token, la voie curl/CMR est plus simple.

## 5. NDVI MODIS MOD13C2 v061 — `05_modis_ndvi/`

**Bloqué depuis cette machine.** Le granule est servi via une redirection
`data.lpdaac.earthdatacloud.nasa.gov` → `*.cloudfront.net`, et ce domaine est
injoignable ici (handshake TLS avorté, timeout), alors que le reste de NASA
répond. Contournements testés et écartés : accès S3 direct (le rôle
`s3-same-region-access-role` refuse hors `us-west-2`), OPeNDAP (302 vers
`/login/urs`, le token n'y est pas accepté), Harmony (aucun service déclaré
pour cette collection), e4ftl01/USGS (service retiré).

Deux scripts prêts à lancer depuis un autre réseau (ou un VPN) :
- `download_mod13c2.sh` — CMR + curl + token, `--test` pour un seul fichier ;
- `download_mod13c2.py` — variante `earthaccess` (nécessite login/mot de passe).


## 6. MERRA-2 (optionnel, compte requis)

`M2TMNXSLV` (Merra-2 tavgM_2d_slv_Nx, mensuel, vent 10 m + humidité 2 m) :

```python
import earthaccess
earthaccess.login(strategy="netrc")
g = earthaccess.search_data(short_name="M2TMNXSLV", temporal=("1980-01-01", "2026-10-31"))
earthaccess.download(g, local_path="06_merra2")
```

## 7. Prétraitement — `pretraiter.py`

Le script `../pretraiter.py` lit les fichiers présents et produit des cubes
NetCDF prêts à l'emploi plus une table de features par (mois, maille) :

```bash
.venv/bin/python pretraiter.py                        # produits satellite → out/
.venv/bin/python etl.py                               # telecharger/in → out/features_modele.csv
```

Le DAG d'ETL ne lance que `etl.py`. `pretraiter.py` reste le script des GeoTIFF et des indices, hors du modèle.

Sorties du modèle dans `out/` :

- `chirps_sahel_mensuel.nc`, `imerg_sahel_mensuel.nc` — pluie (time, lat, lon) en mm/mois ;
- `indices_enso_mensuel.nc` — ONI, RONI, SOI, Niño, Atl3 + champ SST ERSST v6 ;
- `features_modele.csv` — CSV unique lu par le modèle et par l'API, produit par `etl.py` ;
- `features_chirps_sahel.csv`, `features_imerg_sahel.csv` — tables historiques du script satellite ;
- `rapport_controle.txt` — statistiques par mois, masques, contrôles croisés.

Le script est idempotent : il prend tous les fichiers présents, donc il suffit
de le relancer après chaque vague de téléchargements. Options utiles :
`--keep-ocean` (conserver les mailles océaniques, vides pour CHIRPS),
`--no-csv` (cubes seulement).

## Notes

- ERSST v5 (PO.DAAC) est arrêté : ne pas l'utiliser, la v6 le remplace.
- Secours CHIRPS avec découpage avant téléchargement (IRI/LDEO) :
  `https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v3p0/.monthly`
- Le serveur CHC renvoie des HTTP 429 si l'on ouvre plus de 2 connexions
  simultanées : garder `-P 2`.
