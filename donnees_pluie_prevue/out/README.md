# Sorties du prétraitement « pluie prévue »

Générées par `pretraiter.py` le 2026-11-14.

Région : lon [-20.0, 20.0] × lat [-10.0, 25.0] — table à 0.5°.

## Fichiers

| Fichier | Contenu |
|---|---|
| `chirps_sahel_mensuel.nc` | cube `precipitation(time, lat, lon)` en mm/mois, 0,05°, 54 mois |
| `imerg_sahel_mensuel.nc` | cube `precipitation(time, lat, lon)` en mm/mois, 0,1°, 15 mois |
| `indices_enso_mensuel.nc` | séries ONI/RONI/SOI/Niño + `sst(time, lat_sst, lon_sst)` et `sst_anom` |
| `features_chirps_sahel.csv` | table (mois × maille) CHIRPS + indices, 182,736 lignes |
| `features_imerg_sahel.csv` | idem pour IMERG, 84,000 lignes |
| `rapport_controle.txt` | statistiques par mois, masques, contrôles croisés |

Les NetCDF sont en CF-1.10 : `time` en jours depuis 1970-01-01, `lat`/`lon`
croissants, NaN = valeur manquante.

## Lecture

```python
# cube (netCDF4, sans dépendance supplémentaire)
from netCDF4 import Dataset
import numpy as np
d = Dataset("chirps_sahel_mensuel.nc")
pluie = np.ma.filled(d["precipitation"][:], np.nan)   # (time, lat, lon)
lat, lon = d["lat"][:], d["lon"][:]

# table (pandas)
import pandas as pd
df = pd.read_csv("features_chirps_sahel.csv", parse_dates=["date"])

# avec xarray, si installé : xr.open_dataset("chirps_sahel_mensuel.nc")
```

## Colonnes des tables

| Colonne | Sens |
|---|---|
| `date`, `year`, `month` | mois concerné (premier jour du mois) |
| `lat`, `lon` | centre de la maille à 0.5° |
| `rain_mm` | pluie du mois (mm) |
| `clim_mm` | normale du mois de l'année (années complètes disponibles) |
| `anom_mm` | `rain_mm - clim_mm` |
| `anom_z` | anomalie standardisée par l'écart-type interannuel du mois |
| `pct_clim` | % de la normale |
| `rain_3m_mm`, `anom_3m_mm` | cumul et anomalie des 3 derniers mois (mois courant inclus) |
| `rain_lag1/3/6/12_mm` | pluie décalée de 1, 3, 6, 12 mois |
| `oni`, `roni`, `soi` | indices ENSO du mois (CPC) |
| `nino12_anom` … `nino34_anom`, `atl3_anom` | anomalies SST régionales recalculées depuis ERSST v6 (base 1991-2020) |
| `oni_lag3`, `oni_lag6` | ONI décalé de 3 et 6 mois |
| `month_sin`, `month_cos` | encodage cyclique du mois |
| `rain_next1_mm`, `rain_next3_mm` | cibles : pluie du mois suivant / des 3 mois suivants |

## Limites à connaître

- CHIRPS ne couvre que les **terres** : les mailles océaniques ont été écartées
  de la table (le cube les conserve en NaN).
- La climatologie repose sur les seules années complètes présentes :
  CHIRPS 4 années (1981, 1982, 1983, 1984) ; IMERG moins de 2 années complètes → colonnes d'anomalie laissées vides. Elle devra être recalculée quand les téléchargements seront
  terminés.
- CHIRPS (1981-1985) et IMERG (1998-1999) **ne se recouvrent pas** : les deux
  tables ne doivent pas être fusionnées telles quelles.
- IMERG est fourni en intensité mm/heure ; la conversion en mm/mois utilise les
  heures réelles du mois.
