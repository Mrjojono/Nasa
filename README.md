# Monthly rainfall forecast for West Africa

FastAPI service that serves a monthly rainfall forecast for the Sahel. An XGBoost model, trained on satellite rainfall and ocean indices, predicts the next month and the sum of the next three months.

Live API: https://pluie-prevue.fastapicloud.dev

API documentation: https://pluie-prevue.fastapicloud.dev/docs

Web app: https://stationeteo.vercel.app

Frontend source: https://github.com/Mrjojono/MeteoFront

## What it serves

| Method | Path | Content |
| --- | --- | --- |
| GET | `/api/meta` | Grid, time range, and layer descriptions |
| GET | `/api/frames?layer=` | Last 12 observed months for one layer |
| GET | `/api/forecast?lead=1` or `3` | Forecast grid |
| GET | `/api/point?lat=&lon=` | One cell, observed and forecast |
| GET | `/api/alerts` | Regional alert snapshot |
| GET | `/api/signals` | ONI, tropical North Atlantic, Gulf of Guinea |
| GET | `/api/metrics` | Hold-out scores against climatology |
| GET | `/docs` | Interactive API docs |

The hold-out test starts in January 1984. On that test the seasonal normal is more accurate than the model: skill versus climatology is −0.40 at one month (RMSE 44.5 mm, against 31.7 mm for climatology) and −0.49 at three months (RMSE 101.3 mm, against 68.0 mm).

## Use of AI

Parts of this code and text were produced with AI assistance (Claude, Anthropic).

## Data

The service reads `donnees_pluie_prevue/out/features_chirps_sahel.csv` and the two joblib models next to it.

| Source | Version | Role |
| --- | --- | --- |
| CHIRPS monthly rainfall | v3.0 | Target and rainfall features. UCSB Climate Hazards Center. |
| NOAA CPC indices | ONI, RONI, SOI, Niño regions | Ocean predictors. |
| NOAA ERSST | v6 | Sea-surface temperature used for the Atlantic and Gulf of Guinea series. |

NASA IMERG Final V07 was downloaded with an Earthdata login and is stored in this repository. It is not an input of the model that is online. NASA POWER is not used.

## Run locally

Python 3.12. The satellite files are not in the Git repository.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python preparer_donnees.py --entrainer
.venv/bin/python api.py --host 127.0.0.1 --port 8000
```

`preparer_donnees.py` downloads CHIRPS v3.0 (January 1981 to June 1985), the NOAA CPC indices, and ERSST v6, then builds the feature table. Add `--seulement-telecharger` to stop before processing. Files already on disk are skipped.

Open http://127.0.0.1:8000/docs. The first request loads the CSV and the models. No database is required: if `PLUIE_DATABASE_URL` is unset, the API reads the CSV.
