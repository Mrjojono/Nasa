#!/usr/bin/env python3
"""Prétraitement du jeu de données « pluie prévue » (NASA HACK).

Produit, à partir des fichiers présents dans donnees_pluie_prevue/ :

  1. un cube NetCDF par produit de pluie (CHIRPS, IMERG), découpé sur la
     région demandée, en (time, lat, lon) ;
  2. un cube NetCDF d'indices ENSO (séries + champ SST anomalie) ;
  3. une table de features aplatie (CSV) par (mois, maille) prête pour
     scikit-learn / XGBoost ;
  4. un rapport de contrôle qualité.

Aucune dépendance hors venv : numpy, netCDF4, rasterio.

Usage :
    .venv/bin/python pretraiter.py                    # lit in/, écrit out/
    .venv/bin/python pretraiter.py --lon-min -20 --lon-max 20 \
                                   --lat-min -10 --lat-max 25 \
                                   --table-res 0.5
"""

from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "donnees_pluie_prevue"
# Racine lue par find_dir. Le script satellite lit les dossiers produits.
DATA_ROOT = DATA

CHIRPS_FILL = -9999.0
IMERG_FILL = -9999.9

SEASONS = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}

EPOCH = dt.date(1970, 1, 1)

COLUMNS = [
    "date", "year", "month",
    "lat", "lon",
    "rain_mm", "clim_mm", "anom_mm", "anom_z", "pct_clim",
    "rain_3m_mm", "anom_3m_mm",
    "rain_lag1_mm", "rain_lag3_mm", "rain_lag6_mm", "rain_lag12_mm",
    "oni", "roni", "soi",
    "nino12_anom", "nino3_anom", "nino4_anom", "nino34_anom", "atl3_anom",
    "oni_lag3", "oni_lag6",
    "month_sin", "month_cos",
    "rain_next1_mm", "rain_next3_mm",
]


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def month_key(y: int, m: int) -> int:
    return y * 12 + (m - 1)


def key_to_date(k: int) -> dt.date:
    return dt.date(k // 12, k % 12 + 1, 1)


def idx_range(coord: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    """Indices des centres de maille dans [lo, hi], grille croissante ou non."""
    if lo > hi:
        lo, hi = hi, lo
    if coord[0] <= coord[-1]:
        i0 = int(np.searchsorted(coord, lo, side="left"))
        i1 = int(np.searchsorted(coord, hi, side="right"))
    else:
        asc = coord[::-1]
        j0 = int(np.searchsorted(asc, lo, side="left"))
        j1 = int(np.searchsorted(asc, hi, side="right"))
        i0, i1 = len(coord) - j1, len(coord) - j0
    return max(i0, 0), min(i1, len(coord))


def find_dir(*candidates: str) -> Path | None:
    for c in candidates:
        p = DATA_ROOT / c
        if p.is_dir():
            return p
    return None


def clean_rain(arr: np.ndarray, fill: float):
    """Met le fill à NaN ; renvoie le tableau et les compteurs de contrôle."""
    a = arr.astype("float32", copy=True)
    n_fill = int(np.count_nonzero(a <= fill + 1e-3))
    a[a <= fill + 1e-3] = np.nan
    n_neg = int(np.count_nonzero(a < 0))
    if n_neg:
        a[a < 0] = 0.0
    n_big = int(np.count_nonzero(a > 1500))
    return a, n_fill, n_neg, n_big


# --------------------------------------------------------------------------
# Lecture CHIRPS (GeoTIFF)
# --------------------------------------------------------------------------

def crop_raster(path: Path, bbox: tuple[float, float, float, float]):
    import rasterio
    from rasterio.windows import Window

    lon_min, lon_max, lat_min, lat_max = bbox
    with rasterio.open(path) as ds:
        t = ds.transform
        if t.b != 0 or t.d != 0:
            raise ValueError(f"raster tourné, non géré : {path.name}")
        lon = t.c + t.a * (np.arange(ds.width) + 0.5)
        lat = t.f + t.e * (np.arange(ds.height) + 0.5)
        j0, j1 = idx_range(lon, lon_min, lon_max)
        i0, i1 = idx_range(lat, lat_min, lat_max)
        if j1 <= j0 or i1 <= i0:
            raise ValueError("bbox hors raster")
        arr = ds.read(1, window=Window(j0, i0, j1 - j0, i1 - i0))
        lon_c, lat_c = lon[j0:j1], lat[i0:i1]
    arr = np.asarray(arr, dtype="float32")
    if lat_c[0] > lat_c[-1]:                      # normalise lat croissant
        arr, lat_c = arr[::-1, :], lat_c[::-1]
    return arr, lon_c, lat_c


def read_chirps(directory: Path, bbox, report: list[str]):
    rx = re.compile(r"chirps-v3\.0\.(\d{4})\.(\d{2})\.tif$")
    items = []
    for p in directory.glob("chirps-v3.0.*.tif"):
        m = rx.search(p.name)
        if m:
            items.append((int(m.group(1)), int(m.group(2)), p))
    items.sort(key=lambda x: (x[0], x[1]))
    if not items:
        return None

    report.append(f"CHIRPS : {len(items)} fichier(s) trouvé(s) dans {directory}")
    report.append(f"  période : {items[0][0]}-{items[0][1]:02d} → {items[-1][0]}-{items[-1][1]:02d}")

    arrays, months, lon, lat = [], [], None, None
    skipped = []
    qc = {"fill": [], "neg": 0, "big": 0}
    for y, m, p in items:
        try:
            a, lo, la = crop_raster(p, bbox)
        except Exception as exc:                   # fichier tronqué / illisible
            skipped.append((p.name, str(exc)))
            continue
        if a.size == 0 or not np.isfinite(a[a > CHIRPS_FILL + 1]).any():
            skipped.append((p.name, "aucune valeur valide"))
            continue
        if lon is None:
            lon, lat = lo, la
        elif a.shape != (lat.size, lon.size):
            skipped.append((p.name, f"grille inattendue {a.shape}"))
            continue
        a, n_fill, n_neg, n_big = clean_rain(a, CHIRPS_FILL)
        qc["fill"].append(n_fill)
        qc["neg"] += n_neg
        qc["big"] += n_big
        arrays.append(a)
        months.append((y, m))
        if len(months) % 10 == 0:
            log(f"  CHIRPS {len(months)}/{len(items)} lus…")

    for name, why in skipped:
        report.append(f"  IGNORÉ {name} : {why}")

    if not arrays:
        return None
    cube = np.stack(arrays)
    npx = cube.shape[1] * cube.shape[2]
    land = np.isfinite(cube).any(axis=0)
    report.append(
        f"  masque : {int(land.sum()):,} mailles terrestres / {npx:,} "
        f"({land.mean():.1%}) ; fill moyen {np.mean(qc['fill']) / npx:.1%} par mois "
        f"(CHIRPS ne couvre que les terres)"
    )
    if qc["neg"]:
        report.append(f"  {qc['neg']:,} valeurs négatives ramenées à 0")
    if qc["big"]:
        report.append(f"  {qc['big']:,} valeurs > 1500 mm/mois (à vérifier, "
                      f"possible artefact de maille côtière/insulaire)")
    return {
        "cube": cube,
        "months": months,
        "lon": lon,
        "lat": lat,
        "land": land,
        "skipped": skipped,
    }


# --------------------------------------------------------------------------
# Lecture IMERG (HDF5 via netCDF4)
# --------------------------------------------------------------------------

def read_imerg(directory: Path, bbox, report: list[str]):
    from netCDF4 import Dataset

    rx = re.compile(r"3B-MO\.MS\.MRG\.3IMERG\.(\d{4})(\d{2})\d{2}-")
    items = []
    for p in directory.glob("*.HDF5"):
        m = rx.search(p.name)
        if m:
            items.append((int(m.group(1)), int(m.group(2)), p))
    items.sort(key=lambda x: (x[0], x[1]))
    if not items:
        return None

    lon_min, lon_max, lat_min, lat_max = bbox
    report.append(f"IMERG : {len(items)} granule(s) trouvé(s) dans {directory}")
    report.append(f"  période : {items[0][0]}-{items[0][1]:02d} → {items[-1][0]}-{items[-1][1]:02d}")

    arrays, months, lon, lat = [], [], None, None
    skipped = []
    qc = {"fill": [], "neg": 0, "big": 0}
    for y, m, p in items:
        try:
            with Dataset(p) as ds:
                g = ds["Grid"]
                pv = g["precipitation"]
                dims = tuple(pv.dimensions)
                if lon is None:
                    la = np.asarray(g["lat"][:], dtype="float64")
                    lo = np.asarray(g["lon"][:], dtype="float64")
                    i0, i1 = idx_range(la, lat_min, lat_max)
                    j0, j1 = idx_range(lo, lon_min, lon_max)
                    lat, lon = la[i0:i1], lo[j0:j1]
                arr = np.asarray(pv[0], dtype="float32")
                # IMERG stocke (time, lon, lat) : on normalise en (lat, lon)
                if dims[-2:] == ("lat", "lon"):
                    a = arr[i0:i1, j0:j1]
                elif dims[-2:] == ("lon", "lat"):
                    a = arr[j0:j1, i0:i1].T
                else:
                    raise ValueError(f"ordre de dimensions inattendu : {dims}")
        except Exception as exc:
            skipped.append((p.name, str(exc)))
            continue
        # IMERG donne une intensité moyenne en mm/heure → mm/mois
        hours = calendar.monthrange(y, m)[1] * 24
        a, n_fill, n_neg, n_big = clean_rain(a, IMERG_FILL)
        qc["fill"].append(n_fill)
        qc["neg"] += n_neg
        qc["big"] += n_big
        arrays.append(a * np.float32(hours))
        months.append((y, m))
        log(f"  IMERG {len(months)}/{len(items)} lus…")

    for name, why in skipped:
        report.append(f"  IGNORÉ {name} : {why}")

    if not arrays:
        return None
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        arrays = [x[::-1, :] for x in arrays]
    cube = np.stack(arrays)
    npx = cube.shape[1] * cube.shape[2]
    land = np.isfinite(cube).any(axis=0)
    report.append(
        f"  fill : {np.mean(qc['fill']) / npx:.1%} de la grille en moyenne par mois"
    )
    if qc["neg"]:
        report.append(f"  {qc['neg']:,} valeurs négatives ramenées à 0")
    if qc["big"]:
        report.append(f"  {qc['big']:,} valeurs > 1500 mm/mois (fortes pluies tropicales)")
    return {
        "cube": cube,
        "months": months,
        "lon": lon,
        "lat": lat,
        "land": land,
        "skipped": skipped,
    }


# --------------------------------------------------------------------------
# Indices CPC (ASCII) et SST (NetCDF)
# --------------------------------------------------------------------------

def _floats(tokens):
    out = []
    for t in tokens:
        try:
            out.append(float(t))
        except ValueError:
            out.append(np.nan)
    return out


def parse_seasonal(path: Path, value_col: int) -> dict[int, float]:
    """ONI / RONI : lignes « SEAS YR [TOTAL] ANOM », centrées sur le mois médian."""
    out: dict[int, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        p = line.split()
        if len(p) < 3 or p[0] not in SEASONS:
            continue
        try:
            year = int(p[1])
            val = float(p[value_col])
        except (ValueError, IndexError):
            continue
        if abs(val) >= 999:
            continue
        out[month_key(year, SEASONS[p[0]])] = val
    return out


def parse_monthly_ascii(path: Path, names: list[str]) -> dict[int, dict[str, float]]:
    """Fichiers « YR MON v1 v2 … » (nino mth, sstoi)."""
    out: dict[int, dict[str, float]] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        p = line.split()
        if len(p) < 3 or not p[0].isdigit() or len(p[0]) != 4:
            continue
        try:
            year, month = int(p[0]), int(p[1])
        except ValueError:
            continue
        if not 1 <= month <= 12:
            continue
        vals = _floats(p[2:])
        row = {}
        for i, name in enumerate(names):
            v = vals[i] if i < len(vals) else np.nan
            row[name] = np.nan if abs(v) >= 999 else v
        out[month_key(year, month)] = row
    return out


def parse_soi(path: Path) -> dict[int, float]:
    """SOI : année en colonne 0-4 puis 12 champs de 6 caractères."""
    out: dict[int, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        if len(line) < 6 or not line[:4].strip().isdigit():
            continue
        year = int(line[:4])
        body = line[4:].ljust(72)
        for m in range(12):
            chunk = body[6 * m:6 * m + 6].strip()
            try:
                v = float(chunk)
            except ValueError:
                continue
            if abs(v) >= 999:
                continue
            out[month_key(year, m + 1)] = v
    return out


def read_ersst(path: Path, kmin: int, kmax: int, report: list[str]):
    """SST absolue + anomalie (base 1991-2020) + séries régionales Niño/Atl3."""
    from netCDF4 import Dataset

    if not path.exists():
        return None
    with Dataset(path) as ds:
        lat = np.asarray(ds["lat"][:], dtype="float64")
        lon = np.asarray(ds["lon"][:], dtype="float64")
        time = np.asarray(ds["time"][:], dtype="float64")
        units = getattr(ds["time"], "units", "")
        sst = np.ma.filled(ds["sst"][:].astype("float32"), np.nan)

    m = re.search(r"days since (\d{4})-(\d{1,2})-(\d{1,2})", units or "")
    base = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else dt.date(1800, 1, 1)
    dates = [base + dt.timedelta(days=float(x)) for x in time]
    keys = np.array([month_key(d.year, d.month) for d in dates])

    sel = np.where((keys >= kmin) & (keys <= kmax))[0]
    if sel.size == 0:
        return None
    keys, sst = keys[sel], sst[sel]
    months = np.array([k % 12 + 1 for k in keys])
    years = np.array([k // 12 for k in keys])

    # anomalie : climatologie mensuelle 1991-2020 par maille
    base_mask = (years >= 1991) & (years <= 2020)
    clim = np.full((12, lat.size, lon.size), np.nan, dtype="float32")
    for mm in range(1, 13):
        s = base_mask & (months == mm)
        if s.any():
            with np.errstate(invalid="ignore"):
                clim[mm - 1] = np.nanmean(sst[s], axis=0)
    anom = sst - clim[months - 1]
    report.append(f"ERSST v6 : {sel.size} mois chargés, base clim. 1991-2020 "
                  f"({int(base_mask.sum())} mois)")

    # moyennes régionales (bassins océaniques)
    def region(lat_lo, lat_hi, lon_lo, lon_hi):
        i0, i1 = idx_range(lat, lat_lo, lat_hi)
        j0, j1 = idx_range(lon, lon_lo, lon_hi)
        with np.errstate(invalid="ignore"):
            return np.nanmean(anom[:, i0:i1, j0:j1], axis=(1, 2))

    series = {
        "nino12_anom": region(-10, 0, 270, 280),
        "nino3_anom": region(-5, 5, 210, 270),
        "nino4_anom": region(-5, 5, 160, 210),
        "nino34_anom": region(-5, 5, 190, 240),
        "atl3_anom": region(-3, 3, 340, 360),
    }
    return {
        "keys": keys, "lat": lat, "lon": lon,
        "sst": sst, "anom": anom, "clim": clim,
        "series": series, "months": months, "dates": dates,
    }


# --------------------------------------------------------------------------
# Écriture NetCDF
# --------------------------------------------------------------------------

def write_rain_nc(path: Path, cube, months, lon, lat, title, source, notes):
    from netCDF4 import Dataset

    keys = np.array([month_key(y, m) for y, m in months])
    days = np.array([(key_to_date(int(k)) - EPOCH).days for k in keys], dtype="float64")

    ds = Dataset(path, "w", format="NETCDF4")
    ds.title = title
    ds.source = source
    ds.history = f"créé le {dt.date.today()} par pretraiter.py"
    ds.notes = notes
    ds.Conventions = "CF-1.10"

    ds.createDimension("time", keys.size)
    ds.createDimension("lat", lat.size)
    ds.createDimension("lon", lon.size)
    ds.createDimension("nbnds", 2)

    tv = ds.createVariable("time", "f8", ("time",))
    tv.units = "days since 1970-01-01 00:00:00"
    tv.calendar = "standard"
    tv.standard_name = "time"
    tv.bounds = "time_bnds"
    tv[:] = days

    tb = ds.createVariable("time_bnds", "f8", ("time", "nbnds"))
    starts = np.array([(key_to_date(int(k)) - EPOCH).days for k in keys], dtype="float64")
    ends = np.array([(key_to_date(int(k) + 1) - EPOCH).days for k in keys], dtype="float64")
    tb[:, 0], tb[:, 1] = starts, ends

    vlat = ds.createVariable("lat", "f4", ("lat",))
    vlat.units = "degrees_north"
    vlat.standard_name = "latitude"
    vlat.axis = "Y"
    vlat[:] = lat

    vlon = ds.createVariable("lon", "f4", ("lon",))
    vlon.units = "degrees_east"
    vlon.standard_name = "longitude"
    vlon.axis = "X"
    vlon[:] = lon

    crs = ds.createVariable("crs", "i4")
    crs.grid_mapping_name = "latitude_longitude"
    crs.epsg_code = "EPSG:4326"
    crs.semi_major_axis = 6378137.0
    crs.inverse_flattening = 298.257223563

    v = ds.createVariable(
        "precipitation", "f4", ("time", "lat", "lon"),
        zlib=True, complevel=4, shuffle=True,
        chunksizes=(1, lat.size, lon.size), fill_value=np.float32(np.nan),
    )
    v.units = "mm month-1"
    v.long_name = "précipitations mensuelles"
    v.standard_name = "precipitation_amount"
    v.grid_mapping = "crs"
    v.set_auto_mask(False)
    v[:] = cube
    ds.close()
    log(f"  → {path.name}  {cube.shape}  {path.stat().st_size / 1e6:.1f} Mo")


def write_indices_nc(path: Path, ers, cpc, nino_mth, kmin, kmax):
    from netCDF4 import Dataset

    keys = np.arange(kmin, kmax + 1)
    days = np.array([(key_to_date(int(k)) - EPOCH).days for k in keys], dtype="float64")

    ds = Dataset(path, "w", format="NETCDF4")
    ds.title = "Indices ENSO mensuels et SST ERSST v6"
    ds.source = ("NOAA CPC (ONI, RONI, SOI, indices Niño) et NOAA PSL (ERSST v6) — "
                 "https://www.cpc.ncep.noaa.gov/data/indices/")
    ds.history = f"créé le {dt.date.today()} par pretraiter.py"
    ds.Conventions = "CF-1.10"

    ds.createDimension("time", keys.size)
    ds.createDimension("nbnds", 2)
    tv = ds.createVariable("time", "f8", ("time",))
    tv.units = "days since 1970-01-01 00:00:00"
    tv.calendar = "standard"
    tv.standard_name = "time"
    tv.bounds = "time_bnds"
    tv[:] = days
    tb = ds.createVariable("time_bnds", "f8", ("time", "nbnds"))
    tb[:, 0] = days
    tb[:, 1] = np.array([(key_to_date(int(k) + 1) - EPOCH).days for k in keys], "f8")

    def series(name, mapping, units, long_name, comment=""):
        arr = np.full(keys.size, np.nan, dtype="float32")
        for i, k in enumerate(keys):
            v = mapping.get(int(k))
            if v is not None:
                arr[i] = v
        var = ds.createVariable(name, "f4", ("time",), zlib=True,
                                fill_value=np.float32(np.nan))
        var.set_auto_mask(False)
        var.units = units
        var.long_name = long_name
        if comment:
            var.comment = comment
        var[:] = arr

    series("oni", cpc["oni"], "degC",
           "Oceanic Niño Index — anomalie Niño 3.4, moyenne glissante 3 mois",
           "base 1991-2020, source CPC")
    series("roni", cpc["roni"], "degC",
           "Relative Oceanic Niño Index",
           "anomalie relative à une moyenne mobile 30 ans, source CPC")
    series("soi", cpc["soi"], "1", "Southern Oscillation Index (Tahiti - Darwin)")

    for nm, label in (("nino12", "Niño 1+2"), ("nino3", "Niño 3"),
                      ("nino4", "Niño 4"), ("nino34", "Niño 3.4")):
        series(nm, {k: v[nm] for k, v in nino_mth.items()}, "degC",
               f"SST {label} — valeur absolue (CPC)")
        series(f"{nm}_anom_cpc", {k: v[f"{nm}_anom"] for k, v in nino_mth.items()},
               "degC", f"anomalie SST {label} (CPC, base 1991-2020)")

    if ers is not None:
        for name, vals in ers["series"].items():
            series(name, dict(zip(ers["keys"].tolist(), vals.tolist())), "degC",
                   f"anomalie SST {name} — recalculée depuis ERSST v6",
                   "climatologie mensuelle 1991-2020, moyenne spatiale du bassin")

        ds.createDimension("lat_sst", ers["lat"].size)
        ds.createDimension("lon_sst", ers["lon"].size)
        vlat = ds.createVariable("lat_sst", "f4", ("lat_sst",))
        vlat.units, vlat.standard_name, vlat.axis = "degrees_north", "latitude", "Y"
        vlat[:] = ers["lat"]
        vlon = ds.createVariable("lon_sst", "f4", ("lon_sst",))
        vlon.units, vlon.standard_name, vlon.axis = "degrees_east", "longitude", "X"
        vlon[:] = ers["lon"]

        for name, data, lname in (
            ("sst", ers["sst"], "SST ERSST v6 (valeur absolue)"),
            ("sst_anom", ers["anom"], "anomalie SST ERSST v6 (base 1991-2020)"),
        ):
            var = ds.createVariable(name, "f4", ("time", "lat_sst", "lon_sst"),
                                    zlib=True, complevel=4, shuffle=True,
                                    fill_value=np.float32(np.nan))
            var.set_auto_mask(False)
            var.units = "degC"
            var.long_name = lname
            var[:] = data

    ds.close()
    log(f"  → {path.name}  {path.stat().st_size / 1e6:.1f} Mo")


# --------------------------------------------------------------------------
# Agrégation et features
# --------------------------------------------------------------------------

def block_mean(cube: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return cube
    t, h, w = cube.shape
    h2, w2 = h // f, w // f
    c = cube[:, : h2 * f, : w2 * f]
    with np.errstate(invalid="ignore"):
        return np.nanmean(c.reshape(t, h2, f, w2, f), axis=(2, 4))


def block_centers(coord: np.ndarray, f: int) -> np.ndarray:
    n = (coord.size // f) * f
    return coord[:n].reshape(-1, f).mean(axis=1)


def monthly_clim(cube: np.ndarray, months, complete_years):
    t = cube.shape[0]
    clim = np.full((12,) + cube.shape[1:], np.nan, dtype="float32")
    std = np.full_like(clim, np.nan)
    for mm in range(1, 13):
        idx = [i for i, (y, m) in enumerate(months) if m == mm and y in complete_years]
        if len(idx) >= 1:
            with np.errstate(invalid="ignore"):
                clim[mm - 1] = np.nanmean(cube[idx], axis=0)
        if len(idx) >= 2:
            with np.errstate(invalid="ignore"):
                std[mm - 1] = np.nanstd(cube[idx], axis=0)
    return clim, std


def build_features(cube, months, lon, lat, factor, cpc, ers, out_csv: Path, label: str,
                   land=None, drop_ocean: bool = True):
    """Table (mois × maille) aplatie à la résolution demandée."""
    C = block_mean(cube, factor)
    lon_c = block_centers(lon, factor)
    lat_c = block_centers(lat, factor)

    # maille « terrestre » = majorité de sous-mailles valides au moins une fois
    if land is not None and drop_ocean:
        land_c = block_mean(land[None].astype("float32"), factor)[0] > 0.5
    else:
        land_c = np.ones(C.shape[1:], dtype=bool)
    keep = land_c.reshape(-1)

    present = set(months)
    years = sorted({y for y, _ in months})
    complete = [y for y in years if all((y, m) in present for m in range(1, 13))]
    if len(complete) < 2:
        # une seule année complète : « climatologie » = cette année → fuite de
        # la cible. On laisse les colonnes dérivées vides plutôt que fausses.
        clim = np.full((12,) + C.shape[1:], np.nan, dtype="float32")
        std = np.full_like(clim, np.nan)
        complete = []
    else:
        clim, std = monthly_clim(C, months, complete)

    mkeys = [month_key(y, m) for y, m in months]
    idx_by_key = {k: i for i, k in enumerate(mkeys)}

    # séries ERSST recalculées sur le même axe temporel
    ers_by_key = {}
    if ers is not None:
        for name, vals in ers["series"].items():
            for k, v in zip(ers["keys"], vals):
                ers_by_key.setdefault(int(k), {})[name] = float(v)

    def get(series, k):
        return series.get(k, np.nan)

    h, w = C.shape[1], C.shape[2]
    lat_flat = np.repeat(lat_c, w)
    lon_tile = np.tile(lon_c, h)

    n_rows = 0
    with out_csv.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(COLUMNS)
        for t, (y, m) in enumerate(months):
            k = mkeys[t]
            rain = C[t]
            a = rain - clim[m - 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                z = a / std[m - 1]
                pct = 100.0 * rain / clim[m - 1]

            def shifted(dt_months):
                i = idx_by_key.get(k + dt_months)
                return C[i] if i is not None else None

            def trail(nper):
                idx = [idx_by_key.get(k - i) for i in range(nper)]
                if any(i is None for i in idx):
                    return None
                return np.sum([C[i] for i in idx], axis=0)

            lag1, lag3, lag6, lag12 = (shifted(-1), shifted(-3), shifted(-6), shifted(-12))
            r3 = trail(3)
            a3 = (r3 - 3.0 * clim[m - 1]) if r3 is not None else None
            nxt1 = shifted(1)
            idx3 = [idx_by_key.get(k + i) for i in (1, 2, 3)]
            nxt3 = np.sum([C[i] for i in idx3], axis=0) if all(i is not None for i in idx3) else None

            col = lambda arr: arr.reshape(-1) if arr is not None else np.full(h * w, np.nan, "float32")

            oni = np.full(h * w, get(cpc["oni"], k), "float32")
            roni = np.full(h * w, get(cpc["roni"], k), "float32")
            soi = np.full(h * w, get(cpc["soi"], k), "float32")
            lag3k, lag6k = k - 3, k - 6
            oni_l3 = np.full(h * w, get(cpc["oni"], lag3k), "float32")
            oni_l6 = np.full(h * w, get(cpc["oni"], lag6k), "float32")
            reg = {name: np.full(h * w, ers_by_key.get(k, {}).get(name, np.nan), "float32")
                   for name in ("nino12_anom", "nino3_anom", "nino4_anom",
                                "nino34_anom", "atl3_anom")}

            date = f"{y:04d}-{m:02d}-01"
            date_col = [date] * (h * w)
            year_col = np.full(h * w, y, "int16").tolist()
            month_col = np.full(h * w, m, "int8").tolist()

            def fmt(arr, dec):
                return np.round(np.asarray(arr, dtype="float64"), dec).tolist()

            cols = [
                date_col, year_col, month_col,
                fmt(lat_flat, 3), fmt(lon_tile, 3),
                fmt(rain.reshape(-1), 2),
                fmt(clim[m - 1].reshape(-1), 2),
                fmt(a.reshape(-1), 2),
                fmt(z.reshape(-1), 3),
                fmt(pct.reshape(-1), 1),
                fmt(col(r3), 2),
                fmt(col(a3), 2),
                fmt(col(lag1), 2),
                fmt(col(lag3), 2),
                fmt(col(lag6), 2),
                fmt(col(lag12), 2),
                fmt(oni, 3), fmt(roni, 3), fmt(soi, 3),
                fmt(reg["nino12_anom"], 3),
                fmt(reg["nino3_anom"], 3),
                fmt(reg["nino4_anom"], 3),
                fmt(reg["nino34_anom"], 3),
                fmt(reg["atl3_anom"], 3),
                fmt(oni_l3, 3), fmt(oni_l6, 3),
                [round(float(np.sin(2 * np.pi * m / 12)), 4)] * (h * w),
                [round(float(np.cos(2 * np.pi * m / 12)), 4)] * (h * w),
                fmt(col(nxt1), 2),
                fmt(col(nxt3), 2),
            ]
            if drop_ocean and land is not None:
                cols = [np.asarray(c)[keep].tolist() for c in cols]
            wr.writerows(zip(*cols))
            n_rows += int(keep.sum())
            if (t + 1) % 10 == 0:
                log(f"  table {label} : {t + 1}/{len(months)} mois ({n_rows:,} lignes)")

    n_cells = int(keep.sum())
    log(f"  → {out_csv.name}  {n_rows:,} lignes × {len(COLUMNS)} colonnes  "
        f"{out_csv.stat().st_size / 1e6:.1f} Mo")
    return {
        "rows": n_rows, "shape": (len(months), int(land_c.sum()), int((~land_c).sum())),
        "complete_years": complete, "climatology_years": len(complete),
        "cells": n_cells, "dropped_cells": int(land_c.size - n_cells),
    }


# --------------------------------------------------------------------------
# Notice de sortie
# --------------------------------------------------------------------------

README = """# Sorties du prétraitement « pluie prévue »

Générées par `pretraiter.py` le {date}.

Région : lon [{lon_min}, {lon_max}] × lat [{lat_min}, {lat_max}] — table à {res}°.

## Fichiers

| Fichier | Contenu |
|---|---|
| `chirps_sahel_mensuel.nc` | cube `precipitation(time, lat, lon)` en mm/mois, 0,05°, {n_chirps} mois |
| `imerg_sahel_mensuel.nc` | cube `precipitation(time, lat, lon)` en mm/mois, 0,1°, {n_imerg} mois |
| `indices_enso_mensuel.nc` | séries ONI/RONI/SOI/Niño + `sst(time, lat_sst, lon_sst)` et `sst_anom` |
| `features_chirps_sahel.csv` | table (mois × maille) CHIRPS + indices, {rows_chirps} lignes |
| `features_imerg_sahel.csv` | idem pour IMERG, {rows_imerg} lignes |
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
| `lat`, `lon` | centre de la maille à {res}° |
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
  {clim_txt}. Elle devra être recalculée quand les téléchargements seront
  terminés.
- CHIRPS (1981-1985) et IMERG (1998-1999) **ne se recouvrent pas** : les deux
  tables ne doivent pas être fusionnées telles quelles.
- IMERG est fourni en intensité mm/heure ; la conversion en mm/mois utilise les
  heures réelles du mois.
"""


def write_readme(out: Path, bbox, args, chirps, imerg, stats):
    parts = []
    for key, label in (("chirps", "CHIRPS"), ("imerg", "IMERG")):
        if key not in stats:
            continue
        s = stats[key]
        if s["climatology_years"] >= 2:
            parts.append(f"{label} {s['climatology_years']} années "
                         f"({', '.join(map(str, s['complete_years']))})")
        else:
            parts.append(f"{label} moins de 2 années complètes → colonnes "
                         f"d'anomalie laissées vides")
    clim_txt = " ; ".join(parts) if parts else "—"
    text = README.format(
        date=dt.date.today().isoformat(),
        lon_min=bbox[0], lon_max=bbox[1], lat_min=bbox[2], lat_max=bbox[3],
        res=args.table_res,
        n_chirps=len(chirps["months"]) if chirps else 0,
        n_imerg=len(imerg["months"]) if imerg else 0,
        rows_chirps=f"{stats['chirps']['rows']:,}" if "chirps" in stats else "—",
        rows_imerg=f"{stats['imerg']['rows']:,}" if "imerg" in stats else "—",
        clim_txt=clim_txt,
    )
    (out / "README.md").write_text(text)


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------

def ecrire_contrat(out: Path) -> None:
    """Écrit le format de colonnes que le modèle et l'API doivent recevoir."""
    import json

    payload = {
        "columns": COLUMNS,
        "targets": ["rain_next1_mm", "rain_next3_mm"],
        "id": ["date", "year", "month", "lat", "lon"],
    }
    (out / "contrat_colonnes.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def verifier_table(path: Path) -> dict:
    """Échoue si le CSV n'a pas exactement les colonnes du contrat."""
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Table absente ou vide : {path}")
    with path.open(newline="") as handle:
        header = next(csv.reader(handle), None)
        first = next(csv.reader(handle), None)
    if header != COLUMNS:
        raise ValueError(
            f"{path.name} : colonnes {header} au lieu de {COLUMNS}"
        )
    if first is None:
        raise ValueError(f"{path.name} : aucune ligne de données")
    return {"path": str(path), "columns": len(header), "bytes": path.stat().st_size}


def main() -> int:
    global DATA_ROOT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lon-min", type=float, default=-20.0)
    ap.add_argument("--lon-max", type=float, default=20.0)
    ap.add_argument("--lat-min", type=float, default=-10.0)
    ap.add_argument("--lat-max", type=float, default=25.0)
    ap.add_argument("--table-res", type=float, default=0.5,
                    help="résolution de la table de features, en degrés (0.5 par défaut)")
    ap.add_argument("--data", type=Path, default=DATA,
                    help="dossier des produits satellite (CHIRPS, IMERG, indices)")
    ap.add_argument("--out", type=Path, default=DATA / "out")
    ap.add_argument("--no-csv", action="store_true", help="ne pas écrire les tables CSV")
    ap.add_argument("--keep-ocean", action="store_true",
                    help="conserver les mailles océaniques (vides pour CHIRPS)")
    args = ap.parse_args()

    DATA_ROOT = args.data
    if not DATA_ROOT.is_dir():
        log(f"Dossier d'entrée introuvable : {DATA_ROOT}")
        return 1

    bbox = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    report: list[str] = []
    t0 = dt.datetime.now()

    log("=" * 72)
    log("Prétraitement « pluie prévue »")
    log(f"  bbox lon [{bbox[0]}, {bbox[1]}]  lat [{bbox[2]}, {bbox[3]}]")
    log(f"  entrée : {DATA_ROOT}")
    log(f"  sortie : {out}")
    log("=" * 72)

    # --- indices CPC ---
    cpc_dir = find_dir("03_indices_cpc")
    cpc = {"oni": {}, "roni": {}, "soi": {}}
    nino_mth: dict[int, dict[str, float]] = {}
    if cpc_dir:
        cpc["oni"] = parse_seasonal(cpc_dir / "oni.ascii.txt", 3)
        cpc["roni"] = parse_seasonal(cpc_dir / "RONI.ascii.txt", 2)
        cpc["soi"] = parse_soi(cpc_dir / "soi")
        nino_mth = parse_monthly_ascii(
            cpc_dir / "ersst5.nino.mth.91-20.ascii",
            ["nino12", "nino12_anom", "nino3", "nino3_anom",
             "nino4", "nino4_anom", "nino34", "nino34_anom"],
        )
        report.append(
            f"CPC : ONI {len(cpc['oni'])} mois, RONI {len(cpc['roni'])} mois, "
            f"SOI {len(cpc['soi'])} mois, Niño {len(nino_mth)} mois"
        )
    else:
        report.append("CPC : dossier 03_indices_cpc introuvable")

    # --- produits pluie ---
    chirps = None
    d = find_dir("01_chirps_v3_mensuel", "00_docs/01_chirps_v3_mensuel")
    if d:
        log("\n[1/4] Lecture CHIRPS…")
        chirps = read_chirps(d, bbox, report)
    else:
        report.append("CHIRPS : dossier introuvable")

    imerg = None
    d = find_dir("04_imerg_final")
    if d:
        log("\n[2/4] Lecture IMERG…")
        imerg = read_imerg(d, bbox, report)
    else:
        report.append("IMERG : dossier introuvable")

    available = [p for p in (chirps, imerg) if p]
    if not available:
        log("Aucun produit de pluie exploitable : arrêt.")
        (out / "rapport_controle.txt").write_text("\n".join(report) + "\n")
        return 1

    kmins = [month_key(y, m) for y, m in available[0]["months"]]
    kmaxs = [month_key(y, m) for y, m in available[0]["months"]]
    for p in available[1:]:
        kk = [month_key(y, m) for y, m in p["months"]]
        kmins.append(min(kk))
        kmaxs.append(max(kk))
    kmin, kmax = min(kmins), max(kmaxs)

    # --- écriture des cubes pluie ---
    log("\n[3/4] Écriture des cubes NetCDF…")
    if chirps:
        write_rain_nc(
            out / "chirps_sahel_mensuel.nc", chirps["cube"], chirps["months"],
            chirps["lon"], chirps["lat"],
            title="CHIRPS v3.0 mensuel — précipitations",
            source="UCSB CHC — https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/",
            notes="mm/mois ; NaN = valeur manquante (fill -9999 d'origine) ; "
                  "lat et lon croissants ; EPSG:4326.",
        )
    if imerg:
        write_rain_nc(
            out / "imerg_sahel_mensuel.nc", imerg["cube"], imerg["months"],
            imerg["lon"], imerg["lat"],
            title="IMERG Final V07 mensuel — précipitations",
            source="NASA GES DISC — 3B-MO.MS.MRG.3IMERG (V07B)",
            notes="intensité moyenne mm/h convertie en mm/mois via les heures du mois ; "
                  "0,1° ; NaN = valeur manquante (fill -9999.9 d'origine).",
        )

    # --- ERSST + indices ---
    log("\n[4/4] Indices ENSO et SST…")
    ers = None
    d = find_dir("02_ersst_v6")
    if d:
        ers = read_ersst(d / "sst.mnmean.nc", kmin, kmax, report)
    else:
        report.append("ERSST : dossier introuvable")

    write_indices_nc(out / "indices_enso_mensuel.nc", ers, cpc, nino_mth, kmin, kmax)
    if ers is not None:
        # contrôle croisé : Niño 3.4 recalculé (moyenne glissante 3 mois) vs ONI
        vals = ers["series"]["nino34_anom"].astype("float64")
        rm = np.full_like(vals, np.nan)
        for i in range(1, len(vals) - 1):
            rm[i] = np.nanmean(vals[i - 1:i + 2])
        oni = np.array([cpc["oni"].get(int(k), np.nan) for k in ers["keys"]])
        ok = np.isfinite(rm) & np.isfinite(oni)
        if ok.sum() > 24:
            r = float(np.corrcoef(rm[ok], oni[ok])[0, 1])
            report.append(f"Contrôle croisé Niño 3.4 recalculé (ERSST v6, glissant 3 mois) "
                          f"vs ONI CPC : r = {r:.3f} sur {int(ok.sum())} mois")
            if r < 0.85:
                report.append("  ATTENTION : corrélation faible, vérifier la climatologie SST")

    # --- tables de features ---
    stats = {}
    if not args.no_csv:
        if chirps:
            f = max(1, round(args.table_res / 0.05))
            stats["chirps"] = build_features(
                chirps["cube"], chirps["months"], chirps["lon"], chirps["lat"],
                f, cpc, ers, out / "features_chirps_sahel.csv", "CHIRPS",
                land=chirps.get("land"), drop_ocean=not args.keep_ocean)
        if imerg:
            f = max(1, round(args.table_res / 0.1))
            stats["imerg"] = build_features(
                imerg["cube"], imerg["months"], imerg["lon"], imerg["lat"],
                f, cpc, ers, out / "features_imerg_sahel.csv", "IMERG",
                land=imerg.get("land"), drop_ocean=not args.keep_ocean)

    # --- rapport ---
    lines = []
    lines.append("RAPPORT DE CONTRÔLE — prétraitement « pluie prévue »")
    lines.append(f"Exécuté le {t0:%Y-%m-%d %H:%M}  (durée {(dt.datetime.now() - t0).seconds} s)")
    lines.append(f"bbox lon [{bbox[0]}, {bbox[1]}] lat [{bbox[2]}, {bbox[3]}]")
    lines.append(f"sortie : {out}")
    lines.append("")
    lines.append("## Sources lues")
    lines.extend(report)
    lines.append("")

    for name, prod in (("CHIRPS", chirps), ("IMERG", imerg)):
        if not prod:
            continue
        cube = prod["cube"]
        months = prod["months"]
        lines.append(f"## {name} — contrôle qualité")
        lines.append(f"grille : {cube.shape[2]} lon × {cube.shape[1]} lat, "
                     f"{cube.shape[0]} mois, résolution "
                     f"{abs(prod['lon'][1] - prod['lon'][0]):.3g}°")
        lines.append("mois | moyenne mm | médiane mm | max mm | % NaN")
        for i, (y, m) in enumerate(months):
            a = cube[i]
            v = a[np.isfinite(a)]
            pct = 100.0 * (a.size - v.size) / a.size
            if v.size:
                lines.append(f"{y}-{m:02d} | {v.mean():8.1f} | {np.median(v):8.1f} | "
                             f"{v.max():8.1f} | {pct:5.1f}")
            else:
                lines.append(f"{y}-{m:02d} | {'—':>8} | {'—':>8} | {'—':>8} | {pct:5.1f}")
        # mois manquants dans l'intervalle couvert
        kk = {month_key(y, m) for y, m in months}
        holes = [k for k in range(min(kk), max(kk) + 1) if k not in kk]
        if holes:
            lines.append("mois manquants dans l'intervalle : "
                         + ", ".join(f"{k // 12}-{k % 12 + 1:02d}" for k in holes))
        lines.append("")

    if chirps and imerg:
        kc = {month_key(y, m) for y, m in chirps["months"]}
        ki = {month_key(y, m) for y, m in imerg["months"]}
        inter = sorted(kc & ki)
        lines.append("## Recouvrement CHIRPS / IMERG")
        if inter:
            lines.append(f"{len(inter)} mois communs : "
                         f"{inter[0] // 12}-{inter[0] % 12 + 1:02d} → "
                         f"{inter[-1] // 12}-{inter[-1] % 12 + 1:02d}")
        else:
            lines.append("AUCUN mois commun : les deux produits ne peuvent pas être")
            lines.append("fusionnés dans une même table. Utiliser CHIRPS pour la")
            lines.append("période 1981-1985 et IMERG pour 1998-1999, ou télécharger")
            lines.append("davantage de granules IMERG (1998→présent).")
        lines.append("")

    if stats:
        lines.append("## Tables de features")
        for name, st in stats.items():
            if st["climatology_years"] >= 2:
                clim_txt = (f"climatologie sur {st['climatology_years']} années complètes "
                            f"({', '.join(map(str, st['complete_years']))})")
            else:
                clim_txt = ("moins de 2 années complètes : clim_mm, anom_* et pct_clim "
                            "laissés vides (une seule année = fuite de la cible)")
            lines.append(f"{name} : {st['shape'][0]} mois × {st['cells']} mailles "
                         f"(résolution table) = {st['rows']:,} lignes ; "
                         f"{st['dropped_cells']} mailles vides écartées ; {clim_txt}")
        lines.append("")
        lines.append("Rappel : la climatologie est calculée sur les seules années")
        lines.append("complètes présentes sur le disque. Elle devra être recalculée")
        lines.append("quand les téléchargements seront terminés.")
        lines.append("")

    lines.append("## Avertissements")
    if chirps and len(chirps["months"]) < 550:
        lines.append(f"- CHIRPS incomplet : {len(chirps['months'])}/550 mois.")
    if imerg and len(imerg["months"]) < 333:
        lines.append(f"- IMERG incomplet : {len(imerg['months'])}/333 granules.")
    lines.append("- Les scripts avancement.sh et resume_all.sh pointent vers")
    lines.append("  « 01_chirps_v3_mensuel/ » à la racine, alors que les fichiers")
    lines.append("  sont dans « 00_docs/01_chirps_v3_mensuel/ » : la reprise des")
    lines.append("  téléchargements échouera tant que ce chemin n'est pas corrigé.")
    lines.append("- MODIS NDVI : aucun granule présent (dossier scripts uniquement).")

    (out / "rapport_controle.txt").write_text("\n".join(lines) + "\n")
    ecrire_contrat(out)
    if not args.no_csv and chirps:
        verifier_table(out / "features_chirps_sahel.csv")
    write_readme(out, bbox, args, chirps, imerg, stats)
    log(f"\n  → rapport_controle.txt")
    log(f"\nTerminé en {(dt.datetime.now() - t0).seconds} s. Sorties dans {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
