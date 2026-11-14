#!/usr/bin/env python3
"""Génère les fichiers mock du contrat de données à partir des sorties de
`pretraiter.py`.

Lit   : donnees_pluie_prevue/preprocessed/chirps_sahel_mensuel.nc
Écrit : contract/mock/*.json  (une réponse d'API par fichier)
        contract/mock/index.json (manifeste endpoint → fichier)

Le « modèle » utilisé ici est une référence volontairement simple
(persistance d'anomalie amortie) : elle produit des nombres plausibles et
surtout des scores d'évaluation réels. Le front ne dépend que du contrat,
pas de ce modèle : remplacer `model.id` suffira.

Usage :
    .venv/bin/python contract/generer_mock.py
    .venv/bin/python contract/generer_mock.py --res 1.0 --frames 6
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "donnees_pluie_prevue"
PRE = DATA / "preprocessed"
OUT = HERE / "mock"

CONTRACT_VERSION = "1.0.0"
EPOCH = dt.date(1970, 1, 1)
Z90 = 1.2815515655446004

# --- palettes (identiques à celles que le front devra appliquer) ------------
PAL_RAIN = ["#ffffe5", "#f7fcb9", "#d9f0a3", "#addd8e",
            "#78c679", "#41ab5d", "#238443", "#006837", "#004529"]
PAL_DIV = ["#8c510a", "#d8b365", "#f6e8c3", "#f5f5f5",
           "#c7eae5", "#5ab4ac", "#01665e"]
PAL_PROB = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1",
            "#6baed6", "#3182bd", "#08519c"]

CLASSES = [
    ("deficit_severe", "Déficit sévère", "#8c510a"),
    ("deficit", "Déficit", "#d8b365"),
    ("normal", "Normal", "#e8e8e8"),
    ("excedent", "Excédent", "#5ab4ac"),
    ("excedent_fort", "Excédent fort", "#01665e"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def iso(d: dt.date) -> str:
    return d.isoformat()


def num(x, dec: int = 1):
    """float arrondi ou None (JSON n'accepte pas NaN)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, dec)


def flat(arr: np.ndarray, dec: int) -> list:
    """Tableau 2D → liste plate avec None, arrondie."""
    out = []
    for v in np.asarray(arr, dtype="float64").ravel():
        out.append(None if not math.isfinite(v) else round(float(v), dec))
    return out


def stops(values: list[float], colors: list[str]) -> list[dict]:
    return [{"value": round(float(v), 3), "color": c} for v, c in zip(values, colors)]


def lerp_stops(vmin: float, vmax: float, colors: list[str]) -> list[dict]:
    n = len(colors)
    vals = [vmin + (vmax - vmin) * i / (n - 1) for i in range(n)]
    return stops(vals, colors)


def block_mean(cube: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return cube
    t, h, w = cube.shape
    h2, w2 = h // f, w // f
    c = cube[:, : h2 * f, : w2 * f]
    return np.nanmean(c.reshape(t, h2, f, w2, f), axis=(2, 4))


def block_centers(coord: np.ndarray, f: int) -> np.ndarray:
    n = (coord.size // f) * f
    return coord[:n].reshape(-1, f).mean(axis=1)


def cf(x: float) -> float:
    """CDF normale centrée réduite."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.5, help="résolution des mocks (degrés)")
    ap.add_argument("--frames", type=int, default=12, help="nombre de mois inclus par couche")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from netCDF4 import Dataset

    src = PRE / "chirps_sahel_mensuel.nc"
    if not src.exists():
        log(f"Fichier absent : {src}\nLance d'abord : cd donnees_pluie_prevue && "
            f".venv/bin/python ../pretraiter.py")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- lecture
    with Dataset(src) as d:
        raw = np.ma.filled(d["precipitation"][:], np.nan).astype("float32")
        lat0 = np.asarray(d["lat"][:], dtype="float64")
        lon0 = np.asarray(d["lon"][:], dtype="float64")
        tdays = np.asarray(d["time"][:], dtype="float64")
        product = "chirps"
    months = [EPOCH + dt.timedelta(days=float(x)) for x in tdays]

    factor = max(1, round(args.res / 0.05))
    cube = block_mean(raw, factor)
    lat = block_centers(lat0, factor)
    lon = block_centers(lon0, factor)
    ny, nx = cube.shape[1], cube.shape[2]
    land = np.isfinite(cube).any(axis=0)
    log(f"grille mock : {ny} lat × {nx} lon à {abs(lon[1]-lon[0]):.2f}°, "
        f"{len(months)} mois, {land.sum()} mailles terrestres")

    # ------------------------------------------------- climatologie 1981-1984
    keys = [(m.year, m.month) for m in months]
    present = set(keys)
    years = sorted({y for y, _ in keys})
    complete = [y for y in years if all((y, mm) in present for mm in range(1, 13))]
    if len(complete) < 2:
        log("ERREUR : moins de 2 années complètes, climatologie impossible.")
        return 1
    log(f"climatologie sur {complete}")

    clim = np.full((12, ny, nx), np.nan, dtype="float32")
    std = np.full((12, ny, nx), np.nan, dtype="float32")
    frac_above = np.full((12, ny, nx), np.nan, dtype="float32")
    for mm in range(1, 13):
        idx = [i for i, (y, m) in enumerate(keys) if m == mm and y in complete]
        sel = cube[idx]
        clim[mm - 1] = np.nanmean(sel, axis=0)
        std[mm - 1] = np.nanstd(sel, axis=0)
        with np.errstate(invalid="ignore"):
            frac_above[mm - 1] = np.nanmean(sel > clim[mm - 1][None], axis=0)

    m_idx = np.array([m.month - 1 for m in months])
    anom = cube - clim[m_idx]

    def pct_of(a, c):
        with np.errstate(invalid="ignore", divide="ignore"):
            p = 100.0 * a / c
        p[~np.isfinite(p)] = np.nan
        p[c <= 0] = np.nan
        return p

    anom_pct = np.array([pct_of(anom[i], clim[m_idx[i]]) for i in range(len(months))])

    # -------------------------------------------------------- modèle de référence
    train_mask = np.array([y in complete for y, _ in keys])
    ti = np.where(train_mask)[0]
    a1 = anom[ti[1:]].ravel()
    a0 = anom[ti[:-1]].ravel()
    ok = np.isfinite(a1) & np.isfinite(a0)
    rho = float(np.corrcoef(a0[ok], a1[ok])[0, 1]) if ok.sum() > 100 else 0.3
    rho = float(np.clip(rho, 0.0, 0.8))
    log(f"autocorrélation lag-1 des anomalies (1981-1984) : rho = {rho:.3f}")

    def add_months(d: dt.date, n: int) -> dt.date:
        y = d.year + (d.month - 1 + n) // 12
        m = (d.month - 1 + n) % 12 + 1
        return dt.date(y, m, 1)

    def forecast_field(t_origin: int, lead: int, target: dt.date | None = None):
        """Prévision pour toutes les mailles depuis le mois `t_origin`, échéance `lead`.

        `target` permet de forcer le mois visé (rétro-test) ; sinon il est
        calculé, éventuellement au-delà du dernier mois observé.
        """
        if target is None:
            target = add_months(months[t_origin], lead)
        tm = target.month - 1
        c = clim[tm]
        s = std[tm]
        fa = (rho ** lead) * anom[t_origin]
        fc = c + fa
        with np.errstate(invalid="ignore", divide="ignore"):
            prob = np.array([[cf(float(fa[i, j] / s[i, j])) if np.isfinite(fa[i, j]) and np.isfinite(s[i, j]) and s[i, j] > 0 else np.nan
                              for j in range(nx)] for i in range(ny)], dtype="float32")
        p10 = fc - Z90 * s
        p90 = fc + Z90 * s
        return {
            "target": target, "clim": c, "std": s,
            "anom_fc": fa, "rain_fc": fc, "p10": p10, "p90": p90, "prob": prob,
        }

    # évaluation : rho estimé sur 1981-1984, test sur 1985
    test = [i for i, (y, _) in enumerate(keys) if y > max(complete)]
    per_lead = {}
    for lead in (1, 3):
        errs, errs_clim, n = 0.0, 0.0, 0
        bias = 0.0
        for t in test:
            if t - lead < 0:
                continue
            f = forecast_field(t - lead, lead, target=months[t])
            obs = cube[t]
            okm = np.isfinite(obs) & np.isfinite(f["rain_fc"])
            if not okm.any():
                continue
            d = f["rain_fc"][okm] - obs[okm]
            dc = f["clim"][okm] - obs[okm]
            errs += float(np.sum(d * d)); errs_clim += float(np.sum(dc * dc))
            bias += float(np.sum(d)); n += int(okm.sum())
        rmse = math.sqrt(errs / n) if n else None
        rmse_c = math.sqrt(errs_clim / n) if n else None
        per_lead[str(lead)] = {
            "rmse": num(rmse, 2), "mae": None,
            "bias": num(bias / n if n else None, 2),
            "skill_vs_climatology": num(1 - rmse / rmse_c if rmse and rmse_c else None, 3),
            "n": n,
        }
        log(f"  lead {lead} : RMSE={rmse:.1f} mm (clim {rmse_c:.1f})  n={n}")

    # MAE du lead 1 (calcul séparé, plus lisible)
    mae_n, mae_s = 0, 0.0
    for t in test:
        if t < 1:
            continue
        f = forecast_field(t - 1, 1)
        obs = cube[t]
        okm = np.isfinite(obs) & np.isfinite(f["rain_fc"])
        mae_s += float(np.sum(np.abs(f["rain_fc"][okm] - obs[okm])))
        mae_n += int(okm.sum())
    if mae_n:
        per_lead["1"]["mae"] = num(mae_s / mae_n, 2)

    t_last = len(months) - 1
    issued = months[t_last]

    # ------------------------------------------------------------- méta / grille
    grid = {
        "type": "regular_latlon",
        "nx": int(nx), "ny": int(ny),
        "resolution": round(float(abs(lon[1] - lon[0])), 6),
        "lon": {"start": round(float(lon[0]), 4), "step": round(float(lon[1] - lon[0]), 6),
                "count": int(nx), "order": "ascending"},
        "lat": {"start": round(float(lat[0]), 4), "step": round(float(lat[1] - lat[0]), 6),
                "count": int(ny), "order": "ascending"},
        "layout": "row_major_lat_then_lon",
        "nodata": None,
    }

    valid_rain = cube[np.isfinite(cube)]
    rmax = float(np.nanpercentile(valid_rain, 99))
    rmax = max(200.0, math.ceil(rmax / 50.0) * 50.0)
    amax = float(np.nanpercentile(np.abs(anom), 99))
    amax = max(50.0, math.ceil(amax / 25.0) * 25.0)
    log(f"échelles : pluie 0-{rmax:.0f} mm, anomalie ±{amax:.0f} mm")

    layers = [
        {
            "id": "rain_mm", "label": "Pluie", "unit": "mm/mois",
            "type": "continuous", "value_scale": "linear",
            "range": {"min": 0.0, "max": rmax}, "clamp": True, "precision": 1,
            "palette": {"kind": "sequential", "stops": lerp_stops(0, rmax, PAL_RAIN)},
            "legend": {"ticks": [round(t, 0) for t in np.linspace(0, rmax, 5)]},
            "scope": ["observation", "forecast"], "available": True,
        },
        {
            "id": "anomaly_mm", "label": "Anomalie", "unit": "mm/mois",
            "type": "continuous", "value_scale": "linear",
            "range": {"min": -amax, "max": amax}, "clamp": True, "precision": 1,
            "palette": {"kind": "diverging", "stops": lerp_stops(-amax, amax, PAL_DIV)},
            "legend": {"ticks": [-amax, -amax / 2, 0.0, amax / 2, amax]},
            "scope": ["observation", "forecast"], "available": True,
        },
        {
            "id": "anomaly_pct", "label": "Écart à la normale", "unit": "%",
            "type": "continuous", "value_scale": "linear",
            "range": {"min": -100.0, "max": 200.0}, "clamp": True, "precision": 0,
            "palette": {"kind": "diverging",
                        "stops": stops([-100, -50, 0, 50, 100, 150, 200], PAL_DIV)},
            "legend": {"ticks": [-100.0, -50.0, 0.0, 50.0, 100.0, 200.0]},
            "scope": ["observation"], "available": True,
        },
        {
            "id": "anomaly_z", "label": "Anomalie standardisée", "unit": "σ",
            "type": "continuous", "value_scale": "linear",
            "range": {"min": -2.5, "max": 2.5}, "clamp": True, "precision": 2,
            "palette": {"kind": "diverging",
                        "stops": stops([-2.5, -1.5, -0.75, 0, 0.75, 1.5, 2.5], PAL_DIV)},
            "legend": {"ticks": [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]},
            "scope": ["observation"], "available": True,
        },
        {
            "id": "prob_above_normal", "label": "Probabilité de pluie > normale", "unit": "0-1",
            "type": "continuous", "value_scale": "linear",
            "range": {"min": 0.0, "max": 1.0}, "clamp": True, "precision": 2,
            "palette": {"kind": "sequential", "stops": lerp_stops(0, 1, PAL_PROB)},
            "legend": {"ticks": [0.0, 0.25, 0.5, 0.75, 1.0],
                       "labels": ["sec", "plutôt sec", "incertain", "plutôt humide", "humide"]},
            "scope": ["forecast"], "available": True,
        },
        {
            "id": "spi3", "label": "SPI 3 mois", "unit": "σ",
            "type": "continuous", "value_scale": "linear",
            "range": {"min": -2.5, "max": 2.5}, "clamp": True, "precision": 2,
            "palette": {"kind": "diverging",
                        "stops": stops([-2.5, -1.5, -0.75, 0, 0.75, 1.5, 2.5], PAL_DIV)},
            "legend": {"ticks": [-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0]},
            "scope": ["observation"], "available": False,
            "reason": "nécessite 30 ans d'historique pour ajuster la loi gamma",
        },
    ]

    generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    sources = [
        {"id": "chirps", "label": "CHIRPS v3.0 (mensuel)",
         "url": "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/",
         "attribution": "© UCSB Climate Hazards Center — CHIRPS v3.0"},
        {"id": "cpc", "label": "NOAA CPC — indices ONI, RONI, SOI, Niño",
         "url": "https://www.cpc.ncep.noaa.gov/data/indices/",
         "attribution": "© NOAA CPC"},
        {"id": "ersst", "label": "NOAA PSL — ERSST v6",
         "url": "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v6/",
         "attribution": "© NOAA PSL — ERSST v6"},
    ]

    meta = {
        "resource": "meta",
        "contract_version": CONTRACT_VERSION,
        "dataset": {
            "id": "pluie-prevue-sahel",
            "title": "Pluie prévue — Afrique de l'Ouest",
            "product": product,
            "generated_at": generated.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "language": "fr",
        },
        "region": {"id": "sahel", "label": "Afrique de l'Ouest / Sahel",
                   "bbox": [round(float(lon[0] - 0.25), 3), round(float(lat[0] - 0.25), 3),
                            round(float(lon[-1] + 0.25), 3), round(float(lat[-1] + 0.25), 3)],
                   "crs": "EPSG:4326"},
        "grid": grid,
        "time": {
            "step": "P1M",
            "start": iso(months[0]),
            "end": iso(months[-1]),
            "available": [iso(m) for m in months],
            "default": iso(months[-1]),
        },
        "layers": layers,
        "sources": sources,
        "render": {"mode": "inline_grid", "tile_url": None, "style": "heatmap"},
    }

    # ------------------------------------------------------------ frames 12 mois
    sel = list(range(max(0, len(months) - args.frames), len(months)))
    panel = {
        "rain_mm": cube,
        "anomaly_mm": anom,
        "anomaly_pct": anom_pct,
        "anomaly_z": np.array([np.divide(anom[i], std[m_idx[i]],
                                         out=np.full((ny, nx), np.nan, dtype="float32"),
                                         where=std[m_idx[i]] > 0) for i in range(len(months))]),
    }
    layer_by_id = {l["id"]: l for l in layers}
    files = {}

    for lid, arr in panel.items():
        frames = []
        for i in sel:
            a = arr[i]
            v = a[np.isfinite(a)]
            frames.append({
                "time": iso(months[i]),
                "values": flat(a, layer_by_id[lid]["precision"]),
                "stats": {
                    "min": num(v.min(), 2) if v.size else None,
                    "max": num(v.max(), 2) if v.size else None,
                    "mean": num(v.mean(), 2) if v.size else None,
                    "valid": int(v.size),
                    "coverage": round(float(v.size) / (nx * ny), 4),
                },
            })
        payload = {
            "resource": "frames",
            "contract_version": CONTRACT_VERSION,
            "layer": lid,
            "unit": layer_by_id[lid]["unit"],
            "grid": grid,
            "frames": frames,
        }
        name = f"frames_{lid}.json"
        write(args.out / name, payload, pretty=False)
        files[lid] = name
        mb = (args.out / name).stat().st_size / 1e6
        log(f"  → {name}  {len(frames)} mois  {mb:.2f} Mo")

    # ---------------------------------------------------------------- prévisions
    model = {
        "id": f"baseline-persistence-anomaly-v1",
        "label": "Persistance d'anomalie amortie (référence)",
        "kind": "baseline",
        "status": "placeholder",
        "version": "0.1.0",
        "training": {
            "start": iso(months[0]),
            "end": iso(months[[i for i, (y, _) in enumerate(keys) if y == max(complete)][-1]]),
            "parameters": {"rho": round(rho, 3), "decay": "rho^lead", "interval": 0.8},
        },
        "evaluation": {
            "method": (f"holdout temporel : rho estimé sur {complete[0]}-{complete[-1]}, "
                       f"testé sur {min(y for y, _ in keys if y > max(complete))}"),
            "per_lead": per_lead,
        },
    }

    for lead in (1, 3):
        f = forecast_field(t_last, lead)
        okc = np.isfinite(f["rain_fc"]) & land
        mean_fc = float(np.nanmean(f["rain_fc"][okc])) if okc.any() else float("nan")
        mean_cl = float(np.nanmean(f["clim"][okc])) if okc.any() else float("nan")
        prob_ok = f["prob"][okc]
        prob_mean = float(np.nanmean(prob_ok)) if okc.any() else float("nan")
        area_wet = 100.0 * float(np.mean(f["anom_fc"][okc] > 0)) if okc.any() else float("nan")
        payload = {
            "resource": "forecast",
            "contract_version": CONTRACT_VERSION,
            "model": model,
            "issued_at": iso(issued),
            "target": iso(f["target"]),
            "lead_months": lead,
            "grid": grid,
            "layers": {
                "rain_mm": {
                    "values": flat(f["rain_fc"], 1),
                    "p10": flat(f["p10"], 1),
                    "p90": flat(f["p90"], 1),
                },
                "anomaly_mm": {"values": flat(f["anom_fc"], 1)},
                "prob_above_normal": {"values": flat(f["prob"], 2)},
            },
            "summary": {
                "region_mean_mm": num(mean_fc, 1),
                "region_clim_mm": num(mean_cl, 1),
                "region_anomaly_pct": num(100 * (mean_fc - mean_cl) / mean_cl if mean_cl else None, 1),
                "prob_wetter_than_normal": num(prob_mean, 2),
                "area_wetter_pct": num(area_wet, 1),
                "area_drier_pct": num(100 - area_wet, 1) if math.isfinite(area_wet) else None,
            },
            "baseline": {"id": "climatology", "label": "Climatologie 1981-1984",
                         "rmse": per_lead[str(lead)]["rmse"]},
        }
        name = f"forecast_lead{lead}.json"
        write(args.out / name, payload, pretty=False)
        files[f"forecast_{lead}"] = name
        log(f"  → {name}  {iso(issued)} → {iso(f['target'])}")

    # ------------------------------------------------------------------ alertes
    def level_from(v: float) -> str:
        a = abs(v) if math.isfinite(v) else 0.0
        if a < 2:   return "none"
        if a < 10:  return "low"
        if a < 25:  return "moderate"
        if a < 45:  return "high"
        return "extreme"

    def alert_item(i: int) -> dict:
        p = anom_pct[i]
        z = panel["anomaly_z"][i]
        okc = np.isfinite(p) & land
        ap = float(np.nanmean(p[okc])) if okc.any() else float("nan")
        az = float(np.nanmean(z[okc])) if okc.any() else float("nan")
        d_sev = float(np.mean(p[okc] <= -50) * 100) if okc.any() else float("nan")
        below = float(np.mean(p[okc] < 0) * 100) if okc.any() else float("nan")
        rain_ok = np.isfinite(cube[i]) & land
        rmean = float(np.nanmean(cube[i][rain_ok])) if rain_ok.any() else float("nan")
        cmean = float(np.nanmean(clim[m_idx[i]][rain_ok])) if rain_ok.any() else float("nan")

        counts = {cid: 0 for cid, _, _ in CLASSES}
        if okc.any():
            pv = p[okc]
            counts["deficit_severe"] = int(np.sum(pv <= -50))
            counts["deficit"] = int(np.sum((pv > -50) & (pv <= -20)))
            counts["normal"] = int(np.sum((pv > -20) & (pv < 20)))
            counts["excedent"] = int(np.sum((pv >= 20) & (pv < 50)))
            counts["excedent_fort"] = int(np.sum(pv >= 50))
        total = max(1, sum(counts.values()))

        # top 5 mailles par |z|
        order = np.argsort(-np.abs(np.where(okc, z, np.nan)).ravel())[:5]
        top = []
        for k in order:
            iy, ix = divmod(int(k), nx)
            if not math.isfinite(z[iy, ix]):
                continue
            top.append({"lat": round(float(lat[iy]), 2), "lon": round(float(lon[ix]), 2),
                        "anomaly_pct": num(p[iy, ix], 1), "anomaly_z": num(z[iy, ix], 2),
                        "level": level_from(float(z[iy, ix]))})

        lvl = level_from(ap)
        label = {"none": "Aucune alerte", "low": "Vigilance", "moderate": "Modéré",
                 "high": "Élevé", "extreme": "Extrême"}[lvl]
        return {
            "resource": "alerts",
            "contract_version": CONTRACT_VERSION,
            "time": iso(months[i]),
            "severity": {"level": lvl, "label": label,
                         "score": num(min(1.0, abs(ap) / 45.0) if math.isfinite(ap) else 0.0, 2)},
            "indicators": {
                "rain_mean_mm": num(rmean, 1), "clim_mean_mm": num(cmean, 1),
                "anomaly_pct": num(ap, 1), "anomaly_z_mean": num(az, 2),
                "area_below_normal_pct": num(below, 1),
                "area_deficit_severe_pct": num(d_sev, 1),
            },
            "classes": [{"id": cid, "label": lab,
                         "count": counts[cid],
                         "area_pct": round(100.0 * counts[cid] / total, 1),
                         "color": col} for cid, lab, col in CLASSES],
            "top_cells": top,
            "history": [{"time": iso(months[k]),
                         "anomaly_pct": num(float(np.nanmean(anom_pct[k][np.isfinite(anom_pct[k]) & land])), 1),
                         "level": level_from(float(np.nanmean(anom_pct[k][np.isfinite(anom_pct[k]) & land])))}
                        for k in range(max(0, t_last - 11), t_last + 1)],
        }

    payload = alert_item(t_last)
    write(args.out / "alerts.json", payload, pretty=True)
    files["alerts"] = "alerts.json"
    log(f"  → alerts.json  {iso(months[t_last])}  niveau {payload['severity']['level']}")

    # ------------------------------------------------------------------- points
    sites = [
        ("niamey", "Niamey", 13.5, 2.1),
        ("ouagadougou", "Ouagadougou", 12.4, -1.5),
        ("dakar", "Dakar", 14.7, -17.4),
    ]
    fc1 = forecast_field(t_last, 1)
    fc3 = forecast_field(t_last, 3)
    for slug, label, sla, slo in sites:
        iy = int(np.argmin(np.abs(lat - sla)))
        ix = int(np.argmin(np.abs(lon - slo)))
        series = []
        for i, m in enumerate(months):
            r = cube[i, iy, ix]
            c = clim[m_idx[i], iy, ix]
            s = std[m_idx[i], iy, ix]
            a = anom[i, iy, ix]
            series.append({
                "time": iso(m), "kind": "observation",
                "rain_mm": num(r, 1), "clim_mm": num(c, 1),
                "anomaly_mm": num(a, 1), "anomaly_z": num(a / s if s and s > 0 else None, 2),
                "forecast_mm": None, "forecast_p10": None, "forecast_p90": None,
            })
        for f in (fc1, fc3):
            c = f["clim"][iy, ix]
            series.append({
                "time": iso(f["target"]), "kind": "forecast",
                "rain_mm": None, "clim_mm": num(c, 1),
                "anomaly_mm": None, "anomaly_z": None,
                "forecast_mm": num(f["rain_fc"][iy, ix], 1),
                "forecast_p10": num(f["p10"][iy, ix], 1),
                "forecast_p90": num(f["p90"][iy, ix], 1),
            })
        payload = {
            "resource": "point",
            "contract_version": CONTRACT_VERSION,
            "cell": {"lat": round(float(lat[iy]), 2), "lon": round(float(lon[ix]), 2),
                     "is_land": bool(land[iy, ix])},
            "label": label,
            "unit": "mm/mois",
            "series": series,
        }
        name = f"point_{slug}.json"
        write(args.out / name, payload, pretty=True)
        files[f"point_{slug}"] = name
        log(f"  → {name}  ({payload['cell']['lat']}, {payload['cell']['lon']})")

    # ------------------------------------------------------------------- méta
    write(args.out / "meta.json", meta, pretty=True)
    files["meta"] = "meta.json"
    log("  → meta.json")

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": meta["dataset"]["generated_at"],
        "schema": "../schema.json",
        "endpoints": [
            {"method": "GET", "path": "/api/meta", "resource": "meta", "file": "meta.json"},
            {"method": "GET", "path": "/api/frames?layer=rain_mm", "resource": "frames", "file": "frames_rain_mm.json"},
            {"method": "GET", "path": "/api/frames?layer=anomaly_mm", "resource": "frames", "file": "frames_anomaly_mm.json"},
            {"method": "GET", "path": "/api/frames?layer=anomaly_pct", "resource": "frames", "file": "frames_anomaly_pct.json"},
            {"method": "GET", "path": "/api/frames?layer=anomaly_z", "resource": "frames", "file": "frames_anomaly_z.json"},
            {"method": "GET", "path": "/api/forecast?lead=1", "resource": "forecast", "file": "forecast_lead1.json"},
            {"method": "GET", "path": "/api/forecast?lead=3", "resource": "forecast", "file": "forecast_lead3.json"},
            {"method": "GET", "path": "/api/point?lat=13.5&lon=2.1", "resource": "point", "file": "point_niamey.json"},
            {"method": "GET", "path": "/api/point?lat=12.4&lon=-1.5", "resource": "point", "file": "point_ouagadougou.json"},
            {"method": "GET", "path": "/api/point?lat=14.7&lon=-17.4", "resource": "point", "file": "point_dakar.json"},
            {"method": "GET", "path": "/api/alerts", "resource": "alerts", "file": "alerts.json"},
        ],
    }
    write(args.out / "index.json", manifest, pretty=True)
    log(f"\n{len(files) + 1} fichiers mock + manifeste dans {args.out}")
    return 0


def write(path: Path, obj, pretty: bool) -> None:
    if pretty:
        text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    path.write_text(text)


if __name__ == "__main__":
    sys.exit(main())
