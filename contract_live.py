"""Réponses au contrat frontend, calculées depuis le CSV et les modèles XGBoost.

Même formes que contract/mock : meta, frames, forecast, point, alerts.
Les prévisions viennent des modèles entraînés, pas du fichier mock.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

CONTRACT_VERSION = "1.0.0"
Z90 = 1.2815515655446004
FRAME_MONTHS = 12

FEATURES = [
    "year", "month", "lat", "lon",
    "rain_mm", "clim_mm", "anom_mm", "anom_z", "pct_clim",
    "rain_3m_mm", "anom_3m_mm",
    "rain_lag1_mm", "rain_lag3_mm", "rain_lag6_mm", "rain_lag12_mm",
    "oni", "roni", "soi",
    "nino12_anom", "nino3_anom", "nino4_anom", "nino34_anom", "atl3_anom",
    "oni_lag3", "oni_lag6",
    "month_sin", "month_cos",
]

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
CITIES = (
    ("Niamey", 13.5, 2.1),
    ("Ouagadougou", 12.4, -1.5),
    ("Dakar", 14.7, -17.4),
)
LEVEL_LABEL = {
    "none": "Aucune alerte",
    "low": "Vigilance",
    "moderate": "Modéré",
    "high": "Élevé",
    "extreme": "Extrême",
}


def num(x, dec: int):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, dec)


def flat(arr: np.ndarray, dec: int) -> list:
    out = []
    for v in np.asarray(arr, dtype="float64").ravel():
        out.append(None if not math.isfinite(v) else round(float(v), dec))
    return out


def _cdf(z: np.ndarray) -> np.ndarray:
    out = np.full(z.shape, np.nan, dtype="float64")
    ok = np.isfinite(z)
    zz = np.clip(z[ok], -8.0, 8.0)
    out[ok] = 0.5 * (1.0 + np.vectorize(math.erf)(zz / math.sqrt(2.0)))
    return out


def _stops(values, colors):
    return [{"value": round(float(v), 3), "color": c} for v, c in zip(values, colors)]


def _lerp_stops(vmin: float, vmax: float, colors: list[str]):
    n = len(colors)
    vals = [vmin + (vmax - vmin) * i / (n - 1) for i in range(n)]
    return _stops(vals, colors)


def _add_months(d: date, n: int) -> date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return date(y, m, 1)


def _level(v: float) -> str:
    a = abs(v) if math.isfinite(v) else 0.0
    if a < 2:
        return "none"
    if a < 10:
        return "low"
    if a < 25:
        return "moderate"
    if a < 45:
        return "high"
    return "extreme"


class Catalog:
    """Grilles et prévisions gardées en mémoire après le chargement du CSV."""

    def __init__(self, df: pd.DataFrame, models: dict[int, object]):
        self.df = df
        self.models = models
        # Même grille que le contrat frontend : 0,5°, océan compris (mailles vides = null).
        self.lat = np.round(-9.75 + 0.5 * np.arange(70), 4)
        self.lon = np.round(-19.75 + 0.5 * np.arange(80), 4)
        self.ny = int(self.lat.size)
        self.nx = int(self.lon.size)
        self.lat_pos = {float(v): i for i, v in enumerate(self.lat)}
        self.lon_pos = {float(v): i for i, v in enumerate(self.lon)}
        self.months = sorted(pd.to_datetime(df["date"].unique()))
        self.step = float(self.lon[1] - self.lon[0]) if self.nx > 1 else 0.5
        self.clim = self._monthly_field("clim_mm")
        self.std = self._std_field()
        self.scores, self.clim_rmse = self._scores()
        self.forecasts = {lead: self._forecast(lead) for lead in (1, 3) if lead in models}
        self.meta = self._meta()
        self.frames = {
            layer: self._frames(layer)
            for layer in ("rain_mm", "anomaly_mm", "anomaly_pct", "anomaly_z")
        }
        self.alerts = self._alerts()

    def place(self, sub: pd.DataFrame, values: np.ndarray) -> np.ndarray:
        grid = np.full((self.ny, self.nx), np.nan, dtype="float64")
        lat = np.round(sub["lat"].to_numpy(dtype="float64"), 4)
        lon = np.round(sub["lon"].to_numpy(dtype="float64"), 4)
        vals = np.asarray(values, dtype="float64")
        iy = np.fromiter((self.lat_pos.get(float(v), -1) for v in lat), dtype=np.int32, count=lat.size)
        ix = np.fromiter((self.lon_pos.get(float(v), -1) for v in lon), dtype=np.int32, count=lon.size)
        ok = (iy >= 0) & (ix >= 0) & np.isfinite(vals)
        grid[iy[ok], ix[ok]] = vals[ok]
        return grid

    def _monthly_field(self, column: str) -> np.ndarray:
        out = np.full((12, self.ny, self.nx), np.nan, dtype="float64")
        for month, sub in self.df.groupby("month"):
            out[int(month) - 1] = self.place(sub, sub[column].to_numpy(dtype="float64"))
        return out

    def _std_field(self) -> np.ndarray:
        z = self.df["anom_z"].to_numpy(dtype="float64")
        a = self.df["anom_mm"].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(np.abs(z) > 1e-3, a / z, np.nan)
        tmp = self.df.assign(_std=ratio)
        med = tmp.groupby(["lat", "lon", "month"], as_index=False)["_std"].median()
        out = np.full((12, self.ny, self.nx), np.nan, dtype="float64")
        for month, sub in med.groupby("month"):
            out[int(month) - 1] = self.place(sub, sub["_std"].to_numpy(dtype="float64"))
        return out

    def _month_grid(self, when: pd.Timestamp, column: str) -> np.ndarray:
        sub = self.df[self.df["date"] == when]
        if column == "anomaly_pct":
            clim = sub["clim_mm"].to_numpy(dtype="float64")
            anom = sub["anom_mm"].to_numpy(dtype="float64")
            with np.errstate(invalid="ignore", divide="ignore"):
                values = np.where(clim > 0, 100.0 * anom / clim, np.nan)
            return self.place(sub, values)
        return self.place(sub, sub[column].to_numpy(dtype="float64"))

    def _clim_sum(self, months_1based: np.ndarray, iy: np.ndarray, ix: np.ndarray, lead: int) -> np.ndarray:
        total = np.zeros(len(months_1based), dtype="float64")
        for k in range(1, lead + 1):
            mi = (months_1based - 1 + k) % 12
            total += self.clim[mi, iy, ix]
        return total

    def _scores(self) -> tuple[dict, dict]:
        scores: dict[str, dict] = {}
        clim_rmse: dict[int, float | None] = {}
        for lead, target in ((1, "rain_next1_mm"), (3, "rain_next3_mm")):
            empty = {"rmse": None, "mae": None, "bias": None, "skill_vs_climatology": None, "n": 0}
            model = self.models.get(lead)
            if model is None or target not in self.df.columns:
                scores[str(lead)] = empty
                clim_rmse[lead] = None
                continue
            test = self.df.dropna(subset=[target])
            test = test[test["date"] >= "1984-01-01"]
            if test.empty:
                scores[str(lead)] = empty
                clim_rmse[lead] = None
                continue
            X = test[FEATURES].to_numpy(dtype="float32")
            X = np.where(np.isfinite(X), X, 0.0)
            pred = np.clip(model.predict(X).astype("float64"), 0.0, None)
            y = test[target].to_numpy(dtype="float64")
            err = pred - y
            rmse = float(np.sqrt(np.mean(err * err)))
            iy = test["lat"].round(4).map(self.lat_pos)
            ix = test["lon"].round(4).map(self.lon_pos)
            ok = iy.notna() & ix.notna()
            base = np.full(len(test), np.nan)
            if ok.any():
                base[ok.to_numpy()] = self._clim_sum(
                    test.loc[ok, "month"].to_numpy(dtype=int),
                    iy[ok].astype(int).to_numpy(),
                    ix[ok].astype(int).to_numpy(),
                    lead,
                )
            finite = np.isfinite(base) & np.isfinite(y)
            rmse_c = None
            skill = None
            if int(finite.sum()) > 10:
                rmse_c = float(np.sqrt(np.mean((base[finite] - y[finite]) ** 2)))
                if rmse_c > 0:
                    skill = 1.0 - rmse / rmse_c
            scores[str(lead)] = {
                "rmse": num(rmse, 2),
                "mae": num(float(np.mean(np.abs(err))), 2),
                "bias": num(float(np.mean(err)), 2),
                "skill_vs_climatology": num(skill, 3),
                "n": int(len(y)),
            }
            clim_rmse[lead] = num(rmse_c, 2)
        return scores, clim_rmse

    def _grid(self) -> dict:
        return {
            "type": "regular_latlon",
            "nx": self.nx,
            "ny": self.ny,
            "resolution": round(abs(self.step), 6),
            "lon": {
                "start": round(float(self.lon[0]), 4),
                "step": round(float(self.step), 6),
                "count": self.nx,
                "order": "ascending",
            },
            "lat": {
                "start": round(float(self.lat[0]), 4),
                "step": round(float(self.lat[1] - self.lat[0]) if self.ny > 1 else self.step, 6),
                "count": self.ny,
                "order": "ascending",
            },
            "layout": "row_major_lat_then_lon",
            "nodata": None,
        }

    def _layers(self, rain: np.ndarray, anom: np.ndarray) -> list[dict]:
        valid = rain[np.isfinite(rain)]
        rmax = float(np.nanpercentile(valid, 99)) if valid.size else 200.0
        rmax = max(200.0, math.ceil(rmax / 50.0) * 50.0)
        av = np.abs(anom[np.isfinite(anom)])
        amax = float(np.nanpercentile(av, 99)) if av.size else 50.0
        amax = max(50.0, math.ceil(amax / 25.0) * 25.0)
        return [
            {
                "id": "rain_mm", "label": "Pluie", "unit": "mm/mois",
                "type": "continuous", "value_scale": "linear",
                "range": {"min": 0.0, "max": rmax}, "clamp": True, "precision": 1,
                "palette": {"kind": "sequential", "stops": _lerp_stops(0, rmax, PAL_RAIN)},
                "legend": {"ticks": [round(float(t), 0) for t in np.linspace(0, rmax, 5)]},
                "scope": ["observation", "forecast"], "available": True,
            },
            {
                "id": "anomaly_mm", "label": "Anomalie", "unit": "mm/mois",
                "type": "continuous", "value_scale": "linear",
                "range": {"min": -amax, "max": amax}, "clamp": True, "precision": 1,
                "palette": {"kind": "diverging", "stops": _lerp_stops(-amax, amax, PAL_DIV)},
                "legend": {"ticks": [-amax, -amax / 2, 0.0, amax / 2, amax]},
                "scope": ["observation", "forecast"], "available": True,
            },
            {
                "id": "anomaly_pct", "label": "Écart à la normale", "unit": "%",
                "type": "continuous", "value_scale": "linear",
                "range": {"min": -100.0, "max": 200.0}, "clamp": True, "precision": 0,
                "palette": {"kind": "diverging", "stops": _stops([-100, -50, 0, 50, 100, 150, 200], PAL_DIV)},
                "legend": {"ticks": [-100.0, -50.0, 0.0, 50.0, 100.0, 200.0]},
                "scope": ["observation"], "available": True,
            },
            {
                "id": "anomaly_z", "label": "Anomalie standardisée", "unit": "σ",
                "type": "continuous", "value_scale": "linear",
                "range": {"min": -2.5, "max": 2.5}, "clamp": True, "precision": 2,
                "palette": {"kind": "diverging", "stops": _stops([-2.5, -1.5, -0.75, 0, 0.75, 1.5, 2.5], PAL_DIV)},
                "legend": {"ticks": [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]},
                "scope": ["observation"], "available": True,
            },
            {
                "id": "prob_above_normal", "label": "Probabilité de pluie > normale", "unit": "0-1",
                "type": "continuous", "value_scale": "linear",
                "range": {"min": 0.0, "max": 1.0}, "clamp": True, "precision": 2,
                "palette": {"kind": "sequential", "stops": _lerp_stops(0, 1, PAL_PROB)},
                "legend": {
                    "ticks": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "labels": ["sec", "plutôt sec", "incertain", "plutôt humide", "humide"],
                },
                "scope": ["forecast"], "available": True,
            },
            {
                "id": "spi3", "label": "SPI 3 mois", "unit": "σ",
                "type": "continuous", "value_scale": "linear",
                "range": {"min": -2.5, "max": 2.5}, "clamp": True, "precision": 2,
                "palette": {"kind": "diverging", "stops": _stops([-2.5, -1.5, -0.75, 0, 0.75, 1.5, 2.5], PAL_DIV)},
                "legend": {"ticks": [-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0]},
                "scope": ["observation"], "available": False,
                "reason": "nécessite 30 ans d'historique pour ajuster la loi gamma",
            },
        ]

    def _meta(self) -> dict:
        # échelles calculées sur les 12 derniers mois, comme les frames servies
        recent = self.months[-FRAME_MONTHS:]
        rains = [self._month_grid(m, "rain_mm") for m in recent]
        anoms = [self._month_grid(m, "anom_mm") for m in recent]
        layers = self._layers(np.stack(rains), np.stack(anoms))
        half = self.step / 2.0
        generated = datetime.now(timezone.utc).replace(microsecond=0)
        return {
            "resource": "meta",
            "contract_version": CONTRACT_VERSION,
            "dataset": {
                "id": "pluie-prevue-sahel",
                "title": "Pluie prévue — Afrique de l'Ouest",
                "product": "chirps",
                "generated_at": generated.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "language": "fr",
            },
            "region": {
                "id": "sahel",
                "label": "Afrique de l'Ouest / Sahel",
                "bbox": [
                    round(float(self.lon[0] - half), 3),
                    round(float(self.lat[0] - half), 3),
                    round(float(self.lon[-1] + half), 3),
                    round(float(self.lat[-1] + half), 3),
                ],
                "crs": "EPSG:4326",
            },
            "grid": self._grid(),
            "time": {
                "step": "P1M",
                "start": self.months[0].strftime("%Y-%m-%d"),
                "end": self.months[-1].strftime("%Y-%m-%d"),
                "available": [m.strftime("%Y-%m-%d") for m in self.months],
                "default": self.months[-1].strftime("%Y-%m-%d"),
            },
            "layers": layers,
            "sources": [
                {
                    "id": "chirps",
                    "label": "CHIRPS v3.0 (mensuel)",
                    "url": "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/",
                    "attribution": "© UCSB Climate Hazards Center — CHIRPS v3.0",
                },
                {
                    "id": "cpc",
                    "label": "NOAA CPC — indices ONI, RONI, SOI, Niño",
                    "url": "https://www.cpc.ncep.noaa.gov/data/indices/",
                    "attribution": "© NOAA CPC",
                },
                {
                    "id": "ersst",
                    "label": "NOAA PSL — ERSST v6",
                    "url": "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v6/",
                    "attribution": "© NOAA PSL — ERSST v6",
                },
            ],
            "render": {"mode": "inline_grid", "tile_url": None, "style": "heatmap"},
        }

    def _frames(self, layer: str) -> dict:
        precision = {"rain_mm": 1, "anomaly_mm": 1, "anomaly_pct": 0, "anomaly_z": 2}[layer]
        column = {"rain_mm": "rain_mm", "anomaly_mm": "anom_mm", "anomaly_pct": "anomaly_pct", "anomaly_z": "anom_z"}[layer]
        unit = next(item["unit"] for item in self.meta["layers"] if item["id"] == layer)
        frames = []
        for when in self.months[-FRAME_MONTHS:]:
            grid = self._month_grid(when, column)
            valid = grid[np.isfinite(grid)]
            frames.append({
                "time": when.strftime("%Y-%m-%d"),
                "values": flat(grid, precision),
                "stats": {
                    "min": num(valid.min(), 2) if valid.size else None,
                    "max": num(valid.max(), 2) if valid.size else None,
                    "mean": num(valid.mean(), 2) if valid.size else None,
                    "valid": int(valid.size),
                    "coverage": round(float(valid.size) / (self.nx * self.ny), 4),
                },
            })
        return {
            "resource": "frames",
            "contract_version": CONTRACT_VERSION,
            "layer": layer,
            "unit": unit,
            "grid": self._grid(),
            "frames": frames,
        }

    def _model_block(self) -> dict:
        return {
            "id": "xgb-rain-v1",
            "label": "XGBoost — pluie mensuelle",
            "kind": "trained",
            "status": "ready",
            "version": "1.0.0",
            "training": {
                "start": self.months[0].strftime("%Y-%m-%d"),
                "end": "1983-12-01",
                "parameters": {
                    "n_estimators": 400,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "lead_3_aggregation": "sum_next_3_months",
                },
            },
            "evaluation": {
                "method": "holdout temporel : entraîné avant 1984-01, testé de 1984-01 à 1985-06",
                "per_lead": self.scores,
            },
        }

    def _forecast(self, lead: int) -> dict:
        issued = self.months[-1]
        target = _add_months(issued.date(), lead)
        sub = self.df[self.df["date"] == issued].copy()
        X = sub[FEATURES].to_numpy(dtype="float32")
        X = np.where(np.isfinite(X), X, 0.0)
        pred = np.clip(self.models[lead].predict(X).astype("float64"), 0.0, None)
        rain = self.place(sub, pred)
        iy = sub["lat"].round(4).map(self.lat_pos)
        ix = sub["lon"].round(4).map(self.lon_pos)
        ok = iy.notna() & ix.notna()
        ref = np.full(len(sub), np.nan)
        spread = np.full(len(sub), np.nan)
        if ok.any():
            months = sub.loc[ok, "month"].to_numpy(dtype=int)
            iyi = iy[ok].astype(int).to_numpy()
            ixi = ix[ok].astype(int).to_numpy()
            ref[ok.to_numpy()] = self._clim_sum(months, iyi, ixi, lead)
            var = np.zeros(int(ok.sum()), dtype="float64")
            for k in range(1, lead + 1):
                mi = (months - 1 + k) % 12
                s = self.std[mi, iyi, ixi]
                var += np.where(np.isfinite(s), s * s, np.nan)
            spread[ok.to_numpy()] = np.sqrt(var)
        clim = self.place(sub, ref)
        std = self.place(sub, spread)
        anom = rain - clim
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(std > 0, anom / std, np.nan)
        p10 = np.clip(rain - Z90 * std, 0.0, None)
        p90 = rain + Z90 * std
        p10 = np.where(np.isfinite(std), p10, np.nan)
        p90 = np.where(np.isfinite(std), p90, np.nan)
        prob = _cdf(z)
        land = np.isfinite(rain)
        mean_fc = float(np.nanmean(rain[land])) if land.any() else float("nan")
        mean_cl = float(np.nanmean(clim[land])) if land.any() else float("nan")
        prob_mean = float(np.nanmean(prob[land])) if land.any() else float("nan")
        area_wet = 100.0 * float(np.mean(anom[land] > 0)) if land.any() else float("nan")
        return {
            "resource": "forecast",
            "contract_version": CONTRACT_VERSION,
            "model": self._model_block(),
            "issued_at": issued.strftime("%Y-%m-%d"),
            "target": target.isoformat(),
            "lead_months": lead,
            "grid": self._grid(),
            "layers": {
                "rain_mm": {"values": flat(rain, 1), "p10": flat(p10, 1), "p90": flat(p90, 1)},
                "anomaly_mm": {"values": flat(anom, 1)},
                "prob_above_normal": {"values": flat(prob, 2)},
            },
            "summary": {
                "region_mean_mm": num(mean_fc, 1),
                "region_clim_mm": num(mean_cl, 1),
                "region_anomaly_pct": num(100.0 * (mean_fc - mean_cl) / mean_cl if mean_cl else None, 1),
                "prob_wetter_than_normal": num(prob_mean, 2),
                "area_wetter_pct": num(area_wet, 1),
                "area_drier_pct": num(100.0 - area_wet, 1) if math.isfinite(area_wet) else None,
            },
            "baseline": {
                "id": "climatology",
                "label": "Climatologie des mois visés",
                "rmse": self.clim_rmse.get(lead),
            },
        }

    def point(self, lat: float, lon: float) -> dict:
        iy = int(np.argmin(np.abs(self.lat - lat)))
        ix = int(np.argmin(np.abs(self.lon - lon)))
        cell_lat = float(self.lat[iy])
        cell_lon = float(self.lon[ix])
        sub = self.df[
            (self.df["lat"].round(4) == round(cell_lat, 4))
            & (self.df["lon"].round(4) == round(cell_lon, 4))
        ].sort_values("date")
        is_land = not sub.empty and bool(np.isfinite(sub["rain_mm"]).any())
        label = f"{cell_lat:.2f}, {cell_lon:.2f}"
        for name, cla, clo in CITIES:
            if abs(cell_lat - cla) <= 0.5 and abs(cell_lon - clo) <= 0.5:
                label = name
                break
        series = []
        for row in sub.itertuples(index=False):
            series.append({
                "time": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                "kind": "observation",
                "rain_mm": num(row.rain_mm, 1),
                "clim_mm": num(row.clim_mm, 1),
                "anomaly_mm": num(row.anom_mm, 1),
                "anomaly_z": num(row.anom_z, 2),
                "forecast_mm": None,
                "forecast_p10": None,
                "forecast_p90": None,
            })
        for lead in (1, 3):
            fc = self.forecasts.get(lead)
            if fc is None:
                continue
            index = iy * self.nx + ix
            rain_v = fc["layers"]["rain_mm"]["values"][index]
            p10 = fc["layers"]["rain_mm"]["p10"][index]
            p90 = fc["layers"]["rain_mm"]["p90"][index]
            anom_v = fc["layers"]["anomaly_mm"]["values"][index]
            series.append({
                "time": fc["target"],
                "kind": "forecast",
                "rain_mm": None,
                "clim_mm": num((rain_v - anom_v) if rain_v is not None and anom_v is not None else None, 1),
                "anomaly_mm": None,
                "anomaly_z": None,
                "forecast_mm": rain_v,
                "forecast_p10": p10,
                "forecast_p90": p90,
            })
        return {
            "resource": "point",
            "contract_version": CONTRACT_VERSION,
            "cell": {"lat": round(cell_lat, 2), "lon": round(cell_lon, 2), "is_land": is_land},
            "label": label,
            "unit": "mm/mois",
            "series": series or [{
                "time": self.months[-1].strftime("%Y-%m-%d"),
                "kind": "observation",
                "rain_mm": None, "clim_mm": None, "anomaly_mm": None, "anomaly_z": None,
                "forecast_mm": None, "forecast_p10": None, "forecast_p90": None,
            }],
        }

    def _alerts(self) -> dict:
        when = self.months[-1]
        rain = self._month_grid(when, "rain_mm")
        clim = self.clim[when.month - 1]
        pct = self._month_grid(when, "anomaly_pct")
        z = self._month_grid(when, "anom_z")
        land = np.isfinite(rain)
        ok = np.isfinite(pct) & land
        ap = float(np.nanmean(pct[ok])) if ok.any() else float("nan")
        az = float(np.nanmean(z[np.isfinite(z) & land])) if land.any() else float("nan")
        below = float(np.mean(pct[ok] < 0) * 100) if ok.any() else float("nan")
        severe = float(np.mean(pct[ok] <= -50) * 100) if ok.any() else float("nan")
        rmean = float(np.nanmean(rain[land])) if land.any() else float("nan")
        cmean = float(np.nanmean(clim[land])) if land.any() else float("nan")
        counts = {cid: 0 for cid, _, _ in CLASSES}
        if ok.any():
            pv = pct[ok]
            counts["deficit_severe"] = int(np.sum(pv <= -50))
            counts["deficit"] = int(np.sum((pv > -50) & (pv <= -20)))
            counts["normal"] = int(np.sum((pv > -20) & (pv < 20)))
            counts["excedent"] = int(np.sum((pv >= 20) & (pv < 50)))
            counts["excedent_fort"] = int(np.sum(pv >= 50))
        total = max(1, sum(counts.values()))
        score = np.where(ok, np.abs(z), np.nan)
        order = np.argsort(-np.nan_to_num(score, nan=-1.0).ravel())[:5]
        top = []
        for k in order:
            jy, jx = divmod(int(k), self.nx)
            if not math.isfinite(z[jy, jx]):
                continue
            top.append({
                "lat": round(float(self.lat[jy]), 2),
                "lon": round(float(self.lon[jx]), 2),
                "anomaly_pct": num(pct[jy, jx], 1),
                "anomaly_z": num(z[jy, jx], 2),
                "level": _level(float(z[jy, jx])),
            })
        history = []
        for past in self.months[-12:]:
            pgrid = self._month_grid(past, "anomaly_pct")
            mask = np.isfinite(pgrid)
            mean = float(np.nanmean(pgrid[mask])) if mask.any() else float("nan")
            history.append({
                "time": past.strftime("%Y-%m-%d"),
                "anomaly_pct": num(mean, 1),
                "level": _level(mean),
            })
        lvl = _level(ap)
        return {
            "resource": "alerts",
            "contract_version": CONTRACT_VERSION,
            "time": when.strftime("%Y-%m-%d"),
            "severity": {
                "level": lvl,
                "label": LEVEL_LABEL[lvl],
                "score": num(min(1.0, abs(ap) / 45.0) if math.isfinite(ap) else 0.0, 2),
            },
            "indicators": {
                "rain_mean_mm": num(rmean, 1),
                "clim_mean_mm": num(cmean, 1),
                "anomaly_pct": num(ap, 1),
                "anomaly_z_mean": num(az, 2),
                "area_below_normal_pct": num(below, 1),
                "area_deficit_severe_pct": num(severe, 1),
            },
            "classes": [
                {
                    "id": cid, "label": lab, "count": counts[cid],
                    "area_pct": round(100.0 * counts[cid] / total, 1),
                    "color": col,
                }
                for cid, lab, col in CLASSES
            ],
            "top_cells": top,
            "history": history,
        }
