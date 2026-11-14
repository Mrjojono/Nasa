-- Table des features nettoyées, lue par l'API.
-- Même colonnes que le CSV du modèle.

CREATE TABLE IF NOT EXISTS features (
    date date NOT NULL,
    year integer,
    month integer,
    lat double precision NOT NULL,
    lon double precision NOT NULL,
    rain_mm double precision,
    clim_mm double precision,
    anom_mm double precision,
    anom_z double precision,
    pct_clim double precision,
    rain_3m_mm double precision,
    anom_3m_mm double precision,
    rain_lag1_mm double precision,
    rain_lag3_mm double precision,
    rain_lag6_mm double precision,
    rain_lag12_mm double precision,
    oni double precision,
    roni double precision,
    soi double precision,
    nino12_anom double precision,
    nino3_anom double precision,
    nino4_anom double precision,
    nino34_anom double precision,
    atl3_anom double precision,
    oni_lag3 double precision,
    oni_lag6 double precision,
    month_sin double precision,
    month_cos double precision,
    rain_next1_mm double precision,
    rain_next3_mm double precision,
    PRIMARY KEY (date, lat, lon)
);

CREATE TABLE IF NOT EXISTS chargement (
    id bigserial PRIMARY KEY,
    charge_le timestamptz NOT NULL DEFAULT now(),
    lignes integer NOT NULL
);
