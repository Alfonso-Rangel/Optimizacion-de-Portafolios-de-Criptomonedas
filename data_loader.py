import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_DIR = "data"
T_BILL_FILE = os.path.join(DATA_DIR, "_DTB4WK.csv")

# ARCHIVOS será generado por curated_data.py; por defecto esperar archivo con top assets
ARCHIVOS = [
    os.path.join(DATA_DIR, "BTCUSD_1h.csv"),
    os.path.join(DATA_DIR, "ETHUSD_1h.csv"),
    os.path.join(DATA_DIR, "USDCUSD_1h.csv"),
    os.path.join(DATA_DIR, "DOGEUSD_1h.csv"),
    os.path.join(DATA_DIR, "SOLUSD_1h.csv"),
    os.path.join(DATA_DIR, "LTCUSD_1h.csv"),
    os.path.join(DATA_DIR, "LINKUSD_1h.csv"),
    os.path.join(DATA_DIR, "AMPUSD_1h.csv"),
    os.path.join(DATA_DIR, "BCHUSD_1h.csv"),
    os.path.join(DATA_DIR, "FILUSD_1h.csv"),
]

FECHA_INICIO_SIMULACION = "2022-02-23"
FECHA_FIN_SIMULACION = "2025-12-01"


def leer_precios_con_limpieza(archivos):
    dfs = []
    for archivo in archivos:
        try:
            df = pd.read_csv(archivo, skiprows=1)
        except Exception as e:
            print(f"Error leyendo {archivo}: {e}")
            continue

        required = ["date", "open", "high", "low", "close"]
        if not all(col in df.columns for col in required):
            print(f"Skipping {archivo} missing cols")
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date").sort_index()
        # select and coerce
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # pick volume USD
        vol_cols = [c for c in df.columns if "Volume" in c and "USD" in c]
        if vol_cols:
            df["vol_usd"] = pd.to_numeric(df[vol_cols[0]], errors="coerce")
        else:
            df["vol_usd"] = np.nan

        # Replace non-positive prices with NaN
        for c in ["open", "high", "low", "close"]:
            df.loc[df[c] <= 0, c] = np.nan

        # simple ffill small gaps <= 4h
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill(limit=4)

        # drop remaining NaNs
        df = df.dropna(subset=["open", "high", "low", "close"])
        if df.empty:
            continue

        fname = os.path.basename(archivo)
        symbol = fname.split("_")[0].replace("USD", "")
        dfs.append(df[["close"]].rename(columns={"close": symbol}))

    if not dfs:
        raise RuntimeError("No data after cleaning")

    # join on inner (common timestamps)
    precios = pd.concat(dfs, axis=1, join="inner")
    return precios


def cargar_datos_experimento():
    print("Cargando y limpiando precios...")
    precios = leer_precios_con_limpieza(ARCHIVOS)

    # calcular retornos simples
    retornos = precios.pct_change().dropna()
    retornos = retornos.loc[FECHA_INICIO_SIMULACION:FECHA_FIN_SIMULACION]
    if retornos.empty:
        raise RuntimeError("No returns in requested date range")

    fechas_indice = retornos.index
    n_activos = retornos.shape[1]

    # leer T-Bill
    if not os.path.exists(T_BILL_FILE):
        raise FileNotFoundError(f"T-Bill file not found: {T_BILL_FILE}")

    try:
        rf = pd.read_csv(T_BILL_FILE, sep=None, engine="python", header=0)
    except Exception:
        rf = pd.read_csv(T_BILL_FILE, sep="\t", engine="python", header=0)

    # normalize rf columns
    rf_cols = rf.columns.tolist()
    rf = rf.iloc[:, :2]
    rf.columns = ["DATE", "VALUE"]
    rf["DATE"] = pd.to_datetime(rf["DATE"], errors="coerce")
    rf = rf.set_index("DATE").sort_index()
    rf["VALUE"] = pd.to_numeric(rf["VALUE"], errors="coerce").ffill().bfill()

    # convert annual percent to hourly simple rate approximation
    rf["rf_hourly"] = (1.0 + rf["VALUE"] / 100.0) ** (1.0 / 8760.0) - 1.0
    rf_aligned = rf.reindex(fechas_indice, method="ffill")
    rf_aligned["rf_hourly"] = rf_aligned["rf_hourly"].fillna(0.0)

    rf_horaria = rf_aligned["rf_hourly"].to_numpy()

    print(f"Loaded returns: {retornos.shape[0]} periods, {n_activos} assets")
    return retornos, rf_horaria, fechas_indice, n_activos


if __name__ == "__main__":
    r, rf, fechas, n = cargar_datos_experimento()
    print("OK")
