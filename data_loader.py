import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', message='The default fill_method')

# --- Paths ---
DATA_DIR = "data"
T_BILL_FILE = os.path.join(DATA_DIR, "DTB4WK.csv")

# ================ Config ================
ARCHIVOS = [
    os.path.join(DATA_DIR, "Gemini_ANKRUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_BATUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_BTCUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_COMPUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_CRVUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_DOGEUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_ETHUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_FETUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_GRTUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_LINKUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_LTCUSD_1h.csv"),
    os.path.join(DATA_DIR, "Gemini_UMAUSD_1h.csv")
]

#FECHA_INICIO_SIMULACION = "2022-02-23"
FECHA_INICIO_SIMULACION = "2025-01-01"
FECHA_FIN_SIMULACION = "2025-11-02"


def leer_precios(archivos):
    """Lee y concatena los archivos CSV de precios en un solo DataFrame."""
    dfs = []
    for archivo in archivos:
        df = pd.read_csv(archivo, skiprows=1)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        simbolo = os.path.basename(archivo).split("_")[1].replace("USD", "")
        dfs.append(df[["close"]].rename(columns={"close": simbolo}))
    precios = pd.concat(dfs, axis=1)
    return precios


def cargar_datos_experimento():
    """
    Carga precios, calcula retornos simples horarios y procesa la tasa libre de riesgo.
    Retorna:
        retornos (retornos simples, frac.),
        tasa libre de riesgo horaria (fracción),
        fechas de índice,
        número de activos.
    """
    print("Leyendo precios...")
    precios = leer_precios(ARCHIVOS)

    retornos = precios.pct_change().dropna()

    # Filtrar rango
    retornos = retornos.loc[FECHA_INICIO_SIMULACION:FECHA_FIN_SIMULACION]
    fechas_indice = retornos.index
    n_activos = retornos.shape[1]

    print("Leyendo T-Bill (4-week)...")

    # Leer T-Bill
    rf = pd.read_csv(
        T_BILL_FILE,
        sep=None,
        engine="python",
        na_values=[".", ""]
    )

    if rf.shape[1] == 1 and '\t' in rf.columns[0]:
        rf = rf[rf.columns[0]].str.split('\t', expand=True)
    rf.columns = [c.strip() for c in rf.columns[:2]]
    rf = rf.iloc[:, :2]
    rf.columns = ["DATE", "VALUE"]

    rf["DATE"] = pd.to_datetime(rf["DATE"])
    rf = rf.set_index("DATE").sort_index()

    # Convertir VALUE a float y forward-fill para huecos
    rf["VALUE"] = pd.to_numeric(rf["VALUE"], errors="coerce").ffill()

    # Convertir tasa anual (%) a tasa horaria compuesta (fracción)
    # 8760 = 365 * 24
    rf["rf_hourly"] = (1.0 + rf["VALUE"] / 100.0) ** (1.0 / 8760.0) - 1.0

    # Alinear la tasa libre de riesgo con los retornos horarios (ffill)
    rf = rf.reindex(retornos.index, method="ffill")

    rf_horaria = rf["rf_hourly"].to_numpy()

    return retornos, rf_horaria, fechas_indice, n_activos