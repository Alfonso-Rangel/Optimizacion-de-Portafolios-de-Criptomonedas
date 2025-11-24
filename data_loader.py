# data_loader.py
import pandas as pd
import numpy as np
import os
from config import ARCHIVOS, FECHA_INICIO_SIMULACION, FECHA_FIN_SIMULACION, T_BILL_FILE

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
    Carga precios, calcula retornos y procesa la tasa libre de riesgo.
    Retorna retornos (fracción), tasa libre de riesgo horaria (fracción),
    fechas de índice y número de activos.
    """
    print("Leyendo precios...")
    precios = leer_precios(ARCHIVOS)
    
    retornos = np.log(precios / precios.shift(1)).dropna()
    
    # Filtrar por rango de simulación
    retornos = retornos.loc[FECHA_INICIO_SIMULACION:FECHA_FIN_SIMULACION]
    fechas_indice = retornos.index
    n_activos = retornos.shape[1]

    print("Leyendo T-Bill (4-week)...")
    rf = pd.read_csv(T_BILL_FILE)
    rf["DATE"] = pd.to_datetime(rf["DATE"])
    rf = rf.set_index("DATE").sort_index()

    # Convertir tasa anual (%) a tasa horaria
    rf["rf_hourly"] = (1 + rf["DTB4WK"] / 100.0)**(1 / (365 * 24)) - 1

    # Alinear la tasa libre de riesgo con los retornos (relleno hacia adelante)
    rf = rf.reindex(retornos.index, method="ffill")
    rf_horaria = rf["rf_hourly"].ffill().to_numpy()

    return retornos, rf_horaria, fechas_indice, n_activos
