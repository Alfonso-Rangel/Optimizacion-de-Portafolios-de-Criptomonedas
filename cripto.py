import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from tqdm import trange
import matplotlib.pyplot as plt
import os
import jpype

# =====================================================
# Iniciar JVM y tipos Java
# =====================================================
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=["."])
JDouble = jpype.JDouble
JDoubleArray = jpype.JArray(JDouble)
JDouble2DArray = jpype.JArray(JDoubleArray)
#======================================================

ARCHIVOS = [
    "data/Gemini_BTCUSD_1h.csv",
    "data/Gemini_ETHUSD_1h.csv",
    "data/Gemini_DOGEUSD_1h.csv",
    "data/Gemini_LINKUSD_1h.csv",
    "data/Gemini_LTCUSD_1h.csv",
    "data/Gemini_MATICUSD_1h.csv",
]

def leer_precios(archivos):
    dfs = []
    for archivo in archivos:
        df = pd.read_csv(archivo, skiprows=1)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        simbolo = archivo.split("_")[1].replace("USD", "")
        dfs.append(df[["close"]].rename(columns={"close": simbolo}))
    precios = pd.concat(dfs, axis=1)
    return precios

print("Leyendo archivos...")
precios = leer_precios(ARCHIVOS)

# =====================================================
# Calcular retornos
# =====================================================
retornos = np.log(precios / precios.shift(1)).dropna() * 100

FECHA_INICIO_SIMULACION = "2022-01-01"
FECHA_FIN_SIMULACION = "2025-10-01"
retornos = retornos.loc[FECHA_INICIO_SIMULACION:FECHA_FIN_SIMULACION]

# =====================================================
# Rebalanceo
# =====================================================
# 4 semanas = 4 * 7 * 24 = 672 horas
VENTANA_HORAS = 4 * 7 * 24 # 4 semanas de rebalanceo
PASO_HORAS = 4 * 7 * 24 # La ventana se desliza 4 semanas
n_activos = retornos.shape[1]

# Listas para guardar los resultados de cada ciclo
rendimientos_backtest = pd.Series(dtype=float)
pesos_historicos = {}
fechas_indice = retornos.index

print(f"Iniciando Backtesting con ventana de {VENTANA_HORAS} horas ({VENTANA_HORAS/ (7*24)} semanas)...")

for i in trange(0, len(retornos) - VENTANA_HORAS, PASO_HORAS):
    idx_fin_entrenamiento = i + VENTANA_HORAS
    retornos_entrenamiento = retornos.iloc[i:idx_fin_entrenamiento]
    
    idx_fin_inversion = min(idx_fin_entrenamiento + PASO_HORAS, len(retornos))
    retornos_inversion = retornos.iloc[idx_fin_entrenamiento:idx_fin_inversion]

    if retornos_inversion.empty:
        break

    retornos_np = retornos_entrenamiento.to_numpy()
    retornos_java = JDouble2DArray(
        [JDoubleArray(row.tolist()) for row in retornos_np]
    )
    
    PSO = jpype.JClass("PSO")
    pso = PSO(retornos_java)
    
    pso.minimizarKurtosis()
    
    pesos_optimos = np.array(pso.getMejorPosicion())
    print("\n****************** ")
    print(pesos_optimos)
    print(" ******************\n")
    
    fecha_inicio_inversion = fechas_indice[idx_fin_entrenamiento]
    pesos_historicos[fecha_inicio_inversion] = pesos_optimos
    retornos_periodo = (retornos_inversion * pesos_optimos).sum(axis=1)
    rendimientos_backtest = pd.concat([rendimientos_backtest, retornos_periodo])


port_ret = rendimientos_backtest
kurtosis_port_total = kurtosis(port_ret.values, fisher=False)
media_port_anualizada = port_ret.mean() * 24 * 365 # Rendimiento promedio por hora * horas al año
desv_port = port_ret.std()

print("\n Resultados del Portafolio Óptimo (Backtest):")
print(f"  Número total de horas simuladas: {len(port_ret)}")
print(f"  Curtosis total de la serie de retornos: {kurtosis_port_total:.4f}")
print(f"  Retorno medio anualizado (aprox): {media_port_anualizada:.2f}%")
print(f"  Desviación estándar (volatilidad horaria): {desv_port:.4f}%")

# Graficar distribución
plt.figure(figsize=(8,5))
plt.hist(port_ret, bins=80, color="steelblue", alpha=0.7)
plt.title("Distribución de retornos del portafolio (minimizando curtosis)")
plt.xlabel("Retorno horario (%)")
plt.ylabel("Frecuencia")
plt.grid(alpha=0.3)
plt.show()