import pandas as pd
import numpy as np
from tqdm import trange
import sys
import os
import warnings
from scipy.stats import kurtosis
import multiprocessing as mp

# Suprimir warnings no deseados
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils
from data_loader import cargar_datos_experimento
from utils import (
    iniciar_jvm,
    np_a_java_2darray,
    np_a_java_array,
    java_list_of_doublearrays_to_numpy
)

from evaluation import (
    mostrar_resumen_metrica,
    graficar_retornos_acumulados,
    graficar_frente_pareto_global,
    summary_metrics,
    mostrar_tabla_comparativa
)

# Importar la versión paralela con pool global
from parallel_executor import (
    procesar_ventana_parallel_mp,
    init_global_pool,
    close_global_pool
)

# ---------------- Configuración de frecuencias ----------------
FRECUENCIAS = {
    'media_semana': 84,
    'semana': 168,
    'dos_semanas': 336,
    'cuatro_semanas': 672,
    'ocho_semanas': 1344
}

FRECUENCIA_SELECCIONADA = 'ocho_semanas'
VENTANA_HORAS = 4 * 7 * 24
PASO_HORAS = FRECUENCIAS[FRECUENCIA_SELECCIONADA]

# ---------------- Parámetros de experimento ----------------
SHARPE_RUNS = 10
KURT_RUNS = 10
MOPSO_RUNS = 1

# ---------------- Helpers ----------------
def seleccionar_punto_utopia(frente_obj):
    if len(frente_obj) == 0:
        return -1
    obj = np.array(frente_obj)
    mins = np.min(obj, axis=0)
    maxs = np.max(obj, axis=0)
    rango = maxs - mins
    rango[rango == 0] = 1.0
    norm = (obj - mins) / rango
    dist = np.sum(norm ** 2, axis=1)
    return np.argmin(dist)


def normalizar_pesos(w):
    w = np.asarray(w, dtype=float)
    w[w < 0] = 0.0
    s = w.sum()
    if s == 0:
        return np.full_like(w, 1.0 / len(w))
    return w / s


# ---------------- MONO OBJETIVO ----------------
def ejecutar_mono_multi_runs(retornos_ent, rf_ent, objetivo, n_runs):
    """
    Ejecuta n_runs corridas del PSO mono-objetivo.
    Selecciona la mejor corrida basada en el fitness calculado en Python.
    """
    retornos_java = np_a_java_2darray(retornos_ent.to_numpy())

    PSO = utils.PSO
    if PSO is None:
        raise RuntimeError("PSO no disponible")

    mejor_w = None
    mejor_fitness = None
    rf_series = pd.Series(rf_ent, index=retornos_ent.index)

    for r in range(n_runs):
        try:
            # Usar constructor por defecto (sin semilla)
            pso = PSO(retornos_java)

            # Ejecutar optimización
            if objetivo == "sharpe":
                pso.maximizarSharpe(np_a_java_array(rf_ent))
            elif objetivo == "kurtosis":
                pso.minimizarKurtosis()
            else:
                raise ValueError(f"Objetivo desconocido: {objetivo}")

            # Obtener pesos
            w = np.array(pso.getMejorPosicion(), float)
            w = normalizar_pesos(w)

            # Calcular fitness en Python
            port = (retornos_ent * w).sum(axis=1)
            exceso = port - rf_series

            if objetivo == "sharpe":
                mean = exceso.mean()
                std = exceso.std(ddof=0)
                fitness = mean / std if std > 0 else -np.inf
            else:  # curtosis
                fitness = kurtosis(exceso.values, fisher=True, bias=False)

        except Exception as e:
            print(f"[Warning] Error en corrida {r+1} ({objetivo}): {e}")
            continue

        # Seleccionar mejor según criterio
        if w is None:
            continue

        if mejor_w is None:
            mejor_w = w
            mejor_fitness = fitness
        else:
            if objetivo == "sharpe":
                if fitness > mejor_fitness:
                    mejor_fitness = fitness
                    mejor_w = w
            else:  # curtosis (menor es mejor)
                if fitness < mejor_fitness:
                    mejor_fitness = fitness
                    mejor_w = w

    # Fallback a pesos naive si todas las corridas fallaron
    if mejor_w is None:
        n_act = retornos_ent.shape[1]
        return np.full(n_act, 1.0 / n_act)

    return mejor_w


# ---------------- MULTIOBJETIVO ----------------
def ejecutar_mopso_single_run(retornos_ent, rf_ent):
    """
    Ejecuta una sola corrida de MOPSO.
    Selecciona punto de utopía del frente Pareto.
    """
    retornos_java = np_a_java_2darray(retornos_ent.to_numpy())
    rf_java = np_a_java_array(rf_ent)

    PSO = utils.PSO
    if PSO is None:
        raise RuntimeError("PSO no disponible")
    
    try:
        # Usar constructor por defecto (sin semilla)
        pso = PSO(retornos_java)

        pso.optimizar(rf_java)

        # Obtener frente Pareto
        frente_pos = java_list_of_doublearrays_to_numpy(pso.getFrentePos())
        frente_obj = java_list_of_doublearrays_to_numpy(pso.getFrenteObj())

        if len(frente_obj) == 0:
            n_act = retornos_ent.shape[1]
            return np.full(n_act, 1.0 / n_act), []

        # Seleccionar punto de utopía
        idx = seleccionar_punto_utopia(frente_obj)
        w = np.array(frente_pos[idx], float)
        w = normalizar_pesos(w)
        
        return w, [np.array(frente_obj)]

    except Exception as e:
        print(f"[Warning] Error en MOPSO: {e}")
        n_act = retornos_ent.shape[1]
        return np.full(n_act, 1.0 / n_act), []


# ---------------- PROCESAMIENTO POR VENTANA ----------------
def procesar_ventana_multi(retornos_ent, rf_ent, n_activos):
    """
    Procesa una ventana usando el pool global.
    """
    resultados, frentes_runs = procesar_ventana_parallel_mp(
        retornos_ent,
        rf_ent,
        n_activos
    )
    return resultados, frentes_runs


# ---------------- MAIN ----------------
def main():
    print("=" * 60)
    print(f"EXPERIMENTO CONFIGURACIÓN")
    print(f"Frecuencia rebalanceo: {FRECUENCIA_SELECCIONADA}")
    print(f"Ventana: {VENTANA_HORAS} horas, Paso: {PASO_HORAS} horas")
    print(f"Corridas: Sharpe={SHARPE_RUNS}, Kurtosis={KURT_RUNS}, MOPSO={MOPSO_RUNS}")
    print("=" * 60)

    # Iniciar JVM en proceso principal
    iniciar_jvm()
    
    # Inicializar pool global
    print("\nInicializando pool global de procesos...")
    init_global_pool(
        sharpe_runs=SHARPE_RUNS,
        kurt_runs=KURT_RUNS,
        mopso_runs=MOPSO_RUNS
    )

    # Cargar datos
    retornos, rf_horaria, fechas_indice, n_activos = cargar_datos_experimento()
    print(f"\nDatos cargados: {retornos.shape[0]} horas, {retornos.shape[1]} activos")

    # Inicializar estructuras de resultados
    retornos_por_estrategia = {k: [] for k in ["naive", "sharpe", "kurt", "comp"]}
    pesos_por_estrategia = {k: {} for k in ["naive", "sharpe", "kurt", "comp"]}
    frentes_pareto = []

    # Procesar ventanas
    iteraciones = range(0, len(retornos) - VENTANA_HORAS, PASO_HORAS)
    pbar = trange(len(iteraciones), desc="Optimizando")

    for i, _ in enumerate(pbar):
        ini = i * PASO_HORAS
        fin_ent = ini + VENTANA_HORAS
        fin_inv = min(fin_ent + PASO_HORAS, len(retornos))

        ret_ent = retornos.iloc[ini:fin_ent]
        rf_ent = rf_horaria[ini:fin_ent]
        ret_inv = retornos.iloc[fin_ent:fin_inv]
        rf_inv = rf_horaria[fin_ent:fin_inv]

        if ret_inv.empty:
            break

        fecha_inv = fechas_indice[fin_ent]

        # Procesar ventana
        pesos_finales, frentes_runs = procesar_ventana_multi(
            ret_ent, rf_ent, n_activos
        )

        # Guardar pesos
        for strat, w in pesos_finales.items():
            pesos_por_estrategia[strat][fecha_inv] = w

        # Acumular frentes Pareto
        for f in frentes_runs:
            frentes_pareto.append(f)

        # Calcular retornos de inversión
        rf_inv_series = pd.Series(rf_inv, index=ret_inv.index)
        for strat, w in pesos_finales.items():
            port = (ret_inv * w).sum(axis=1)
            port_exceso = port - rf_inv_series
            retornos_por_estrategia[strat].append(port_exceso)

        pbar.set_postfix({'Fecha': fecha_inv.strftime('%Y-%m-%d')})

    # Cerrar pool global
    print("\nCerrando pool global...")
    close_global_pool()

    # Consolidar resultados
    for strat in retornos_por_estrategia:
        lst = retornos_por_estrategia[strat]
        retornos_por_estrategia[strat] = pd.concat(lst) if lst else pd.Series(dtype=float)

    # Calcular métricas
    metricas = [
        summary_metrics(retornos_por_estrategia["naive"], "Naive 1/N"),
        summary_metrics(retornos_por_estrategia["sharpe"], "Sharpe-PSO"),
        summary_metrics(retornos_por_estrategia["kurt"], "Kurtosis-PSO"),
        summary_metrics(retornos_por_estrategia["comp"], "MOPSO")
    ]

    mostrar_resumen_metrica(metricas)
    mostrar_tabla_comparativa(metricas)
    graficar_frente_pareto_global(frentes_pareto)
    graficar_retornos_acumulados(
        retornos_por_estrategia,
        titulo=f"Desempeño - {FRECUENCIA_SELECCIONADA}"
    )


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        mp.freeze_support()
    main()