import multiprocessing as mp
import os
import sys
import warnings
import numpy as np
import pandas as pd
from tqdm import trange

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import iniciar_jvm, np_a_java_array, np_a_java_2darray, java_list_of_doublearrays_to_numpy
from data_loader import cargar_datos_experimento
from parallel_executor import run_mono_parallel
import utils as U

from evaluation import (
    mostrar_resumen_metrica,
    graficar_retornos_acumulados,
    graficar_frente_pareto_global,
    summary_metrics,
    mostrar_tabla_comparativa,
    graficar_barras_pesos
)

# Config
FRECUENCIAS = {'media_semana': 84, 'semana':168, 'dos_semanas':336, 'cuatro_semanas':672, 'ocho_semanas':1344}
FRECUENCIA_SELECCIONADA = 'ocho_semanas'
VENTANA_HORAS = 4 * 7 * 24
PASO_HORAS = FRECUENCIAS[FRECUENCIA_SELECCIONADA]

SHARPE_RUNS = 6
KURT_RUNS = 6
MOPSO_RUNS = 1  # MOPSO sequential single run per window

SMOOTH_ALPHA = 0.7

def seleccionar_punto_utopia(frente_obj):
    if len(frente_obj) == 0:
        return -1
    obj = np.array(frente_obj)
    mins = np.min(obj, axis=0)
    maxs = np.max(obj, axis=0)
    range_vals = maxs - mins
    range_vals[range_vals == 0] = 1.0
    norm = (obj - mins) / range_vals
    dist = np.sum(norm**2, axis=1)
    return np.argmin(dist)


def normalizar_pesos(w):
    w = np.asarray(w, dtype=float)
    w[w < 0] = 0.0
    s = w.sum()
    if s == 0:
        return np.full_like(w, 1.0 / len(w))
    return w / s


def ejecutar_mono_seleccion_mejor(ret_ent, rf_ent, objetivo, n_runs):
    """
    Ejecuta n_runs en paralelo usando run_mono_parallel, luego selecciona la mejor corrida según fitness.
    Para Sharpe: fitness = Sharpe (mayor mejor). Se aplica regla: si Sharpe <= 0 => naive.
    Para Kurtosis: fitness = curtosis (menor mejor).
    """
    results = run_mono_parallel(ret_ent, rf_ent, objetivo, n_runs)
    # filter successful
    valids = [r for r in results if r.get("w") is not None]
    if not valids:
        n_act = ret_ent.shape[1]
        return np.full(n_act, 1.0 / n_act), results

    if objetivo == "sharpe":
        best = max(valids, key=lambda x: (x.get("fitness") if x.get("fitness") is not None else -1e9))
        w = np.array(best["w"], dtype=float)
        w = normalizar_pesos(w)
        # compute final sharpe to apply rule
        port = (ret_ent * w).sum(axis=1)
        exces = port - pd.Series(rf_ent, index=ret_ent.index)
        mean = exces.mean()
        std = exces.std(ddof=0)
        sharpe_final = mean / std if std > 0 else -np.inf
        if sharpe_final <= 0:
            n_act = ret_ent.shape[1]
            return np.full(n_act, 1.0 / n_act), results
        return w, results
    else:  # kurtosis
        best = min(valids, key=lambda x: (x.get("fitness") if x.get("fitness") is not None else 1e9))
        w = np.array(best["w"], dtype=float)
        w = normalizar_pesos(w)
        return w, results


def ejecutar_mopso_single_run(ret_ent, rf_ent):
    """
    Ejecuta 1 corrida MOPSO secuencial y selecciona punto de utopía.
    """
    iniciar_jvm()
    PSO = U.PSO
    retornos_java = np_a_java_2darray(ret_ent.to_numpy())
    rf_java = np_a_java_array(rf_ent)
    try:
        pso = PSO(retornos_java)
        pso.optimizar(rf_java)
        front_pos = java_list_of_doublearrays_to_numpy(pso.getFrentePos())
        front_obj = java_list_of_doublearrays_to_numpy(pso.getFrenteObj())
        if len(front_obj) == 0:
            n_act = ret_ent.shape[1]
            return np.full(n_act, 1.0 / n_act), []
        idx = seleccionar_punto_utopia(front_obj)
        w = np.array(front_pos[idx], dtype=float)
        w = normalizar_pesos(w)
        return w, [np.array(front_obj)]
    except Exception:
        n_act = ret_ent.shape[1]
        return np.full(n_act, 1.0 / n_act), []

def procesar_ventana(ret_ent, rf_ent, prev_weights=None):
    """
    Procesa una ventana: filtra activos con muchos gaps/zeros estáticos,
    ejecuta Sharpe (mejor corrida), Kurtosis (mejor corrida) en paralelo,
    ejecuta MOPSO secuencialmente (1 corrida utopía).
    Devuelve pesos dict y frentes.
    """
    # Filtrado simple por activos con demasiados NaN en la ventana
    valid_cols = [c for c in ret_ent.columns if ret_ent[c].isna().sum() == 0]
    if len(valid_cols) < max(2, int(0.5 * len(ret_ent.columns))):
        # fallback: keep original
        valid_cols = list(ret_ent.columns)

    ret_filtered = ret_ent[valid_cols]

    # Sharpe best
    w_sharpe, sharpe_runs = ejecutar_mono_seleccion_mejor(ret_filtered, rf_ent, "sharpe", SHARPE_RUNS)

    # Kurtosis best
    w_kurt, kurt_runs = ejecutar_mono_seleccion_mejor(ret_filtered, rf_ent, "kurtosis", KURT_RUNS)

    # MOPSO single run (on filtered set), returns full length weights for original asset set
    w_comp_sub, frentes = ejecutar_mopso_single_run(ret_filtered, rf_ent)

    # expand weights back to original asset order (missing assets -> 0)
    def expand(w_sub, sub_cols, all_cols):
        w_full = np.zeros(len(all_cols), dtype=float)
        for i, c in enumerate(all_cols):
            if c in sub_cols:
                idx = sub_cols.index(c)
                w_full[i] = w_sub[idx]
        # renormalize
        return normalizar_pesos(w_full)

    all_cols = list(ret_ent.columns)
    w_sharpe_full = expand(w_sharpe, list(ret_filtered.columns), all_cols)
    w_kurt_full = expand(w_kurt, list(ret_filtered.columns), all_cols)
    w_comp_full = expand(w_comp_sub, list(ret_filtered.columns), all_cols)
    w_naive = np.full(len(all_cols), 1.0 / len(all_cols))

    resultados = {
        "naive": w_naive,
        "sharpe": w_sharpe_full,
        "kurt": w_kurt_full,
        "comp": w_comp_full
    }

    # smoothing
    if prev_weights is not None:
        for k in resultados.keys():
            if k in prev_weights and prev_weights[k] is not None:
                resultados[k] = normalizar_pesos(SMOOTH_ALPHA * resultados[k] + (1.0 - SMOOTH_ALPHA) * prev_weights[k])

    return resultados, frentes


def main():
    iniciar_jvm()
    retornos, rf_horaria, fechas_indice, n_activos = cargar_datos_experimento()

    retornos_por_estrategia = {k: [] for k in ["naive", "sharpe", "kurt", "comp"]}
    pesos_por_estrategia = {k: {} for k in ["naive", "sharpe", "kurt", "comp"]}
    frentes_pareto = []

    prev_weights = {k: None for k in ["naive", "sharpe", "kurt", "comp"]}

    iteraciones = range(0, len(retornos) - VENTANA_HORAS, PASO_HORAS)
    pbar = trange(len(iteraciones), desc=f"Optimizando ({FRECUENCIA_SELECCIONADA})")

    for i, _ in enumerate(pbar):
        ini = i * PASO_HORAS
        fin_ent = ini + VENTANA_HORAS
        fin_inv = min(fin_ent + PASO_HORAS, len(retornos))

        ret_ent = retornos.iloc[ini:fin_ent]
        rf_ent = rf_horaria[ini:fin_ent].astype(float)

        ret_inv = retornos.iloc[fin_ent:fin_inv]
        rf_inv = rf_horaria[fin_ent:fin_inv].astype(float)

        if ret_inv.empty:
            break

        fecha_inv = fechas_indice[fin_ent]

        pesos_avg, frentes_runs = procesar_ventana(ret_ent, rf_ent, prev_weights)

        for strat, w in pesos_avg.items():
            pesos_por_estrategia[strat][fecha_inv] = w

        for f in frentes_runs:
            frentes_pareto.append(f)

        rf_inv_series = pd.Series(rf_inv, index=ret_inv.index)

        for strat, w in pesos_avg.items():
            port = (ret_inv * w).sum(axis=1)
            port_exceso = port - rf_inv_series
            retornos_por_estrategia[strat].append(port_exceso)

        prev_weights = pesos_avg
        pbar.set_postfix({'Ventana': f"{ini}-{fin_ent}", 'Fecha': fecha_inv.strftime('%Y-%m-%d')})

    # consolidate
    for strat in retornos_por_estrategia:
        lst = retornos_por_estrategia[strat]
        retornos_por_estrategia[strat] = pd.concat(lst) if lst else pd.Series(dtype=float)

    metricas = [
        summary_metrics(retornos_por_estrategia["naive"], "Naive 1/N"),
        summary_metrics(retornos_por_estrategia["sharpe"], "Sharpe-PSO"),
        summary_metrics(retornos_por_estrategia["kurt"], "Kurtosis-PSO"),
        summary_metrics(retornos_por_estrategia["comp"], "MOPSO")
    ]

    mostrar_resumen_metrica(metricas)
    mostrar_tabla_comparativa(metricas)
    #graficar_frente_pareto_global(frentes_pareto)
    graficar_retornos_acumulados(retornos_por_estrategia, titulo=f"Desempeño - {FRECUENCIA_SELECCIONADA}")

    nombres_activos = retornos.columns.tolist()

    frecuencia_texto = {
        'media_semana': '0.5 semanas',
        'semana': '1 semana',
        'dos_semanas': '2 semanas',
        'cuatro_semanas': '4 semanas',
        'ocho_semanas': '8 semanas'
    }.get(FRECUENCIA_SELECCIONADA, FRECUENCIA_SELECCIONADA)
    
    titulo_general = f"Evolución de Portafolios de Criptomonedas (Rebalanceo: {frecuencia_texto})"
    
    graficar_barras_pesos(
        pesos_por_estrategia=pesos_por_estrategia,
        nombres_activos=nombres_activos,
        titulo_general=titulo_general,
        max_barras=40 
    )
    
    # También mostrar información sobre el período total
    print(f"\nPERÍODO TOTAL DE ANÁLISIS:")
    print(f"  Fecha inicio: {fechas_indice[0].strftime('%Y-%m-%d')}")
    print(f"  Fecha fin: {fechas_indice[-1].strftime('%Y-%m-%d')}")
    print(f"  Total períodos (horas): {len(fechas_indice)}")
    print(f"  Activos analizados: {nombres_activos}")


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        mp.freeze_support()
    main()
