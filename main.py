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
SMOOTH_ALPHA = 0.7

ESTRATEGIAS = ["naive", "sharpe", "curt", "comp"]


def seleccionar_del_frente(frente_obj, criterio='utopia'):
    """Selecciona punto del frente según criterio unificado."""
    if len(frente_obj) == 0:
        return -1
    
    obj = np.array(frente_obj)
    
    if criterio == 'sharpe':
        return np.argmax(-obj[:, 1])  # Maximizar Sharpe (segunda columna es -sharpe)
    elif criterio == 'kurtosis':
        return np.argmin(obj[:, 0])   # Minimizar Kurtosis
    else:  # utopia
        mins, maxs = obj.min(axis=0), obj.max(axis=0)
        range_vals = np.where(maxs - mins == 0, 1.0, maxs - mins)
        norm = (obj - mins) / range_vals
        return np.argmin(np.sum(norm**2, axis=1))


def normalizar_pesos(w):
    w = np.asarray(w, dtype=float)
    w[w < 0] = 0.0
    s = w.sum()
    if s == 0:
        return np.full_like(w, 1.0 / len(w))
    return w / s


def ejecutar_mopso_y_seleccionar_puntos(ret_ent, rf_ent):
    """
    Ejecuta 1 corrida MOPSO y selecciona 3 puntos del frente:
    1. Punto de utopía (compromiso óptimo)
    2. Mejor Sharpe (máximo ratio de Sharpe)
    3. Mejor Kurtosis (mínima curtosis)
    
    Devuelve: (dict_pesos, frentes)
    """
    iniciar_jvm()
    retornos_java = np_a_java_2darray(ret_ent.to_numpy())
    rf_java = np_a_java_array(rf_ent)
    
    try:
        pso = U.PSO(retornos_java)
        pso.optimizar(rf_java)
        front_pos = java_list_of_doublearrays_to_numpy(pso.getFrentePos())
        front_obj = java_list_of_doublearrays_to_numpy(pso.getFrenteObj())
        
        if len(front_obj) == 0:
            naive_w = np.full(ret_ent.shape[1], 1.0 / ret_ent.shape[1])
            return {k: naive_w for k in ['utopia', 'sharpe', 'kurtosis']}, []
        
        # Seleccionar los 3 puntos con un loop
        criterios = ['utopia', 'sharpe', 'kurtosis']
        pesos = {}
        for criterio in criterios:
            idx = seleccionar_del_frente(front_obj, criterio)
            pesos[criterio] = normalizar_pesos(np.array(front_pos[idx], dtype=float))
        
        return pesos, [np.array(front_obj)]
        
    except Exception as e:
        print(f"Error en MOPSO: {e}")
        naive_w = np.full(ret_ent.shape[1], 1.0 / ret_ent.shape[1])
        return {k: naive_w for k in ['utopia', 'sharpe', 'kurtosis']}, []


def expandir_pesos(w_sub, sub_cols, all_cols):
    """Expande pesos de subset a todos los activos."""
    w_full = np.zeros(len(all_cols))
    col_to_idx = {c: i for i, c in enumerate(sub_cols)}
    for i, c in enumerate(all_cols):
        if c in col_to_idx:
            w_full[i] = w_sub[col_to_idx[c]]
    return normalizar_pesos(w_full)


def procesar_ventana(ret_ent, rf_ent, n_activos, prev_weights=None):
    """
    Procesa una ventana: ejecuta MOPSO una vez y obtiene los 3 puntos clave.
    Devuelve pesos dict y frentes.
    """
    # Filtrado simplificado
    valid_cols = [c for c in ret_ent.columns if ret_ent[c].notna().all()]
    if len(valid_cols) < max(2, len(ret_ent.columns) // 2):
        valid_cols = list(ret_ent.columns)

    ret_filtered = ret_ent[valid_cols]
    pesos_dict, frentes = ejecutar_mopso_y_seleccionar_puntos(ret_filtered, rf_ent)

    # Expandir todos los pesos de una vez
    all_cols = list(ret_ent.columns)
    resultados = {
        "naive": np.full(len(all_cols), 1.0 / len(all_cols)),
        "sharpe": expandir_pesos(pesos_dict['sharpe'], valid_cols, all_cols),
        "curt": expandir_pesos(pesos_dict['kurtosis'], valid_cols, all_cols),
        "comp": expandir_pesos(pesos_dict['utopia'], valid_cols, all_cols)
    }

    # Suavizado simplificado
    if prev_weights:
        for k in resultados:
            if prev_weights.get(k) is not None:
                resultados[k] = normalizar_pesos(
                    SMOOTH_ALPHA * resultados[k] + (1 - SMOOTH_ALPHA) * prev_weights[k]
                )

    return resultados, frentes


def main():
    iniciar_jvm()
    retornos, rf_horaria, fechas_indice, n_activos = cargar_datos_experimento()

    retornos_por_estrategia = {k: [] for k in ESTRATEGIAS}
    pesos_por_estrategia = {k: {} for k in ESTRATEGIAS}
    frentes_pareto = []
    prev_weights = None

    num_ventanas = (len(retornos) - VENTANA_HORAS) // PASO_HORAS
    pbar = trange(num_ventanas, desc=f"Optimizando ({FRECUENCIA_SELECCIONADA})")

    for i in pbar:
        ini = i * PASO_HORAS
        fin_ent = ini + VENTANA_HORAS
        fin_inv = min(fin_ent + PASO_HORAS, len(retornos))

        if fin_inv - fin_ent == 0:
            break

        ret_ent = retornos.iloc[ini:fin_ent]
        rf_ent = rf_horaria[ini:fin_ent].astype(float)
        ret_inv = retornos.iloc[fin_ent:fin_inv]
        rf_inv = rf_horaria[fin_ent:fin_inv].astype(float)

        fecha_inv = fechas_indice[fin_ent]
        pesos_ventana, frentes = procesar_ventana(ret_ent, rf_ent, n_activos, prev_weights)

        # Guardar pesos y calcular retornos
        rf_inv_series = pd.Series(rf_inv, index=ret_inv.index)
        for strat in ESTRATEGIAS:
            w = pesos_ventana[strat]
            pesos_por_estrategia[strat][fecha_inv] = w
            port_exceso = (ret_inv * w).sum(axis=1) - rf_inv_series
            retornos_por_estrategia[strat].append(port_exceso)

        frentes_pareto.extend(frentes)
        prev_weights = pesos_ventana
        pbar.set_postfix({'Ventana': f"{ini}-{fin_ent}", 'Fecha': fecha_inv.strftime('%Y-%m-%d')})

    # Consolidar resultados
    for strat in ESTRATEGIAS:
        lst = retornos_por_estrategia[strat]
        retornos_por_estrategia[strat] = pd.concat(lst) if lst else pd.Series(dtype=float)

    metricas = [
        summary_metrics(retornos_por_estrategia["naive"], "Naive 1/N"),
        summary_metrics(retornos_por_estrategia["sharpe"], "Max Sharpe (del frente)"),
        summary_metrics(retornos_por_estrategia["curt"], "Min Curtosis (del frente)"),
        summary_metrics(retornos_por_estrategia["comp"], "Punto de Utopía")
    ]

    mostrar_resumen_metrica(metricas)
    mostrar_tabla_comparativa(metricas)
    
    graficar_frente_pareto_global(frentes_pareto)
    graficar_retornos_acumulados(retornos_por_estrategia, titulo=f"Desempeño - {FRECUENCIA_SELECCIONADA}")

    nombres_activos = retornos.columns.tolist()

    frecuencia_texto = {
        'media_semana': '0.5 semanas',
        'semana': '1 semana',
        'dos_semanas': '2 semanas',
        'cuatro_semanas': '4 semanas',
        'ocho_semanas': '8 semanas'
    }.get(FRECUENCIA_SELECCIONADA, FRECUENCIA_SELECCIONADA)
    
    titulo_general = f"Evolución de la distribución del portafolio (Rebalanceo: {frecuencia_texto})"
    
    graficar_barras_pesos(
        pesos_por_estrategia=pesos_por_estrategia,
        nombres_activos=nombres_activos,
        titulo_general=titulo_general
    )


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        mp.freeze_support()
    main()