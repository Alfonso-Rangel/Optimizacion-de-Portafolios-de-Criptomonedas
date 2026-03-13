import multiprocessing as mp
import os
import sys
import warnings
import numpy as np
import pandas as pd
from tqdm import trange
from pymoo.indicators.hv import HV

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
    graficar_hipervolumen,
    graficar_drawdown,
    graficar_boxplot_retornos,
    graficar_pesos_promedio
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
N_CORRIDAS = 3  # Número de ejecuciones solicitado

ESTRATEGIAS = ["naive", "sharpe", "curt", "comp"]

# ---------------- Funciones Auxiliares ----------------

def seleccionar_del_frente(frente_obj, criterio='utopia'):
    if len(frente_obj) == 0: return -1
    obj = np.array(frente_obj)
    if criterio == 'sharpe':
        return np.argmax(-obj[:, 1])
    elif criterio == 'kurtosis':
        return np.argmin(obj[:, 0])
    else:  # utopia
        mins, maxs = obj.min(axis=0), obj.max(axis=0)
        range_vals = np.where(maxs - mins == 0, 1.0, maxs - mins)
        norm = (obj - mins) / range_vals
        return np.argmin(np.sum(norm**2, axis=1))

def normalizar_pesos(w):
    w = np.asarray(w, dtype=float)
    w[w < 0] = 0.0
    s = w.sum()
    if s == 0: return np.full_like(w, 1.0 / len(w))
    return w / s

def ejecutar_NSGAII(ret_ent, rf_ent):
    iniciar_jvm()
    retornos_java = np_a_java_2darray(ret_ent.to_numpy())
    rf_java = np_a_java_array(rf_ent)
    try:
        nsga = U.NSGAII(retornos_java)
        nsga.optimizar(rf_java)
        front_pos = java_list_of_doublearrays_to_numpy(nsga.getFrentePos())
        front_obj = java_list_of_doublearrays_to_numpy(nsga.getFrenteObj())
        if len(front_obj) == 0:
            return {k: np.full(ret_ent.shape[1], 1.0/ret_ent.shape[1]) for k in ['utopia', 'sharpe', 'kurtosis']}, []
        
        criterios = ['utopia', 'sharpe', 'kurtosis']
        pesos = {c: normalizar_pesos(np.array(front_pos[seleccionar_del_frente(front_obj, c)], dtype=float)) for c in criterios}
        return pesos, [np.array(front_obj)]
    except Exception as e:
        print(f"Error en MOPSO: {e}")
        return {k: np.full(ret_ent.shape[1], 1.0/ret_ent.shape[1]) for k in ['utopia', 'sharpe', 'kurtosis']}, []

def ejecutar_MOPSO(ret_ent, rf_ent):
    iniciar_jvm()
    retornos_java = np_a_java_2darray(ret_ent.to_numpy())
    rf_java = np_a_java_array(rf_ent)
    try:
        pso = U.PSO(retornos_java)
        pso.optimizar(rf_java)
        front_pos = java_list_of_doublearrays_to_numpy(pso.getFrentePos())
        front_obj = java_list_of_doublearrays_to_numpy(pso.getFrenteObj())
        if len(front_obj) == 0:
            return {k: np.full(ret_ent.shape[1], 1.0/ret_ent.shape[1]) for k in ['utopia', 'sharpe', 'kurtosis']}, []
        
        criterios = ['utopia', 'sharpe', 'kurtosis']
        pesos = {c: normalizar_pesos(np.array(front_pos[seleccionar_del_frente(front_obj, c)], dtype=float)) for c in criterios}
        return pesos, [np.array(front_obj)]
    except Exception as e:
        print(f"Error en MOPSO: {e}")
        return {k: np.full(ret_ent.shape[1], 1.0/ret_ent.shape[1]) for k in ['utopia', 'sharpe', 'kurtosis']}, []

def expandir_pesos(w_sub, sub_cols, all_cols):
    w_full = np.zeros(len(all_cols))
    col_to_idx = {c: i for i, c in enumerate(sub_cols)}
    for i, c in enumerate(all_cols):
        if c in col_to_idx: w_full[i] = w_sub[col_to_idx[c]]
    return normalizar_pesos(w_full)

def procesar_ventana(ret_ent, rf_ent, n_activos, prev_weights=None):
    valid_cols = [c for c in ret_ent.columns if ret_ent[c].notna().all()]
    if len(valid_cols) < max(2, len(ret_ent.columns) // 2): valid_cols = list(ret_ent.columns)
    
    pesos_dict, frentes = ejecutar_MOPSO(ret_ent[valid_cols], rf_ent)
    #pesos_dict, frentes = ejecutar_NSGAII(ret_ent[valid_cols], rf_ent)
    all_cols = list(ret_ent.columns)
    resultados = {
        "naive": np.full(len(all_cols), 1.0 / len(all_cols)),
        "sharpe": expandir_pesos(pesos_dict['sharpe'], valid_cols, all_cols),
        "curt": expandir_pesos(pesos_dict['kurtosis'], valid_cols, all_cols),
        "comp": expandir_pesos(pesos_dict['utopia'], valid_cols, all_cols)
    }
    if prev_weights:
        for k in resultados:
            if prev_weights.get(k) is not None:
                resultados[k] = normalizar_pesos(SMOOTH_ALPHA * resultados[k] + (1 - SMOOTH_ALPHA) * prev_weights[k])
    return resultados, frentes

# ---------------- Main Logic ----------------

def main():
    iniciar_jvm()
    retornos, rf_horaria, fechas_indice, n_activos = cargar_datos_experimento()
    
    # Almacén de frentes para cálculo de HV global: [corrida][ventana]
    matriz_frentes = [[] for _ in range(N_CORRIDAS)]
    fechas_ventanas = []
    
    # Para métricas finales de la última corrida (o podrías promediarlas)
    retornos_por_estrategia = {k: [] for k in ESTRATEGIAS}
    pesos_por_estrategia = {k: {} for k in ESTRATEGIAS}

    for n in range(N_CORRIDAS):
        print(f"\n>>> Iniciando Corrida {n+1}/{N_CORRIDAS}")
        prev_weights = None
        num_ventanas = (len(retornos) - VENTANA_HORAS) // PASO_HORAS
        pbar = trange(num_ventanas, desc=f"Corrida {n+1}")

        for i in pbar:
            ini = i * PASO_HORAS
            fin_ent = ini + VENTANA_HORAS
            fin_inv = min(fin_ent + PASO_HORAS, len(retornos))
            if fin_inv - fin_ent == 0: break

            ret_ent, rf_ent = retornos.iloc[ini:fin_ent], rf_horaria[ini:fin_ent].astype(float)
            pesos_ventana, frentes = procesar_ventana(ret_ent, rf_ent, n_activos, prev_weights)
            
            # Guardamos el frente (solo objetivos para ahorrar memoria)
            matriz_frentes[n].append(frentes[0] if frentes else np.array([]))
            
            if n == 0: fechas_ventanas.append(fechas_indice[fin_ent])

            # Solo guardamos retornos de la última corrida para el resumen visual estándar
            if n == N_CORRIDAS - 1:
                ret_inv, rf_inv = retornos.iloc[fin_ent:fin_inv], rf_horaria[fin_ent:fin_inv].astype(float)
                rf_inv_series = pd.Series(rf_inv, index=ret_inv.index)
                fecha_inv = fechas_indice[fin_ent]
                for strat in ESTRATEGIAS:
                    w = pesos_ventana[strat]
                    pesos_por_estrategia[strat][fecha_inv] = w
                    retornos_por_estrategia[strat].append((ret_inv * w).sum(axis=1) - rf_inv_series)
            
            prev_weights = pesos_ventana

    # --- CÁLCULO DE HIPERVOLUMEN GLOBAL ---
    print("\nCalculando métricas de Hipervolumen Global...")
    
    # 1. Encontrar el peor punto entre TODAS las corridas y fechas
    puntos_totales = []
    for c in range(N_CORRIDAS):
        for f in matriz_frentes[c]:
            if f.size > 0: puntos_totales.append(f)
    
    if puntos_totales:
        puntos_concat = np.vstack(puntos_totales)
        nadir_point = np.max(puntos_concat, axis=0)
        # Sugerencia: Margen de seguridad del 5%
        ref_point = nadir_point + (np.abs(nadir_point) * 0.05)
        # Caso especial si el valor es 0
        ref_point = np.where(ref_point == 0, 0.1, ref_point)
        
        print(f"Punto de Referencia Global: Curtosis={ref_point[0]:.4f}, NegSharpe={ref_point[1]:.4f}")
        
        # 2. Calcular matriz de HV [Fecha x Corrida]
        hv_obj = HV(ref_point=ref_point)
        hv_results = np.zeros((len(fechas_ventanas), N_CORRIDAS))
        
        for c in range(N_CORRIDAS):
            for f in range(len(fechas_ventanas)):
                frente = matriz_frentes[c][f]
                hv_results[f, c] = hv_obj.do(frente) if frente.size > 0 else 0.0

        # 3. Crear DataFrame de Hipervolumen
        df_hv = pd.DataFrame(hv_results, index=fechas_ventanas, columns=[f'Corrida_{i+1}' for i in range(N_CORRIDAS)])
        df_hv['Promedio'] = df_hv.mean(axis=1)
        print("\nMatriz de Hipervolumen (Primeras filas):")
        print(df_hv.head())
        
        # Guardar a CSV para análisis posterior
        df_hv.to_csv("resultados_hipervolumen.csv")
        graficar_hipervolumen(df_hv)
    
    # --- VISUALIZACIONES FINALES (Usando la última corrida) ---
    for strat in ESTRATEGIAS:
        lst = retornos_por_estrategia[strat]
        retornos_por_estrategia[strat] = pd.concat(lst) if lst else pd.Series(dtype=float)

    metricas = [summary_metrics(retornos_por_estrategia[s], s) for s in ESTRATEGIAS]
    mostrar_resumen_metrica(metricas)
    mostrar_tabla_comparativa(metricas)
    
    # Consolidar todos los frentes de la última corrida para el gráfico de Pareto
    frentes_finales = [f for f in matriz_frentes[-1] if f.size > 0]
    graficar_frente_pareto_global(frentes_finales)
    graficar_retornos_acumulados(retornos_por_estrategia, titulo=f"Desempeño - {FRECUENCIA_SELECCIONADA}")
    graficar_drawdown(retornos_por_estrategia)
    graficar_boxplot_retornos(retornos_por_estrategia)
    graficar_pesos_promedio(pesos_por_estrategia)

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        mp.freeze_support()
    main()