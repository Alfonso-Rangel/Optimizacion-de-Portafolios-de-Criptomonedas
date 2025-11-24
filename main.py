import pandas as pd
import numpy as np
from tqdm import trange
import jpype
from scipy.stats import kurtosis

import utils
from config import VENTANA_HORAS, PASO_HORAS
from data_loader import cargar_datos_experimento
from utils import iniciar_jvm, np_a_java_2darray, np_a_java_array, java_list_of_doublearrays_to_numpy 
from evaluation import (
    mostrar_resumen_metrica,
    graficar_retornos_acumulados,
    graficar_pesos_promedio,
    graficar_distribucion_kde,
    graficar_box_plot,
    graficar_distribucion_retornos,
    mostrar_tabla_comparativa,
    graficar_frente_pareto_global,
    summary_metrics
)

def seleccionar_punto_utopia(frente_obj):
    """
    Selecciona un portafolio de compromiso del Frente de Pareto
    utilizando la distancia Euclidiana (L2) al Punto de Utopía (Ideal Point).
    """
    objs = np.array(frente_obj)
    if len(objs) == 0:
        return -1 # Retorna un índice inválido si no hay soluciones

    # 1. Normalizar min-max
    # Los objetivos son [Curtosis, -Sharpe]. 
    # El punto ideal (min Curtosis, max Sharpe) se mapea a (0, 0).
    min_vals = np.min(objs, axis=0)
    max_vals = np.max(objs, axis=0)
    
    # Evitar división por cero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0 
    
    # front_norm[i, m] es la distancia normalizada del portafolio i al ideal (mínimo)
    front_norm = (objs - min_vals) / range_vals
    
    # 2. Calcular la Distancia Euclidiana (L2) al Punto de Utopía (0, 0)
    # Distancia = sqrt(obj1_norm^2 + obj2_norm^2)
    # Nota: No necesitamos calcular la raíz cuadrada, solo minimizamos la suma de los cuadrados
    # para ahorrar cálculo, ya que np.argmin() dará el mismo resultado.
    
    distancias_utopia_sq = np.sum(front_norm**2, axis=1)
    
    # 3. Seleccionar el punto que minimiza la distancia
    idx_seleccionado = np.argmin(distancias_utopia_sq)
    
    return idx_seleccionado

def seleccionar_knee_point(frente_obj):
    """
    Selecciona el punto de codo (knee point) normalizando primero los objetivos
    para evitar que la escala de la Curtosis (10-100) opaque al Sharpe (0.001).
    """
    objs = np.array(frente_obj)
    if len(objs) <= 2:
        return 0

    # 1. Normalizar min-max para poner ambos objetivos en escala [0, 1]
    min_vals = np.min(objs, axis=0)
    max_vals = np.max(objs, axis=0)
    rango = max_vals - min_vals
    
    # Evitar división por cero si todos los puntos son iguales en una dimensión
    rango[rango == 0] = 1.0 
    
    objs_norm = (objs - min_vals) / rango

    # 2. Ordenar basado en el primer objetivo (Curtosis)
    idx_sorted = np.argsort(objs_norm[:, 0])
    front_norm = objs_norm[idx_sorted]
    indices_originales = idx_sorted

    # 3. Trazar línea recta desde el primero al último punto
    p1 = front_norm[0]
    p2 = front_norm[-1]
    
    vec_linea = p2 - p1
    norm_linea = np.linalg.norm(vec_linea)
    
    if norm_linea == 0:
        return 0

    vec_linea_unitario = vec_linea / norm_linea

    # 4. Calcular distancia perpendicular de cada punto a la línea
    distancias = []
    for p in front_norm:
        vec_puntos = p - p1
        proyeccion = np.dot(vec_puntos, vec_linea_unitario) * vec_linea_unitario
        perpendicular = vec_puntos - proyeccion
        dist = np.linalg.norm(perpendicular)
        distancias.append(dist)

    # 5. El Knee point es el que tiene mayor distancia a la línea recta
    knee_local_idx = np.argmax(distancias)
    
    return int(indices_originales[knee_local_idx])


iniciar_jvm()
PSO = utils.PSO

if PSO is None:
    raise RuntimeError("Clase PSO no encontrada")

retornos, rf_horaria, fechas_indice, n_activos = cargar_datos_experimento()

retornos_por_estrategia = {"naive": [], "sharpe": [], "kurt": [], "comp": []}
pesos_por_estrategia = {"naive": {}, "sharpe": {}, "kurt": {}, "comp": {}}

frentes_obj_por_ventana = []

print("Iniciando rolling experiment (fracciones)...")

for i in trange(0, len(retornos) - VENTANA_HORAS, PASO_HORAS):
    idx_fin_entrenamiento = i + VENTANA_HORAS
    retornos_ent = retornos.iloc[i:idx_fin_entrenamiento]
    rf_ent = rf_horaria[i:idx_fin_entrenamiento]

    idx_fin_inv = min(idx_fin_entrenamiento + PASO_HORAS, len(retornos))
    retornos_inv = retornos.iloc[idx_fin_entrenamiento:idx_fin_inv]
    rf_inv = rf_horaria[idx_fin_entrenamiento:idx_fin_inv]

    if retornos_inv.empty:
        break

    fecha_inv = fechas_indice[idx_fin_entrenamiento]

    w_naive = np.full(n_activos, 1.0 / n_activos)
    pesos_por_estrategia["naive"][fecha_inv] = w_naive

    retornos_java = np_a_java_2darray(retornos_ent.to_numpy())
    rf_java = np_a_java_array(rf_ent)

    # --- Estrategia Max Sharpe (Single Objective) ---
    pso_sh = PSO(retornos_java)
    pso_sh.maximizarSharpe(rf_java)
    w_sh = np.array(pso_sh.getMejorPosicion(), float)
    w_sh = np.maximum(w_sh, 0)
    w_sh = w_sh / w_sh.sum() if w_sh.sum() != 0 else np.full(n_activos, 1 / n_activos)
    pesos_por_estrategia["sharpe"][fecha_inv] = w_sh

    # --- Estrategia Min Curtosis (Single Objective) ---
    pso_k = PSO(retornos_java)
    pso_k.minimizarKurtosis()
    w_k = np.array(pso_k.getMejorPosicion(), float)
    w_k = np.maximum(w_k, 0)
    w_k = w_k / w_k.sum() if w_k.sum() != 0 else np.full(n_activos, 1 / n_activos)
    pesos_por_estrategia["kurt"][fecha_inv] = w_k

    # --- Estrategia Compuesta (Multi Objective MOPSO) ---
    mopso = PSO(retornos_java)
    mopso.optimizar(rf_java)

    frente_pos_java = mopso.getFrentePos()
    frente_obj_java = mopso.getFrenteObj()

    frente_pos = java_list_of_doublearrays_to_numpy(frente_pos_java)
    frente_obj = java_list_of_doublearrays_to_numpy(frente_obj_java)

    w_comp = np.full(n_activos, 1.0 / n_activos) # Fallback por defecto

    if len(frente_obj) > 0:
        frentes_obj_por_ventana.append(frente_obj.copy())
        idx_knee = seleccionar_punto_utopia(frente_obj)
        w_comp = np.array(frente_pos[idx_knee], float)
    
    w_comp = np.maximum(w_comp, 0)
    w_comp = w_comp / w_comp.sum() if w_comp.sum() != 0 else np.full(n_activos, 1 / n_activos)

    pesos_por_estrategia["comp"][fecha_inv] = w_comp

    # --- Calcular Retornos del Periodo ---
    pesos_dict = {"naive": w_naive, "sharpe": w_sh, "kurt": w_k, "comp": w_comp}
    rf_inv_series = pd.Series(rf_inv, index=retornos_inv.index)

    for key, w in pesos_dict.items():
        r = (retornos_inv * w).sum(axis=1)
        ex_r = r - rf_inv_series
        retornos_por_estrategia[key].append(ex_r)

print("Consolidando resultados...")

for key in retornos_por_estrategia:
    if len(retornos_por_estrategia[key]) == 0:
        retornos_por_estrategia[key] = pd.Series(dtype=float)
    else:
        retornos_por_estrategia[key] = pd.concat(retornos_por_estrategia[key])

print("Obteniendo métricas finales...")
metricas_finales = [
    summary_metrics(retornos_por_estrategia["naive"], "Naive 1/N"),
    summary_metrics(retornos_por_estrategia["sharpe"], "Sharpe-PSO"),
    summary_metrics(retornos_por_estrategia["kurt"], "Kurtosis-PSO"),
    summary_metrics(retornos_por_estrategia["comp"], "Compuesto (MOPSO)")
]

mostrar_resumen_metrica(metricas_finales)
graficar_frente_pareto_global(frentes_obj_por_ventana)
graficar_retornos_acumulados(retornos_por_estrategia)