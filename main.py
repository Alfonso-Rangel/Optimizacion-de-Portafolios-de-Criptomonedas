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
    summary_metrics,
    mostrar_tabla_comparativa,
    graficar_hipervolumen_comparado,
    graficar_drawdown,
    graficar_boxplot_retornos,
    graficar_barras_pesos,
    graficar_frente_promedio,
    graficar_dominancia_frentes,
    graficar_frentes_pareto
)

FRECUENCIAS = {
    'media_semana': 84,
    'semana': 168,
    'dos_semanas': 336,
    'cuatro_semanas': 672,
    'ocho_semanas': 1344
}

VENTANA_HORAS = FRECUENCIAS["dos_semanas"]
PASO_HORAS = FRECUENCIAS['cuatro_semanas']

N_CORRIDAS = 3

ALGORITMOS = ["mopso", "nsga2"]
ESTRATEGIAS = ["mopso", "nsga2", "sharpe"]

def seleccionar_del_frente(frente_obj):

    if len(frente_obj) == 0:
        return -1

    obj = np.array(frente_obj)

    mins, maxs = obj.min(axis=0), obj.max(axis=0)
    rango = np.where(maxs - mins == 0, 1.0, maxs - mins)

    norm = (obj - mins) / rango

    return np.argmin(np.sum(norm**2, axis=1))

def seleccionar_max_sharpe(frente_obj):

    if len(frente_obj) == 0:
        return -1

    obj = np.array(frente_obj)

    # f1 = -Sharpe
    return np.argmin(obj[:,1])

def num_solutions(front):

    if front is None or len(front) == 0:
        return 0

    return len(front)


def spacing(front):

    if len(front) < 2:
        return 0

    d = []

    for i in range(len(front)):

        dist = []

        for j in range(len(front)):

            if i != j:
                dist.append(np.sum(np.abs(front[i] - front[j])))

        d.append(min(dist))

    d = np.array(d)

    return np.sqrt(np.sum((d - d.mean())**2) / (len(front)-1))

def spread(front):

    if len(front) < 2:
        return 0

    front = front[np.argsort(front[:,0])]

    distances = np.linalg.norm(np.diff(front, axis=0), axis=1)

    d_bar = distances.mean()

    df = distances[0]
    dl = distances[-1]

    delta = (df + dl + np.sum(np.abs(distances - d_bar))) / (
        df + dl + (len(distances)) * d_bar
    )

    return delta

def normalizar_pesos(w):

    w = np.asarray(w, dtype=float)
    w[w < 0] = 0.0

    s = w.sum()

    if s == 0:
        return np.full_like(w, 1.0 / len(w))

    return w / s


def ejecutar_NSGAII(ret_ent, rf_ent):

    iniciar_jvm()

    retornos_java = np_a_java_2darray(ret_ent.to_numpy())
    rf_java = np_a_java_array(rf_ent)

    nsga = U.NSGAII(retornos_java)
    nsga.optimizar(rf_java)

    front_pos = java_list_of_doublearrays_to_numpy(nsga.getFrentePos())
    front_obj = java_list_of_doublearrays_to_numpy(nsga.getFrenteObj())

    if len(front_obj) == 0:
        w = np.full(ret_ent.shape[1], 1.0/ret_ent.shape[1])
        return w, w, []

    idx_utopia = seleccionar_del_frente(front_obj)
    idx_sharpe = seleccionar_max_sharpe(front_obj)

    w_utopia = normalizar_pesos(np.array(front_pos[idx_utopia]))
    w_sharpe = normalizar_pesos(np.array(front_pos[idx_sharpe]))

    return w_utopia, w_sharpe, [np.array(front_obj)]

def ejecutar_MOPSO(ret_ent, rf_ent):

    iniciar_jvm()

    retornos_java = np_a_java_2darray(ret_ent.to_numpy())
    rf_java = np_a_java_array(rf_ent)

    pso = U.PSO(retornos_java)
    pso.optimizar(rf_java)

    front_pos = java_list_of_doublearrays_to_numpy(pso.getFrentePos())
    front_obj = java_list_of_doublearrays_to_numpy(pso.getFrenteObj())

    if len(front_obj) == 0:
        w = np.full(ret_ent.shape[1], 1.0/ret_ent.shape[1])
        return w, w, []

    idx_utopia = seleccionar_del_frente(front_obj)
    idx_sharpe = seleccionar_max_sharpe(front_obj)

    w_utopia = normalizar_pesos(np.array(front_pos[idx_utopia]))
    w_sharpe = normalizar_pesos(np.array(front_pos[idx_sharpe]))

    return w_utopia, w_sharpe, [np.array(front_obj)]

def expandir_pesos(w_sub, sub_cols, all_cols):

    w_full = np.zeros(len(all_cols))

    col_to_idx = {c: i for i, c in enumerate(sub_cols)}

    for i, c in enumerate(all_cols):
        if c in col_to_idx:
            w_full[i] = w_sub[col_to_idx[c]]

    return normalizar_pesos(w_full)


def procesar_ventana(ret_ent, rf_ent):

    valid_cols = [c for c in ret_ent.columns if ret_ent[c].notna().all()]

    if len(valid_cols) < max(2, len(ret_ent.columns)//2):
        valid_cols = list(ret_ent.columns)

    w_mopso, w_mopso_sharpe, fr_mopso = ejecutar_MOPSO(ret_ent[valid_cols], rf_ent)
    w_nsga, w_nsga_sharpe, fr_nsga = ejecutar_NSGAII(ret_ent[valid_cols], rf_ent)

    all_cols = list(ret_ent.columns)

    resultados = {
        "mopso": expandir_pesos(w_mopso, valid_cols, all_cols),
        "nsga2": expandir_pesos(w_nsga, valid_cols, all_cols),

        # nuevo benchmark
        "sharpe": expandir_pesos(w_nsga_sharpe, valid_cols, all_cols)
    }

    frentes = {
        "mopso": fr_mopso,
        "nsga2": fr_nsga
    }

    return resultados, frentes


def main():

    os.makedirs("figures", exist_ok=True)

    iniciar_jvm()

    retornos, rf_horaria, fechas_indice, n_activos = cargar_datos_experimento()

    matriz_frentes = {
        alg: [[] for _ in range(N_CORRIDAS)]
        for alg in ALGORITMOS
    }

    fechas_ventanas = []

    retornos_por_estrategia = {k: [] for k in ESTRATEGIAS}
    pesos_por_estrategia = {k: {} for k in ESTRATEGIAS}

    for n in range(N_CORRIDAS):

        print(f"\n>>> Iniciando Corrida {n+1}/{N_CORRIDAS}")

        num_ventanas = (len(retornos) - VENTANA_HORAS) // PASO_HORAS

        pbar = trange(num_ventanas)

        for i in pbar:

            ini = i * PASO_HORAS
            fin_ent = ini + VENTANA_HORAS
            fin_inv = min(fin_ent + PASO_HORAS, len(retornos))

            if fin_inv - fin_ent == 0:
                break

            ret_ent = retornos.iloc[ini:fin_ent]
            rf_ent = rf_horaria[ini:fin_ent].astype(float)

            pesos_ventana, frentes = procesar_ventana(ret_ent, rf_ent)

            for alg in ALGORITMOS:
                frente = frentes[alg][0] if frentes[alg] else np.array([])
                matriz_frentes[alg][n].append(frente)

            if n == 0:
                fechas_ventanas.append(fechas_indice[fin_ent])

            if n == N_CORRIDAS - 1:

                ret_inv = retornos.iloc[fin_ent:fin_inv]
                rf_inv = rf_horaria[fin_ent:fin_inv].astype(float)

                rf_series = pd.Series(rf_inv, index=ret_inv.index)

                fecha_inv = fechas_indice[fin_ent]

                for strat in ESTRATEGIAS:
                    w = pesos_ventana[strat]
                    pesos_por_estrategia[strat][fecha_inv] = w
                    retornos_por_estrategia[strat].append(
                        (ret_inv * w).sum(axis=1) - rf_series
                    )

    # HIPERVOLUMEN

    hv_dict = {}

    for alg in ALGORITMOS:

        puntos = []

        for c in range(N_CORRIDAS):
            for f in matriz_frentes[alg][c]:
                if f.size > 0:
                    puntos.append(f)

        if not puntos:
            continue

        puntos_concat = np.vstack(puntos)

        nadir = np.max(puntos_concat, axis=0)
        ref = nadir + np.abs(nadir) * 0.05

        hv = HV(ref_point=ref)

        hv_results = np.zeros((len(fechas_ventanas), N_CORRIDAS))

        for c in range(N_CORRIDAS):
            for f in range(len(fechas_ventanas)):

                frente = matriz_frentes[alg][c][f]

                hv_results[f, c] = hv.do(frente) if frente.size > 0 else 0

        df = pd.DataFrame(
            hv_results,
            index=fechas_ventanas,
            columns=[f"Corrida_{i+1}" for i in range(N_CORRIDAS)]
        )

        df["Promedio"] = df.mean(axis=1)

        hv_dict[alg] = df

    graficar_hipervolumen_comparado(hv_dict["mopso"], hv_dict["nsga2"])

    for strat in ESTRATEGIAS:
        lst = retornos_por_estrategia[strat]
        retornos_por_estrategia[strat] = pd.concat(lst)
    
    metricas_frente = {
        alg: {"spacing":[], "spread":[], "n":[]}
        for alg in ALGORITMOS
    }

    for alg in ALGORITMOS:
        for c in range(N_CORRIDAS):
            for f in matriz_frentes[alg][c]:
                if f.size == 0:
                    continue

                metricas_frente[alg]["spacing"].append(spacing(f))
                metricas_frente[alg]["spread"].append(spread(f))
                metricas_frente[alg]["n"].append(len(f))
    for alg in ALGORITMOS:
        print("\n", alg)
        print("solutions:", np.mean(metricas_frente[alg]["n"]))
        print("spacing:", np.mean(metricas_frente[alg]["spacing"]))
        print("spread:", np.mean(metricas_frente[alg]["spread"]))

    metricas = [summary_metrics(retornos_por_estrategia[s], s) for s in ESTRATEGIAS]

    mostrar_resumen_metrica(metricas)
    mostrar_tabla_comparativa(metricas)

    graficar_retornos_acumulados(retornos_por_estrategia)
    graficar_drawdown(retornos_por_estrategia)
    graficar_boxplot_retornos(retornos_por_estrategia)

    graficar_barras_pesos(pesos_por_estrategia, list(retornos.columns))

    graficar_frente_promedio(matriz_frentes)

    graficar_dominancia_frentes(matriz_frentes)

    # ejemplo con primera ventana
    graficar_frentes_pareto(
        matriz_frentes["mopso"][0][0],
        matriz_frentes["nsga2"][0][0]
    )


if __name__ == "__main__":

    if sys.platform.startswith('win'):
        mp.freeze_support()

    main()