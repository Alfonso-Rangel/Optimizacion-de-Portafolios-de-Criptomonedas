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
    graficar_retornos_acumulados, resumen_metricas, mostrar_tabla_comparativa,
    graficar_hipervolumen_comparado, graficar_drawdown, graficar_barras_pesos
)

# ================= CONFIG =================
FRECUENCIAS = {'media_semana': 84, 'semana': 168, 'dos_semanas': 336, 'cuatro_semanas': 672, 'ocho_semanas': 1344}

VENTANA_HORAS = FRECUENCIAS["dos_semanas"]
PASO_HORAS    = FRECUENCIAS['cuatro_semanas']
N_CORRIDAS    = 5
ALGORITMOS    = ["mopso", "nsga2"]

ESTRATEGIAS = ["mopso", "nsga2", "curtosis"]
#ESTRATEGIAS = ["naive", "sharpe", "curtosis"]
#ESTRATEGIAS = ["mopso", "nsga2"]

# ================= NORMALIZACIÓN =================

def normalizar_frente(f):
    if len(f) == 0:
        return f
    min_ = f.min(axis=0)
    max_ = f.max(axis=0)
    rango = np.where((max_ - min_) == 0, 1, (max_ - min_))
    return (f - min_) / rango

# ================= MÉTRICAS =================

def spacing(front):
    if len(front) < 2:
        return 0
    d = np.array([
        min(np.sum(np.abs(front[i] - front[j]))
            for j in range(len(front)) if i != j)
        for i in range(len(front))
    ])
    return np.sqrt(np.sum((d - d.mean()) ** 2) / (len(front) - 1))


def spread(front):
    if len(front) < 2:
        return 0
    front = front[np.argsort(front[:, 0])]
    dist  = np.linalg.norm(np.diff(front, axis=0), axis=1)
    d_bar = dist.mean()
    return (dist[0] + dist[-1] + np.sum(np.abs(dist - d_bar))) / \
           (dist[0] + dist[-1] + len(dist) * d_bar)


def igd(front, ref):
    if len(front) == 0 or len(ref) == 0:
        return np.inf
    return np.mean([
        np.min(np.linalg.norm(front - r, axis=1)) for r in ref
    ])

# ================= SELECCIÓN =================
def portafolio_naive(n_assets):
    return np.full(n_assets, 1.0 / n_assets)

def seleccionar_indice(frente_obj, modo="utopia"):
    if len(frente_obj) == 0:
        return -1

    obj = np.array(frente_obj)

    if modo == "sharpe":
        return np.argmin(obj[:, 1])
    
    if modo == "curtosis":
        return np.argmin(obj[:, 0])

    rango = np.where((r := obj.max(0) - obj.min(0)) == 0, 1.0, r)
    return np.argmin(np.sum(((obj - obj.min(0)) / rango) ** 2, axis=1))

# ================= UTILIDADES =================

def normalizar_pesos(w):
    w = np.maximum(np.asarray(w, float), 0)
    s = w.sum()
    return w / s if s else np.full_like(w, 1 / len(w))

# ================= EJECUCIÓN =================

def ejecutar_algoritmo(ret_ent, rf_ent, algoritmo):
    iniciar_jvm()
    ret_java = np_a_java_2darray(ret_ent.to_numpy())
    rf_java  = np_a_java_array(rf_ent)

    solver = U.MOPSO(ret_java) if algoritmo == "mopso" else U.NSGAII(ret_java)
    solver.optimizar(rf_java)

    front_pos = java_list_of_doublearrays_to_numpy(solver.getFrentePos())
    front_obj = java_list_of_doublearrays_to_numpy(solver.getFrenteObj())

    if len(front_obj) == 0:
        w = np.full(ret_ent.shape[1], 1.0 / ret_ent.shape[1])
        return w, w, np.array([])

    w_utopia = normalizar_pesos(front_pos[seleccionar_indice(front_obj, "utopia")])
    w_sharpe = normalizar_pesos(front_pos[seleccionar_indice(front_obj, "sharpe")])
    w_curtosis   = normalizar_pesos(front_pos[seleccionar_indice(front_obj, "curtosis")])

    return w_utopia, w_sharpe, w_curtosis, np.array(front_obj)

# ================= MAIN =================

def main():
    os.makedirs("figures", exist_ok=True)
    iniciar_jvm()
    retornos, rf_horaria, fechas_indice, _ = cargar_datos_experimento()
    matriz_frentes = {alg: [[] for _ in range(N_CORRIDAS)] for alg in ALGORITMOS}
    fechas_ventanas = []

    retornos_runs = {
        strat: [[] for _ in range(N_CORRIDAS)]
        for strat in ESTRATEGIAS
    }

    pesos_runs = {
        strat: [{} for _ in range(N_CORRIDAS)]
        for strat in ESTRATEGIAS
    }

    num_ventanas = (len(retornos) - VENTANA_HORAS) // PASO_HORAS

    # ===== RUNS =====
    for n in range(N_CORRIDAS):
        print(f"Corrida {n+1}")

        for i in trange(num_ventanas):
            ini, fin_ent = i * PASO_HORAS, i * PASO_HORAS + VENTANA_HORAS
            fin_inv = min(fin_ent + PASO_HORAS, len(retornos))
            if fin_inv == fin_ent:
                break

            ret_ent = retornos.iloc[ini:fin_ent]
            rf_ent  = rf_horaria[ini:fin_ent].astype(float)

            resultados = {}
            resultados["naive"] = portafolio_naive(ret_ent.shape[1])

            for alg in ALGORITMOS:
                w_u, w_s, w_k, f = ejecutar_algoritmo(ret_ent, rf_ent, alg)

                resultados[alg] = w_u
                matriz_frentes[alg][n].append(f)

                if alg == "mopso":
                    resultados["sharpe"]   = w_s
                    resultados["curtosis"] = w_k

            # inversión en TODAS las corridas
            ret_inv = retornos.iloc[fin_ent:fin_inv]
            rf_inv  = pd.Series(rf_horaria[fin_ent:fin_inv].astype(float), index=ret_inv.index)
            fecha_inv = fechas_indice[fin_ent]

            for strat in ESTRATEGIAS:
                pesos_runs[strat][n][fecha_inv] = resultados[strat]

                serie = (ret_inv * resultados[strat]).sum(axis=1) - rf_inv
                retornos_runs[strat][n].append(serie)

    # ===== MÉTRICAS POR VENTANA =====
    hv_dict = {alg: [] for alg in ALGORITMOS}

    for w in range(num_ventanas):
        union = []
        for n in range(N_CORRIDAS):
            for alg in ALGORITMOS:
                f = matriz_frentes[alg][n][w]
                if len(f) > 0:
                    union.append(f)

        if not union:
            continue

        ref = normalizar_frente(np.vstack(union))
        ref_point = np.max(ref, axis=0) * 1.05
        hv = HV(ref_point=ref_point)

        for alg in ALGORITMOS:
            vals = []
            for n in range(N_CORRIDAS):
                f = matriz_frentes[alg][n][w]
                if len(f) == 0:
                    continue
                min_ref = ref.min(axis=0)
                max_ref = ref.max(axis=0)
                rango = np.where((max_ref - min_ref) == 0, 1, (max_ref - min_ref))
                f_norm = (f - min_ref) / rango
                vals.append(hv.do(f_norm))

            hv_dict[alg].append(np.mean(vals) if vals else 0)

    df_mopso = pd.DataFrame({
        "Promedio": hv_dict["mopso"]
    })

    df_nsga = pd.DataFrame({
        "Promedio": hv_dict["nsga2"]
    })
    graficar_hipervolumen_comparado(df_mopso, df_nsga)

    # ===== MÉTRICAS AGREGADAS =====
    for alg in ALGORITMOS:
        all_f = [f for c in range(N_CORRIDAS) for f in matriz_frentes[alg][c] if len(f) > 0]
        print(f"\n{alg}")
        print("spacing:", np.mean([spacing(normalizar_frente(f)) for f in all_f]))
        print("spread:",  np.mean([spread(normalizar_frente(f))  for f in all_f]))

    # ===== FINANZAS =====
    retornos_por_corrida = {s: [] for s in ESTRATEGIAS}

    for strat in ESTRATEGIAS:
        for n in range(N_CORRIDAS):
            if retornos_runs[strat][n]:
                retornos_por_corrida[strat].append(
                    pd.concat(retornos_runs[strat][n])
                )
    retornos_promedio = {}

    for strat in ESTRATEGIAS:
        df = pd.concat(retornos_por_corrida[strat], axis=1)
        # promedio por timestamp (ignora NaN si alguna corrida no tiene ese punto)
        retornos_promedio[strat] = df.mean(axis=1)

    pesos_por_estrategia = {}

    for strat in ESTRATEGIAS:
        acumulado = {}

        for n in range(N_CORRIDAS):
            for fecha, w in pesos_runs[strat][n].items():
                acumulado.setdefault(fecha, []).append(w)

        pesos_por_estrategia[strat] = {
            fecha: np.mean(ws, axis=0)
            for fecha, ws in acumulado.items()
        }

    metricas = [
        resumen_metricas(retornos_promedio[s], s)
        for s in ESTRATEGIAS
    ]

    mostrar_tabla_comparativa(metricas)
    graficar_retornos_acumulados(retornos_promedio)
    graficar_drawdown(retornos_promedio)
    graficar_barras_pesos(pesos_por_estrategia, list(retornos.columns))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()