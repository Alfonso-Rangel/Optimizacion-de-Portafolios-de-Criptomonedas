import multiprocessing as mp
import numpy as np
import pandas as pd
import sys
import os
import traceback
import atexit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ====== POOL GLOBAL ======
_GLOBAL_POOL = None
_INITIALIZED = False

# Parámetros globales por worker
_SHARPE_RUNS = None
_KURT_RUNS = None
_MOPSO_RUNS = None

def init_global_pool(sharpe_runs=3, kurt_runs=3, mopso_runs=1):
    """Inicializa el pool global una sola vez"""
    global _GLOBAL_POOL, _INITIALIZED, _SHARPE_RUNS, _KURT_RUNS, _MOPSO_RUNS
    
    if _INITIALIZED:
        return _GLOBAL_POOL
    
    _SHARPE_RUNS = sharpe_runs
    _KURT_RUNS = kurt_runs
    _MOPSO_RUNS = mopso_runs
    
    # Crear pool con 3 procesos
    _GLOBAL_POOL = mp.Pool(
        processes=3,
        initializer=_init_worker,
        initargs=(sharpe_runs, kurt_runs, mopso_runs)
    )
    
    _INITIALIZED = True
    atexit.register(close_global_pool)
    
    print(f"Pool global inicializado con {3} procesos")
    return _GLOBAL_POOL

def close_global_pool():
    """Cierra el pool global"""
    global _GLOBAL_POOL, _INITIALIZED
    
    if _GLOBAL_POOL is not None:
        if _INITIALIZED:
            print("Cerrando pool global...")
        _GLOBAL_POOL.close()
        _GLOBAL_POOL.join()
        _GLOBAL_POOL = None
        _INITIALIZED = False

def _init_worker(sharpe_runs, kurt_runs, mopso_runs):
    """Inicializa cada worker del pool"""
    global _SHARPE_RUNS, _KURT_RUNS, _MOPSO_RUNS
    
    _SHARPE_RUNS = sharpe_runs
    _KURT_RUNS = kurt_runs
    _MOPSO_RUNS = mopso_runs
    
    # Inicializar JVM en cada worker
    from utils import iniciar_jvm
    iniciar_jvm()


# ------- Workers -------
def worker_sharpe(retornos_data, rf_data):
    try:
        from main import ejecutar_mono_multi_runs
        
        # Reconstruir datos
        retornos_ent = pd.DataFrame(
            retornos_data[0],
            index=pd.to_datetime(retornos_data[1]),
            columns=retornos_data[2]
        )
        rf_ent = np.array(rf_data[0], float)
        
        w = ejecutar_mono_multi_runs(retornos_ent, rf_ent, "sharpe", _SHARPE_RUNS)
        return ("sharpe", w.tolist())
    except Exception as e:
        print(f"[ERROR worker_sharpe] {e}")
        traceback.print_exc()
        return ("sharpe", None)


def worker_kurtosis(retornos_data, rf_data):
    try:
        from main import ejecutar_mono_multi_runs
        
        retornos_ent = pd.DataFrame(
            retornos_data[0],
            index=pd.to_datetime(retornos_data[1]),
            columns=retornos_data[2]
        )
        rf_ent = np.array(rf_data[0], float)
        
        w = ejecutar_mono_multi_runs(retornos_ent, rf_ent, "kurtosis", _KURT_RUNS)
        return ("kurt", w.tolist())
    except Exception as e:
        print(f"[ERROR worker_kurtosis] {e}")
        traceback.print_exc()
        return ("kurt", None)


def worker_mopso(retornos_data, rf_data):
    try:
        from main import ejecutar_mopso_single_run
        
        retornos_ent = pd.DataFrame(
            retornos_data[0],
            index=pd.to_datetime(retornos_data[1]),
            columns=retornos_data[2]
        )
        rf_ent = np.array(rf_data[0], float)
        
        w, frentes = ejecutar_mopso_single_run(retornos_ent, rf_ent)
        
        frentes_serializable = [f.tolist() for f in frentes]
        return ("comp", w.tolist(), frentes_serializable)
    except Exception as e:
        print(f"[ERROR worker_mopso] {e}")
        traceback.print_exc()
        return ("comp", None, [])


# ------- Procesamiento paralelo -------
def procesar_ventana_parallel_mp(retornos_ent, rf_ent, n_activos):
    """
    Procesa una ventana usando el pool global.
    """
    global _GLOBAL_POOL
    
    if _GLOBAL_POOL is None:
        raise RuntimeError("Pool global no inicializado. Llama a init_global_pool() primero.")
    
    # Preparar datos para serialización
    retornos_data = (
        retornos_ent.to_numpy().astype(float),
        retornos_ent.index.astype(str).tolist(),
        retornos_ent.columns.tolist()
    )
    rf_data = (rf_ent.astype(float).tolist(),)
    
    resultados = {
        "naive": np.full(n_activos, 1.0 / n_activos).tolist()
    }
    
    frentes_runs = []
    
    # Enviar tareas al pool global
    tasks = [
        _GLOBAL_POOL.apply_async(worker_sharpe, (retornos_data, rf_data)),
        _GLOBAL_POOL.apply_async(worker_kurtosis, (retornos_data, rf_data)),
        _GLOBAL_POOL.apply_async(worker_mopso, (retornos_data, rf_data))
    ]
    
    # Recoger resultados
    for task in tasks:
        try:
            result = task.get(timeout=300)  # timeout de 5 minutos
        except mp.TimeoutError:
            print("[ERROR] Worker timeout; usando naive")
            continue
        except Exception as e:
            print(f"[ERROR] Error en worker: {e}")
            continue
        
        if result[0] == "comp":
            estrategia, w, frentes = result
            if frentes:
                for f in frentes:
                    frentes_runs.append(np.array(f, dtype=float))
        else:
            estrategia, w = result
        
        if w is not None:
            resultados[estrategia] = w
        else:
            resultados[estrategia] = resultados["naive"]
    
    # Convertir a numpy arrays
    for key in resultados:
        resultados[key] = np.array(resultados[key], dtype=float)
    
    return resultados, frentes_runs