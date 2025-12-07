import multiprocessing as mp
import traceback
import numpy as np
import pandas as pd


# worker initializer to start JVM per process
def _init_worker():
    # Lazy import to avoid importing jpype in master process on Windows spawn
    from utils import iniciar_jvm
    iniciar_jvm()


def _worker_mono_once(args):
    """
    Ejecuta una sola corrida mono-objetivo PSO y devuelve dic con
    {'w': list, 'fitness': float, 'objetivo': str}
    args = (retornos_array, index_strings, cols, rf_array, objetivo, run_id)
    """
    try:
        from utils import np_a_java_2darray, np_a_java_array, java_list_of_doublearrays_to_numpy, iniciar_jvm
        import utils as U
        iniciar_jvm()
        retornos_np, index_strs, cols, rf_arr, objetivo, run_id = args
        retornos_df = pd.DataFrame(retornos_np, index=pd.to_datetime(index_strs), columns=cols)
        # call PSO
        PSO = U.PSO
        pso = PSO(U.np_a_java_2darray(retornos_df.to_numpy())) if hasattr(PSO, "__call__") else PSO(retornos_df.to_numpy())
        if objetivo == "sharpe":
            pso.maximizarSharpe(np_a_java_array(rf_arr))
        else:
            pso.minimizarKurtosis()
        w = np.array(pso.getMejorPosicion(), dtype=float)
        w = np.clip(w, 0, None)
        s = w.sum()
        if s == 0:
            w = np.full_like(w, 1.0 / len(w))
        else:
            w = w / s

        # compute fitness in python (paired)
        port = (retornos_df * w).sum(axis=1)
        exceso = port - pd.Series(rf_arr, index=retornos_df.index)
        if objetivo == "sharpe":
            mean = exceso.mean()
            std = exceso.std(ddof=0)
            fitness = float(mean / std) if std > 0 else float(-1e9)
        else:
            from scipy.stats import kurtosis
            fitness = float(kurtosis(exceso.values, fisher=True, bias=False))

        return {"w": w.tolist(), "fitness": fitness, "objetivo": objetivo, "run_id": run_id}
    except Exception as e:
        traceback.print_exc()
        return {"w": None, "fitness": None, "objetivo": objetivo, "run_id": run_id, "error": str(e)}


def run_mono_parallel(retornos_ent, rf_ent, objetivo, n_runs, processes=None, timeout=300):
    """
    Ejecuta n_runs en paralelo (pool) y devuelve lista de resultados por corrida.
    """
    retornos_data = (retornos_ent.to_numpy().astype(float), retornos_ent.index.astype(str).tolist(), retornos_ent.columns.tolist())
    rf_arr = rf_ent.astype(float)
    args = []
    for i in range(n_runs):
        args.append((retornos_data[0], retornos_data[1], retornos_data[2], rf_arr, objetivo, i))

    pool = mp.Pool(processes=processes or min(mp.cpu_count(), n_runs), initializer=_init_worker)
    try:
        results = [pool.apply_async(_worker_mono_once, (a,)) for a in args]
        out = [r.get(timeout=timeout) for r in results]
    finally:
        pool.close()
        pool.join()
    return out
