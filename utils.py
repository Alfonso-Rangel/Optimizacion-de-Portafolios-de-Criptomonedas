import jpype
import jpype.imports
import numpy as np
import threading
import os
import warnings

warnings.filterwarnings('ignore', message='The default fill_method')

JDouble = None
JDoubleArray = None
JDouble2DArray = None
PSO = None
NSGAII = None

_jvm_lock = threading.Lock()
_jvm_started = False

def iniciar_jvm(classpath=None, force=False):
    global JDouble, JDoubleArray, JDouble2DArray, PSO, NSGAII, _jvm_started
    with _jvm_lock:
        if jpype.isJVMStarted() and not force:
            JDouble = jpype.JDouble
            JDoubleArray = jpype.JArray(JDouble)
            JDouble2DArray = jpype.JArray(JDoubleArray)
            PSO = jpype.JClass("PSO")
            NSGAII = jpype.JClass("NSGAII")
            _jvm_started = True
            return

        if classpath is None:
            # Agregar la carpeta pso al classpath
            pso_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pso")
            classpath = [".", pso_dir]
        elif isinstance(classpath, str):
            classpath = [classpath]

        try:
            jpype.startJVM(
                jpype.getDefaultJVMPath(),
                "-ea",
                "--enable-native-access=ALL-UNNAMED",
                "-Djava.class.path=" + os.pathsep.join(classpath)
            )
        except Exception:
            jpype.startJVM(jpype.getDefaultJVMPath())

        JDouble = jpype.JDouble
        JDoubleArray = jpype.JArray(JDouble)
        JDouble2DArray = jpype.JArray(JDoubleArray)
        PSO = jpype.JClass("PSO")
        NSGAII = jpype.JClass("NSGAII")
        _jvm_started = True


def np_a_java_array(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=float)
    if JDoubleArray is None:
        iniciar_jvm()
    return JDoubleArray(arr.astype(float).tolist())


def np_a_java_2darray(arr2d):
    if not isinstance(arr2d, np.ndarray):
        arr2d = np.asarray(arr2d, dtype=float)
    if JDouble2DArray is None:
        iniciar_jvm()
    filas = [JDoubleArray(fila.tolist()) for fila in arr2d.astype(float)]
    return JDouble2DArray(filas)


def java_list_of_doublearrays_to_numpy(java_list):
    if java_list is None:
        return []
    return [np.array([float(v) for v in elem], dtype=float) for elem in java_list]