# utils.py

import jpype
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

# Variables globales para los tipos Java
JDouble = None
JDoubleArray = None
JDouble2DArray = None
PSO = None

def iniciar_jvm(classpath=["."]):
    """Inicializa la JVM y configura los tipos Java necesarios."""
    global JDouble, JDoubleArray, JDouble2DArray, PSO 
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=classpath)
    
    JDouble = jpype.JDouble
    JDoubleArray = jpype.JArray(JDouble)
    JDouble2DArray = jpype.JArray(JDoubleArray)
    PSO = jpype.JClass("PSO") 
    print("JVM inicializada y tipos Java configurados. Clase PSO (unificada) cargada.")

def np_a_java_2darray(numpy_array):
    """Convierte un array 2D de NumPy a un JDouble2DArray de Java."""
    if JDouble2DArray is None:
        raise RuntimeError("JVM no inicializada. Llama a iniciar_jvm() primero.")
    
    return JDouble2DArray([JDoubleArray(row.tolist()) for row in numpy_array])


def np_a_java_array(numpy_array):
    """Convierte un array 1D de NumPy a un JDoubleArray de Java."""
    if JDoubleArray is None:
        raise RuntimeError("JVM no inicializada. Llama a iniciar_jvm() primero.")
        
    return JDoubleArray(numpy_array.tolist())


def java_list_of_doublearrays_to_numpy(java_list):
    """
    Convierte una lista Java (ArrayList) de arrays double[] a una lista de numpy arrays.
    Ejemplo de uso: frente de Pareto devuelto por PSO_MOPSO.getFrentePos()
    """
    if java_list is None:
        return []
    py_list = []
    for jarr in java_list:
        # jarr es un array Java; convertir a lista de floats
        py_list.append(np.array([float(x) for x in jarr], dtype=float))
    return py_list