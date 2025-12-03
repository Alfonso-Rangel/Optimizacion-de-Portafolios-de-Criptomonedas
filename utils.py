import jpype
import jpype.imports
import numpy as np
import threading
import os 

# Objetos Java inicializados cuando se arranca la JVM
JDouble = None
JDoubleArray = None
JDouble2DArray = None
PSO = None

# Lock para inicialización thread-safe
_jvm_lock = threading.Lock()
_jvm_started = False
_jvm_started_per_process = False  # Para multiprocessing

def iniciar_jvm(classpath=None, force=False):
    """
    Inicia la JVM una única vez por proceso.
    El classpath debe apuntar a la carpeta donde se encuentra PSO.class
    o al jar que contiene la clase.
    
    Args:
        classpath: lista de paths o string
        force: forzar reinicio (útil para testing)
    """
    global JDouble, JDoubleArray, JDouble2DArray, PSO, _jvm_started, _jvm_started_per_process
    
    # Para multiprocessing, cada proceso debe iniciar su propia JVM
    if not force and _jvm_started_per_process:
        return
    
    with _jvm_lock:
        if jpype.isJVMStarted() and not force:
            # Ya iniciada, sólo cargar clases
            JDouble = jpype.JDouble
            JDoubleArray = jpype.JArray(JDouble)
            JDouble2DArray = jpype.JArray(JDoubleArray)
            PSO = jpype.JClass("PSO")
            _jvm_started_per_process = True
            return

        if classpath is None:
            classpath = ["."]
        elif isinstance(classpath, str):
            classpath = [classpath]

        jvm_args = [
            jpype.getDefaultJVMPath(),
            "-ea",  # Enable assertions
            "--enable-native-access=ALL-UNNAMED",
            "-Djava.class.path=" + ":".join(classpath),
            "-Xms256m",  # Memoria inicial
            "-Xmx1024m", # Memoria máxima
             "-XX:+IgnoreUnrecognizedVMOptions",
        ]
        
        try:
            jpype.startJVM(*jvm_args)
        except Exception as e:
            print(f"Error iniciando JVM: {e}")
            # Intentar sin argumentos de memoria
            jpype.startJVM(jpype.getDefaultJVMPath(), "-Djava.class.path=" + ":".join(classpath))

        JDouble = jpype.JDouble
        JDoubleArray = jpype.JArray(JDouble)
        JDouble2DArray = jpype.JArray(JDoubleArray)
        PSO = jpype.JClass("PSO")
        
        _jvm_started = True
        _jvm_started_per_process = True
        print(f"JVM inicializada en proceso {os.getpid()}. Clase PSO cargada correctamente.")

def np_a_java_array(arr):
    """
    Convierte np.array 1D → JDoubleArray
    Requiere haber llamado iniciar_jvm() antes.
    """
    if JDoubleArray is None:
        iniciar_jvm()

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=float)

    return JDoubleArray(arr.astype(float).tolist())


def np_a_java_2darray(arr2d):
    """
    Convierte np.array 2D → JDouble2DArray.
    """
    if JDouble2DArray is None:
        iniciar_jvm()

    if not isinstance(arr2d, np.ndarray):
        arr2d = np.asarray(arr2d, dtype=float)

    filas = []
    for fila in arr2d.astype(float):
        filas.append(JDoubleArray(fila.tolist()))

    return JDouble2DArray(filas)


def java_list_of_doublearrays_to_numpy(java_list):
    """
    Convierte una lista Java (ArrayList<double[]>) → lista de np.array.
    """
    if java_list is None:
        return []

    resultados = []
    for elem in java_list:
        fila = np.array([float(v) for v in elem], dtype=float)
        resultados.append(fila)

    return resultados