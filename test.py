import jpype
import jpype.imports
from jpype.types import *

# Iniciar la JVM
jpype.startJVM(classpath=["."])  # o el path a tus clases compiladas

# Importar la clase Java
PSO = jpype.JClass("PSO")

# Crear datos de ejemplo
import numpy as np
retornos = np.array([0.1, 0.2, 0.15])
matriz_cov = np.array([[0.01, 0.001, 0.002],
                       [0.001, 0.02, 0.003],
                       [0.002, 0.003, 0.015]])

# Convertir numpy a arrays Java
JDoubleArray = jpype.JArray(jpype.JDouble)
JDouble2DArray = jpype.JArray(JDoubleArray)

retornos_java = JDoubleArray(retornos.tolist())
cov_java = JDouble2DArray([JDoubleArray(row.tolist()) for row in matriz_cov])

# Crear instancia de PSO
pso = PSO(retornos_java, cov_java)

# Ejecutar optimización
pso.maximizarRetorno(0.003)

# Obtener la mejor solución
mejor = pso.getMejorPosicion()
print("Mejor portafolio:", list(mejor))
print("Resultado:", pso.toString())

# Apagar la JVM
jpype.shutdownJVM()
