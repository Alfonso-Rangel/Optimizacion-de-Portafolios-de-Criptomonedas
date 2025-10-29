import numpy as np
import matplotlib.pyplot as plt
from datos import leer_datos 
from tqdm import trange  # para barra de progreso opcional
import jpype
import jpype.imports
from jpype.types import *
#---------------------------------------------------------------------------------------#
# Iniciar la JVM
#---------------------------------------------------------------------------------------#
jpype.startJVM(classpath=["."])
PSO = jpype.JClass("PSO")
# Convertir numpy a arrays Java
JDoubleArray = jpype.JArray(jpype.JDouble)
JDouble2DArray = jpype.JArray(JDoubleArray)
#---------------------------------------------------------------------------------------#
# Leer los datos
#---------------------------------------------------------------------------------------#
archivo_port = "port1.txt"
archivo_portef = "portef1.txt"
retornos, matriz_cov, frontera_ef = leer_datos(archivo_port, archivo_portef)
# Convertir datos a Java
retornos_java = JDoubleArray(retornos.tolist())
cov_java = JDouble2DArray([JDoubleArray(row.tolist()) for row in matriz_cov])
# --------------------------------------------------------------------------------------#
# Generar portafolios con PSO
# --------------------------------------------------------------------------------------#
riesgo_asumido = 0.003
n_portafolios = 10
# Arrays para almacenar resultados
retornos_pso = np.zeros(n_portafolios)
riesgos_pso = np.zeros(n_portafolios)
# Crear instancia de PSO
pso = PSO(retornos_java, cov_java)
# Barra de progreso para ver avance
for i in trange(n_portafolios, desc="Generando portafolios PSO"):
    pso.maximizarRetorno(riesgo_asumido) # Ejecutar PSO (esto actualiza la mejor solución interna de pso)
    port_x = pso.getMejorPosicion()  # Esto devuelve un double[] de Java
    # Calcular retorno y riesgo usando los métodos estáticos de la clase Java
    retornos_pso[i] = PSO.calcularRetorno(port_x)
    riesgos_pso[i]  = PSO.calcularVarianza(port_x)
# --------------------------------------------------------------------------------------#
# Graficar la frontera eficiente y los portafolios PSO
# --------------------------------------------------------------------------------------#
plt.figure(figsize=(10, 6))
plt.plot(frontera_ef[:, 1], frontera_ef[:, 0], color="darkslategray", lw=2, label="Frontera Eficiente")
plt.xlabel("Riesgo (Varianza)")
plt.ylabel("Retorno esperado")
plt.title("Máxima rentabilidad bajo riesgo asumido")
plt.grid(True)
plt.scatter(riesgos_pso, retornos_pso, color="royalblue", alpha=0.5, s=10, label="Portafolios Factibles")
plt.legend()
plt.show()