import numpy as np

def leer_datos(archivo_port: str, archivo_portef: str):
    """
    Lee los archivos de datos de portafolio y frontera eficiente.
    
    Args:
        archivo_port: ruta del archivo con medias, desviaciones y correlación.
        archivo_portef: ruta del archivo con la frontera eficiente (retornos y varianzas).
    
    Returns:
        medias: array de retornos esperados (N,)
        covarianza: matriz de covarianza (N,N)
        frontera_eficiente: array de la frontera eficiente (M,2)
    """
    with open(archivo_port, "r") as f:
        # Número de activos
        N = int(f.readline().strip())

        # Inicializamos vectores
        medias = np.zeros(N)
        desviaciones = np.zeros(N)

        # Leemos medias y desviaciones
        for i in range(N):
            linea = f.readline().strip()
            medias[i], desviaciones[i] = map(float, linea.split())

        # Inicializamos matriz de correlación
        correlacion = np.eye(N)

        # Leemos los pares i, j, rho
        for linea in f:
            i, j, rho = linea.strip().split()
            i, j = int(i)-1, int(j)-1  # convertir a índice 0-based
            rho = float(rho)
            correlacion[i, j] = rho
            correlacion[j, i] = rho  # simétrica

    # Calculamos la matriz de covarianza
    D = np.diag(desviaciones)
    covarianza = D @ correlacion @ D

    # Cargamos la frontera eficiente
    frontera_eficiente = np.loadtxt(archivo_portef)

    return medias, covarianza, frontera_eficiente


if __name__ == "__main__":
    archivo_port = "port1.txt"
    archivo_portef = "portef1.txt"

    retornos, matriz_cov, frontera_ef = leer_datos(archivo_port, archivo_portef)

    print("Medias de retorno:")
    print(retornos)
    print("\nMatriz de covarianza:")
    print(matriz_cov)
    print("\nFrontera eficiente:")
    print(frontera_ef)