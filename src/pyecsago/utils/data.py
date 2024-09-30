import numpy as np


def generar_datos_sinteticos(num_clusters=10, puntos_por_cluster=50, dimensiones=2, dispersión=0.05, semilla=None):
    """
    Genera datos sintéticos de manera dinámica con varios clusters.
    
    Parámetros:
    - num_clusters (int): Número de clusters a generar.
    - puntos_por_cluster (int): Cantidad de puntos por cluster.
    - dimensiones (int): Dimensiones del espacio (por defecto 2D).
    - dispersión (float): Dispersión de los puntos alrededor del centro del cluster.
    - semilla (int): Semilla para la generación aleatoria (opcional para reproducibilidad).
    
    Retorna:
    - datos (np.array): Puntos de datos generados.
    - centros_reales (np.array): Centros reales de los clusters generados.
    """
    if semilla is not None:
        np.random.seed(semilla)

    # Generar los centros de los clusters aleatoriamente
    centros_reales = np.random.rand(num_clusters, dimensiones)

    # Generar puntos alrededor de cada centro
    datos = []
    for centro in centros_reales:
        puntos_cluster = np.random.normal(loc=centro, scale=dispersión, size=(puntos_por_cluster, dimensiones))
        datos.append(puntos_cluster)
    
    # Combinar todos los puntos generados
    datos = np.vstack(datos)
    
    return datos, centros_reales
