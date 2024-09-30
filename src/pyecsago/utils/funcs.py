import matplotlib.pyplot as plt
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

def visualizar_resultados_(datos, centros_reales=None, centros_refinados=None, titulo="Visualización de Clustering"):
    """Visualiza los datos, los centros reales (si están disponibles) y los centros refinados."""
    
    # Dibujar los datos
    plt.scatter(datos[:, 0], datos[:, 1], c='lightblue', label='Datos')
    
    # Dibujar los centros reales si están disponibles
    if centros_reales is not None:
        plt.scatter(centros_reales[:, 0], centros_reales[:, 1], c='green', marker='x', label='Centros Reales', s=100)
    
    # Dibujar los centros refinados si están disponibles
    if centros_refinados is not None:
        plt.scatter(centros_refinados[:, 0], centros_refinados[:, 1], c='red', marker='o', label='Centros Refinados', s=100)
    
    # Añadir leyenda y título
    plt.legend()
    plt.title(titulo)
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    
    # Mostrar gráfico
    plt.show()

def visualizar_resultados(datos, centros_reales=None, centros_refinados=None, sigmas_refinados=None, titulo="Visualización de Clustering"):
    """Visualiza los datos, los centros reales (si están disponibles) y los centros refinados, incluyendo los radios."""
    # Dibujar los datos
    plt.scatter(datos[:, 0], datos[:, 1], c='lightblue', label='Datos')
    
    # Dibujar los centros reales si están disponibles
    if centros_reales is not None:
        plt.scatter(centros_reales[:, 0], centros_reales[:, 1], c='green', marker='x', label='Centros Reales', s=100)

    # Dibujar los centros refinados si están disponibles
    if centros_refinados is not None:
        plt.scatter(centros_refinados[:, 0], centros_refinados[:, 1], c='red', marker='o', label='Centros Refinados', s=100)
        # Dibujar círculos que representan el radio de cada centro refinado
        if sigmas_refinados is not None:
            for centro, sigma in zip(centros_refinados, sigmas_refinados):
                circle = plt.Circle(centro, sigma, color='red', fill=False, linestyle='--', linewidth=1.5)
                plt.gca().add_patch(circle)

    # Añadir leyenda y título
    plt.legend()
    plt.title(titulo)
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    
    # Mostrar gráfico
    plt.show()
