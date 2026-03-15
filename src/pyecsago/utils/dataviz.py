from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def visualizar_resultados_(datos: np.ndarray, centros_reales: np.ndarray | None = None, centros_refinados: np.ndarray | None = None, titulo: str = "Visualización de Clustering") -> None:
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

def visualizar_resultados(datos: np.ndarray, centros_reales: np.ndarray | None = None, centros_refinados: np.ndarray | None = None, sigmas_refinados: np.ndarray | None = None, titulo: str = "Visualización de Clustering") -> None:
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
