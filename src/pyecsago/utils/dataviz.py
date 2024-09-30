import matplotlib.pyplot as plt


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
