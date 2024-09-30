from pyecsago.ea.deterministic_crowding import DeterministicCrowding
from pyecsago.ea.haea import HAEA
from pyecsago.ea.individual import GeneraIndividuo
from pyecsago.ea.population import GeneraPoblacion
from pyecsago.utils.data import generar_datos_sinteticos


# Generar datos sintéticos
datos_sinteticos, centros_reales = generar_datos_sinteticos(num_clusters=5, puntos_por_cluster=50, dimensiones=2, semilla=42)

# Inicializar la población
poblacion = GeneraPoblacion(
    num_individuos=30, 
    individuo_class=GeneraIndividuo, 
    niching_strategy=DeterministicCrowding(), 
    operadores_strategy=HAEA(),
    datos=datos_sinteticos,  
    dimensiones=2,  # Definir el número de dimensiones del genoma
    weight_threshold=0.3,
    tasa_mutacion=0.1, 
    tasa_cruce=0.7, 
    sigma2=0.05
)

# Evaluar el fitness inicial de la población
poblacion.evaluar_fitness_poblacion()

# Evolucionar la población durante varias generaciones
poblacion.evolucionar(num_generaciones=10)

# Extraer y refinar los prototipos
prototipos_refinados = poblacion.extraer_y_refinar_prototipos(      
    umbral_fitness=0.8, # Ajustar a un valor más bajo si es necesario
    kmin=0.1,
    iteraciones=10
)

# Mostrar la visualización de los prototipos refinados junto con los datos y los centros reales
poblacion.mostrar_visualizacion(centros_reales=centros_reales, prototipos_refinados=prototipos_refinados)
