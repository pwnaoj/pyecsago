import numpy as np

from pyecsago.interface.base import Poblacion
from pyecsago.utils.funcs import visualizar_resultados


class GeneraPoblacion(Poblacion):
    def __init__(self, num_individuos, individuo_class, niching_strategy, operadores_strategy, datos, dimensiones, weight_threshold, *args, **kwargs):
        """
        Inicializa la población concreta con una estrategia de niching, operadores y datos.
        :param num_individuos: Número de individuos en la población
        :param individuo_class: Clase de los individuos
        :param niching_strategy: Estrategia de niching (Deterministic Crowding)
        :param operadores_strategy: Estrategia de operadores evolutivos (HAEA)
        :param datos: Datos con los que se trabajará
        :param dimensiones: Dimensiones del genoma
        """
        # Inicializar la población con individuos, generando un genoma aleatorio para cada uno
        self.individuos = [individuo_class(genoma=np.random.rand(dimensiones), *args, **kwargs) for _ in range(num_individuos)]
        
        # Guardar otros parámetros
        self.niching_strategy = niching_strategy
        self.operadores_strategy = operadores_strategy
        self.datos = datos
        self.generaciones = 0
        self.weight_threshold = weight_threshold

    def evolucionar(self, num_generaciones):
        """Evoluciona la población durante varias generaciones aplicando niching y operadores evolutivos."""
        for _ in range(num_generaciones):
            nuevos_individuos = []
            for _ in range(len(self.individuos) // 2):
                # Usar la estrategia de niching para seleccionar padres
                padre1, padre2 = self.niching_strategy.seleccionar_padres(self.individuos)

                # Aplicar cruce y mutación
                hijo1, hijo2 = self.operadores_strategy.cruzar(padre1, padre2)
                self.operadores_strategy.mutar(hijo1)
                self.operadores_strategy.mutar(hijo2)

                hijo1.calcular_fitness(self.datos, self.weight_threshold)
                hijo2.calcular_fitness(self.datos, self.weight_threshold)

                # Evaluar si los operadores fueron exitosos
                recompensa1 = self.operadores_strategy.evaluar_operador(padre1, hijo1)
                recompensa2 = self.operadores_strategy.evaluar_operador(padre2, hijo2)

                self.operadores_strategy.ajustar_tasas(padre1, recompensa1)
                self.operadores_strategy.ajustar_tasas(padre2, recompensa2)

                # Usar niching strategy para reemplazar individuos
                nuevos_individuos += self.niching_strategy.reemplazar([padre1, padre2], [hijo1, hijo2])

            self.individuos = nuevos_individuos
            self.generaciones += 1

    def evaluar_fitness_poblacion(self):
        """Evalúa el fitness de cada individuo en la población"""
        for individuo in self.individuos:
            individuo.calcular_fitness(datos=self.datos, weight_threshold=self.weight_threshold)

    def extraer_prototipos(self, umbral_fitness, kmin):
        """
        Selecciona los individuos con mejor fitness que superen el umbral y cuya distancia genética supere kmin.
        :param umbral_fitness: Umbral mínimo de fitness para considerar a un individuo como prototipo.
        :param kmin: Umbral mínimo de distancia genética para garantizar diversidad genética entre los prototipos.
        :return: Lista de prototipos (individuos seleccionados).
        """
        # Filtrar individuos que cumplen con el umbral de fitness
        candidatos = [individuo for individuo in self.individuos if individuo.fitness >= umbral_fitness]
        
        # Ordenar los candidatos por fitness de mayor a menor
        candidatos.sort(key=lambda ind: ind.fitness, reverse=True)
        
        prototipos = []
        
        # Seleccionar los prototipos asegurando que cumplan con la distancia genética mínima (kmin)
        for candidato in candidatos:
            # Verificar que la distancia genética entre el candidato y los prototipos seleccionados sea mayor que kmin * min(σ²)
            if all(np.linalg.norm(candidato.genoma - prototipo.genoma) > kmin * min(np.min(candidato.sigma2), np.min(prototipo.sigma2)) for prototipo in prototipos):
                prototipos.append(candidato)

        return prototipos

    def refinar_prototipos(self, prototipos, iteraciones=10, kmin=0.05):
        """
        Refinar los prototipos utilizando Maximal Density Estimator (MDE), asegurando que la distancia genética mínima
        y la dispersión genética se respeten.
        :param prototipos: Lista de individuos que representan los prototipos.
        :param iteraciones: Número de iteraciones para refinar los prototipos.
        :param kmin: Umbral mínimo de distancia genética entre prototipos.
        :return: Prototipos refinados.
        """
        for _ in range(iteraciones):
            # Inicializar clusters vacíos para cada prototipo
            clusters = {i: [] for i in range(len(prototipos))}

            # Asignar cada punto de datos al prototipo más cercano
            for punto in self.datos:
                distancias = [np.linalg.norm(punto - prototipo.genoma) for prototipo in prototipos]
                if len(distancias) > 0:
                    cluster_idx = np.argmin(distancias)  # Encontrar el prototipo más cercano
                    clusters[cluster_idx].append(punto)

            # Ajustar la posición de cada prototipo basado en los puntos asignados a su cluster
            for i, prototipo in enumerate(prototipos):
                if clusters[i]:  # Si el cluster no está vacío
                    # Calcular la nueva posición del prototipo como la media ponderada (MDE)
                    puntos_cluster = np.array(clusters[i])
                    distancias_puntos = np.linalg.norm(puntos_cluster - prototipo.genoma, axis=1)
                    
                    # Pesos inversamente proporcionales a las distancias (opcional)
                    w_ij = 1 / (distancias_puntos + 1e-6)
                    nuevo_centro = np.average(puntos_cluster, axis=0, weights=w_ij)
                    
                    # Calcular nueva σ² según la fórmula MDE
                    num_sigma = np.sum(w_ij * (distancias_puntos ** 4))
                    denom_sigma = 3 * np.sum(w_ij * (distancias_puntos ** 2))
                    nueva_sigma = num_sigma / denom_sigma if denom_sigma != 0 else prototipo.sigma2
                    
                    # Verificar la distancia genética mínima (kmin) antes de actualizar el prototipo
                    if all(np.linalg.norm(nuevo_centro - otro_prototipo.genoma) > kmin for j, otro_prototipo in enumerate(prototipos) if j != i):
                        prototipo.genoma = nuevo_centro  # Actualizar la posición del prototipo
                        prototipo.sigma2 = nueva_sigma  # Actualizar la dispersión genética

        return prototipos

    def extraer_y_refinar_prototipos(self, umbral_fitness, kmin, iteraciones=10):
        """
        Realiza la extracción y refinamiento de prototipos.
        :param umbral_fitness: Umbral mínimo de fitness para la selección de prototipos.
        :param kmin: Distancia genética mínima para garantizar diversidad entre prototipos.
        :param iteraciones: Número de iteraciones para refinar los prototipos.
        :return: Prototipos refinados.
        """
        # Fase de extracción de prototipos
        prototipos = self.extraer_prototipos(umbral_fitness, kmin)

        # Fase de refinamiento de prototipos usando MDE
        prototipos_refinados = self.refinar_prototipos(prototipos, iteraciones, kmin)
        
        return prototipos_refinados

    def mostrar_visualizacion(self, centros_reales=None, prototipos_refinados=None):
        """Muestra la visualización de los resultados."""
        # Extraer los centros refinados (prototipos) después de la evolución
        centros_refinados = np.array([individuo.genoma for individuo in prototipos_refinados])
        sigmas_refinados = np.array([individuo.sigma2 for individuo in prototipos_refinados])
        
        # Visualizar los resultados
        visualizar_resultados(self.datos, centros_reales=centros_reales, centros_refinados=centros_refinados, sigmas_refinados=sigmas_refinados)
