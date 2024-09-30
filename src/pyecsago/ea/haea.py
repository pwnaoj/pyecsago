import numpy as np

from pyecsago.interface.base import OperadoresEvolutivos
from .individual import GeneraIndividuo


class HAEA(OperadoresEvolutivos):
    def __init__(self, tasa_aprendizaje=None):
        if tasa_aprendizaje is None:
            self.tasa_aprendizaje = np.random.uniform(0, 1)
        else:
            self.tasa_aprendizaje = tasa_aprendizaje

    def mutar(self, individuo, tipo_mutacion='gaussiana_adaptativa'):
        """ Aplica la mutación al individuo según el tipo especificado. """
        if tipo_mutacion == 'gaussiana':
            self._mutacion_gaussiana(individuo)
        elif tipo_mutacion == 'gaussiana_adaptativa':
            self._mutacion_gaussiana_adaptativa(individuo)
        return individuo

    def cruzar(self, padre1, padre2, tipo_cruce='LCD'):
        """ Aplica el cruce entre dos padres usando el tipo especificado. """
        if tipo_cruce == 'LC':
            return self._linear_crossover(padre1, padre2)
        elif tipo_cruce == 'LCD':
            return self._linear_crossover_per_dimension(padre1, padre2)

    def ajustar_tasas(self, individuo, recompensa=True):
        if recompensa:
            individuo.tasas_operadores *= (1.0 + self.tasa_aprendizaje)
        else:
            individuo.tasas_operadores *= (1.0 - self.tasa_aprendizaje)
        
        individuo.normalizar_tasas()
         
    def evaluar_operador(self, padre, hijo):
        return hijo.fitness > padre.fitness

    # Implementaciones de mutación gaussiana
    def _mutacion_gaussiana(self, individuo):
        for i in range(len(individuo.genoma)):
            if np.random.rand() < individuo.tasa_mutacion:
                individuo.genoma[i] += np.random.normal(0, individuo.sigma2)

    # Implementaciones de mutación gaussiana adaptativa
    def _mutacion_gaussiana_adaptativa(self, individuo):
        for i in range(len(individuo.genoma)):
            if np.random.rand() < individuo.tasa_mutacion:
                adaptacion_sigma = individuo.sigma2 * (1 + np.random.randn() * 0.1)
                individuo.genoma[i] += np.random.normal(0, adaptacion_sigma)

    # Implementaciones de cruce LC
    def _linear_crossover(self, padre1, padre2):
        """Linear Crossover entre dos padres"""
        alpha = np.random.rand()
        hijo1_genoma = alpha * padre1.genoma + (1 - alpha) * padre2.genoma
        hijo2_genoma = (1 - alpha) * padre1.genoma + alpha * padre2.genoma

        # Crear dos nuevos individuos (hijos)
        hijo1 = GeneraIndividuo(genoma=hijo1_genoma, tasa_mutacion=padre1.tasa_mutacion, tasa_cruce=padre1.tasa_cruce)
        hijo2 = GeneraIndividuo(genoma=hijo2_genoma, tasa_mutacion=padre2.tasa_mutacion, tasa_cruce=padre2.tasa_cruce)

        return hijo1, hijo2  # Devolver dos hijos

    # Implementaciones de cruce LCD
    def _linear_crossover_per_dimension(self, padre1, padre2):
        """Linear Crossover per Dimension"""
        nuevo_genoma1 = np.copy(padre1.genoma)
        nuevo_genoma2 = np.copy(padre2.genoma)
        for i in range(len(padre1.genoma)):
            alpha = np.random.rand()
            nuevo_genoma1[i] = alpha * padre1.genoma[i] + (1 - alpha) * padre2.genoma[i]
            nuevo_genoma2[i] = (1 - alpha) * padre1.genoma[i] + alpha * padre2.genoma[i]

        # Crear dos nuevos individuos (hijos)
        hijo1 = GeneraIndividuo(genoma=nuevo_genoma1, tasa_mutacion=padre1.tasa_mutacion, tasa_cruce=padre1.tasa_cruce)
        hijo2 = GeneraIndividuo(genoma=nuevo_genoma2, tasa_mutacion=padre2.tasa_mutacion, tasa_cruce=padre2.tasa_cruce)

        return hijo1, hijo2  # Devolver dos hijos
