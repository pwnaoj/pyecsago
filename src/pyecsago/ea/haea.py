import numpy as np

from pyecsago.interface.base import OperadoresEvolutivos
from .individual import GeneraIndividuo


class HAEA(OperadoresEvolutivos):
    def __init__(self, tasa_aprendizaje=None):
        self.tasa_aprendizaje = np.random.uniform(0, 1) if tasa_aprendizaje is None else tasa_aprendizaje

    def seleccionar_operador(self, tasas_operadores):
        """ Selecciona un operador basado en las tasas del individuo """
        operadores = list(tasas_operadores.keys())
        probabilidades = list(tasas_operadores.values())
        operador_seleccionado = np.random.choice(operadores, p=probabilidades)

        return operador_seleccionado

    def aplicar_operador(self, individuo, padre2=None):
        """ Aplica el operador seleccionado al individuo """
        operador = self.seleccionar_operador(individuo.tasas_operadores)

        if operador == 'mutacion_gaussiana':
            self._mutacion_gaussiana(individuo)
        elif operador == 'mutacion_gaussiana_adaptativa':
            self._mutacion_gaussiana_adaptativa(individuo)
        elif operador == 'cruce_lc' and padre2 is not None:
            return self._linear_crossover(individuo, padre2)
        elif operador == 'cruce_lcd' and padre2 is not None:
            return self._linear_crossover_per_dimension(individuo, padre2)
        return individuo
    
    def ajustar_tasas(self, individuo, operador, recompensa=True):
        """ Ajusta las tasas del operador seleccionado, recompensando o penalizando """
        if recompensa:
            individuo.tasas_operadores[operador] *= (1.0 + self.tasa_aprendizaje)
        else:
            individuo.tasas_operadores[operador] *= (1.0 - self.tasa_aprendizaje)

        # Normalizar tasas después de ajustar
        individuo.normalizar_tasas()

    def evaluar_operador(self, padre, hijo):
        return hijo.fitness > padre.fitness
    
    # Implementación de mutación gaussiana 
    def _mutacion_gaussiana(self, individuo):
        """Aplica la mutación gaussiana clásica al individuo"""
        tasa_mutacion = individuo.tasas_operadores['mutacion_gaussiana']
        for i in range(len(individuo.genoma)):
            if np.random.rand() < tasa_mutacion:
                individuo.genoma[i] += np.random.normal(0, individuo.sigma2)

    # Implementación de mutación gaussiana adaptativa
    def _mutacion_gaussiana_adaptativa(self, individuo):
        """Aplica la mutación gaussiana adaptativa al individuo"""
        tasa_mutacion_adaptativa = individuo.tasas_operadores['mutacion_gaussiana_adaptativa']
        for i in range(len(individuo.genoma)):
            if np.random.rand() < tasa_mutacion_adaptativa:
                adaptacion_sigma = individuo.sigma2 * (1 + np.random.randn() * 0.1)
                individuo.genoma[i] += np.random.normal(0, adaptacion_sigma)

    # Implementación de cruce LC (Linear Crossover)
    def _linear_crossover(self, padre1, padre2):
        """
        Linear Crossover entre dos padres.
        Genera dos hijos combinando los genomas de los padres con un factor aleatorio alpha.
        """
        tasa_cruce_lc = padre1.tasas_operadores['cruce_lc']
        if np.random.rand() < tasa_cruce_lc:
            alpha = np.random.rand()  # Factor de mezcla aleatorio
            hijo1_genoma = alpha * padre1.genoma + (1 - alpha) * padre2.genoma
            hijo2_genoma = (1 - alpha) * padre1.genoma + alpha * padre2.genoma

            # Crear dos nuevos individuos (hijos) con el genoma cruzado y las tasas heredadas
            hijo1 = GeneraIndividuo(genoma=hijo1_genoma, tasas_operadores=padre1.tasas_operadores)
            hijo2 = GeneraIndividuo(genoma=hijo2_genoma, tasas_operadores=padre2.tasas_operadores)

            return hijo1, hijo2  # Devolver los dos hijos
        return padre1, padre2
    
    # Implementación de cruce LCD (Linear Crossover per Dimension)
    def _linear_crossover_per_dimension(self, padre1, padre2):
        """
        Linear Crossover per Dimension.
        Realiza un cruce independiente por cada dimensión del genoma, utilizando un alpha distinto para cada uno.
        """
        tasa_cruce_lcd = padre1.tasas_operadores['cruce_lcd']
        if np.random.rand() < tasa_cruce_lcd:
            nuevo_genoma1 = np.copy(padre1.genoma)
            nuevo_genoma2 = np.copy(padre2.genoma)
            
            for i in range(len(padre1.genoma)):
                alpha = np.random.rand()  # Un alpha diferente para cada dimensión
                nuevo_genoma1[i] = alpha * padre1.genoma[i] + (1 - alpha) * padre2.genoma[i]
                nuevo_genoma2[i] = (1 - alpha) * padre1.genoma[i] + alpha * padre2.genoma[i]

            # Crear dos nuevos individuos (hijos) con el genoma cruzado y las tasas heredadas
            hijo1 = GeneraIndividuo(genoma=nuevo_genoma1, tasas_operadores=padre1.tasas_operadores)
            hijo2 = GeneraIndividuo(genoma=nuevo_genoma2, tasas_operadores=padre2.tasas_operadores)

            return hijo1, hijo2  # Devolver los dos hijos
        return padre1, padre2
