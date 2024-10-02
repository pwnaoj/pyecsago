import numpy as np

from abc import ABC, abstractmethod


class Individuo(ABC):
    def __init__(self, genoma, sigma2=1.0, tasas_operadores=None):
        """Inicializa un individuo con genoma, fitness, sigma2 y tasas de operadores genéticos. Además normaliza las tasas."""
        self.genoma = np.array(genoma)
        self.fitness = 0
        self.sigma2 = np.array(sigma2)
        if tasas_operadores is None: self.tasas_operadores = {'mutacion_gaussiana': 0.3, 'mutacion_gaussiana_adaptativa': 0.3, 'cruce_lc': 0.2, 'cruce_lcd': 0.2} 
        else: self.tasas_operadores = tasas_operadores
        self.normalizar_tasas()

    @abstractmethod
    def calcular_fitness(self, datos):
        """Calcula el fitness del individuo."""
        pass

    def normalizar_tasas(self):
        """Normaliza las tasas de los operadores genéticos."""
        suma = sum(self.tasas_operadores.values())
        if suma > 0:
            for operador in self.tasas_operadores:
                self.tasas_operadores[operador] /= suma

class Poblacion(ABC):
    def __init__(self, num_individuos, individuo_class, *args, **kwargs):
        """Inicializa la población con una lista de individuos y el número de generaciones a evolucionar."""
        self.individuos = [individuo_class(*args, **kwargs) for _ in range(num_individuos)]
        self.generaciones = 0

    @abstractmethod
    def evolucionar(self, num_generaciones):
        """Inicia proceso evolutivo."""
        pass

class OperadoresEvolutivos(ABC):
    @abstractmethod
    def seleccionar_operador(self, individuo):
        """Selecciona un operador basado en las tasas del individuo."""
        pass

    @abstractmethod
    def aplicar_operador(self, padre1, padre2):
        """Aplica el operador seleccionado al individuo."""
        pass

    @abstractmethod
    def ajustar_tasas(self, individuo, recompensa=True):
        """Ajusta las tasas de los operadores basándose en su éxito."""
        pass

class NichingStrategy(ABC):
    @abstractmethod
    def seleccionar_padres(self, individuos):
        """Selecciona padres para reproducción."""
        pass

    @abstractmethod
    def reemplazar(self, padres, hijos):
        """Aplica la estrategia de reemplazo."""
        pass
