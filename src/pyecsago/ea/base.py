import numpy as np

from abc import ABC, abstractmethod


class Individuo(ABC):
    def __init__(self, genoma, tasa_mutacion=0.01, tasa_cruce=0.7, sigma2=1.0):
        self.genoma = np.array(genoma)
        self.fitness = 0
        self.tasa_mutacion = tasa_mutacion
        self.tasa_cruce = tasa_cruce
        self.sigma2 = np.array(sigma2)
        self.tasas_operadores = np.array([self.tasa_cruce, self.tasa_mutacion])
        self.normalizar_tasas()

    @abstractmethod
    def calcular_fitness(self, datos):
        pass

    def normalizar_tasas(self):
        suma = np.sum(self.tasas_operadores)
        if suma > 0:
            self.tasas_operadores /= suma

class Poblacion(ABC):
    def __init__(self, num_individuos, individuo_class, *args, **kwargs):
        self.individuos = [individuo_class(*args, **kwargs) for _ in range(num_individuos)]
        self.generaciones = 0

    @abstractmethod
    def evolucionar(self, num_generaciones):
        pass

class OperadoresEvolutivos(ABC):
    @abstractmethod
    def mutar(self, individuo):
        """Aplica la mutación al individuo."""
        pass

    @abstractmethod
    def cruzar(self, padre1, padre2):
        """Aplica el cruce entre dos individuos."""
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
        """Aplica la estrategia de reemplazo, en este caso Deterministic Crowding."""
        pass
