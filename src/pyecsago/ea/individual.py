import numpy as np

from pyecsago.interface.base import Individuo
from pyecsago.utils.metrics import (
    euclidean, 
    minkowski, 
    cosine, 
    jaccard
)


class GeneraIndividuo(Individuo):
    def __init__(self, genoma, sigma2=1.0, *args, **kawargs):
        super().__init__(genoma, sigma2, *args, **kawargs)
        self.normalizar_tasas()

    def calcular_fitness(self, datos, weight_threshold, tipo_metrica='euclidiana', p_minkowski=2):
        """ Calcula el fitness usando diferentes métricas de distancia. """
        if tipo_metrica == 'euclidiana':
            distancias = np.array([euclidean(self.genoma, punto) for punto in datos])
        elif tipo_metrica == 'minkowski':
            distancias = np.array([minkowski(self.genoma, punto, p=p_minkowski) for punto in datos])
        elif tipo_metrica == 'coseno':
            distancias = np.array([cosine(self.genoma, punto) for punto in datos])
        elif tipo_metrica == 'jaccard':
            distancias = np.array([jaccard(self.genoma, punto) for punto in datos])
        
        # Calcular pesos y fitness
        distancias2 = distancias ** 2
        pesos = np.exp(-distancias2 / (2 * self.sigma2))
        pesos_bin = np.where(pesos > weight_threshold, 1, 0).astype(float)
        self.sigma2 = np.sum(pesos_bin * distancias2) / np.sum(pesos_bin) if np.sum(pesos_bin) != 0 else 1e-10
        self.fitness = np.sum(pesos_bin) / self.sigma2
        
        return self.fitness
