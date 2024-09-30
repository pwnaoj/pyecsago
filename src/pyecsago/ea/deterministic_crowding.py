import numpy as np

from pyecsago.interface.base import NichingStrategy


class DeterministicCrowding(NichingStrategy):
    def seleccionar_padres(self, individuos):
        """Seleccionar padres aleatoriamente."""
        return np.random.choice(individuos, 2, replace=False)

    def reemplazar(self, padres, hijos):
        """Aplicar la estrategia de niching de deterministic crowding."""
        poblacion_final = []
        for i in range(2):
            padre = padres[i]
            hijo = hijos[i]

            # Comparar similitud genética y fitness
            if np.linalg.norm(hijo.genoma - padre.genoma) < np.linalg.norm(hijo.genoma - padres[1-i].genoma):
                if hijo.fitness > padre.fitness:
                    poblacion_final.append(hijo)
                else:
                    poblacion_final.append(padre)
            else:
                if hijo.fitness > padres[1-i].fitness:
                    poblacion_final.append(hijo)
                else:
                    poblacion_final.append(padres[1-i])

        return poblacion_final
    
    def reemplazar_(self, padres, hijos):
        """Aplicar la estrategia de niching de deterministic crowding."""
        poblacion_final = []
        for i in range(2):  # Asumiendo dos hijos y dos padres
            hijo = hijos[i]
            padre = padres[i]
            otro_padre = padres[1 - i]
            
            # Distancias genéticas
            d_hijo_padre = np.linalg.norm(hijo.genoma - padre.genoma)
            d_hijo_otro_padre = np.linalg.norm(hijo.genoma - otro_padre.genoma)

            # Determinar el padre más cercano
            if d_hijo_padre <= d_hijo_otro_padre:
                mejor_padre = padre
            else:
                mejor_padre = otro_padre

            # Comparar fitness
            if hijo.fitness > mejor_padre.fitness:
                poblacion_final.append(hijo)
            else:
                poblacion_final.append(mejor_padre)

        return poblacion_final
    
    def reemplazar__(self, padres, hijos):
        """Aplicar la estrategia de niching de deterministic crowding."""
        poblacion_final = []

        # Cálculo de distancias cruzadas
        d1 = np.linalg.norm(hijos[0].genoma - padres[0].genoma) + np.linalg.norm(hijos[1].genoma - padres[1].genoma)
        d2 = np.linalg.norm(hijos[0].genoma - padres[1].genoma) + np.linalg.norm(hijos[1].genoma - padres[0].genoma)

        # Determinar el emparejamiento de hijos a padres basado en menor distancia cruzada
        if d1 <= d2:
            emparejamientos = [(hijos[0], padres[0]), (hijos[1], padres[1])]
        else:
            emparejamientos = [(hijos[0], padres[1]), (hijos[1], padres[0])]

        # Reemplazar padres con hijos si el hijo es más apto
        for hijo, padre in emparejamientos:
            if hijo.fitness > padre.fitness:
                poblacion_final.append(hijo)
            else:
                poblacion_final.append(padre)

        return poblacion_final
