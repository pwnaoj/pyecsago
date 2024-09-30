import unittest
import numpy as np

from pyecsago.ea.ecsago import IndividuoConcreto, PoblacionConcreta, HAEA, DeterministicCrowding, generar_datos_sinteticos

class TestECSAGO(unittest.TestCase):

    def setUp(self):
        # Generar datos sintéticos
        self.datos_sinteticos, self.centros_reales = generar_datos_sinteticos(num_clusters=5, puntos_por_cluster=50, dimensiones=2, semilla=42)
        self.poblacion = PoblacionConcreta(
            num_individuos=30,
            individuo_class=IndividuoConcreto,
            niching_strategy=DeterministicCrowding(),
            operadores_strategy=HAEA(),
            datos=self.datos_sinteticos,
            dimensiones=2,
            weight_treshold=0.3,
            tasa_mutacion=0.1,
            tasa_cruce=0.7,
            sigma2=0.05
        )

    def test_calculo_fitness(self):
        # Verificar que el fitness se calcula correctamente para un individuo
        individuo = self.poblacion.individuos[0]
        fitness = individuo.calcular_fitness(self.datos_sinteticos, 0.3)
        self.assertGreater(fitness, 0, "El fitness debería ser mayor que 0")

    def test_generacion_descendencia(self):
        # Verificar la correcta generación de descendencia usando cruce y mutación
        padre1, padre2 = self.poblacion.niching_strategy.seleccionar_padres(self.poblacion.individuos)
        hijo1, hijo2 = self.poblacion.operadores_strategy.cruzar(padre1, padre2)
        self.poblacion.operadores_strategy.mutar(hijo1)
        self.poblacion.operadores_strategy.mutar(hijo2)
        
        self.assertIsInstance(hijo1, IndividuoConcreto, "Hijo1 debería ser un IndividuoConcreto")
        self.assertIsInstance(hijo2, IndividuoConcreto, "Hijo2 debería ser un IndividuoConcreto")

    def test_evolucion_poblacion(self):
        # Verificar la evolución de la población
        generaciones_antes = self.poblacion.generaciones
        self.poblacion.evolucionar(5)
        generaciones_despues = self.poblacion.generaciones
        
        self.assertEqual(generaciones_despues, generaciones_antes + 5, "El número de generaciones debería haber incrementado en 5")

    def test_extraer_prototipos(self):
        # Verificar la extracción de prototipos
        prototipos = self.poblacion.extraer_prototipos(umbral_fitness=0.3, kmin=0.1)
        self.assertGreater(len(prototipos), 0, "Deberían extraerse prototipos con un umbral de fitness de 0.5")

    def test_refinamiento_prototipos(self):
        # Verificar el refinamiento de prototipos
        prototipos = self.poblacion.extraer_prototipos(umbral_fitness=0.5, kmin=0.1)
        prototipos_refinados = self.poblacion.refinar_prototipos(prototipos, iteraciones=5, kmin=0.1)
        self.assertEqual(len(prototipos), len(prototipos_refinados), "El número de prototipos refinados debería ser el mismo que el de los extraídos")

    def test_auto_deteccion_clusters(self):
        # Verificar la detección automática del número de clusters
        prototipos_refinados = self.poblacion.extraer_y_refinar_prototipos(umbral_fitness=0.8, kmin=0.1, iteraciones=10)
        self.assertGreater(len(prototipos_refinados), 0, "Deberían detectarse prototipos refinados")

if __name__ == '__main__':
    unittest.main()
