"""Tests for pyecsago.utils.data and pyecsago.utils.funcs."""

import numpy as np

from pyecsago.utils.data import generar_datos_sinteticos
from pyecsago.utils.funcs import generate_synthetic_data


class TestGenerarDatosSinteticos:
    def test_default_parameters(self):
        data, centers = generar_datos_sinteticos(semilla=42)
        assert data.shape == (500, 2)  # 10 clusters * 50 points
        assert centers.shape == (10, 2)

    def test_custom_parameters(self):
        data, centers = generar_datos_sinteticos(
            num_clusters=3, puntos_por_cluster=20, dimensiones=4, semilla=0
        )
        assert data.shape == (60, 4)
        assert centers.shape == (3, 4)

    def test_reproducibility(self):
        d1, c1 = generar_datos_sinteticos(semilla=42)
        d2, c2 = generar_datos_sinteticos(semilla=42)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(c1, c2)

    def test_without_seed(self):
        data, centers = generar_datos_sinteticos(num_clusters=2, puntos_por_cluster=10)
        assert data.shape == (20, 2)
        assert centers.shape == (2, 2)


class TestGenerateSyntheticData:
    def test_default_parameters(self):
        data, centers = generate_synthetic_data(seed=42)
        assert data.shape == (500, 2)
        assert centers.shape == (10, 2)

    def test_custom_parameters(self):
        data, centers = generate_synthetic_data(
            num_clusters=5, points_per_cluster=30, dimensions=3, seed=0
        )
        assert data.shape == (150, 3)
        assert centers.shape == (5, 3)

    def test_reproducibility(self):
        d1, c1 = generate_synthetic_data(seed=7)
        d2, c2 = generate_synthetic_data(seed=7)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(c1, c2)

    def test_without_seed(self):
        data, centers = generate_synthetic_data(num_clusters=2, points_per_cluster=10)
        assert data.shape == (20, 2)
