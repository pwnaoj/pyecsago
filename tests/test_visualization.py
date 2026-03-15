"""Tests for visualization utilities (matplotlib-based, non-interactive)."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import pytest

from pyecsago.utils.funcs import visualize_results_simple, visualize_results
from pyecsago.utils.dataviz import visualizar_resultados_, visualizar_resultados


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')


class TestVisualizeResultsSimple:
    def test_data_only(self):
        data = np.random.randn(50, 2)
        visualize_results_simple(data)

    def test_with_real_centers(self):
        data = np.random.randn(50, 2)
        centers = np.array([[0, 0], [1, 1]])
        visualize_results_simple(data, real_centers=centers)

    def test_with_refined_centers(self):
        data = np.random.randn(50, 2)
        refined = np.array([[0.1, 0.1], [0.9, 0.9]])
        visualize_results_simple(data, refined_centers=refined)

    def test_with_all(self):
        data = np.random.randn(50, 2)
        real = np.array([[0, 0], [1, 1]])
        refined = np.array([[0.1, 0.1], [0.9, 0.9]])
        visualize_results_simple(data, real_centers=real, refined_centers=refined, title="Test")


class TestVisualizeResultsWithRealAndRefined:
    def test_real_without_refined(self):
        data = np.random.randn(50, 2)
        real = np.array([[0, 0]])
        visualize_results(data, real_centers=real)


class TestVisualizeResults:
    def test_data_only(self):
        data = np.random.randn(50, 2)
        visualize_results(data)

    def test_with_sigmas(self):
        data = np.random.randn(50, 2)
        refined = np.array([[0.0, 0.0], [1.0, 1.0]])
        sigmas = np.array([0.5, 0.3])
        visualize_results(data, refined_centers=refined, refined_sigmas=sigmas)

    def test_without_sigmas(self):
        data = np.random.randn(50, 2)
        refined = np.array([[0.0, 0.0]])
        visualize_results(data, refined_centers=refined)


class TestVisualizarResultados:
    def test_simple(self):
        data = np.random.randn(50, 2)
        visualizar_resultados_(data)

    def test_with_centers(self):
        data = np.random.randn(50, 2)
        centros = np.array([[0, 0]])
        visualizar_resultados_(data, centros_reales=centros, centros_refinados=centros)

    def test_extended(self):
        data = np.random.randn(50, 2)
        centros = np.array([[0, 0]])
        sigmas = np.array([0.5])
        visualizar_resultados(data, centros_refinados=centros, sigmas_refinados=sigmas)

    def test_extended_no_sigmas(self):
        data = np.random.randn(50, 2)
        centros = np.array([[0, 0]])
        visualizar_resultados(data, centros_reales=centros, centros_refinados=centros)
