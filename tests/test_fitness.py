"""Tests for pyecsago.strategies.fitness.ecsago — distance metrics and edge cases."""

import numpy as np
import pytest

from pyecsago import ECSAGOFitnessCalculator, ECSAGOIndividual
from pyecsago.utils.data_types import NumPyStrategy


class TestFitnessDistanceMetrics:
    @pytest.fixture
    def calc(self):
        return ECSAGOFitnessCalculator(dtype_strategy=NumPyStrategy())

    def test_minkowski_distance(self, calc):
        genome = np.array([0.0, 0.0])
        data = np.array([[3.0, 4.0], [1.0, 0.0]])
        distances = calc._compute_distances(genome, data, metric='minkowski', p_minkowski=2)
        assert distances[0] == pytest.approx(5.0, rel=1e-5)
        assert distances[1] == pytest.approx(1.0, rel=1e-5)

    def test_cosine_distance(self, calc):
        genome = np.array([1.0, 0.0])
        data = np.array([[0.0, 1.0], [1.0, 0.0]])
        distances = calc._compute_distances(genome, data, metric='cosine')
        assert distances[0] == pytest.approx(1.0, rel=1e-5)  # orthogonal
        assert distances[1] == pytest.approx(0.0, abs=1e-10)   # identical direction

    def test_jaccard_distance(self, calc):
        genome = np.array([1.0, 0.0, 1.0])
        data = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        distances = calc._compute_distances(genome, data, metric='jaccard')
        assert distances[0] == pytest.approx(0.0, abs=1e-10)  # identical

    def test_unsupported_metric_raises(self, calc):
        genome = np.array([0.0, 0.0])
        data = np.array([[1.0, 1.0]])
        with pytest.raises(ValueError, match="Unsupported metric"):
            calc._compute_distances(genome, data, metric='manhattan')

    def test_update_scale_zero_weights(self, calc):
        result = calc._update_scale(np.float64(0.0), np.float64(0.0))
        assert float(result) == 0.0

    def test_update_scale_nonzero(self, calc):
        result = calc._update_scale(np.float64(10.0), np.float64(50.0))
        assert float(result) == pytest.approx(5.0)

    def test_calculate_with_minkowski(self, cpu_context, fitness_calculator, operator_rates):
        ind = ECSAGOIndividual(
            genome=np.array([0.0, 0.0]),
            fitness_calculator=fitness_calculator,
            operator_rates=operator_rates.copy(),
            context=cpu_context,
        )
        data = cpu_context.get_data()
        result = fitness_calculator.calculate(ind, data, weight_threshold=0.3, metric='minkowski', p_minkowski=3)
        assert result is not None

    def test_select_best_offspring_with_better_offspring(self, cpu_context, fitness_calculator, operator_rates):
        parent = ECSAGOIndividual(
            genome=np.array([0.0, 0.0]),
            fitness_calculator=fitness_calculator,
            operator_rates=operator_rates.copy(),
            context=cpu_context,
        )
        parent.fitness = np.float64(5.0)

        child = ECSAGOIndividual(
            genome=np.array([0.1, 0.1]),
            fitness_calculator=fitness_calculator,
            operator_rates=operator_rates.copy(),
            context=cpu_context,
        )
        child.fitness = np.float64(10.0)

        best = fitness_calculator._select_best_offspring(parent, [child])
        assert best is child

    def test_select_best_offspring_parent_wins(self, cpu_context, fitness_calculator, operator_rates):
        parent = ECSAGOIndividual(
            genome=np.array([0.0, 0.0]),
            fitness_calculator=fitness_calculator,
            operator_rates=operator_rates.copy(),
            context=cpu_context,
        )
        parent.fitness = np.float64(10.0)

        child = ECSAGOIndividual(
            genome=np.array([0.1, 0.1]),
            fitness_calculator=fitness_calculator,
            operator_rates=operator_rates.copy(),
            context=cpu_context,
        )
        child.fitness = np.float64(3.0)

        best = fitness_calculator._select_best_offspring(parent, [child])
        assert best is parent

    def test_select_best_offspring_identical_genomes(self, cpu_context, fitness_calculator, operator_rates):
        parent = ECSAGOIndividual(
            genome=np.array([0.0, 0.0]),
            fitness_calculator=fitness_calculator,
            operator_rates=operator_rates.copy(),
            context=cpu_context,
        )
        parent.fitness = np.float64(5.0)

        # offspring with identical genome
        child = ECSAGOIndividual(
            genome=np.array([0.0, 0.0]),
            fitness_calculator=fitness_calculator,
            operator_rates=operator_rates.copy(),
            context=cpu_context,
        )
        child.fitness = np.float64(10.0)

        best = fitness_calculator._select_best_offspring(parent, [child])
        assert best is parent  # all identical => returns parent
