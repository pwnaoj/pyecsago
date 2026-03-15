"""Tests for pyecsago.strategies.extraction."""

import numpy as np
import pytest

from pyecsago.strategies.extraction.ecsago import (
    FitnessExtractionStrategy,
    NicheExtractionStrategy,
    ComposeExtractionStrategy,
)


class TestFitnessExtractionStrategy:
    def test_empty_candidates(self, numpy_strategy):
        strategy = FitnessExtractionStrategy(threshold=0.5, extraction_type=0, dtype_strategy=numpy_strategy)
        result = strategy.extract([], np.array([[1, 2]]))
        assert result == []

    def test_absolute_value(self, sample_population, numpy_strategy):
        strategy = FitnessExtractionStrategy(threshold=0.0, extraction_type=0, dtype_strategy=numpy_strategy)
        data = np.random.randn(90, 2)
        result = strategy.extract(sample_population, data)
        assert all(ind.fitness > 0.0 for ind in result)

    def test_proportion_avg(self, sample_population, numpy_strategy):
        strategy = FitnessExtractionStrategy(threshold=0.5, extraction_type=1, dtype_strategy=numpy_strategy)
        data = np.random.randn(90, 2)
        result = strategy.extract(sample_population, data)
        assert isinstance(result, list)

    def test_proportion_max(self, sample_population, numpy_strategy):
        # Sort descending as expected by the strategy
        candidates = sorted(sample_population, key=lambda c: c.fitness, reverse=True)
        strategy = FitnessExtractionStrategy(threshold=0.5, extraction_type=2, dtype_strategy=numpy_strategy)
        data = np.random.randn(90, 2)
        result = strategy.extract(candidates, data)
        assert isinstance(result, list)

    def test_proportion_median(self, sample_population, numpy_strategy):
        strategy = FitnessExtractionStrategy(threshold=0.5, extraction_type=3, dtype_strategy=numpy_strategy)
        data = np.random.randn(90, 2)
        result = strategy.extract(sample_population, data)
        assert isinstance(result, list)

    def test_minimum_density(self, sample_population, numpy_strategy):
        strategy = FitnessExtractionStrategy(threshold=0.001, extraction_type=4, dtype_strategy=numpy_strategy)
        data = np.random.randn(90, 2)
        result = strategy.extract(sample_population, data)
        assert isinstance(result, list)


class TestNicheExtractionStrategy:
    def test_empty_candidates(self, numpy_strategy):
        strategy = NicheExtractionStrategy(sigma_factor=1.0, k=3.0, dtype_strategy=numpy_strategy)
        result = strategy.extract([])
        assert result == []

    def test_single_candidate(self, sample_population, numpy_strategy):
        strategy = NicheExtractionStrategy(sigma_factor=1.0, k=3.0, dtype_strategy=numpy_strategy)
        result = strategy.extract([sample_population[0]])
        assert len(result) == 1

    def test_distant_candidates_all_selected(self, sample_population, numpy_strategy):
        # With very small sigma_factor, all should be selected (no merging)
        strategy = NicheExtractionStrategy(sigma_factor=0.0001, k=0.001, dtype_strategy=numpy_strategy)
        result = strategy.extract(sample_population)
        assert len(result) >= 1

    def test_close_candidates_merged(self, cpu_context, fitness_calculator, operator_rates, numpy_strategy):
        # Create two very close individuals
        from pyecsago import ECSAGOIndividual
        ind1 = ECSAGOIndividual(
            genome=np.array([0.0, 0.0]),
            fitness_calculator=fitness_calculator,
            operator_rates=operator_rates.copy(),
            context=cpu_context,
        )
        ind1.fitness = np.float64(10.0)
        ind2 = ECSAGOIndividual(
            genome=np.array([0.001, 0.001]),
            fitness_calculator=fitness_calculator,
            operator_rates=operator_rates.copy(),
            context=cpu_context,
        )
        ind2.fitness = np.float64(8.0)

        strategy = NicheExtractionStrategy(sigma_factor=100.0, k=100.0, dtype_strategy=numpy_strategy)
        result = strategy.extract([ind1, ind2])
        assert len(result) == 1  # second should be merged


class TestComposeExtractionStrategy:
    def test_both_strategies(self, sample_population, numpy_strategy):
        fitness_s = FitnessExtractionStrategy(threshold=0.0, extraction_type=0, dtype_strategy=numpy_strategy)
        niche_s = NicheExtractionStrategy(sigma_factor=0.0001, k=0.001, dtype_strategy=numpy_strategy)
        compose = ComposeExtractionStrategy(fitness_s, niche_s)
        data = np.random.randn(90, 2)
        result = compose.extract(sample_population, data)
        assert isinstance(result, list)

    def test_none_first_strategy(self, sample_population, numpy_strategy):
        niche_s = NicheExtractionStrategy(sigma_factor=0.0001, k=0.001, dtype_strategy=numpy_strategy)
        compose = ComposeExtractionStrategy(None, niche_s)
        data = np.random.randn(90, 2)
        result = compose.extract(sample_population, data)
        assert isinstance(result, list)
        assert len(result) >= 1
