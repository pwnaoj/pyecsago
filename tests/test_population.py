"""Tests for ECSAGOPopulation methods."""

import numpy as np
import pytest

from pyecsago import (
    ECSAGOPopulation,
    ECSAGOStrategyFactory,
    ECSAGOFitnessCalculator,
    AdaptiveGaussianMutation,
    LinearCrossoverPerDimension,
)


@pytest.fixture
def evolution_strategy(cpu_context):
    factory = ECSAGOStrategyFactory()
    config = {
        'weight_threshold': 0.3,
        'mutation_operators': {'adaptive_gaussian_mutation': AdaptiveGaussianMutation},
        'crossover_operators': {'linear_crossover_per_dimension': LinearCrossoverPerDimension},
        'fitness_calculator': ECSAGOFitnessCalculator,
        'context': cpu_context,
    }
    return factory.create_strategy(config)


@pytest.fixture
def population(cpu_context, evolution_strategy):
    np.random.seed(42)
    return ECSAGOPopulation(size=6, context=cpu_context, evolution_strategy=evolution_strategy)


class TestECSAGOPopulation:
    def test_evaluate_population(self, population):
        population.evaluate_population()
        for ind in population.individuals:
            assert ind.fitness is not None

    def test_extract_prototypes(self, population):
        population.evaluate_population()
        protos = population.extract_prototypes(extraction_type={0: 0})
        assert isinstance(protos, list)

    def test_extract_prototypes_default(self, population):
        """extract_prototypes() without args uses default {0: 0}."""
        population.evaluate_population()
        protos = population.extract_prototypes()
        assert isinstance(protos, list)

    def test_extract_prototypes_with_type(self, population):
        population.evaluate_population()
        protos = population.extract_prototypes(extraction_type={2: 0.25})
        assert isinstance(protos, list)

    def test_refine_empty_prototypes(self, population):
        result = population.refine_prototypes(prototypes=[])
        assert result == []

    def test_refine_prototypes(self, population):
        population.evaluate_population()
        protos = population.extract_prototypes(extraction_type={0: 0})
        if protos:
            refined = population.refine_prototypes(protos, iterations=2)
            assert isinstance(refined, list)

    def test_evolve(self, population):
        population.evaluate_population()
        result = population.evolve()
        assert len(result) == len(population.individuals)
        assert population.generation == 1

    def test_evolve_covers_reward_and_punish(self, population):
        """Evolving covers both reward (offspring > parent) and punish paths.

        With the fix >= to > (HAEA Algorithm 1, Gómez 2004), the punish path
        fires when best_child.fitness <= individual.fitness, which includes
        the common case where _select_best_offspring returns the parent itself.
        """
        population.evaluate_population()
        for _ in range(3):
            population.evolve()
        assert population.generation == 3
