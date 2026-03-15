"""Shared fixtures for pyecsago tests."""

import numpy as np
import pytest

from pyecsago import (
    ECSAGOIndividual,
    ECSAGOFitnessCalculator,
    ECSAGOPopulation,
    ECSAGOStrategy,
    ECSAGOStrategyFactory,
    AdaptiveGaussianMutation,
    LinearCrossoverPerDimension,
    HAEA,
    DeterministicCrowding,
)
from pyecsago.implementations.ecsago.context import StandardAlgorithmContext
from pyecsago.utils.data_types import NumPyStrategy


@pytest.fixture
def numpy_strategy():
    """Create a NumPy data type strategy."""
    return NumPyStrategy()


@pytest.fixture
def cpu_context():
    """Create a CPU context with simple 2D test data (3 Gaussian blobs)."""
    ctx = StandardAlgorithmContext(use_cuda=False)
    np.random.seed(42)
    blob1 = np.random.randn(30, 2) + np.array([0, 0])
    blob2 = np.random.randn(30, 2) + np.array([5, 5])
    blob3 = np.random.randn(30, 2) + np.array([10, 0])
    data = np.vstack([blob1, blob2, blob3])
    ctx.set_data(data)
    return ctx


@pytest.fixture
def fitness_calculator(cpu_context):
    """Create a CPU fitness calculator."""
    return ECSAGOFitnessCalculator(dtype_strategy=cpu_context.dtype_strategy)


@pytest.fixture
def operator_rates():
    """Default operator rates for test individuals."""
    return {
        'adaptive_gaussian_mutation': 0.5,
        'linear_crossover_per_dimension': 0.5,
    }


@pytest.fixture
def sample_individual(cpu_context, fitness_calculator, operator_rates):
    """Create a sample individual at the origin."""
    return ECSAGOIndividual(
        genome=np.array([0.0, 0.0]),
        fitness_calculator=fitness_calculator,
        operator_rates=operator_rates,
        context=cpu_context,
    )


@pytest.fixture
def sample_population(cpu_context, fitness_calculator, operator_rates):
    """Create a small population of 6 individuals with computed fitness."""
    individuals = []
    positions = [[0, 0], [5, 5], [10, 0], [1, 1], [6, 6], [9, 1]]
    for pos in positions:
        ind = ECSAGOIndividual(
            genome=np.array(pos, dtype=np.float64),
            fitness_calculator=fitness_calculator,
            operator_rates=operator_rates.copy(),
            context=cpu_context,
        )
        ind.calculate_fitness(weight_threshold=0.3)
        individuals.append(ind)
    return individuals
