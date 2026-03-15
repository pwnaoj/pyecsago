"""Tests for pyecsago."""

import numpy as np
import pytest

from pyecsago import (
    ECSAGO,
    ECSAGOIndividual,
    ECSAGOFitnessCalculator,
    LinearCrossoverPerDimension,
    AdaptiveGaussianMutation,
    GaussianMutation,
    DeterministicCrowding,
    HAEA,
    ECSAGOStrategy,
    ECSAGOStrategyFactory,
    ECSAGOPopulation,
)
from pyecsago.implementations.ecsago.context import StandardAlgorithmContext
from pyecsago.core.exceptions import ConfigurationError


# --- Fixtures ---

@pytest.fixture
def cpu_context():
    """Create a CPU context with simple 2D test data."""
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
def sample_individual(cpu_context, fitness_calculator):
    """Create a sample individual at the origin."""
    operator_rates = {
        'adaptive_gaussian_mutation': 0.5,
        'linear_crossover_per_dimension': 0.5,
    }
    return ECSAGOIndividual(
        genome=np.array([0.0, 0.0]),
        fitness_calculator=fitness_calculator,
        operator_rates=operator_rates,
        context=cpu_context,
    )


@pytest.fixture
def sample_population(cpu_context, fitness_calculator):
    """Create a small population of 6 individuals."""
    operator_rates = {
        'adaptive_gaussian_mutation': 0.5,
        'linear_crossover_per_dimension': 0.5,
    }
    individuals = []
    positions = [[0, 0], [5, 5], [10, 0], [1, 1], [6, 6], [9, 1]]
    for pos in positions:
        ind = ECSAGOIndividual(
            genome=np.array(pos, dtype=np.float64),
            fitness_calculator=fitness_calculator,
            operator_rates=operator_rates.copy(),
            context=cpu_context,
        )
        individuals.append(ind)
    return individuals


# --- DataTypeStrategy Tests ---

class TestDataTypeStrategy:
    def test_numpy_strategy_array(self, cpu_context):
        arr = cpu_context.dtype_strategy.array([1.0, 2.0, 3.0])
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64

    def test_numpy_strategy_module(self, cpu_context):
        assert cpu_context.dtype_strategy.module is np

    def test_numpy_strategy_random_choice(self, cpu_context):
        result = cpu_context.dtype_strategy.random_choice([1, 2, 3], size=2, replace=True)
        assert len(result) == 2


# --- Context Tests ---

class TestContext:
    def test_context_initialization(self, cpu_context):
        assert cpu_context.use_cuda is False
        assert cpu_context.cuda_context is None

    def test_sigma_limits(self, cpu_context):
        sigma2_max = cpu_context.get_sigma2_max()
        sigma2_min = cpu_context.get_sigma2_min()
        sigma2_initial = cpu_context.get_sigma2_initial()

        assert float(sigma2_max) == pytest.approx(6.8375, rel=1e-3)
        assert float(sigma2_min) == pytest.approx(6.8375 / 100, rel=1e-3)
        assert float(sigma2_initial) == pytest.approx(6.8375 / 10, rel=1e-3)

    def test_data_retrieval(self, cpu_context):
        data = cpu_context.get_data()
        assert data.shape == (90, 2)

    def test_execute_runs_function(self, cpu_context):
        result = cpu_context.execute(lambda: 42)
        assert result == 42


# --- Individual Tests ---

class TestIndividual:
    def test_individual_creation(self, sample_individual):
        assert sample_individual.genome is not None
        assert len(sample_individual.genome) == 2
        assert sample_individual.sigma2 is not None
        assert sample_individual.operator_rates is not None

    def test_individual_sigma_limits(self, sample_individual):
        sigma2_max = sample_individual.context.get_sigma2_max()
        sigma2_min = sample_individual.context.get_sigma2_min()
        assert float(sample_individual.sigma2) >= float(sigma2_min)
        assert float(sample_individual.sigma2) <= float(sigma2_max)

    def test_operator_rates_normalized(self, sample_individual):
        rates_sum = sum(float(v) for v in sample_individual.operator_rates.values())
        assert rates_sum == pytest.approx(1.0)

    def test_clone(self, sample_individual):
        clone = sample_individual.clone()
        assert np.allclose(clone.genome, sample_individual.genome)
        assert clone is not sample_individual

    def test_get_radius(self, sample_individual):
        radius = sample_individual.get_radius(k=13.8)
        assert float(radius) > 0

    def test_calculate_fitness(self, sample_individual):
        fitness = sample_individual.calculate_fitness(weight_threshold=0.3)
        assert fitness is not None


# --- Operator Tests ---

class TestMutation:
    def test_adaptive_gaussian_mutation(self, sample_individual, cpu_context):
        op = AdaptiveGaussianMutation()
        sample_individual.sigma2 = cpu_context.dtype_strategy.array(1.0)
        mutant = op.mutate(sample_individual, cpu_context.dtype_strategy)

        assert mutant is not sample_individual
        assert len(mutant.genome) == len(sample_individual.genome)
        assert not np.allclose(mutant.genome, sample_individual.genome)

    def test_gaussian_mutation(self, sample_individual, cpu_context):
        op = GaussianMutation()
        sample_individual.sigma2 = cpu_context.dtype_strategy.array(1.0)
        mutant = op.mutate(sample_individual, cpu_context.dtype_strategy)

        assert mutant is not sample_individual
        assert not np.allclose(mutant.genome, sample_individual.genome)


class TestCrossover:
    def test_linear_crossover_produces_two_children(self, sample_population, cpu_context):
        op = LinearCrossoverPerDimension()
        parent1 = sample_population[0]
        parent2 = sample_population[1]
        children = op.crossover(parent1, parent2, cpu_context.dtype_strategy)

        assert len(children) == 2
        for child in children:
            assert len(child.genome) == len(parent1.genome)

    def test_crossover_children_between_parents(self, sample_population, cpu_context):
        op = LinearCrossoverPerDimension()
        parent1 = sample_population[0]
        parent2 = sample_population[1]
        children = op.crossover(parent1, parent2, cpu_context.dtype_strategy)

        p_min = np.minimum(parent1.genome, parent2.genome)
        p_max = np.maximum(parent1.genome, parent2.genome)
        for child in children:
            assert np.all(child.genome >= p_min - 1e-10)
            assert np.all(child.genome <= p_max + 1e-10)


# --- Fitness Calculator Tests ---

class TestFitnessCalculator:
    def test_compute_distances_euclidean(self, fitness_calculator, cpu_context):
        data = cpu_context.get_data()
        genome = np.array([0.0, 0.0])
        distances = fitness_calculator._compute_distances(genome, data)
        assert len(distances) == len(data)
        assert np.all(distances >= 0)

    def test_compute_weights(self, fitness_calculator):
        distances = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
        weights_sum, weights_sum_d2 = fitness_calculator._compute_weights(
            distances, sigma=1.0, weight_threshold=0.3
        )
        assert float(weights_sum) >= 0
        assert float(weights_sum_d2) >= 0

    def test_calculate_full(self, sample_individual, cpu_context):
        data = cpu_context.get_data()
        fitness = fitness_calculator = sample_individual.fitness_calculator
        result = fitness_calculator.calculate(
            sample_individual, data, weight_threshold=0.3,
            metric='euclidean', p_minkowski=2
        )
        assert result is not None

    def test_select_best_offspring(self, fitness_calculator, sample_population):
        parent = sample_population[0]
        parent.fitness = np.float64(10.0)

        offspring = [sample_population[1], sample_population[2]]
        offspring[0].fitness = np.float64(5.0)
        offspring[1].fitness = np.float64(15.0)

        best = fitness_calculator._select_best_offspring(parent, offspring)
        assert best is not None

    def test_select_best_offspring_empty(self, fitness_calculator, sample_population):
        parent = sample_population[0]
        best = fitness_calculator._select_best_offspring(parent, [])
        assert best is parent


# --- HAEA Tests ---

class TestHAEA:
    def test_select_operator(self, cpu_context):
        haea = HAEA(
            mutation_operators={'mut': AdaptiveGaussianMutation},
            crossover_operators={'xover': LinearCrossoverPerDimension},
            context=cpu_context,
        )
        rates = {'mut': 0.7, 'xover': 0.3}
        op = haea.select_operator(rates)
        assert op in ['mut', 'xover']

    def test_apply_mutation_operator(self, cpu_context, sample_individual):
        haea = HAEA(
            mutation_operators={'mut': AdaptiveGaussianMutation},
            crossover_operators={'xover': LinearCrossoverPerDimension},
            context=cpu_context,
        )
        sample_individual.sigma2 = cpu_context.dtype_strategy.array(1.0)
        result = haea.apply_operator(cpu_context.dtype_strategy, 'mut', sample_individual)
        assert len(result) == 1

    def test_adjust_rates(self, cpu_context):
        haea = HAEA(
            mutation_operators={'mut': AdaptiveGaussianMutation},
            crossover_operators={'xover': LinearCrossoverPerDimension},
            context=cpu_context,
        )
        rates = {'mut': 0.5, 'xover': 0.5}
        new_rates = haea.adjust_rates(rates, 'mut', reward=True)
        assert 'mut' in new_rates


# --- Deterministic Crowding Tests ---

class TestDeterministicCrowding:
    def test_select_parents_unary(self, cpu_context, sample_population):
        haea = HAEA(
            mutation_operators={'mut': AdaptiveGaussianMutation},
            crossover_operators={'xover': LinearCrossoverPerDimension},
            context=cpu_context,
        )
        dc = DeterministicCrowding(operators_strategy=haea, context=cpu_context)

        parents = dc.select_parents(sample_population[0], sample_population, 'mut')
        assert len(parents) == 1
        assert parents[0] is sample_population[0]

    def test_select_parents_binary(self, cpu_context, sample_population):
        haea = HAEA(
            mutation_operators={'mut': AdaptiveGaussianMutation},
            crossover_operators={'xover': LinearCrossoverPerDimension},
            context=cpu_context,
        )
        dc = DeterministicCrowding(operators_strategy=haea, context=cpu_context)

        parents = dc.select_parents(sample_population[0], sample_population, 'xover')
        assert len(parents) == 2
        assert parents[0] is sample_population[0]


# --- Integration Tests ---

class TestIntegration:
    def test_ecsago_run_cpu(self):
        """Full integration test: evolve, extract, refine on 3 Gaussian blobs."""
        np.random.seed(42)
        blob1 = np.random.randn(50, 2) + np.array([0, 0])
        blob2 = np.random.randn(50, 2) + np.array([5, 5])
        blob3 = np.random.randn(50, 2) + np.array([10, 0])
        data = np.vstack([blob1, blob2, blob3])

        config = {
            'population_size': 20,
            'weight_threshold': 0.3,
            'max_generations': 5,
            'iterations': 3,
            'extraction_type': {0: 0},
            'k': 13.8,
            'use_cuda': False,
        }

        ecsago = ECSAGO(config)
        results = ecsago.run(data)

        assert 'final_population' in results
        assert 'prototypes' in results
        assert 'refined_prototypes' in results
        assert 'cluster_assignments' in results
        assert len(results['final_population']) == config['population_size']

    def test_ecsago_validate_config_missing_param(self):
        """Test that missing config params raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Missing required parameter"):
            ECSAGO({'population_size': 20})

    def test_population_evolve(self, cpu_context, fitness_calculator):
        """Test that population evolves without errors."""
        evolution_config = {
            'weight_threshold': 0.3,
            'mutation_operators': {
                'adaptive_gaussian_mutation': AdaptiveGaussianMutation,
            },
            'crossover_operators': {
                'linear_crossover_per_dimension': LinearCrossoverPerDimension,
            },
            'fitness_calculator': ECSAGOFitnessCalculator,
            'context': cpu_context,
        }

        factory = ECSAGOStrategyFactory()
        strategy = factory.create_strategy(evolution_config)

        population = ECSAGOPopulation(
            size=10,
            context=cpu_context,
            evolution_strategy=strategy,
        )

        initial_gen = population.generation
        population.evolve()
        assert population.generation == initial_gen + 1
        assert len(population.individuals) == 10
