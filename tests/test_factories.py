"""Tests for factory classes."""

import numpy as np
import pytest

from pyecsago import (
    ECSAGOFitnessCalculator,
    ECSAGOStrategyFactory,
    ECSAGOStrategy,
    AdaptiveGaussianMutation,
    LinearCrossoverPerDimension,
)
from pyecsago.strategies.context.factory import AlgorithmContextFactory
from pyecsago.strategies.extraction.factory import ExtractionStrategyFactory
from pyecsago.strategies.extraction.ecsago import (
    FitnessExtractionStrategy,
    NicheExtractionStrategy,
    ComposeExtractionStrategy,
)
from pyecsago.strategies.refinement.factory import RefinementStrategyFactory
from pyecsago.strategies.refinement.mde import MDE
from pyecsago.implementations.ecsago.context import StandardAlgorithmContext


class TestAlgorithmContextFactory:
    def test_create_cpu_context(self):
        ctx = AlgorithmContextFactory.create_context({'use_cuda': False})
        assert isinstance(ctx, StandardAlgorithmContext)
        assert ctx.use_cuda is False

    def test_create_default_context(self):
        ctx = AlgorithmContextFactory.create_context({})
        assert isinstance(ctx, StandardAlgorithmContext)
        assert ctx.use_cuda is False


class TestExtractionStrategyFactory:
    def test_create_fitness_extraction(self, numpy_strategy):
        strategy = ExtractionStrategyFactory.create_fitness_extraction(0.5, 2, numpy_strategy)
        assert isinstance(strategy, FitnessExtractionStrategy)

    def test_create_niche_extraction(self, numpy_strategy):
        strategy = ExtractionStrategyFactory.create_niche_extraction(1.0, 3.0, numpy_strategy)
        assert isinstance(strategy, NicheExtractionStrategy)

    def test_create_default_strategy_type_0(self, numpy_strategy):
        strategy = ExtractionStrategyFactory.create_default_strategy(
            data_size=100, sigma_max=1.0, sigma_factor=1.0,
            extraction_type={0: 0}, dtype_strategy=numpy_strategy
        )
        assert isinstance(strategy, ComposeExtractionStrategy)

    def test_create_default_strategy_type_2(self, numpy_strategy):
        strategy = ExtractionStrategyFactory.create_default_strategy(
            data_size=100, sigma_max=1.0, sigma_factor=1.0,
            extraction_type={2: 0.25}, dtype_strategy=numpy_strategy
        )
        assert isinstance(strategy, ComposeExtractionStrategy)


class TestRefinementStrategyFactory:
    def test_create_cpu_mde(self):
        strategy = RefinementStrategyFactory.create_mde(
            weight_threshold=0.3, sigma_factor=13.8,
            cuda_context=None, use_cuda=False
        )
        assert isinstance(strategy, MDE)

    def test_create_cpu_mde_when_no_cuda_context(self):
        strategy = RefinementStrategyFactory.create_mde(
            weight_threshold=0.3, sigma_factor=13.8,
            cuda_context=None, use_cuda=True
        )
        assert isinstance(strategy, MDE)


class TestECSAGOStrategyFactory:
    def test_create_strategy(self, cpu_context):
        factory = ECSAGOStrategyFactory()
        config = {
            'weight_threshold': 0.3,
            'mutation_operators': {'mut': AdaptiveGaussianMutation},
            'crossover_operators': {'xover': LinearCrossoverPerDimension},
            'fitness_calculator': ECSAGOFitnessCalculator,
            'context': cpu_context,
        }
        strategy = factory.create_strategy(config)
        assert isinstance(strategy, ECSAGOStrategy)

    def test_create_strategy_missing_context(self):
        factory = ECSAGOStrategyFactory()
        with pytest.raises(ValueError, match="(?i)context"):
            factory.create_strategy({'context': None})

    def test_create_strategy_defaults(self, cpu_context):
        factory = ECSAGOStrategyFactory()
        config = {
            'context': cpu_context,
            'fitness_calculator': ECSAGOFitnessCalculator,
        }
        strategy = factory.create_strategy(config)
        assert isinstance(strategy, ECSAGOStrategy)
