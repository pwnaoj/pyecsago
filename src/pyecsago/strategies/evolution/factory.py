"""Factory for creating evolution strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .ecsago import ECSAGOStrategy
from .base import EvolutionStrategy
from ..niching.deterministic_crowding import DeterministicCrowding
from ..operators.haea import HAEA


class EvolutionStrategyFactory(ABC):
    """Abstract factory for creating evolution strategies."""

    @abstractmethod
    def create_strategy(self, config: dict[str, Any]) -> EvolutionStrategy:
        """Creates a new evolution strategy instance."""
        pass

class ECSAGOStrategyFactory(EvolutionStrategyFactory):
    """Concrete factory for creating ECSAGO strategies."""

    def create_strategy(self, config: dict[str, Any]) -> ECSAGOStrategy:
        """Creates a new ECSAGO strategy instance.

        Args:
            config: Dictionary with parameters such as:
                   - weight_threshold: Binarization threshold.
                   - mutation_operators: Mutation operator configuration.
                   - crossover_operators: Crossover operator configuration.
                   - fitness_calculator: Fitness calculator class.
                   - context: Algorithm context.

        Returns:
            A fully configured ECSAGOStrategy instance.
        """
        # Get execution context
        context = config.get('context')
        if context is None:
            raise ValueError("An AlgorithmContext is required to create the strategy.")

        # Get configuration parameters
        weight_threshold = config.get('weight_threshold', 0.3)
        fitness_calculator_cls = config.get('fitness_calculator')

        # Create and configure operator strategy
        operators_strategy = HAEA(
            mutation_operators=config.get('mutation_operators', {}),
            crossover_operators=config.get('crossover_operators', {}),
            context=context
        )

        # Create niching strategy
        niching_strategy = DeterministicCrowding(
            operators_strategy=operators_strategy,
            context=context
        )

        # Instantiate fitness calculator according to context
        if context.use_cuda:  # pragma: no cover
            fitness_calculator = fitness_calculator_cls(
                context=context,
                dtype_strategy=context.dtype_strategy
            )
        else:
            fitness_calculator = fitness_calculator_cls(
                dtype_strategy=context.dtype_strategy
            )

        return ECSAGOStrategy(
            niching_strategy=niching_strategy,
            operators_strategy=operators_strategy,
            weight_threshold=weight_threshold,
            fitness_calculator=fitness_calculator,
            context=context
        )
