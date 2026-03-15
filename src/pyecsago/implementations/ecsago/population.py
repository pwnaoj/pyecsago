"""ECSAGO population implementation."""

from __future__ import annotations

import numpy as np

from typing import Any, TYPE_CHECKING

from ...core.population import BasePopulation
from .individual import ECSAGOIndividual
from ...strategies.extraction.factory import ExtractionStrategyFactory
from ...strategies.refinement.mde import MDE
from ...strategies.refinement.cuda_mde import CUDAMDE

if TYPE_CHECKING:
    import cupy as cp
    from pyecsago import ECSAGOStrategy
    from pyecsago.core.context import AlgorithmContext
    from pyecsago.strategies.extraction.base import ExtractionStrategy


class ECSAGOPopulation(BasePopulation):
    """Population manager for the ECSAGO algorithm.

    Handles initialization, evolution, prototype extraction, and
    refinement of a population of ECSAGOIndividual instances.
    """

    def __init__(
        self,
        size: int,
        context: AlgorithmContext,
        evolution_strategy: ECSAGOStrategy,
    ) -> None:
        """Initializes the ECSAGO population.

        Selects random data points as initial cluster centers and creates
        individuals with random operator rates.

        Args:
            size: Population size.
            context: Algorithm execution context.
            evolution_strategy: Configured ECSAGO evolution strategy.
        """
        self.context = context
        self.size = size
        self.evolution_strategy = evolution_strategy
        self.generation = 0

        self._initialize_population()

    def _initialize_population(self) -> None:
        """Creates initial individuals from random data points."""
        operators = list(self.evolution_strategy.operators_strategy.mutation_operators.keys()) + \
                    list(self.evolution_strategy.operators_strategy.crossover_operators.keys())

        data = self.context.get_data()

        indices = self.context.dtype_strategy.random_choice(len(data), self.size, replace=False)
        centers = data[indices]

        self.individuals = []
        for center in centers:
            random_rates = self.context.dtype_strategy.module.random.random(len(operators))
            normalized_rates = random_rates / self.context.dtype_strategy.module.sum(random_rates)
            individual_rates = dict(zip(operators, normalized_rates))

            individual = ECSAGOIndividual(
                genome=center,
                fitness_calculator=self.evolution_strategy.fitness_calculator,
                operator_rates=individual_rates,
                context=self.context
            )
            self.individuals.append(individual)

    def evaluate_population(self) -> None:
        """Evaluates fitness for all individuals."""
        for individual in self.individuals:
            individual.calculate_fitness(
                weight_threshold=self.evolution_strategy.weight_threshold
            )

    def evolve(self) -> list[ECSAGOIndividual]:
        """Evolves the population using the configured strategy.

        Returns:
            List of evolved individuals.
        """
        self.individuals = self.evolution_strategy.evolve_population(
            self.individuals
        )

        self.generation += 1

        return self.individuals

    def extract_prototypes(self, extraction_type: dict | None = None, k: float = 13.8) -> list[ECSAGOIndividual]:
        """Extracts prototypes using fitness filtering and niche distance.

        Args:
            extraction_type: Extraction method configuration.
            k: Chi-squared factor for minimum inter-prototype distance.

        Returns:
            List of selected prototypes.
        """
        if extraction_type is None:
            extraction_type = {0: 0}

        strategy = ExtractionStrategyFactory.create_default_strategy(
            data_size=len(self.context.get_data()),
            sigma_max=self.context.get_sigma2_max(),
            sigma_factor=k,
            extraction_type=extraction_type,
            dtype_strategy=self.context.dtype_strategy
        )

        return self.extract_prototypes_with_strategy(strategy)

    def refine_prototypes(self, prototypes: list[ECSAGOIndividual], iterations: int = 10, k: float = 13.8) -> list[ECSAGOIndividual]:
        """Refines prototypes using Maximal Density Estimator (MDE).

        Args:
            prototypes: Prototypes to refine.
            iterations: Number of MDE iterations.
            k: Chi-squared factor for niche radius.

        Returns:
            List of refined prototypes.
        """
        if not prototypes:
            return []

        wt = self.evolution_strategy.weight_threshold
        if self.context.use_cuda and self.context.cuda_context is not None:  # pragma: no cover
            refinement = CUDAMDE(wt, k)
        else:
            refinement = MDE(wt, k)

        return refinement.apply(
            prototypes=prototypes,
            data=self.context.get_data(),
            iterations=iterations,
            dtype_strategy=self.context.dtype_strategy
        )

    def extract_prototypes_with_strategy(self, strategy: ExtractionStrategy, **kwargs: Any) -> list[ECSAGOIndividual]:
        """Extracts prototypes using a custom extraction strategy.

        Args:
            strategy: Extraction strategy to apply.
            **kwargs: Additional arguments for the strategy.

        Returns:
            List of extracted prototypes.
        """
        candidates = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)

        prototypes = strategy.extract(candidates, self.context.get_data())

        return prototypes
