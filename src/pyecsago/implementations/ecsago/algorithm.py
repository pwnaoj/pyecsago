"""Main ECSAGO algorithm implementation."""

from __future__ import annotations

import numpy as np

from typing import Any, TYPE_CHECKING
from tqdm import tqdm

from ...core.algorithm import EvolutionaryAlgorithm
from ...core.exceptions import ConfigurationError, ValidationError
from ...strategies.evolution.factory import ECSAGOStrategyFactory
from ...strategies.fitness.ecsago import ECSAGOFitnessCalculator, CUDAECSAGOFitnessCalculator
from ...strategies.operators.crossover import LinearCrossoverPerDimension
from ...strategies.operators.mutation import AdaptiveGaussianMutation, GaussianMutation
from .context import StandardAlgorithmContext
from .population import ECSAGOPopulation

if TYPE_CHECKING:
    import cupy as cp


class ECSAGO(EvolutionaryAlgorithm):
    """Main entry point for the ECSAGO algorithm.

    Coordinates population evolution, prototype extraction, and refinement
    to find optimal cluster centroids in the data.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initializes the ECSAGO algorithm.

        Args:
            config: Algorithm configuration containing:
                - population_size (int): Population size.
                - weight_threshold (float): Weight binarization threshold.
                - max_generations (int): Maximum number of generations.
                - iterations (int): MDE refinement iterations.
                - extraction_type (dict): Extraction method configuration.
                - k (float): Chi-squared factor for niche radius.
                - use_cuda (bool): Whether to use CUDA acceleration.

        Raises:
            ConfigurationError: If required parameters are missing or invalid.
        """
        super().__init__(config)
        self.validate_config(config)

        self.context = StandardAlgorithmContext(use_cuda=config.get('use_cuda', False))

        self.iterations = self.config.get('iterations', 10)
        self.extraction_type = self.config.get('extraction_type', {})
        self.k = self.config.get('k', 13.8)
        self.population = None
        self.best_prototypes = None

        evolution_config = {
            'weight_threshold': self.config['weight_threshold'],
            'mutation_operators': {
                'adaptive_gaussian_mutation': AdaptiveGaussianMutation,
            },
            'crossover_operators': {
                'linear_crossover_per_dimension': LinearCrossoverPerDimension,
            },
            'fitness_calculator': ECSAGOFitnessCalculator if not self.use_cuda else CUDAECSAGOFitnessCalculator,
            'context': self.context
        }

        strategy_factory = ECSAGOStrategyFactory()
        self.evolution_strategy = strategy_factory.create_strategy(evolution_config)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validates that all required parameters are present and correctly typed.

        Args:
            config: Configuration dictionary to validate.

        Raises:
            ConfigurationError: If a required parameter is missing or has wrong type.
        """
        required_params = {
            'population_size': int,
            'weight_threshold': float,
            'max_generations': int,
            'iterations': int,
            'extraction_type': dict,
            'k': float,
            'use_cuda': bool
        }

        for param, param_type in required_params.items():
            if param not in config:
                raise ConfigurationError(f"Missing required parameter: {param}")
            if not isinstance(config[param], param_type):
                raise ConfigurationError(
                    f"Wrong type for {param}. Expected {param_type}"
                )

    def evolve(self, max_generations: int | None = 30) -> None:
        """Runs the evolutionary loop.

        Args:
            max_generations: Fallback if not set in config.
        """
        generations = self.config.get('max_generations', max_generations)

        self.population = ECSAGOPopulation(
            size=self.config['population_size'],
            context=self.context,
            evolution_strategy=self.evolution_strategy
        )

        for _ in tqdm(range(generations)):
            self.population.evolve()

    def extract_prototypes(self, extraction_type: dict | None = None, k: float = 13.8) -> list:
        """Extracts final prototypes from the evolved population.

        Args:
            extraction_type: Extraction method configuration.
            k: Chi-squared factor for minimum inter-prototype distance.

        Returns:
            List of extracted prototypes.

        Raises:
            ValidationError: If evolve() has not been called yet.
        """
        if extraction_type is None:
            extraction_type = {}

        if self.population is None:
            raise ValidationError("Must call evolve() before extracting prototypes")

        self.best_prototypes = self.population.extract_prototypes(
            extraction_type=extraction_type,
            k=k
        )

        return self.best_prototypes

    def refine_prototypes(self, prototypes: list, iterations: int = 10, k: float = 13.8) -> list:
        """Refines extracted prototypes using MDE.

        Args:
            prototypes: Prototypes to refine.
            iterations: Number of MDE refinement iterations.
            k: Chi-squared factor for niche radius.

        Returns:
            List of refined prototypes.

        Raises:
            ValidationError: If evolve() has not been called yet.
        """
        if self.population is None:
            raise ValidationError("Must call evolve() before refining prototypes")

        return self.population.refine_prototypes(
            prototypes=prototypes,
            iterations=iterations,
            k=k
        )

    def run(self, data: np.ndarray | cp.ndarray) -> dict[str, Any]:
        """Runs the complete ECSAGO pipeline: evolve, extract, refine, assign.

        Args:
            data: Input dataset for clustering.

        Returns:
            Dictionary with keys: final_population, prototypes,
            refined_prototypes, cluster_assignments.
        """
        self.context.set_data(data)

        self.evolve()

        prototypes = self.extract_prototypes(
            extraction_type=self.extraction_type,
            k=self.k
        )

        refined_prototypes = self.refine_prototypes(
            prototypes=prototypes,
            iterations=self.iterations,
            k=self.k
        )

        cluster_assignments = self._assign_clusters(refined_prototypes)

        if self.use_cuda:  # pragma: no cover
            for proto in (*prototypes, *refined_prototypes):
                proto.genome = self.context.to_cpu(proto.genome)
                proto.sigma2 = float(proto.sigma2)
                proto.fitness = float(proto.fitness)

        return {
            'final_population': self.population.individuals,
            'prototypes': prototypes,
            'refined_prototypes': refined_prototypes,
            'cluster_assignments': cluster_assignments,
        }

    def _assign_clusters(self, prototypes: list) -> np.ndarray:
        """Assigns each data point to the nearest prototype.

        Args:
            prototypes: List of prototypes defining cluster centers.

        Returns:
            Array of cluster assignments per data point.
        """
        if not prototypes:
            return self.context.dtype_strategy.array([])

        def calculate_assignments():
            data = self.context.get_data()
            proto_genomes = [
                self.context.to_device(proto.genome) if not hasattr(proto.genome, 'device')
                else proto.genome for proto in prototypes
            ]

            distances = self.context.dtype_strategy.array([
                self.context.dtype_strategy.module.linalg.norm(data - genome, axis=1)
                for genome in proto_genomes
            ])

            return self.context.dtype_strategy.module.argmin(distances, axis=0)

        assignments = self.context.execute(calculate_assignments)

        return self.context.to_cpu(assignments)
