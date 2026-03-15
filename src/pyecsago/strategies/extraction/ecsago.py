"""ECSAGO extraction strategy implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ExtractionStrategy

if TYPE_CHECKING:
    import numpy as np
    import cupy as cp
    from pyecsago.implementations.ecsago.individual import ECSAGOIndividual
    from pyecsago.utils.data_types import DataTypeStrategy


class FitnessExtractionStrategy(ExtractionStrategy):
    """Fitness-based extraction strategy.

    Selects individuals based on different fitness threshold criteria.
    """

    # Extraction type constants
    ABSOLUTE_VALUE = 0    # Absolute threshold
    PROPORTION_AVG = 1    # Proportional to average fitness
    PROPORTION_MAX = 2    # Proportional to maximum fitness
    PROPORTION_MEDIAN = 3 # Proportional to median fitness
    MINIMUM_DENSITY = 4   # Based on minimum density

    def __init__(self, threshold: float, extraction_type: int, dtype_strategy: DataTypeStrategy) -> None:
        """Initializes the fitness extraction strategy.

        Args:
            threshold: Extraction threshold.
            extraction_type: Extraction type (0-4).
        """
        self.threshold = threshold
        self.extraction_type = extraction_type
        self.dtype_strategy = dtype_strategy

    def extract(self, candidates: list[ECSAGOIndividual], data: np.ndarray | cp.ndarray) -> list[ECSAGOIndividual]:
        """Extracts prototypes based on the fitness criterion.

        Args:
            candidates: List of candidate individuals sorted by fitness (descending).
            data: Original dataset.

        Returns:
            List of extracted prototypes.
        """
        if not candidates:
            return []

        # Calculate threshold based on extraction type
        min_value = 0.0

        if self.extraction_type == self.ABSOLUTE_VALUE:
            min_value = self.threshold
        elif self.extraction_type == self.PROPORTION_AVG:
            fitness_values = self.dtype_strategy.array([c.fitness for c in candidates])
            avg_fitness = self.dtype_strategy.module.mean(fitness_values)
            min_value = self.threshold * avg_fitness
        elif self.extraction_type == self.PROPORTION_MAX:
            min_value = self.threshold * candidates[0].fitness
        elif self.extraction_type == self.PROPORTION_MEDIAN:
            median_index = len(candidates) // 2
            min_value = self.threshold * candidates[median_index].fitness
        elif self.extraction_type == self.MINIMUM_DENSITY:
            min_value = self.threshold * len(data)

        # Filter candidates by threshold
        extracted = [c for c in candidates if c.fitness > min_value]

        return extracted

class NicheExtractionStrategy(ExtractionStrategy):
    """Niche-based extraction strategy.

    Selects prototypes while maintaining a minimum distance between them,
    ensuring prototypes are not too close to each other.
    """

    def __init__(self, sigma_factor: float, k: float, dtype_strategy: DataTypeStrategy):
        """Initializes the niche extraction strategy.

        Args:
            sigma_factor: Factor for radius calculation.
            k: Additional factor to adjust the minimum distance.
        """
        self.sigma_factor = sigma_factor
        self.k = k
        self.dtype_strategy = dtype_strategy

    def extract(self, candidates: list[ECSAGOIndividual], data: np.ndarray | cp.ndarray = None) -> list[ECSAGOIndividual]:
        """Extracts prototypes while maintaining a minimum distance between them.

        Args:
            candidates: List of candidate individuals (ideally already filtered by fitness).
            data: Original dataset (unused, present for interface consistency).

        Returns:
            List of extracted prototypes.
        """
        if not candidates:
            return []

        # Sort candidates by fitness in descending order (assumed by the paper)
        candidates.sort(key=lambda c: c.fitness, reverse=True)

        # Start with the first candidate (highest fitness)
        selected_prototypes = [candidates[0]]

        # Convert selected genomes to an array for vectorization
        selected_genomes = self.dtype_strategy.array([p.genome for p in selected_prototypes])

        # Evaluate each remaining candidate
        for candidate in candidates[1:]:
            candidate_genome = self.dtype_strategy.array(candidate.genome)

            # Calculate distances to all selected prototypes at once
            diffs = selected_genomes - candidate_genome
            distances_sq = self.dtype_strategy.module.sum(diffs * diffs, axis=1)
            distances = self.dtype_strategy.module.sqrt(distances_sq)

            # Calculate required minimum radii for comparison
            candidate_radius_base = self.k * self.sigma_factor * candidate.sigma2
            proto_radii_base = self.dtype_strategy.array([self.k * self.sigma_factor * p.sigma2 for p in selected_prototypes])

            # Check if the candidate is too close to any already selected prototype
            # Eq 2.18: max(s_i, s_j) as distance threshold for merging prototypes
            max_radii = self.dtype_strategy.module.maximum(candidate_radius_base, proto_radii_base)

            is_too_close = self.dtype_strategy.module.any(distances <= max_radii)

            # If not too close to any, add to the selection
            if not is_too_close:
                selected_prototypes.append(candidate)
                # Update the selected genomes array for the next iteration
                selected_genomes = self.dtype_strategy.module.vstack([selected_genomes, candidate_genome])

        return selected_prototypes

class ComposeExtractionStrategy(ExtractionStrategy):
    """Composite strategy that applies multiple strategies in sequence."""

    def __init__(self, strategy_a: ExtractionStrategy, strategy_b: ExtractionStrategy):
        """Initializes the composite strategy.

        Args:
            strategy_a: First strategy to apply.
            strategy_b: Second strategy to apply after the first.
        """
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b

    def extract(self, candidates: list[ECSAGOIndividual], data: np.ndarray | cp.ndarray) -> list[ECSAGOIndividual]:
        """Applies the strategies in sequence.

        Args:
            candidates: List of candidate individuals.
            data: Original dataset.

        Returns:
            List of extracted prototypes.
        """
        # If the first strategy is None, apply only the second
        if self.strategy_a is None:
            return self.strategy_b.extract(candidates, data)

        # Apply the first strategy
        intermediate_results = self.strategy_a.extract(candidates, data)

        # Apply the second strategy to the results of the first
        return self.strategy_b.extract(intermediate_results, data)
