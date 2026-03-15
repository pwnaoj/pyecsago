"""Factory for creating extraction strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ecsago import (
    FitnessExtractionStrategy,
    NicheExtractionStrategy,
    ExtractionStrategy,
    ComposeExtractionStrategy
)

if TYPE_CHECKING:
    from pyecsago.utils.data_types import DataTypeStrategy


class ExtractionStrategyFactory:
    """Factory for creating configured extraction strategies."""

    @staticmethod
    def create_fitness_extraction(threshold: float, extraction_type: int, dtype_strategy: DataTypeStrategy) -> FitnessExtractionStrategy:
        """Creates a fitness-based extraction strategy.

        Args:
            threshold: Extraction threshold.
            extraction_type: Extraction type (0-4).

        Returns:
            Configured fitness extraction strategy.
        """
        return FitnessExtractionStrategy(threshold, extraction_type, dtype_strategy)

    @staticmethod
    def create_niche_extraction(sigma_factor: float, k: float, dtype_strategy: DataTypeStrategy) -> NicheExtractionStrategy:
        """Creates a niche-based extraction strategy.

        Args:
            sigma_factor: Factor for radius calculation.
            k: Additional factor to adjust the minimum distance.

        Returns:
            Configured niche extraction strategy.
        """
        return NicheExtractionStrategy(sigma_factor, k, dtype_strategy)

    @staticmethod
    def create_default_strategy(data_size: int, sigma_max: float, sigma_factor: float, extraction_type: dict[int, float], dtype_strategy: DataTypeStrategy) -> ExtractionStrategy:
        """Creates the default extraction strategy.

        Combines a fitness extraction followed by a niche extraction.

        Args:
            data_size: Size of the dataset.
            sigma_max: Maximum sigma value.
            sigma_factor: Sigma factor for distance calculation.
            extraction_type: Extraction method configuration.

        Returns:
            Configured composite extraction strategy.
        """
        ext_type, value = next(iter(extraction_type.items()))

        if ext_type == 0:
            # Calculate threshold
            min_fitness_ext = data_size / sigma_max / sigma_factor / 4.0
        elif ext_type in range(1,5):
            # Assign the configuration value as threshold
            min_fitness_ext = value

        # Create individual strategies
        fitness_strategy = ExtractionStrategyFactory.create_fitness_extraction(min_fitness_ext, ext_type, dtype_strategy)
        niche_strategy = ExtractionStrategyFactory.create_niche_extraction(sigma_factor, 3.0, dtype_strategy)

        # Compose them
        return ComposeExtractionStrategy(fitness_strategy, niche_strategy)
