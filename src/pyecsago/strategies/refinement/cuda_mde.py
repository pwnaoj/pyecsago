"""CUDA-optimized MDE refinement strategy."""

from __future__ import annotations

import logging

from typing import Any, TYPE_CHECKING

from .base import Refinement

if TYPE_CHECKING:
    import cupy as cp
    from pyecsago.implementations.ecsago.individual import ECSAGOIndividual
    from pyecsago.utils.data_types import DataTypeStrategy


class CUDAMDE(Refinement):
    """CUDA-optimized Maximal Density Estimator (MDE) refinement strategy.

    Replaces explicit Python loops with fully vectorized matrix operations,
    leveraging CuPy for massive parallel execution on the GPU.
    """
    def __init__(self, weight_threshold: float, sigma_factor: float = 13.8):
        """Initializes the CUDAMDE refiner.

        Args:
            weight_threshold: Threshold for weight binarization.
            sigma_factor: K-factor for minimum distance between prototypes.
        """
        self.weight_threshold = weight_threshold
        self.sigma_factor = sigma_factor
        logging.info("CUDAMDE strategy initialized.")

    def apply(self, prototypes: list[ECSAGOIndividual], data: cp.ndarray, dtype_strategy: DataTypeStrategy, iterations: int = 10, **kwargs) -> list[ECSAGOIndividual]:
        """Applies the refinement process for a fixed number of iterations."""
        refined_prototypes = [p.clone() for p in prototypes]

        for i in range(iterations):
            logging.debug(f"CUDAMDE Iteration {i+1}/{iterations}")
            if not refined_prototypes:
                break
            refined_prototypes = self.iteration(refined_prototypes, data, dtype_strategy, **kwargs)

        return refined_prototypes

    def iteration(self, prototypes: list[ECSAGOIndividual], data: cp.ndarray, dtype_strategy: DataTypeStrategy, **kwargs) -> list[ECSAGOIndividual]:
        """Performs a single, fully vectorized refinement iteration on the GPU."""

        # 1. Prepare data on GPU
        proto_genomes = dtype_strategy.array([p.genome for p in prototypes])
        proto_sigma2 = dtype_strategy.array([p.sigma2 for p in prototypes])
        num_prototypes = len(prototypes)

        # 2. Assign each point to the closest prototype (Winner-Takes-All)
        diff = dtype_strategy.module.expand_dims(data, 1) - dtype_strategy.module.expand_dims(proto_genomes, 0)
        distances_sq = dtype_strategy.module.sum(diff ** 2, axis=2)
        winner_indices = dtype_strategy.module.argmin(distances_sq, axis=1)
        min_dists_sq = dtype_strategy.module.min(distances_sq, axis=1)

        # 3. Calculate weights and statistics in a vectorized manner
        # Uses continuous membership weights with RBF kernel exp(-d2/(2*sigma2))
        xp = dtype_strategy.module

        winner_sigma2 = proto_sigma2[winner_indices]

        # Continuous membership with RBF kernel exp(-d2/(2*sigma2))
        membership = xp.exp(-min_dists_sq / (2 * winner_sigma2))

        # One-hot encoding for per-prototype aggregation
        one_hot_winners = xp.eye(num_prototypes)[winner_indices]

        # sum(w) per prototype
        sum_weights = xp.sum(one_hot_winners * xp.expand_dims(membership, axis=1), axis=0)

        # sum(w*x) per prototype
        weighted_points = xp.expand_dims(membership, axis=1) * data
        sum_weighted_feat = one_hot_winners.T @ weighted_points

        # sum(w*d2) per prototype
        weighted_dists_sq = membership * min_dists_sq
        sum_weighted_dists_sq = one_hot_winners.T @ weighted_dists_sq

        # Update prototypes (vectorized)
        non_empty_mask = sum_weights > 0
        safe_sum_weights = sum_weights + (sum_weights == 0) * 1e-9

        # mu = sum(w*x) / sum(w)
        new_centers = sum_weighted_feat / xp.expand_dims(safe_sum_weights, axis=1)

        # sigma2 = sum(w*d2) / sum(w)
        new_sigma2 = sum_weighted_dists_sq / safe_sum_weights

        # 5. Assign new values to the prototype list
        for i, proto in enumerate(prototypes):
            if non_empty_mask[i]:
                proto.genome = new_centers[i]
                proto.sigma2 = new_sigma2[i]
                proto._validate_sigma2_limits()

        return prototypes
