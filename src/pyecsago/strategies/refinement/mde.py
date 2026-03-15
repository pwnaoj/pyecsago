"""Maximal Density Estimator (MDE) refinement strategy."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .base import Refinement

if TYPE_CHECKING:
    import numpy as np
    import cupy as cp
    from pyecsago.implementations.ecsago.individual import ECSAGOIndividual
    from pyecsago.utils.data_types import DataTypeStrategy


class MDE(Refinement):
    """Maximal Density Estimator (MDE) for prototype refinement.

    Iteratively updates prototype centers and spreads:
    1. mu = sum(wi * xi) / sum(wi)      -- center update (continuous weights)
    2. sigma2 = sum(wi * di2) / sum(wi)  -- spread update (weighted mean)
    Where wi = exp(-di2 / (2 * sigma2)) is the continuous membership of point i.
    """

    def __init__(self, weight_threshold: float, sigma_factor: float = 13.8):
        """Initializes the MDE refiner.

        Args:
            weight_threshold: Threshold for weight binarization.
            sigma_factor: K-factor for minimum distance between prototypes.
        """
        self.weight_threshold = weight_threshold
        self.sigma_factor = sigma_factor

    def iteration(self, prototypes: list[ECSAGOIndividual], data: np.ndarray | cp.ndarray, dtype_strategy: DataTypeStrategy, **kwargs) -> list[ECSAGOIndividual]:
        """Executes a single MDE refinement iteration."""
        if not prototypes:
            return []

        # Clone prototypes to avoid modifying originals during this iteration
        refined_prototypes = [proto.clone() for proto in prototypes]

        # 1. Vector assignment (Winner-Takes-All)
        # Determines the single closest cluster for each data point.
        closest_centers_indices, min_dists_sq = self._assign_vectors_winner_takes_all(
            refined_prototypes, data, dtype_strategy
        )

        # 2. Statistics accumulation
        # Computes weights and accumulates statistics based on the assignment.
        stats = self._accumulate_stats(
            refined_prototypes, data, closest_centers_indices, min_dists_sq, dtype_strategy
        )

        # 3. Prototype update
        # Updates centers and sigmas with the collected statistics.
        self._update_prototypes(refined_prototypes, stats, dtype_strategy)

        return refined_prototypes

    def _assign_vectors_winner_takes_all(self, prototypes: list[ECSAGOIndividual], data: np.ndarray | cp.ndarray, dtype_strategy: DataTypeStrategy):
        """Assigns each data point to its closest prototype (winner-takes-all).

        Uses broadcasting to compute all distances in a single operation:
        dists_sq[i, j] = ||data[i] - proto[j]||^2
        """
        xp = dtype_strategy.module
        proto_genomes = dtype_strategy.array([p.genome for p in prototypes])

        # (N, 1, D) - (1, M, D) -> (N, M, D) -> sum -> (N, M)
        diffs = data[:, xp.newaxis, :] - proto_genomes[xp.newaxis, :, :]
        dists_sq = xp.sum(diffs ** 2, axis=2)

        closest_indices = xp.argmin(dists_sq, axis=1)
        min_dists_sq = dists_sq[xp.arange(data.shape[0]), closest_indices]

        return closest_indices, min_dists_sq

    def _accumulate_stats(self, prototypes: list[ECSAGOIndividual], data: np.ndarray | cp.ndarray, closest_indices, min_dists_sq, dtype_strategy: DataTypeStrategy):
        """Accumulates continuous-weight statistics using vectorized operations.

        Uses RBF kernel membership weights exp(-d2 / (2 * sigma2)) with
        one-hot encoding for parallel accumulation across all prototypes.
        """
        xp = dtype_strategy.module
        n_prototypes = len(prototypes)
        n_samples, n_features = data.shape

        # Build sigma array indexed by prototype: sigmas[j] = proto_j.sigma2
        sigmas = dtype_strategy.array([p.sigma2 for p in prototypes])

        # Compute per-point membership: exp(-d2 / (2 * sigma2_winner))
        sigma_per_point = sigmas[closest_indices]
        membership = xp.exp(-min_dists_sq / (2 * sigma_per_point))

        # One-hot assignment matrix (N, M)
        one_hot = xp.zeros((n_samples, n_prototypes))
        one_hot[xp.arange(n_samples), closest_indices] = 1.0

        # Weighted one-hot: (N, M) with membership values
        weighted_one_hot = one_hot * membership[:, xp.newaxis]

        stats = {
            'sum_weights': xp.sum(weighted_one_hot, axis=0),
            'sum_weights_feat': weighted_one_hot.T @ data,
            'sum_weights_dist_sq': xp.sum(weighted_one_hot * min_dists_sq[:, xp.newaxis], axis=0),
        }

        return stats

    def apply(self, prototypes: list[ECSAGOIndividual], data: np.ndarray | cp.ndarray, dtype_strategy: DataTypeStrategy, iterations: int = 10, **kwargs) -> list[ECSAGOIndividual]:
        """Applies the full MDE refinement process.

        Args:
            prototypes: List of prototypes to refine.
            data: Original dataset.
            dtype_strategy: Data type strategy.
            iterations: Number of refinement iterations.

        Returns:
            List of refined prototypes.
        """
        refined = prototypes

        for _ in range(iterations):
            refined = self.iteration(refined, data, dtype_strategy, **kwargs)

        return refined

    def _update_prototypes(self, prototypes: list[ECSAGOIndividual], stats: dict[str, Any], dtype_strategy: DataTypeStrategy) -> None:
        """Updates prototype centers and spreads using MDE.

        Sigma formula: sigma2 = sum(w * d2) / sum(w), weighted mean compatible
        with the RBF kernel.
        """
        for j, proto in enumerate(prototypes):
            # Only update if the prototype has assigned points
            if stats['sum_weights'][j] > 0:
                # Update center: mu = sum(w*x) / sum(w)
                new_center = stats['sum_weights_feat'][j] / stats['sum_weights'][j]
                proto.genome = new_center

                # Update spread: sigma2 = sum(w*d2) / sum(w)
                new_sigma2 = stats['sum_weights_dist_sq'][j] / stats['sum_weights'][j]
                proto.sigma2 = dtype_strategy.array(new_sigma2)
                proto._validate_sigma2_limits()
