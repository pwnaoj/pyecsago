"""ECSAGO fitness calculator implementations."""

from __future__ import annotations

import logging

from typing import Literal, TYPE_CHECKING

from .base import FitnessCalculator

if TYPE_CHECKING:
    import numpy as np
    import cupy as cp
    from pyecsago import ECSAGOIndividual
    from pyecsago.utils.data_types import DataTypeStrategy

logger = logging.getLogger(__name__)


class ECSAGOFitnessCalculator(FitnessCalculator):
    """Fitness calculator for ECSAGO supporting multiple distance metrics.

    Supported metrics: euclidean, minkowski, cosine, jaccard.
    """

    def __init__(self, dtype_strategy: DataTypeStrategy) -> None:
        """Initializes the ECSAGO fitness calculator.

        Args:
            dtype_strategy: Data type strategy for array operations.
        """
        self.dtype_strategy = dtype_strategy

    def _compute_distances(self, genome: np.ndarray, data: np.ndarray, metric: Literal['euclidean', 'minkowski', 'cosine', 'jaccard'] = 'euclidean',
                           p_minkowski: int = 2) -> np.ndarray | cp.ndarray:
        """Computes distances from genome to all data points.

        Args:
            genome: N-dimensional cluster center.
            data: Dataset array.
            metric: Distance metric to use. Defaults to 'euclidean'.
            p_minkowski: Minkowski parameter. Defaults to 2.

        Returns:
            Array of distances.
        """
        if metric == 'euclidean':
            distances = self.dtype_strategy.module.linalg.norm(data - genome, axis=1)

        elif metric == 'minkowski':
            distances = self.dtype_strategy.module.sum(self.dtype_strategy.module.abs(data - genome)**p_minkowski, axis=1)**(1/p_minkowski)

        elif metric == 'cosine':
            norm_data = self.dtype_strategy.module.linalg.norm(data, axis=1)
            norm_genome = self.dtype_strategy.module.linalg.norm(genome)
            distances = 1 - self.dtype_strategy.module.dot(data, genome) / (norm_data * norm_genome)

        elif metric == 'jaccard':
            intersection = self.dtype_strategy.module.minimum(data, genome).sum(axis=1)
            union = self.dtype_strategy.module.maximum(data, genome).sum(axis=1)
            distances = 1 - intersection / union

        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return distances

    def _compute_weights(self, distances: np.ndarray | cp.ndarray, sigma: float, weight_threshold: float) -> tuple[np.ndarray | cp.ndarray, np.ndarray | cp.ndarray]:
        """Computes binarized RBF kernel weights: exp(-d^2/(2*sigma2)).

        Args:
            distances: Distance array from genome to data points.
            sigma: Dispersion value (sigma2) of the individual.
            weight_threshold: Threshold for weight binarization.

        Returns:
            Tuple of (sum of binary weights, sum of binary weights * distances^2).
        """
        squared_distances = distances ** 2
        weights = self.dtype_strategy.module.exp(-squared_distances / (2 * sigma))
        binary_weights = self.dtype_strategy.module.where(weights > weight_threshold, 1.0, 0.0)
        weights_sum = self.dtype_strategy.module.sum(binary_weights)
        weights_sum_d2 = self.dtype_strategy.module.sum(binary_weights * squared_distances)

        return weights_sum, weights_sum_d2

    def _update_scale(self, weights_sum: np.ndarray | cp.ndarray, weights_sum_d2: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        """Computes new sigma2 as mean squared distance of weighted points.

        Args:
            weights_sum: Sum of binarized weights.
            weights_sum_d2: Sum of binarized weights times squared distances.

        Returns:
            New sigma2 value.
        """
        if weights_sum == 0.0:
            new_sigma = 0.0
        else:
            new_sigma = weights_sum_d2 / weights_sum

        new_sigma = self.dtype_strategy.array(new_sigma)

        return new_sigma

    def calculate(self, individual: ECSAGOIndividual, data: np.ndarray, weight_threshold: float, metric: str, p_minkowski: int) -> float:
        """Computes and updates the individual's fitness value.

        Args:
            individual: The individual to evaluate.
            data: Dataset array.
            weight_threshold: Threshold for weight binarization.
            metric: Distance metric to use.
            p_minkowski: Minkowski parameter.

        Returns:
            Computed fitness value.
        """
        distances = self._compute_distances(individual.genome, data, metric, p_minkowski)

        weights_sum, weights_sum_d2 = self._compute_weights(distances, individual.sigma2, weight_threshold)

        new_sigma = self._update_scale(weights_sum, weights_sum_d2)

        individual.sigma2 = new_sigma
        individual._validate_sigma2_limits()

        fitness = weights_sum / individual.sigma2 if individual.sigma2 > 0.0 else 0.0

        fitness = self.dtype_strategy.array(fitness)

        individual.fitness = fitness

        return individual.fitness

    def _select_best_offspring(self, parent: ECSAGOIndividual, offspring: list[ECSAGOIndividual]) -> ECSAGOIndividual:
        """Selects the best offspring via deterministic crowding.

        Picks the closest non-identical offspring to the parent, then compares
        fitness. Returns the parent if no offspring improves on it.

        Args:
            parent: Parent individual.
            offspring: List of generated offspring.

        Returns:
            Best individual (parent or closest offspring with higher fitness).
        """
        if not offspring:
            return parent

        offspring_genomes = self.dtype_strategy.array([ind.genome for ind in offspring])

        distances = self.dtype_strategy.module.linalg.norm(offspring_genomes - parent.genome, axis=1)

        non_identical_mask = distances > 0

        if self.dtype_strategy.module.any(non_identical_mask):
            filtered_distances = self.dtype_strategy.module.where(non_identical_mask, distances, self.dtype_strategy.module.inf)

            closest_index = int(self.dtype_strategy.module.argmin(filtered_distances))
            closest_offspring = offspring[closest_index]

            if closest_offspring.fitness >= parent.fitness:
                return closest_offspring
            else:
                return parent
        else:
            return parent


class CUDAECSAGOFitnessCalculator(ECSAGOFitnessCalculator):  # pragma: no cover
    """CUDA-accelerated fitness calculator for ECSAGO.

    Uses optimized CUDA kernels for parallel distance, weight, and sigma
    computation. Inherits _select_best_offspring from the CPU version.
    """

    def __init__(self, context: object, dtype_strategy: DataTypeStrategy) -> None:
        """Initializes the CUDA fitness calculator.

        Args:
            context: Algorithm context for accessing CUDAContext.
            dtype_strategy: CuPy data type strategy.
        """
        from ...utils.cuda.kernels import ECSAGOKernels
        super().__init__(dtype_strategy)
        self._context = context
        self.kernels = ECSAGOKernels()

    @property
    def cuda_context(self):
        """Returns the current CUDAContext from the algorithm context."""
        return self._context.cuda_context

    def _report_nan_inf(self, arr: cp.ndarray, var: str | None = None) -> None:
        """Checks for NaN or Inf values in a GPU array and raises if found.

        Args:
            arr: CuPy array to check.
            var: Variable name for logging context.

        Raises:
            TypeError: If arr is not a CuPy array.
            ValueError: If NaN or Inf values are found.
        """
        if not isinstance(arr, self.dtype_strategy.module.ndarray):
            raise TypeError("Parameter must be a cupy.ndarray")

        mask = self.dtype_strategy.module.logical_or(
            self.dtype_strategy.module.isnan(arr),
            self.dtype_strategy.module.isinf(arr)
        )

        if mask.any():
            logger.warning("NaN/Inf detected in %s", var or "unknown")

            indices = self.dtype_strategy.module.where(mask)
            coords = list(zip(*[c.tolist() for c in indices]))

            results = []
            for coord in coords:
                val = arr[coord].item()
                logger.debug("Position %s has value: %s", coord, val)
                results.append((coord, val))

            logger.debug("All NaN/Inf positions: %s", results)
            raise ValueError(f"NaN or Inf values found at positions: {results}")

    def calculate_with_kernels(self, individuals: list[ECSAGOIndividual], weight_threshold: float, max_iterations: int = 10) -> None:
        """Batch fitness calculation using CUDA kernels with iterative hill-climbing.

        Args:
            individuals: List of individuals to evaluate.
            weight_threshold: Threshold for weight binarization.
            max_iterations: Maximum hill-climbing iterations.
        """
        self._report_nan_inf(self.cuda_context.data_gpu, "data")

        if not individuals:
            return

        N, D = self.cuda_context.data_gpu.shape
        M = len(individuals)

        genomes_list = [ind.genome for ind in individuals]
        sigmas_list = [ind.sigma2 for ind in individuals]

        xp = self.dtype_strategy.module
        genomes_gpu = xp.stack(genomes_list, dtype=xp.float64)
        current_sigmas_gpu = self.dtype_strategy.array(sigmas_list)

        best_sigmas_gpu = xp.copy(current_sigmas_gpu)
        best_fitness_gpu = xp.full(M, -xp.inf, dtype=xp.float64)

        sigma_min = individuals[0].context.get_sigma2_min()
        sigma_max = individuals[0].context.get_sigma2_max()

        distances_squared_gpu = self.kernels.calculate_distances(self.cuda_context.data_gpu, genomes_gpu, N, M, D)
        self._report_nan_inf(distances_squared_gpu, "distances_squared_gpu")

        for _ in range(max_iterations):
            binary_weights_gpu = self.kernels.calculate_weights(
                distances_squared_gpu, current_sigmas_gpu, weight_threshold, N, M
            )

            sum_weights_gpu, sum_weighted_d2_gpu = self.kernels.reduce_sums(
                binary_weights_gpu, distances_squared_gpu, N, M
            )

            new_sigmas_gpu, new_fitness_gpu = self.kernels.update_sigma_fitness(
                sum_weights_gpu, sum_weighted_d2_gpu, sigma_min, sigma_max, M
            )

            improvement_mask = new_fitness_gpu > best_fitness_gpu

            if not self.dtype_strategy.module.any(improvement_mask):
                break

            best_fitness_gpu = self.dtype_strategy.module.where(improvement_mask, new_fitness_gpu, best_fitness_gpu)
            best_sigmas_gpu = self.dtype_strategy.module.where(improvement_mask, new_sigmas_gpu, best_sigmas_gpu)

            current_sigmas_gpu = self.dtype_strategy.module.where(improvement_mask, new_sigmas_gpu, current_sigmas_gpu)

        for i, ind in enumerate(individuals):
            ind.sigma2 = best_sigmas_gpu[i]
            ind.fitness = best_fitness_gpu[i]

            ind._validate_sigma2_limits()
