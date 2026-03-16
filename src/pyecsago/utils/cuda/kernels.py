"""kernels.py"""

import cupy as cp

from .compiler import CUDAKernelCompiler


class ECSAGOKernels:
    """CUDA-accelerated kernels for ECSAGO fitness evaluation.

    Only the distance kernel is a custom CUDA kernel (uses shared memory for genomes).
    Weights, reduction, and update use native CuPy operations.
    """
    def __init__(self) -> None:
        self._compile_kernels()

    def _compile_kernels(self) -> None:
        self.distance_kernel = CUDAKernelCompiler.compile_kernel(r'''
        extern "C" __global__ void calculateDistancesKernel(
            const double* data, const double* genomes,
            double* distances_squared, int N, int M, int D)
        {
            int individual_idx = blockIdx.x;
            if (individual_idx >= M) return;

            extern __shared__ double shared_genome[];

            for (int d = threadIdx.x; d < D; d += blockDim.x) {
                shared_genome[d] = genomes[individual_idx * D + d];
            }
            __syncthreads();

            for (int point_idx = threadIdx.x; point_idx < N; point_idx += blockDim.x) {
                double dist_squared = 0.0;
                for (int d = 0; d < D; d++) {
                    double diff = data[point_idx * D + d] - shared_genome[d];
                    dist_squared += diff * diff;
                }
                distances_squared[point_idx * M + individual_idx] = dist_squared;
            }
        }
        ''', 'calculateDistancesKernel')

    def calculate_distances(self, data_gpu: cp.ndarray, genomes_gpu: cp.ndarray, N: int, M: int, D: int) -> cp.ndarray:
        distances_squared_gpu = cp.empty((N, M), dtype=cp.float64)
        threads_per_block = 256
        blocks_per_grid = M
        shared_mem_size = D * 8  # sizeof(double)
        self.distance_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (data_gpu, genomes_gpu, distances_squared_gpu, N, M, D),
            shared_mem=shared_mem_size
        )
        return distances_squared_gpu

    def calculate_weights(self, distances_squared_gpu: cp.ndarray, sigmas_gpu: cp.ndarray, weight_threshold: float, N: int, M: int) -> cp.ndarray:
        """Calculates binary weights from squared distances and sigmas.

        Args:
            distances_squared_gpu: Squared distances matrix (N, M).
            sigmas_gpu: Sigma values per individual (M,).
            weight_threshold: Threshold for weight binarization.
            N: Number of data points.
            M: Number of individuals.

        Returns:
            Binary weights matrix (N, M).
        """
        weights = cp.exp(-distances_squared_gpu / (2.0 * sigmas_gpu[cp.newaxis, :]))
        return cp.where(weights > weight_threshold, 1.0, 0.0)

    def reduce_sums(self, binary_weights_gpu: cp.ndarray, distances_squared_gpu: cp.ndarray, N: int, M: int) -> tuple[cp.ndarray, cp.ndarray]:
        """Reduce binary weights and weighted distances per individual.

        Args:
            binary_weights_gpu: Binary weights matrix (N, M).
            distances_squared_gpu: Squared distances matrix (N, M).
            N: Number of data points.
            M: Number of individuals.

        Returns:
            Tuple of (sum_weights, sum_weighted_d2) arrays of shape (M,).
        """
        sum_weights = cp.sum(binary_weights_gpu, axis=0)
        sum_weighted_d2 = cp.sum(binary_weights_gpu * distances_squared_gpu, axis=0)
        return sum_weights, sum_weighted_d2

    def update_sigma_fitness(self, sum_weights_gpu: cp.ndarray, sum_weighted_d2_gpu: cp.ndarray, sigma_min: float, sigma_max: float, M: int) -> tuple[cp.ndarray, cp.ndarray]:
        """Compute new sigma2 = mean(d²) and fitness = sum_weights / sigma2 per individual."""
        new_sigma2 = cp.where(sum_weights_gpu > 0, sum_weighted_d2_gpu / sum_weights_gpu, sigma_min)
        new_sigma2 = cp.clip(new_sigma2, sigma_min, sigma_max)
        new_fitness = cp.where(new_sigma2 > 0, sum_weights_gpu / new_sigma2, 0.0)
        return new_sigma2, new_fitness
