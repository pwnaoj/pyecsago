"""kernels.py"""

import cupy as cp

from .compiler import CUDAKernelCompiler


class ECSAGOKernels:
    """
    Centraliza todos los kernels CUDA optimizados para ECSAGO.
    """
    def __init__(self) -> None:
        self._compile_kernels()

    def _compile_kernels(self) -> None:
        """
        Compila todos los kernels CUDA optimizados para ECSAGO.
        """
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
        
        self.weights_kernel = CUDAKernelCompiler.compile_kernel(r'''
        extern "C" __global__ void calculateWeightsKernel(
        const double* distances_squared, const double* sigmas,
        double* binary_weights, double weight_threshold, long long N, long long M) // Usar long long por seguridad
        {
            // --- NUEVO CaLCULO DE iNDICES 2D ---
            long long global_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            long long total_threads = N * M;

            if (global_idx >= total_threads) return;

            // Reconstruir los indices 2D a partir del indice 1D
            int point_idx = global_idx / M;
            int individual_idx = global_idx % M;
            // ------------------------------------

            int idx = point_idx * M + individual_idx; // Este calculo sigue igual
            double dist_squared = distances_squared[idx];
            double sigma2 = sigmas[individual_idx];

            if (sigma2 < 1e-9) {
                binary_weights[idx] = 0.0;
                return;
            }
            double weight = exp(-dist_squared / (2.0 * sigma2));
            binary_weights[idx] = (weight > weight_threshold) ? 1.0 : 0.0;
        }
            ''', 'calculateWeightsKernel')
        
        self.reduction_kernel = CUDAKernelCompiler.compile_kernel(r'''
        extern "C" __global__ void reduceSumsKernelKahan(
            const double* binary_weights, const double* distances_squared,
            double* sum_weights, double* sum_weighted_d2, int N, int M)
        {
            int individual_idx = blockIdx.x;
            if (individual_idx >= M) return;

            int tid = threadIdx.x;
            extern __shared__ double shared_mem[];
            double* shared_weights_sum = shared_mem;
            double* shared_weighted_d2_sum = shared_mem + blockDim.x;
            // Espacio adicional para los errores de compensacion
            double* shared_weights_err = shared_mem + 2 * blockDim.x;
            double* shared_weighted_d2_err = shared_mem + 3 * blockDim.x;

            // Inicializacion con la compensacion de errores
            double local_weights_sum = 0.0;
            double local_weighted_d2_sum = 0.0;
            double weights_err = 0.0;
            double weighted_d2_err = 0.0;

            // Acumulacion con algoritmo de Kahan
            for (int point_idx = tid; point_idx < N; point_idx += blockDim.x) {
                int idx = point_idx * M + individual_idx;
                double weight = binary_weights[idx];

                // Suma compensada para weights
                double y = weight - weights_err;
                double t = local_weights_sum + y;
                weights_err = (t - local_weights_sum) - y;
                local_weights_sum = t;

                if (weight > 0.0) {
                    double weighted_d2 = weight * distances_squared[idx];

                    // Suma compensada para weighted_d2
                    double y2 = weighted_d2 - weighted_d2_err;
                    double t2 = local_weighted_d2_sum + y2;
                    weighted_d2_err = (t2 - local_weighted_d2_sum) - y2;
                    local_weighted_d2_sum = t2;
                }
            }

            // Guardar en memoria compartida
            shared_weights_sum[tid] = local_weights_sum;
            shared_weighted_d2_sum[tid] = local_weighted_d2_sum;
            shared_weights_err[tid] = weights_err;
            shared_weighted_d2_err[tid] = weighted_d2_err;
            __syncthreads();

            // Reduccion en memoria compartida (tambien con compensacion)
            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    double y = shared_weights_sum[tid + s] - shared_weights_err[tid];
                    double t = shared_weights_sum[tid] + y;
                    shared_weights_err[tid] = (t - shared_weights_sum[tid]) - y;
                    shared_weights_sum[tid] = t;

                    double y2 = shared_weighted_d2_sum[tid + s] - shared_weighted_d2_err[tid];
                    double t2 = shared_weighted_d2_sum[tid] + y2;
                    shared_weighted_d2_err[tid] = (t2 - shared_weighted_d2_sum[tid]) - y2;
                    shared_weighted_d2_sum[tid] = t2;
                }
                __syncthreads();
            }

            if (tid == 0) {
                sum_weights[individual_idx] = shared_weights_sum[0];
                sum_weighted_d2[individual_idx] = shared_weighted_d2_sum[0];
            }
        }
        ''','reduceSumsKernelKahan')
        
        self.update_kernel = CUDAKernelCompiler.compile_kernel(r'''
        extern "C" __global__ void updateSigmaFitnessKernel(
            const double* sum_weights, const double* sum_weighted_d2,
            double* output_sigmas, double* output_fitness,
            double sigma_min, double sigma_max, int M)
        {
            int individual_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (individual_idx >= M) return;

            double weights_sum = sum_weights[individual_idx];
            double weighted_d2_sum = sum_weighted_d2[individual_idx];
            double new_sigma2;

            if (weights_sum <= 0.0) {
                new_sigma2 = sigma_min;
            } else {
                new_sigma2 = weighted_d2_sum / weights_sum;
            }
            
            new_sigma2 = fmax(new_sigma2, sigma_min);
            new_sigma2 = fmin(new_sigma2, sigma_max);
            
            output_sigmas[individual_idx] = new_sigma2;
            
            if (new_sigma2 > 0.0) {
                output_fitness[individual_idx] = weights_sum / new_sigma2;
            } else {
                output_fitness[individual_idx] = 0.0;
            }
        }
        ''', 'updateSigmaFitnessKernel')

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
        binary_weights_gpu = cp.empty((N, M), dtype=cp.float64)

        # Calcular el numero de bloques y threads
        threads_per_block = 256
        total_threads = N * M
        blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block
        
        self.weights_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (distances_squared_gpu, sigmas_gpu, binary_weights_gpu,
            cp.float64(weight_threshold), N, M)
        )
        return binary_weights_gpu

    def reduce_sums_kahan(self, binary_weights_gpu: cp.ndarray, distances_squared_gpu: cp.ndarray, N: int, M: int) -> tuple[cp.ndarray, cp.ndarray]:
        """Reduce binary weights and weighted distances using Kahan summation.

        Accumulates sum(binary_weights) and sum(binary_weights * dist_squared)
        per individual, using compensated summation for numerical precision.
        """
        sum_weights_gpu = cp.zeros(M, dtype=cp.float64)
        sum_weighted_d2_gpu = cp.zeros(M, dtype=cp.float64)

        threads_per_block = 256
        blocks_per_grid = M

        # 4 arrays in shared memory (2 for sums, 2 for Kahan error compensation)
        shared_mem_size = threads_per_block * 4 * 8  # 4 arrays × sizeof(double)

        self.reduction_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (binary_weights_gpu, distances_squared_gpu, sum_weights_gpu, sum_weighted_d2_gpu, N, M),
            shared_mem=shared_mem_size
        )

        return sum_weights_gpu, sum_weighted_d2_gpu

    def update_sigma_fitness(self, sum_weights_gpu: cp.ndarray, sum_weighted_d2_gpu: cp.ndarray, sigma_min: float, sigma_max: float, M: int) -> tuple[cp.ndarray, cp.ndarray]:
        """Compute new sigma2 = mean(d²) and fitness = sum_weights / sigma2 per individual."""
        new_sigmas_gpu = cp.empty(M, dtype=cp.float64)
        new_fitness_gpu = cp.empty(M, dtype=cp.float64)
        threads_per_block = 256
        blocks_per_grid = (M + threads_per_block - 1) // threads_per_block
        self.update_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (sum_weights_gpu, sum_weighted_d2_gpu,
             new_sigmas_gpu, new_fitness_gpu,
             cp.float64(sigma_min), cp.float64(sigma_max), M)
        )
        return new_sigmas_gpu, new_fitness_gpu
