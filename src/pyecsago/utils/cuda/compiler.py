"""CUDA kernel compiler with encoding fallback."""

import cupy as cp
import tempfile
import os


class CUDAKernelCompiler:
    @staticmethod
    def compile_kernel(source_code: str, kernel_name: str) -> cp.RawKernel:
        """Compiles a CUDA kernel with encoding fallback.

        Args:
            source_code: Kernel source code.
            kernel_name: Kernel function name.

        Returns:
            Compiled CUDA kernel.
        """
        try:
            return cp.RawKernel(source_code, kernel_name)
        except UnicodeEncodeError:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu',
                                           encoding='utf-8', delete=False) as f:
                f.write(source_code)
                temp_file = f.name

            try:
                kernel = cp.RawKernel(source_code, kernel_name)
                return kernel
            finally:
                os.unlink(temp_file)
