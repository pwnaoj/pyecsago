# utils/cuda/compiler.py
import cupy as cp
import tempfile
import os

class CUDAKernelCompiler:
    @staticmethod
    def compile_kernel(source_code: str, kernel_name: str) -> cp.RawKernel:
        """
        Compila un kernel CUDA manejando apropiadamente la codificación.
        
        Args:
            source_code: Código fuente del kernel
            kernel_name: Nombre del kernel
            
        Returns:
            Kernel compilado
        """
        try:
            # Intentar compilación directa
            return cp.RawKernel(source_code, kernel_name)
        except UnicodeEncodeError:
            # Si falla, usar un archivo temporal con codificación explícita
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', 
                                           encoding='utf-8', delete=False) as f:
                f.write(source_code)
                temp_file = f.name
            
            try:
                # Compilar desde el archivo temporal
                kernel = cp.RawKernel(source_code, kernel_name)
                return kernel
            finally:
                # Limpiar archivo temporal
                os.unlink(temp_file)
