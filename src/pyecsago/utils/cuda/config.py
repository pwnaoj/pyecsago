"""pyecsago/utils/cuda/config.py"""

import cupy as cp

from dataclasses import dataclass


@dataclass
class BlockConfig:
    """
    Almacena la configuración de bloques y threads para kernels CUDA.
    
    Attributes:
        block_size: Número de threads por bloque
        max_threads: Máximo número de threads permitido por la GPU
        max_blocks: Máximo número de bloques permitido
    """
    block_size: int
    max_threads: int
    max_blocks: int

class CUDABlockManager:
    """
    Gestiona la configuración de bloques y threads para kernels CUDA.
    
    Esta clase determina automáticamente los valores óptimos basándose en
    las capacidades del dispositivo y el tamaño de los datos a procesar.
    """
    
    def __init__(self) -> None:
        """
        Inicializa el gestor obteniendo las capacidades del dispositivo.
        """
        device = cp.cuda.Device(0)
        self.max_threads_per_block = device.attributes['MaxThreadsPerBlock']
        self.max_blocks_per_grid = device.attributes['MaxGridDimX']
        
        # Determinamos un tamaño de bloque base que sea potencia de 2
        # y no exceda el máximo permitido
        self.base_block_size = self._get_optimal_block_size()
    
    def _get_optimal_block_size(self) -> int:
        """
        Calcula un tamaño de bloque óptimo basado en las capacidades del dispositivo.
        
        Returns:
            Tamaño de bloque que es potencia de 2 y no excede el máximo permitido
        """
        # Comenzamos con 256 como valor base común
        block_size = 256
        
        # Reducimos el tamaño si excede el máximo permitido
        while block_size > self.max_threads_per_block:
            block_size //= 2
            
        return block_size
    
    def get_block_config(self, data_size: int) -> BlockConfig:
        """
        Calcula la configuración óptima de bloques y threads para un tamaño de datos.
        
        Args:
            data_size: Número total de elementos a procesar
            
        Returns:
            BlockConfig con la configuración óptima
        """
        # Ajustamos el tamaño de bloque si es necesario
        block_size = min(self.base_block_size, data_size)
        
        # Calculamos el número de bloques necesario
        num_blocks = (data_size + block_size - 1) // block_size
        
        # Limitamos el número de bloques al máximo permitido
        num_blocks = min(num_blocks, self.max_blocks_per_grid)
        
        return BlockConfig(
            block_size=block_size,
            max_threads=self.max_threads_per_block,
            max_blocks=num_blocks
        )
    
    def get_grid_spec(self, data_size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        Obtiene la especificación de grid y bloque para lanzar un kernel.
        
        Args:
            data_size: Número total de elementos a procesar
            
        Returns:
            Tupla (grid_spec, block_spec) para lanzar el kernel
        """
        config = self.get_block_config(data_size)
        return ((config.max_blocks,), (config.block_size,))
