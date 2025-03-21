# singleton_kernel.py
from semantic_kernel import Kernel

class SingletonKernel:
    _kernel_instance = None

    @classmethod
    def get_kernel(cls) -> Kernel:
        if cls._kernel_instance is None:
            cls._kernel_instance = Kernel()
        return cls._kernel_instance
