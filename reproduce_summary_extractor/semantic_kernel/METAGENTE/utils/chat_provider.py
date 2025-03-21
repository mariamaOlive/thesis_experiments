
from abc import ABC, abstractmethod
import semantic_kernel as sk

class KernelProvider(ABC):
    @abstractmethod
    def run(self) -> str:
        """Return a configured Semantic Kernel instance."""
        pass