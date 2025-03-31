from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

class BaseAgentCreator:
    """Base class for creating agents with common kernel and prompt template setup."""

    def __init__(self, name):
        """Initializes the agent creator with a kernel and service."""
        self.name = name
        self.kernel = Kernel()

    def _add_chat_completion_kernel(self, service_id: str) -> Kernel:
        """Creates a kernel with a chat completion service."""
        self.kernel.add_service(OpenAIChatCompletion(service_id=service_id))
    
    def add_plugin_kernel(self, name, plugin):
        """Adds plugin to kernel"""
        self.kernel.add_plugin(plugin, plugin_name=name)