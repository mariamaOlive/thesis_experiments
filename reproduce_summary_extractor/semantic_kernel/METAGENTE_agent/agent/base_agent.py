from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings


class BaseAgentCreator:
    """Base class for creating agents with common kernel and prompt template setup."""

    def __init__(self, name):
        """Initializes the agent creator with a kernel and service."""
        self.name = name
        self.kernel = Kernel()

    def _add_chat_completion_kernel(self, service_id: str, type:str = "OpenAI") -> Kernel:
        """Creates a kernel with a chat completion service."""
        
        if type=='OpenAI':
            self.kernel.add_service(OpenAIChatCompletion(service_id=service_id))
        else:
            self.kernel.add_service(OllamaChatCompletion(service_id=service_id, ai_model_id="llama3.2"))
    
    def add_plugin_kernel(self, name, plugin):
        """Adds plugin to kernel"""
        self.kernel.add_plugin(plugin, plugin_name=name)