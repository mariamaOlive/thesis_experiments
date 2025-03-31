from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
import yaml


class BaseAgentCreator:
    """Base class for creating agents with common kernel and prompt template setup."""

    def __init__(self):
        """Initializes the agent creator with a kernel and service."""
        self.kernel = Kernel()

    def _add_chat_completion_kernel(self, service_id: str) -> Kernel:
        """Creates a kernel with a chat completion service."""
        self.kernel.add_service(OpenAIChatCompletion(service_id=service_id))

    def _prompt_template(self, file_path: str) -> PromptTemplateConfig:
        """Reads a YAML file and creates a PromptTemplateConfig."""
        with open(file_path, "r", encoding="utf-8") as file:
            generate_story_yaml = file.read()

        data = yaml.safe_load(generate_story_yaml)
        prompt_template_config = PromptTemplateConfig(**data)
        return prompt_template_config
    
    def add_plugin_kernel(self, name, plugin):
        """Adds plugin to kernel"""
        self.kernel.add_plugin(plugin, plugin_name=name)