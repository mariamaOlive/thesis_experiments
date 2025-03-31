from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments
from .base_agent import BaseAgentCreator 
from utils.prompt_builder import PromptBuilder
from semantic_kernel.connectors.ai.ollama import OllamaChatPromptExecutionSettings

class ExtractorAgent(BaseAgentCreator):
    """Agent for extracting information based on a prompt template."""

    def __init__(self, name):
        """Initializes the ExtractorAgent with specific settings."""
        super().__init__(name) 
        # self.settings = OpenAIChatPromptExecutionSettings(
        #     service_id=name,
        #     ai_model_id="gpt-4o-mini",
        #     temperature=0,
        # )
        
        self.settings = OllamaChatPromptExecutionSettings(
            service_id = name,
            ai_model_id="llama3.2",
            temperature=0,
        )

    def create_agent(self, file_path: str) -> ChatCompletionAgent:
        """Creates a ChatCompletionAgent for extraction."""
        # Create instruction  prompt
        prompt_template = PromptBuilder.prompt_template(file_path)
        # Add chat completion to kernel
        self._add_chat_completion_kernel(self.name, "Ollma")
        # Create Agent
        agent = ChatCompletionAgent(
            kernel=self.kernel,
            name=self.name,
            prompt_template_config=prompt_template,
            arguments=KernelArguments(settings=self.settings),
        )
        return agent
    
