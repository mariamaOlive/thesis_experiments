from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments
from .base_agent import BaseAgentCreator 

class ExtractorAgent(BaseAgentCreator):
    """Agent for extracting information based on a prompt template."""

    def __init__(self):
        """Initializes the ExtractorAgent with specific settings."""
        self.settings = OpenAIChatPromptExecutionSettings(
            service_id="extractor",
            ai_model_id="gpt-4o-mini",
            temperature=0,
        )

    def create_agent(self, file_path: str) -> ChatCompletionAgent:
        """Creates a ChatCompletionAgent for extraction."""
        prompt_template = self._prompt_template(file_path)

        agent_extractor = ChatCompletionAgent(
            kernel=self._create_kernel_with_chat_completion("extractor"),
            name="extractor",
            prompt_template_config=prompt_template,
            arguments=KernelArguments(settings=self.settings),
        )
        return agent_extractor