from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments
from .base_agent import BaseAgentCreator 
from utils.prompt_builder import PromptBuilder

class EvaluatorAgent(BaseAgentCreator):
    """Agent for evaluating the result of the summarizer."""

    def __init__(self, name):
        """Initializes the Agent with specific settings."""
        super().__init__(name) 
        self.settings = OpenAIChatPromptExecutionSettings(
            service_id=name,
            ai_model_id="gpt-4o-mini",
            temperature=0,
        )

    def create_agent(self, file_path: str, ground_truth: str) -> ChatCompletionAgent:
        """Creates a ChatCompletionAgent for evaluation."""
        # Create instruction  prompt
        prompt_template = PromptBuilder.prompt_template(file_path)
        # Add chat completion to kernel
        self._add_chat_completion_kernel(self.name)
        # Create Agent
        agent = ChatCompletionAgent(
            kernel=self.kernel,
            name=self.name,
            prompt_template_config=prompt_template,
            arguments=KernelArguments(ground_truth=ground_truth, settings= self.settings),
        )
        return agent