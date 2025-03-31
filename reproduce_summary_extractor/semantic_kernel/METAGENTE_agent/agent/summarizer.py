from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments
from .base_agent import BaseAgentCreator 

class SummarizerAgent(BaseAgentCreator):
    """Agent for extracting information based on a prompt template."""

    def __init__(self):
        """Initializes the Agent with specific settings."""
        super().__init__() 
        self.settings = OpenAIChatPromptExecutionSettings(
            service_id="summarizer",
            ai_model_id="gpt-4o-mini",
            temperature=0,
        )

    def create_agent(self, file_path: str, extracted_text: str) -> ChatCompletionAgent:
        """Creates a ChatCompletionAgent for extraction."""
        # Create instruction  prompt
        prompt_template = self._prompt_template(file_path)
        # Add chat completion to kernel
        self._add_chat_completion_kernel("summarizer")
        # Create Agent
        agent_summarizer = ChatCompletionAgent(
            kernel=self.kernel,
            name="summarizer",
            prompt_template_config=prompt_template,
            arguments=KernelArguments(extracted_text=extracted_text, settings= self.settings),
        )
        return agent_summarizer