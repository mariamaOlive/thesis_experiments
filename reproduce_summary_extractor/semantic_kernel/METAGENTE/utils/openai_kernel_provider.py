import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.contents import ChatHistory

from utils.chat_provider import KernelProvider


class OpenAIChatProvider(KernelProvider):
    def __init__(self, model: str = "gpt-4", temperature: float= 0.7):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.service_id = None
        
        # Set up prompt execution settings
        self.execution_settings = OpenAIChatPromptExecutionSettings(
            service_id=self.service_id,
            ai_model_id=self.model,
            temperature=self.temperature,
        )

        # Create chat service
        self.chat_completion_service = OpenAIChatCompletion(
            service_id=self.service_id,
            ai_model_id=self.model,
            api_key=self.api_key,
        )
    

    async def run(self, prompt):
        # Create a temporary chat history for this prompt only
        chat_history = ChatHistory()
        chat_history.add_user_message(prompt)

        # Generate response
        response = await self.chat_completion_service.get_chat_message_content(
            chat_history=chat_history,
            settings=self.execution_settings,
        )

        return str(response)