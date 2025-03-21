import os
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings

from semantic_kernel.contents import ChatHistory

class OpenAIChatProvider():
    def __init__(self, model):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.service_id = None

        # Create chat service
        self.chat_completion_service = OpenAIChatCompletion(
            service_id=self.service_id,
            ai_model_id=self.model,
            api_key=self.api_key,
        )
    

    async def run(self, prompt,  temperature: float= 0.7):
        # Create a temporary chat history for this prompt only
        chat_history = ChatHistory()
        chat_history.add_user_message(prompt)
        
        # Set up prompt execution settings
        self.execution_settings = OpenAIChatPromptExecutionSettings(
            service_id=self.service_id,
            ai_model_id=self.model,
            temperature=temperature,
        )

        # Generate response
        response = await self.chat_completion_service.get_chat_message_content(
            chat_history=chat_history,
            settings=self.execution_settings,
        )

        return str(response)
    
    
class OllamaChatProvider(OpenAIChatProvider):
    def __init__(self, model):
        super().__init__(model)
        self.model = model

        self.chat_completion_service = OllamaChatCompletion(
            ai_model_id= model,
            service_id= self.service_id
        )
        
        async def run(self, prompt,  temperature: float= 0.7):
            # Create a temporary chat history for this prompt only
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Set up prompt execution settings
            self.execution_settings = OllamaChatPromptExecutionSettings(
                service_id=self.service_id,
                ai_model_id=self.model,
                temperature=temperature,
            )

            # Generate response
            response = await self.chat_completion_service.get_chat_message_content(
                chat_history=chat_history,
                settings=self.execution_settings,
            )

            return str(response)

