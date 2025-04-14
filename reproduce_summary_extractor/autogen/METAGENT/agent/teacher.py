from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken
from typing import List, Sequence
from string import Template
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.agents import BaseChatAgent
from prompt.prompt import (
    TEACHER_PROMPT,
)
from metric.rouge import ROUGE
import os



class TeacherAgent(BaseChatAgent):
    def __init__(self, name: str, description: str, extracted_text: str, ground_truth: str) -> None:
        super().__init__(name, description=description)
        self._message_history: List[BaseChatMessage] = []
        self.extracted_text = extracted_text
        self.ground_truth = ground_truth

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        # Update the message history.
        self._message_history.extend(messages)

        # Get information to the Teacher
        summarizer_generated = self._message_history[-1].content
        summarizer_prompt = self._message_history[-2].content
        rouge_score = ROUGE.get_RougeL(self.ground_truth, summarizer_generated)
        
        # Generete prompt
        prompt = self._build_prompt(self.extracted_text, self.ground_truth, summarizer_generated, rouge_score, summarizer_prompt)

        # Call the LLM to generate a response
        result = await self._call_llm(prompt)

        # Create a new message with the result.
        response_message = TextMessage(content=str(result.content), source=self.name)

        # Update the message history.
        self._message_history.append(response_message)

        # Return the response.
        print(f"[{self.name} sent]: {response_message.content}\n")
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    def _build_prompt(
        self,
        extracted_text: str,
        ground_truth: str, 
        generated_about: str, 
        rouge_score: float,
        summarizer_prompt: str,
    ) -> str:
        prompt = Template(TEACHER_PROMPT)
        prompt = prompt.substitute(
            extracted_text=extracted_text,
            description=ground_truth,
            generated_about=generated_about,
            rouge_score=rouge_score,
            summarizer_prompt=summarizer_prompt,
        )
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        # Call the LLM model (e.g., OpenAI API) with the constructed prompt
        try:
            # Initialize the OpenAIChatCompletionClient with your API key
            api_key = os.getenv('OPENAI_API_KEY')
            openai_model_client = OpenAIChatCompletionClient(model="gpt-4o", api_key=api_key)
            result = await openai_model_client.create([UserMessage(content=prompt, source="user")])
            await openai_model_client.close()
            return result
        except Exception as e:
            return f"Error in LLM call: {str(e)}"





