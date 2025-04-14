import os
from string import Template
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage, TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient


class ExtractorAgent():
    """Agent for extracting information based on a prompt template."""

    def __init__(self, name):
        self.name = name
        # Create an agent that uses the OpenAI GPT-4o model.
        model_client = OpenAIChatCompletionClient(
            model= "gpt-4o-mini",
            api_key= os.getenv('OPENAI_API_KEY'),
        )
        self.agent = AssistantAgent(
            name=name,
            model_client=model_client,
            # tools=[web_search],
            system_message="Use tools to solve tasks.",
        )
       
    def _build_prompt(self, prompt: str, readme_text: str) -> str:
        prompt = Template(prompt)
        prompt = prompt.substitute(readme_text=readme_text)
        return prompt 

    async def run_agent(self, prompt: str, readme_text) -> str:
        """Creates a ChatCompletionAgent for extraction."""
        # Create instruction  prompt
        prompt = self._build_prompt(prompt, readme_text)
        response = await self.agent.on_messages(
        [TextMessage(content=prompt, source="user")],
        cancellation_token=CancellationToken(),
        )
        response_text = response.chat_message.content
        print(f"[{self.name} sent]: {response_text}\n")
        return response_text
    
