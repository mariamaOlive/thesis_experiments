import os
import re
from string import Template
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage, TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from prompt.prompt import COMBINE_PROMPT

class PromptCombineAgent():
    """Agent for combining prompts into a single one."""

    def __init__(self, name):
        self.name = name
        # Create an agent that uses the OpenAI GPT-4o model.
        model_client = OpenAIChatCompletionClient(
            model= "gpt-4o",
            api_key= os.getenv('OPENAI_API_KEY'),
        )
        self.agent = AssistantAgent(
            name=name,
            model_client=model_client,
            system_message="Use tools to solve tasks.",
        )


    def _build_prompt(self, summarizer_list: str) -> str:
        prompt = Template(COMBINE_PROMPT)
        prompt = prompt.substitute(summarizer_list=summarizer_list)
        return prompt


    def _parse_answer(self, answer: str) -> tuple[str, str]:
        summarizer_prompt = answer
        if "$extracted_text" not in summarizer_prompt:
            summarizer_prompt = re.sub(
                r"\btext\b", "$extracted_text", summarizer_prompt, count=1
            )
        return summarizer_prompt


    def _clean_prompt_list(self, prompt_list: list) -> list:
        ret = ""
        for i, prompt in enumerate(prompt_list):
            ret += f"Extractor Prompts #{i}:\n{prompt}"
        return ret


    async def run_agent(self, prompt_list: list) -> tuple[str, str]:
        """Creates a ChatCompletionAgent for extraction."""
        # Create instruction  prompt
        summarizer_list = self._clean_prompt_list(prompt_list)
        prompt = self._build_prompt(summarizer_list=summarizer_list)
        # Send prompt to agent
        response = await self.agent.on_messages(
        [TextMessage(content=prompt, source="user")],
        cancellation_token=CancellationToken(),
        )
        response_text = response.chat_message.content
        print(f"[{self.name} sent]: {response_text}\n")
        return response_text
    
