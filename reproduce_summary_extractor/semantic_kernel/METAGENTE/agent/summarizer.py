from string import Template

from utils.chat_provider import KernelProvider
from utils.openai_kernel_provider import OpenAIChatProvider


class SummarizerAgent:
    def __init__(self):
        self.llm: KernelProvider = OpenAIChatProvider(model="gpt-4o-mini",  temperature=0)

    def _build_prompt(self, prompt: str, extracted_text: str) -> str:
        prompt = Template(prompt)
        prompt = prompt.substitute(extracted_text=extracted_text)
        return prompt

    async def run(self, prompt: str, extracted_text: str) -> str:
        prompt = self._build_prompt(prompt, extracted_text)
        about = await self.llm.run(prompt)
        return about