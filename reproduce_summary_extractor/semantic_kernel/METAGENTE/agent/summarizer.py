from string import Template
from utils.chat_kernel_provider import OpenAIChatProvider, OllamaChatProvider


class SummarizerAgent:
    def __init__(self):
        # self.llm = OpenAIChatProvider(model="gpt-4o-mini")
        self.llm = OllamaChatProvider(model="llama3.2")

    def _build_prompt(self, prompt: str, extracted_text: str) -> str:
        prompt = Template(prompt)
        prompt = prompt.substitute(extracted_text=extracted_text)
        return prompt

    async def run(self, prompt: str, extracted_text: str) -> str:
        prompt = self._build_prompt(prompt, extracted_text)
        about = await self.llm.run(prompt, temperature=0)
        return about