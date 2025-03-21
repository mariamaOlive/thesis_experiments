from string import Template
from utils.chat_kernel_provider import OpenAIChatProvider, OllamaChatProvider


class ExtractorAgent:
    def __init__(self):
        # self.llm = OpenAIChatProvider(model="gpt-4o-mini")
        self.llm = OllamaChatProvider(model="llama3.2")

    def _build_prompt(self, prompt: str, readme_text: str) -> str:
        prompt = Template(prompt)
        prompt = prompt.substitute(readme_text=readme_text)
        return prompt

    async def run(self, prompt: str, readme_text: str) -> str:
        prompt = self._build_prompt(prompt, readme_text)
        extracted_text = await self.llm.run(prompt,  temperature=0)
        return extracted_text