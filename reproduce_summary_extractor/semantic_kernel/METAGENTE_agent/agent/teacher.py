import re
from string import Template

from prompt.prompt import TEACHER_PROMPT
from utils.chat_kernel_provider import OpenAIChatProvider, OllamaChatProvider


class TeacherAgent:
    def __init__(self):
        self.llm = OpenAIChatProvider(model="gpt-4o")
        # self.llm = OllamaChatProvider(model="llama3.2")

    def _build_prompt(
        self,
        extracted_text: str,
        description: str,
        generated_about: str,
        rouge_score: float,
        summarizer_prompt: str,
    ) -> str:
        prompt = Template(TEACHER_PROMPT)
        prompt = prompt.substitute(
            extracted_text=extracted_text,
            description=description,
            generated_about=generated_about,
            rouge_score=rouge_score,
            summarizer_prompt=summarizer_prompt,
        )
        return prompt

    def _parse_answer(self, answer: str) -> tuple[str, str]:
        summarizer_prompt = answer
        if "$extracted_text" not in summarizer_prompt:
            summarizer_prompt = re.sub(
                r"\btext\b", "$extracted_text", summarizer_prompt, count=1
            )

        return summarizer_prompt

    async def run(
        self,
        extracted_text: str,
        description: str,
        generated_about: str,
        rouge_score: float,
        summarizer_prompt: str,
    ) -> tuple[str, str]:
        prompt = self._build_prompt(
            extracted_text, description, generated_about, rouge_score, summarizer_prompt
        )
        answer = await self.llm.run(prompt, temperature=0.7)

        # print(f"Teacher raw answer: {answer}")

        summarizer_prompt = self._parse_answer(answer)
        return summarizer_prompt
