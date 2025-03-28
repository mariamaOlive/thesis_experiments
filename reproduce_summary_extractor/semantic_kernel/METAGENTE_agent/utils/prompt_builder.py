class PromptBuilder:
    @staticmethod
    def _clean_prompt_list(prompt_list: list) -> str:
        cleaned = ""
        for i, prompt in enumerate(prompt_list):
            cleaned += f"Extractor Prompt #{i}:\n{prompt}\n"
        return cleaned

