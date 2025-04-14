from semantic_kernel.prompt_template import PromptTemplateConfig
import yaml

class PromptBuilder:
    @staticmethod
    def _clean_prompt_list(prompt_list: list) -> str:
        cleaned = ""
        for i, prompt in enumerate(prompt_list):
            cleaned += f"Extractor Prompt #{i}:\n{prompt}\n"
        return cleaned

    @staticmethod
    def prompt_template(file_path: str) -> PromptTemplateConfig:
        """Reads a YAML file and creates a PromptTemplateConfig."""
        with open(file_path, "r", encoding="utf-8") as file:
            generate_story_yaml = file.read()

        data = yaml.safe_load(generate_story_yaml)
        prompt_template_config = PromptTemplateConfig(**data)
        return prompt_template_config
