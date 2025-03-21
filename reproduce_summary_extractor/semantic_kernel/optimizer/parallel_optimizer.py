from metric.rouge import ROUGE
from prompt.prompt import (
    COMBINE_PROMPT,
    EXTRACTOR_PROMPT,
    INITIAL_SUMMARIZER_PROMPT,
    TEACHER_PROMPT,
)


class ParallelOptimizer:
    def __init__(self, threshold: float = 0.7):
        self.extractor_prompt = EXTRACTOR_PROMPT
        self.summarizer_prompt = INITIAL_SUMMARIZER_PROMPT
        # self.extractor_agent = ExtractorAgent()
        # self.teacher_agent = TeacherAgent()
        # self.summarizer_agent = SummarizerAgent()
        # self.prompt_combine = PromptCombineAgent()
        self.debug_result = {}
        self.threshold = threshold
