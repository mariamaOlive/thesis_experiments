from metric.rouge import ROUGE
from semantic_kernel.functions.kernel_function_decorator import kernel_function

class RougePlugin:
    def __init__(self):
        self.rouge = ROUGE()

    @kernel_function(
        name="get_rouge_l",
        description="Computes the ROUGE-L similarity score between two input texts."
    )
    async def get_rouge_l(self, text_1: str, text_2: str) -> float:
        """Gets the ROUGE-L value between two texts"""
        return self.rouge.get_RougeL(text_1, text_2)
