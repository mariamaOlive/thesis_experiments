from rouge_score import rouge_scorer

class ROUGE:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    @staticmethod
    def get_RougeL(string_1: str, string_2: str):
        score = ROUGE.scorer.score(string_1, string_2)
        return score["rougeL"].fmeasure

    @staticmethod
    def get_Rouge1(string_1: str, string_2: str):
        score = ROUGE.scorer.score(string_1, string_2)
        return score["rouge1"].fmeasure

    @staticmethod
    def get_Rouge2(string_1: str, string_2: str):
        score = ROUGE.scorer.score(string_1, string_2)
        return score["rouge2"].fmeasure
