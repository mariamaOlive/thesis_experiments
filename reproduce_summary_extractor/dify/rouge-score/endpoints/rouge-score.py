from typing import Mapping
from werkzeug import Request, Response
from flask import Flask, jsonify
from dify_plugin import Endpoint
from rouge_score import rouge_scorer

app = Flask(__name__)

class RougeScoreEndpoint(Endpoint):
 
    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        # reference = values.get("reference", "")
        # hypothesis = values.get("hypothesis", "")

        # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        # scores = scorer.score(reference, hypothesis)

        # result = {
        #     "rouge1": {
        #         "precision": round(scores['rouge1'].precision, 4),
        #         "recall": round(scores['rouge1'].recall, 4),
        #         "f1": round(scores['rouge1'].fmeasure, 4),
        #     },
        #     "rouge2": {
        #         "precision": round(scores['rouge2'].precision, 4),
        #         "recall": round(scores['rouge2'].recall, 4),
        #         "f1": round(scores['rouge2'].fmeasure, 4),
        #     },
        #     "rougeL": {
        #         "precision": round(scores['rougeL'].precision, 4),
        #         "recall": round(scores['rougeL'].recall, 4),
        #         "f1": round(scores['rougeL'].fmeasure, 4),
        #     }
        # }

        # with app.app_context():
        #     return jsonify(result)
        """
        Invokes the endpoint with the given request.
        """
        app_id = values["app_id"]
        def generator():
            yield f"{app_id} <br>"
        return Response(generator(), status=200, content_type="text/html")

