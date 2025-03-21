from fastapi import FastAPI, Body, HTTPException, Header
from pydantic import BaseModel
from rouge_score import rouge_scorer

app = FastAPI()

# Define a hardcoded API key (replace this with your own key)
EXPECTED_API_KEY = "123456"  # TODO: Replace with your actual API key

# Define request model
class InputData(BaseModel):
    point: str
    params: dict

@app.post("/rouge")
async def dify_receive(data: InputData = Body(...), authorization: str = Header(None)):
    """
    Receive API query data from Dify.
    """
    # Check API key authentication
    auth_scheme, _, api_key = authorization.partition(" ")

    if auth_scheme.lower() != "bearer" or api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    point = data.point

    # Debugging logs
    print(f"Received point: {point}")
    print(f"Params: {data.params}")

    # Ping-Pong Test
    if point == "ping":
        return {"result": "pong"}

    # Call ROUGE computation inside the handler function
    if point == "app.external_data_tool.query":
        return handle_app_external_data_tool_query(params=data.params)

    raise HTTPException(status_code=400, detail="Not implemented")

def handle_app_external_data_tool_query(params: dict):
    """
    Handles external data tool query and computes ROUGE score.
    """
    # Extract expected parameters
    ground_truth = params.get("ground_truth")
    generated = params.get("generated")

    # Debugging logs
    print(f"Ground Truth: {ground_truth}")
    print(f"Generated: {generated}")

    # Validate input
    if not ground_truth or not generated:
        return {"error": "Both 'ground_truth' and 'generated' text are required"}

    # Compute ROUGE scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, generated)

    return {
        "result": {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }
    }
