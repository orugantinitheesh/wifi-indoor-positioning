from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_position


app = FastAPI(
    title="Wi-Fi Indoor Positioning API",
    description="Predicts indoor longitude and latitude from Wi-Fi signal readings.",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    wifi_signals: dict[str, float]


@app.get("/health")
def health():
    """Simple service health check."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Predict indoor coordinates.

    Send detected access points as a JSON dictionary. Any omitted WAP is
    handled as not detected.
    """
    try:
        return predict_position(request.wifi_signals)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error