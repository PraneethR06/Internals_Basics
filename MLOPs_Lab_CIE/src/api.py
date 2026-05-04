from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("../models/best_model.pkl")

# Input schema
class InputData(BaseModel):
    prompt_token_count: int = Field(..., ge=10, le=2000)
    system_prompt_length: int = Field(..., ge=50, le=4000)
    temperature: float = Field(..., ge=0, le=2)
    is_few_shot: int = Field(..., ge=0, le=1)

# Health check
@app.get("/ping")
def ping():
    return {"status": "operational", "service": "PromptLab API"}

# Prediction endpoint
@app.post("/infer")
def infer(data: InputData):
    features = np.array([[
        data.prompt_token_count,
        data.system_prompt_length,
        data.temperature,
        data.is_few_shot
    ]])

    prediction = model.predict(features)[0]

    return {"prediction": float(prediction)}