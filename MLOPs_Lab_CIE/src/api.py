from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
import os
from datetime import datetime

app = FastAPI()

model = joblib.load("../models/best_model.pkl")

# ensure logs folder
os.makedirs("../logs", exist_ok=True)

class InputData(BaseModel):
    prompt_token_count: int = Field(..., ge=10, le=2000)
    system_prompt_length: int = Field(..., ge=50, le=4000)
    temperature: float = Field(..., ge=0, le=2)
    is_few_shot: int = Field(..., ge=0, le=1)

@app.get("/ping")
def ping():
    return {"status": "operational", "service": "PromptLab API"}

@app.post("/infer")
def infer(data: InputData):
    features = np.array([[
        data.prompt_token_count,
        data.system_prompt_length,
        data.temperature,
        data.is_few_shot
    ]])

    prediction = model.predict(features)[0]

    # log prediction
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": data.dict(),
        "prediction": float(prediction)
    }

    with open("../logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": float(prediction)}