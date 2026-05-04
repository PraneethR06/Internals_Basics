import pandas as pd
import json
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow

# Load data
df = pd.read_csv("../data/training_data.csv")

X = df.drop("response_quality_score", axis=1)
y = df["response_quality_score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("promptlab-response-quality-score")

results = []

# ================= SVR =================
with mlflow.start_run(run_name="SVR"):
    svr = SVR()
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mlflow.log_param("model", "SVR")
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.set_tag("domain", "prompt_engineering")

    results.append({
        "name": "SVR",
        "mae": float(mae),
        "rmse": float(rmse)
    })

# ========== Gradient Boosting ==========
with mlflow.start_run(run_name="GradientBoosting"):
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mlflow.log_param("model", "GradientBoosting")
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

    results.append({
        "name": "GradientBoosting",
        "mae": float(mae),
        "rmse": float(rmse)
    })

# Find best model
best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "promptlab-response-quality-score",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

# Save JSON
os.makedirs("../results", exist_ok=True)
with open("../results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

import joblib
os.makedirs("../models", exist_ok=True)
joblib.dump(gbr, "../models/best_model.pkl")
print("✅ step1_s1.json created!")