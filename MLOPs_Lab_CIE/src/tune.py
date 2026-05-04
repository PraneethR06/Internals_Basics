import pandas as pd
import json
import os
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow

# Load data
df = pd.read_csv("../data/training_data.csv")

X = df.drop("response_quality_score", axis=1)
y = df["response_quality_score"]

# Split (as per instructions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("promptlab-response-quality-score")

# Parameter grid
param_dist = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7]
}

# Model
model = GradientBoostingRegressor(random_state=42)

# Random Search (5-fold CV)
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42
)

# Parent MLflow run
with mlflow.start_run(run_name="tuning-promptlab") as parent_run:

    search.fit(X_train, y_train)

    total_trials = len(search.cv_results_["params"])

    # Log each trial as nested run
    for i, params in enumerate(search.cv_results_["params"]):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            rmse = -search.cv_results_["mean_test_score"][i]
            mlflow.log_metric("rmse", rmse)

    # Best model
    best_model = search.best_estimator_
    best_params = search.best_params_

    # Evaluate on test data
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Save results JSON
output = {
    "search_type": "random",
    "n_folds": 5,
    "total_trials": total_trials,
    "best_params": best_params,
    "best_mae": float(mae),
    "best_cv_mae": float(-search.best_score_),
    "parent_run_name": "tuning-promptlab"
}

os.makedirs("../results", exist_ok=True)

with open("../results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ step2_s2.json created!")