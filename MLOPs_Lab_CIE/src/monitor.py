import pandas as pd
import json
import os

# Load logs
logs = []

with open("../logs/predictions.jsonl", "r") as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame([l["input"] for l in logs])

# Training means (given in question)
train_prompt_mean = 589.32
train_system_mean = 1067.6

# Live means
live_prompt_mean = df["prompt_token_count"].mean()
live_system_mean = df["system_prompt_length"].mean()

# Thresholds
prompt_threshold = 157.35
system_threshold = 545.06

alerts = []

# Prompt drift
prompt_shift = abs(live_prompt_mean - train_prompt_mean)
alerts.append({
    "feature": "prompt_token_count",
    "train_mean": train_prompt_mean,
    "live_mean": float(live_prompt_mean),
    "shift": float(prompt_shift),
    "threshold": prompt_threshold,
    "status": "ALERT" if prompt_shift > prompt_threshold else "OK"
})

# System drift
system_shift = abs(live_system_mean - train_system_mean)
alerts.append({
    "feature": "system_prompt_length",
    "train_mean": train_system_mean,
    "live_mean": float(live_system_mean),
    "shift": float(system_shift),
    "threshold": system_threshold,
    "status": "ALERT" if system_shift > system_threshold else "OK"
})

# Final output
output = {
    "total_predictions": len(logs),
    "mean_prediction": float(pd.DataFrame(logs)["prediction"].mean()),
    "drift_detected": any(a["status"] == "ALERT" for a in alerts),
    "alerts": alerts
}

os.makedirs("../results", exist_ok=True)

with open("../results/step4_s5.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ step4_s5.json created!")