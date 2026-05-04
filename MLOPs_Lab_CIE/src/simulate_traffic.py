import requests
import random
import time

URL = "http://127.0.0.1:9000/infer"

def send_request(data):
    try:
        requests.post(URL, json=data)
    except:
        pass

# 30 normal requests
for _ in range(30):
    data = {
        "prompt_token_count": random.randint(50, 1000),
        "system_prompt_length": random.randint(100, 2000),
        "temperature": round(random.uniform(0, 1.5), 2),
        "is_few_shot": random.randint(0, 1)
    }
    send_request(data)
    time.sleep(0.1)

# 20 drifted requests
for _ in range(20):
    data = {
        "prompt_token_count": random.randint(1000, 2000),
        "system_prompt_length": random.randint(2000, 4000),
        "temperature": round(random.uniform(0, 1.5), 2),
        "is_few_shot": random.randint(0, 1)
    }
    send_request(data)
    time.sleep(0.1)

print("✅ Traffic simulation done")