import json
import requests

# === Config ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

# === Load your JSON file ===
with open("all_data.json", "r") as f:
    dataset = json.load(f)

# === Loop over each row and call the LLM ===
for item in dataset:
    row = item["row"]
    prompt = row["instructions"]

    print(f"Processing row_idx: {item['row_idx']}...")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        output = response.json().get("response", "").strip()
        row["mistral-llm-response"] = output
    else:
        print(f"Failed on row {item['row_idx']}: {response.status_code}")
        row["mistral-llm-response"] = f"Error: {response.status_code}"

# === Save output ===
with open("data_with_mistral.json", "a") as f:
    json.dump(dataset, f, indent=2)

print("✅ All rows processed and saved to data_with_mistral.json")

