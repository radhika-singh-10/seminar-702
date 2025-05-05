import json
from huggingface_hub import InferenceClient

# Set your HF token here
HF_TOKEN = "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"

# Initialize HF Inference Client
client = InferenceClient(
    model="google/gemma-7b-it",
    token=HF_TOKEN
)

# Load your JSON file
with open("all_data.json", "r") as f:
    data = json.load(f)

# Loop through each entry
for item in data:
    row = item["row"]
    prompt = row["instructions"]

    print(f"Generating for row_idx: {item['row_idx']}...")

    try:
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["\n\n"]
        )
        row["gemma-hf-response"] = response.strip()

    except Exception as e:
        print(f"Error at row {item['row_idx']}: {e}")
        row["gemma-hf-response"] = f"Error: {str(e)}"

# Save the modified data
with open("data_with_gemma.json", "a") as f:
    json.dump(data, f, indent=2)

print("✅ Done. Output saved to data_with_gemma.json")
