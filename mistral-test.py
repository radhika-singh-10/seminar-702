import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Load Mistral 7B Instruct Model ===
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# === Load your JSON file ===
with open("all_data.json", "r") as f:
    data = json.load(f)

# === Loop through each row ===
for item in data:
    row = item["row"]
    
    # Skip already-processed rows
    if "mistral-local-response" in row:
        continue

    prompt = row["instructions"]
    
    print(f"Processing row_idx: {item['row_idx']}")

    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # Decode and store the result
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    row["mistral-local-response"] = response

# === Save updated JSON ===
with open("data_with_mistral.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ All rows processed and saved to data_with_mistral.json")
