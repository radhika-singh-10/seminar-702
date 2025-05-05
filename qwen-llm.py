import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Choose model ===
# model_id = "deepseek-ai/deepseek-llm-7b-instruct"
model_id = "Qwen/Qwen-7B-Chat"  # Or use DeepSeek

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# === Load your JSON ===
with open("all_data.json", "r") as f:
    data = json.load(f)

# === Loop and generate ===
for index, item in enumerate(data):
    row = item["row"]
    if "qwen-response" in row:
        continue

    prompt = row["instructions"]
    print(f"Generating for row_idx: {item['row_idx']}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    row["qwen-response"] = result

    if index == 100:  # optional limit
        break

# === Save file ===
with open("data_with_qwen.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Saved responses to data_with_qwen.json")
