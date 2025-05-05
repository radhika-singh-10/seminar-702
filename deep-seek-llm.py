import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Choose model ===
model_id = "deepseek-ai/DeepSeek-V2"

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# === Load your JSON ===
with open("all_data.json", "r") as f:
    data = json.load(f)

# === Format prompt if needed ===
def format_prompt(instruction):
    return f"<|user|>\n{instruction.strip()}\n<|assistant|>"

# === Loop and generate ===
for index, item in enumerate(data):
    row = item["row"]

    if "deep-seek-response" in row:
        continue  # Skip if already processed

    prompt = format_prompt(row["instructions"])
    print(f"🔄 Generating for row_idx: {item['row_idx']}")

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        result = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        row["deep-seek-response"] = result

    except Exception as e:
        print(f"❌ Error at row {item['row_idx']}: {e}")
        row["deep-seek-response"] = f"Error: {str(e)}"

    if index == 100:  # Optional limit
        break

# === Save output ===
with open("data_with_deep_seek.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Finished: responses saved to data_with_deep_seek.json")
