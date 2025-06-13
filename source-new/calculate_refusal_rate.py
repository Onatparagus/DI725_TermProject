import os
import json
import pandas as pd

# 🔧 CONFIGURATION
INPUT_DIR = "out/final/multi_promptset_captions_beams1_maxtokens50 3.json"  # 👈 Your folder path
REFUSAL_PHRASES = [
    "Sorry, as a base VLM I am not trained to answer this question.",
    "unanswerable",
    "no",
    "nothing",
    "no image"
]

if INPUT_DIR.endswith(".json"):
    first_json_path = INPUT_DIR     
else:
    # 🔍 Find first JSON file
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError("No JSON files found in the directory.")

    first_json_path = os.path.join(INPUT_DIR, json_files[0])
print(f"Using file: {first_json_path}")

# 📖 Load the data
with open(first_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 📊 Calculate refusal rate
total = 0
refusals = 0
refusal_details = []

for item in data:
    for set_name, captions in item.get("generated_captions", {}).items():
        for caption in captions:
            total += 1
            response = caption.split("\n", 1)[-1].strip().lower()
            if any(response == phrase.lower() for phrase in REFUSAL_PHRASES):
                refusals += 1
                refusal_details.append({
                    "image": item["image"],
                    "prompt_set": set_name,
                    "caption": caption
                })

rate = 100 * refusals / total if total else 0
print(f"\n📈 Refusal rate: {rate:.2f}% ({refusals}/{total})")

# 🧾 Optional: Save details to CSV
if refusal_details:
    df = pd.DataFrame(refusal_details)
    if INPUT_DIR.endswith(".json"):
        head, tail = os.path.split(INPUT_DIR)
        out_csv = os.path.join(os.path.dirname(INPUT_DIR) , "refusal_details for " + tail + ".csv")
    else:
        out_csv = os.path.join(INPUT_DIR, "refusal_details for " + INPUT_DIR + ".csv")
    df.to_csv(out_csv, index=False)
    print(f"📝 Refusal details saved to {out_csv}")
