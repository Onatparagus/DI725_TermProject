import os
os.environ["TORCH_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"

from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
import wandb
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

NUM_BEAMS = 1
MAX_NEW_TOKENS = 50

now = datetime.now().strftime("%Y-%m-%d_%H-%M")
run_name = f"PaliGemma_{now}_beams{NUM_BEAMS}_maxtokens{MAX_NEW_TOKENS}"

wandb.init(
    project="DI725_TermProject",
    name=run_name,
    config={
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": NUM_BEAMS
    }
)
wandb.init(project="DI725_Inference", name="PaliGemma_Prompt_Benchmark")

PROMPT_SETS = {
    "basic": [
        "<image> Describe the image.",
        "<image> Write a caption.",
        "<image> What is happening?",
        "<image> What do you see?",
        "<image> Give a description."
    ],
    "partial": [
        "<image> What are the subject/subjects of this image?",
        "<image> What is the location of this image?",
        "<image> What is the background of this image?",
        "<image> What are the contents of this image?",
        "<image> What are the features of this image?"
    ],
    "descriptive": [
        "<image> Describe this satellite image in detail.",
        "<image> Write a rich caption.",
        "<image> What is shown in this aerial view?",
        "<image> Provide a full description.",
        "<image> Summarize the scene."
    ]    
}

# Configuration
CAPTIONS_CSV = "../data/captions.csv"
IMAGES_DIR = "../data/resized"
OUTPUT_JSON = "out/multi_promptset_captions.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma2-3b-mix-224",
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    attn_implementation="eager"
)
processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")
model.eval()

# Load validation data
df = pd.read_csv(CAPTIONS_CSV)
val_df = df[df["split"] == "val"]

@torch._dynamo.disable
def safe_generate(model, inputs, max_new_tokens, num_beams):
    return model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        use_cache=True
    )

# Process each image across all prompt sets
results = []
for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Generating multi-prompt captions"):
    image_path = Path(IMAGES_DIR) / row["image"]
    if not image_path.exists():
        continue
    image = Image.open(image_path).convert("RGB")

    gen_by_set = {}
    for set_name, prompts in PROMPT_SETS.items():
        generated_captions = []
        for prompt in prompts:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output = safe_generate(model, inputs, MAX_NEW_TOKENS, NUM_BEAMS)
            caption = processor.decode(output[0], skip_special_tokens=True)
            generated_captions.append(caption)
        gen_by_set[set_name] = generated_captions

    wandb.log({
        "image": wandb.Image(image),
        "prompt_set": set_name,
        "prompt": prompt,
        "caption": caption,
        "image_filename": row["image"]
    })
    
    results.append({
        "image": row["image"],
        "reference_captions": [row[f"caption_{i}"] for i in range(1, 6)],
        "generated_captions": gen_by_set
    })
    
print(results[0])

# Save to file
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)