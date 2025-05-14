import os
os.environ["TORCH_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"

from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
#import time
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

PROMPT_SETS = {
    "basic": [
        "<image> Write a caption.",
        "<image> Write a description.",
        "<image> Describe the image",
        "<image> Caption the image.",
        "<image> Add a caption."
    ],
    "partial": [
        "<image> What subjects are visible?",
        "<image> Describe the type of area shown.",
        "<image> What is the background?",
        "<image> What is shown in this image?",
        "<image> What are the features of this image?"
    ],
    "descriptive": [
        "<image> Caption this satellite image.",
        "<image> Write a detailed description.",
        "<image> Describe this image in detail.",
        "<image> Write a detailed caption.",
        "<image> Caption what is happening in this scene?"
    ]    
}

#WORDCOUNT_PROMPT = " Use at least 10 words."


# Configuration
CAPTIONS_CSV = "../data/captions.csv"
IMAGES_DIR = "../data/resized"
OUTPUT_JSON = f"out/multi_promptset_captions_beams{NUM_BEAMS}_maxtokens{MAX_NEW_TOKENS}.json"
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
val_df = df[df["split"] == "val"].sample(n=100, random_state=42)

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
    #start_time = time.time()
    image_path = Path(IMAGES_DIR) / row["image"]
    if not image_path.exists():
        continue
    image = Image.open(image_path).convert("RGB")

    gen_by_set = {}
    for set_name, prompts in PROMPT_SETS.items():
        generated_captions = []
        for prompt in prompts:
            #prompt = prompt + WORDCOUNT_PROMPT
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                output = safe_generate(model, inputs, MAX_NEW_TOKENS, NUM_BEAMS)
            caption = processor.decode(output[0], skip_special_tokens=True)
            generated_captions.append(caption)
        gen_by_set[set_name] = generated_captions

    #print(f"{row['image']} took {time.time() - start_time:.2f} seconds")
    
    '''
    # Log only once per image (or remove entirely for speed)
    if set_name == "basic" and prompt == PROMPT_SETS["basic"][0]:
        wandb.log({
            "image": wandb.Image(image),
            "image_filename": row["image"]
        })
    '''
    
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
os.makedirs("out", exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)