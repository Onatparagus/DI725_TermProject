from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from data_loader import load_risc_dataset
from evaluate import load
from PIL import Image
import torch

import nltk
nltk.download('punkt')

bleu = load("bleu")
meteor = load("meteor")
rouge = load("rouge")
cider = load("cider")

device = "cuda"

model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-mix-224").to(device)
processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

val_data = load_risc_dataset("../data/captions.csv", "../data/resized", "val")

predictions = []
references = []

for image_path, caps in val_data[:200]:  # evaluate on a small subset
    image = Image.open(image_path).convert("RGB")
    prompt = "Describe the image."

    inputs = processor(image, prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=50)
    pred = processor.decode(output[0], skip_special_tokens=True)

    predictions.append(pred)
    references.append([cap for cap in caps])

results = {
    "BLEU": bleu.compute(predictions=predictions, references=references),
    "METEOR": meteor.compute(predictions=predictions, references=references),
    "ROUGE-L": rouge.compute(predictions=predictions, references=references),
    "CIDEr": cider.compute(predictions=predictions, references=references),
}

for k, v in results.items():
    print(k, v)