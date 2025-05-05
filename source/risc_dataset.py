from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import random
from PIL import Image
from augment_text import augment_captions

class RISCDataset(Dataset):
    def __init__(self, data, processor, augment=False):
        self.data = data
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, captions = self.data[idx]
        image = Image.open(image_path).convert("RGB")

        if self.augment:
            captions = augment_captions(captions)

        target = random.choice(captions)
        prompt = "<image> Describe the image."  

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",   
            truncation=True,
            max_length=128          
        )

        labels = self.processor.tokenizer(
            text=target,
            padding="max_length",   
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )["input_ids"]

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }