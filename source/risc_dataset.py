from torch.utils.data import Dataset
from PIL import Image

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
        target = random.choice(captions)

        if self.augment:
            from augment_text import augment_captions
            captions = augment_captions(captions)
            target = random.choice(captions)

        inputs = self.processor(image, target, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0)
        }