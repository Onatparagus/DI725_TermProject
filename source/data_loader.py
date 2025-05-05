import pandas as pd
import os
from PIL import Image

def load_risc_dataset(captions_path, images_dir, split="train"):
    df = pd.read_csv(captions_path)
    df = df[df['split'] == split]

    data = []
    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, row['image'])
        if os.path.exists(image_path):
            captions = [row[f'caption_{i}'] for i in range(1, 6)]
            data.append((image_path, captions))
    return data