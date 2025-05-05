import torch
from torch.utils.data import DataLoader
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from torch.optim import AdamW
from risc_dataset import RISCDataset
from data_loader import load_risc_dataset
import wandb
from datetime import datetime
from tqdm import tqdm

#CONFIG
USE_AUGMENTATION = True
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
BATCH_SIZE = 1

now = datetime.now().strftime("%Y-%m-%d_%H-%M")
run_name = f"PaliGemma_{now}_lr{LEARNING_RATE}_aug{USE_AUGMENTATION}"

wandb.init(
    project="DI725_TermProject",
    name=run_name,
    config={
        "learning_rate": LEARNING_RATE,
        "augmentation": USE_AUGMENTATION,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE
    }
)

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Stack images (they are fixed size)
    pixel_values = torch.stack(pixel_values)

    # Pad text inputs dynamically
    encoded = processor.tokenizer.pad(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        },
        padding=True,
        return_tensors="pt"
    )

    labels_padded = processor.tokenizer.pad(
        {"input_ids": labels},
        padding=True,
        return_tensors="pt"
    )["input_ids"]

    return {
        "pixel_values": pixel_values,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels_padded
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma2-3b-mix-224",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
    offload_folder="offload",
    offload_state_dict=True
)
model.gradient_checkpointing_enable()
model.train()

train_data = load_risc_dataset("../data/captions.csv", "../data/resized", "train")
train_dataset = RISCDataset(train_data, processor, augment=USE_AUGMENTATION)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for step, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
        labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        wandb.log({"train_loss": loss.item(), "epoch": epoch + 1, "step": step})
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Avg loss: {avg_loss:.4f}")