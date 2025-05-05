import torch
from torch.utils.data import DataLoader
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
from torch.optim import AdamW
from risc_dataset import RISCDataset
from data_loader import load_risc_dataset
import wandb
from datetime import datetime

#CONFIG
USE_AUGMENTATION = True
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
BATCH_SIZE = 2

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma2-3b-mix-224",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
model.train()

train_data = load_risc_dataset("../data/captions.csv", "../data/resized", "train")
train_dataset = RISCDataset(train_data, processor, augment=USE_AUGMENTATION)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=labels,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        wandb.log({"train_loss": loss.item(), "epoch": epoch + 1})
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
