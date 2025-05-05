import torch
from torch.utils.data import DataLoader
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor, AdamW
from risc_dataset import RISCDataset
from data_loader import load_risc_dataset
import wandb

wandb.init(project="DI725_ImageCaptioning", name="PaliGemma_Finetune")

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
train_dataset = RISCDataset(train_data, processor, augment=True)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
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
        wandb.log({"train_loss": loss.item()})
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()