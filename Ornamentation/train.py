import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader

from dataloader import SkeletonDataset
from ornamentation_network import OrnamentationUNet, OrnamentationLoss


# =========================
# Config
# =========================
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

TRAIN_RATIO = 0.9
SAVE_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Validation
# =========================
@torch.no_grad()
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        preds = model(inputs)
        loss = criterion(preds, targets)
        total_loss += loss.item()

    return total_loss / len(dataloader)


# =========================
# Training
# =========================
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Dataset
    full_dataset = SkeletonDataset(
        input_dir="skeletons",
        label_dir="labels",
        augment = True
    )

    train_size = int(TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Model
    model = OrnamentationUNet().to(DEVICE)
    criterion = OrnamentationLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    print(f"🚀 Training on {DEVICE}")
    print(f"📊 Train / Val = {train_size} / {val_size}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch}/{NUM_EPOCHS}]",
            ncols=100
        )

        for inputs, targets in pbar:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(inputs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(train_loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        val_loss = validate(model, val_loader, criterion)

        print(
            f"📉 Epoch {epoch} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss
            }, save_path)
            print(f"⭐ Saved best model (val_loss={val_loss:.6f})")

    print("✅ Training finished")


if __name__ == "__main__":
    train()
