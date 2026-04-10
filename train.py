import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset.unified_dataset import UnifiedDataset
from model.segmentation_model import SimpleSegmentationModel
from tqdm import tqdm
import torch.nn.functional as F

BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = UnifiedDataset("data/unified/metadata.json")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = SimpleSegmentationModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    model.eval()
    val_loss = 0
    dice_total = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs)
            dice_total += dice_score(preds, masks).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_dice = dice_total / len(val_loader)

    print(f"\nEpoch {epoch+1}:")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Val Dice: {avg_dice:.4f}")

torch.save(model.state_dict(), "model.pth")
print("\nModel saved as model.pth")