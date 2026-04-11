import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.unified_dataset import UnifiedDataset
from model.segmentation_model import SimpleSegmentationModel
import cv2
import os
from tqdm import tqdm

MODEL_PATH = "model.pth"
OUTPUT_DIR = "outputs"
BATCH_SIZE = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = UnifiedDataset("data/unified/metadata.json", img_size=256)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

model = SimpleSegmentationModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.3).float()

    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.3).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + smooth) / (union + smooth)

dice_total = 0
iou_total = 0

with torch.no_grad():
    for i, (images, masks) in enumerate(tqdm(loader)):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        dice_total += dice_score(outputs, masks).item()
        iou_total += iou_score(outputs, masks).item()

        if i < 20:
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.3).float()

            for j in range(images.shape[0]):
                pred_mask = preds[j][0].cpu().numpy() * 255
                gt_mask = masks[j][0].cpu().numpy() * 255

                cv2.imwrite(f"{OUTPUT_DIR}/pred_{i}_{j}.png", pred_mask)
                cv2.imwrite(f"{OUTPUT_DIR}/gt_{i}_{j}.png", gt_mask)

avg_dice = dice_total / len(loader)
avg_iou = iou_total / len(loader)

print("\n===== FINAL RESULTS =====")
print(f"Mean Dice Score: {avg_dice:.4f}")
print(f"Mean IoU Score: {avg_iou:.4f}")