import json
import cv2
import torch
from torch.utils.data import Dataset


class UnifiedDataset(Dataset):
    def __init__(self, metadata_path, img_size=256):
        with open(metadata_path, 'r') as f:
            self.data = json.load(f)

        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = cv2.imread(item["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(item["mask"], 0)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        image = image / 255.0
        mask = mask / 255.0

        if item["prompt"] == "segment crack":
            prompt_value = 1.0
        else:
            prompt_value = 0.0

        h, w, _ = image.shape
        prompt_channel = torch.full((h, w, 1), prompt_value)

        image = torch.tensor(image, dtype=torch.float32)
        image = torch.cat([image, prompt_channel], dim=2)

        image = image.permute(2, 0, 1)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask