import os
import json
import shutil
from tqdm import tqdm


def merge_dataset(source_dir, target_images, target_masks, prompt):
    images_dir = os.path.join(source_dir, "images")
    masks_dir = os.path.join(source_dir, "masks")

    entries = []

    for img_name in tqdm(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, img_name)

        base_name = img_name.split('.')[0]
        mask_name = f"{base_name}__{prompt.replace(' ', '_')}.png"
        mask_path = os.path.join(masks_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        new_img_name = f"{prompt.replace(' ', '_')}__{img_name}"
        new_mask_name = f"{prompt.replace(' ', '_')}__{mask_name}"

        new_img_path = os.path.join(target_images, new_img_name)
        new_mask_path = os.path.join(target_masks, new_mask_name)

        shutil.copy(img_path, new_img_path)
        shutil.copy(mask_path, new_mask_path)

        entries.append({
            "image": new_img_path,
            "mask": new_mask_path,
            "prompt": prompt
        })

    return entries


def main():
    unified_path = "data/unified"
    images_out = os.path.join(unified_path, "images")
    masks_out = os.path.join(unified_path, "masks")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    all_entries = []

    all_entries += merge_dataset(
        source_dir="data/processed/taping",
        target_images=images_out,
        target_masks=masks_out,
        prompt="segment taping area"
    )

    all_entries += merge_dataset(
        source_dir="data/processed/cracks",
        target_images=images_out,
        target_masks=masks_out,
        prompt="segment crack"
    )

    with open(os.path.join(unified_path, "metadata.json"), "w") as f:
        json.dump(all_entries, f, indent=4)

    print(f"\nTotal samples: {len(all_entries)}")


if __name__ == "__main__":
    main()