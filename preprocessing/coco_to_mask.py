import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil


def create_dirs(base_path):
    images_out = os.path.join(base_path, "images")
    masks_out = os.path.join(base_path, "masks")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    return images_out, masks_out


def convert_coco_to_masks(input_dir, output_dir, prompt_name):
    images_path = os.path.join(input_dir, "images")
    ann_path = os.path.join(input_dir, "annotations.json")

    coco = COCO(ann_path)
    img_ids = coco.getImgIds()

    images_out, masks_out = create_dirs(output_dir)

    print(f"\nProcessing dataset: {input_dir}")
    print(f"Total images: {len(img_ids)}")

    print("Categories:", coco.loadCats(coco.getCatIds()))

    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]

        height, width = img_info["height"], img_info["width"]
        mask = np.zeros((height, width), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        valid_pixel_found = False

        for ann in anns:
            if "segmentation" in ann and len(ann["segmentation"]) > 0:
                try:
                    m = coco.annToMask(ann)
                    if m.sum() > 0:
                        mask = np.maximum(mask, m)
                        valid_pixel_found = True
                except Exception:
                    continue

            elif "bbox" in ann:
                x, y, w, h = ann["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)

                mask[y:y+h, x:x+w] = 1
                valid_pixel_found = True

        mask = (mask * 255).astype(np.uint8)

        src_img_path = os.path.join(images_path, file_name)
        dst_img_path = os.path.join(images_out, file_name)

        if not os.path.exists(dst_img_path):
            shutil.copy(src_img_path, dst_img_path)

        mask_name = file_name.split('.')[0] + f"__{prompt_name}.png"
        mask_path = os.path.join(masks_out, mask_name)

        cv2.imwrite(mask_path, mask)

        if not valid_pixel_found:
            print(f"[WARNING] No valid mask for image: {file_name}")

    print(f"Finished processing: {input_dir}")


if __name__ == "__main__":
    convert_coco_to_masks(
        input_dir="data/raw/taping",
        output_dir="data/processed/taping",
        prompt_name="segment_taping_area"
    )

    convert_coco_to_masks(
        input_dir="data/raw/cracks",
        output_dir="data/processed/cracks",
        prompt_name="segment_crack"
    )