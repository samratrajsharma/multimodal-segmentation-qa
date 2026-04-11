# Prompted Segmentation for Drywall QA

## Overview

This project develops a prompt-conditioned segmentation model capable of
generating binary masks for drywall defects based on natural language
prompts such as: - "segment crack" - "segment taping area"

The system is designed to generalize across multiple defect types using
a unified multimodal pipeline.

------------------------------------------------------------------------

## Goal

Train and fine-tune a segmentation model that takes: - Image - Text
prompt

and outputs: - Binary segmentation mask (PNG, values {0,255})

------------------------------------------------------------------------

## Dataset Preparation

### Datasets Used

-   Crack Dataset (Roboflow)
-   Drywall Taping Dataset (Roboflow)

### Key Preprocessing Steps

-   COCO annotations → binary masks
-   Fallback to bounding boxes where segmentation missing
-   Merged datasets into unified format
-   Fixed filename mismatches (.rf issue)
-   Resolved path inconsistencies (Windows → Colab)
-   Balanced dataset (crack vs taping)

### Final Dataset

-   Total samples after matching: \~6391
-   Balanced dataset: 2044 samples

------------------------------------------------------------------------

## Model Approach

### Baseline Model

-   ResNet18 encoder (pretrained)
-   Simple decoder
-   Prompt conditioning via extra input channel

### Improvements Applied

-   Increased image resolution (256 → 384)
-   Introduced stronger decoder (U-Net style)
-   Combined loss:
    -   BCEWithLogitsLoss
    -   Dice Loss
-   Increased class weighting for imbalance handling
-   Extended training (resume training up to 25 epochs)

------------------------------------------------------------------------

## Training Strategy

### Phase 1 (Baseline)

-   Resolution: 256
-   Epochs: 10
-   Result:
    -   Dice ≈ 0.59
    -   IoU ≈ 0.44

### Phase 2 (Improved Model)

-   Resolution: 384
-   Better decoder
-   Combined loss
-   Initial Result:
    -   Dice ≈ 0.55

### Phase 3 (Resume Training)

-   Continued training from epoch 10 → 25
-   Final Result:
    -   Dice ≈ 0.68
    -   IoU ≈ \~0.50+

------------------------------------------------------------------------

## Final Results

  Metric       Score
  ------------ --------
  Dice Score   \~0.68
  IoU Score    \~0.52

------------------------------------------------------------------------

## Visual Outputs

Each prediction includes: - Original Image - Ground Truth Mask -
Predicted Mask

Saved in:

    outputs/

------------------------------------------------------------------------

## Failure Analysis

-   Thin cracks occasionally under-segmented
-   Coarse boundaries due to bbox-derived masks
-   Resolution vs speed trade-off

------------------------------------------------------------------------

## Runtime & Efficiency

-   Local CPU baseline for debugging
-   GPU training on Google Colab
-   Epoch time reduced after caching (\~40s/epoch)
-   Lightweight model (ResNet18-based)

------------------------------------------------------------------------

## Repository Structure

    dataset/
    model/
    train.py
    evaluate.py
    notebooks/
    README.md

------------------------------------------------------------------------

## Key Engineering Challenges Solved

-   Dataset mismatch and filename inconsistencies
-   Cross-platform path issues
-   Class imbalance handling
-   Efficient GPU training setup
-   Multimodal conditioning implementation

------------------------------------------------------------------------

## Conclusion

A robust prompt-conditioned segmentation pipeline was developed,
achieving strong performance on drywall QA tasks. The system
demonstrates effective handling of real-world data challenges and
scalable model improvements through iterative refinement.

------------------------------------------------------------------------

## Notes

-   Dataset not included due to size constraints
-   Please download datasets from Roboflow links provided
