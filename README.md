# ISIC 2016 Challenge - Lesion Segmentation

## Challenge Overview

This project implements a solution for Part 1 of the ISIC 2016 Challenge: **Lesion Segmentation** from dermoscopic images. The challenge is part of the International Skin Imaging Collaboration (ISIC) and focuses on developing automated image analysis tools for melanoma detection from dermoscopic images.

**Challenge Website:** [ISIC 2016 Challenge](https://challenge.isic-archive.com/landing/2016/)

The primary goal is to develop a model that can automatically segment skin lesions (benign or malignant) from dermoscopic images, separating the lesion area from background normal skin and other structures. The performance is evaluated using metrics such as the Dice coefficient, pixel-level accuracy, specificity, sensitivity, and average precision.

## Dataset

The dataset consists of **~900 dermoscopic images** from the ISIC Archive, with expert-annotated ground truth segmentations. The dataset contains a representative mix of both malignant and benign skin lesions.

### Dataset Structure

```
data/
├── original/      # Original dermoscopic images (.jpg)
└── gt/            # Ground truth segmentation masks (.png)
```

- **`data/original/`**: Contains the original dermoscopic images in JPG format
- **`data/gt/`**: Contains the ground truth binary segmentation masks in PNG format. Each mask file is named as `{image_name}_Segmentation.png`

The dataset is automatically split into three sets:
- **Training**: 70% of the data (630 images)
- **Validation**: 20% of the data (180 images)
- **Evaluation**: 10% of the data (90 images)

## Baseline Model

The baseline model is a **U-Net architecture** designed for binary semantic segmentation of skin lesions from dermoscopic images.

### Architecture Details

- **Input**: RGB images of size 256×256 pixels
- **Output**: Binary segmentation masks (single channel) of size 256×256 pixels

#### Encoder Path (Contracting)
- 4 encoder blocks with skip connections
- Channel progression: 3 → 64 → 128 → 256 → 512 → 1024
- Each encoder block consists of two 3×3 convolutions with batch normalization, ReLU activation, and max pooling

#### Decoder Path (Expanding)
- 4 decoder blocks with skip connections from corresponding encoder levels
- Channel progression: 1024 → 512 → 256 → 128 → 64 → 32 → 1
- Each decoder block performs upsampling, concatenation with skip connections, followed by two 3×3 convolutions

### Training Configuration

- **Loss Function**: Binary Cross Entropy Loss (BCELoss)
- **Optimizer**: Adam with learning rate 0.001
- **Epochs**: 10
- **Batch Size**: 8
- **Image Preprocessing**: 
  - Resize to 256×256
  - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Mask Preprocessing**: Resize to 256×256 and convert to binary (0/1)

The model is trained for 10 epochs and saves the best performing model based on validation loss. The trained model weights are saved as `best_segmentation_model.pth`.


## Files

- `baseline.ipynb`: Jupyter notebook containing the complete implementation, training, and evaluation pipeline for a baseline model - UNet
- `improved.ipynb`: (TO BE CREATED) Jupyter notebook containting the complete implementation, training, and evaluation pipeline for an improved model
- `dataset.py`: Custom PyTorch dataset class for loading and preprocessing the segmentation data


## References

Gutman, David; Codella, Noel C. F.; Celebi, Emre; Helba, Brian; Marchetti, Michael; Mishra, Nabin; Halpern, Allan. "Skin Lesion Analysis toward Melanoma Detection: A Challenge at the International Symposium on Biomedical Imaging (ISBI) 2016, hosted by the International Skin Imaging Collaboration (ISIC)". arXiv preprint arXiv:1605.01397. 2016.

