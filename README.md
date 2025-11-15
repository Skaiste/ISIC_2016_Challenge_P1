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
├── training/
│   ├── original/   # Training dermoscopic images (.jpg)
│   └── gt/         # Training ground truth segmentation masks (.png)
└── testing/
    ├── original/   # Testing dermoscopic images (.jpg)
    └── gt/         # Testing ground truth segmentation masks (.png)
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
- **Validation**: Selected best performing model based on accuracy metric
- **Epochs**: 10
- **Batch Size**: 8
- **Image Preprocessing**: Resize to 256×256
- **Mask Preprocessing**: Resize to 256×256 and convert to binary (0/1)

### Training Results
- **Sensitivity**: 0.6668
- **Specificity**: 0.9869
- **Accuracy**: 0.8967
- **IoU**: 0.6547
- **Dice**: 0.7544

## Improved Model

The model architecture and training has been adjusted in stages.

### Stage 1:
- **Image & Mask Processing**: increased the size of the images to 512x512px

#### Training Results
Trained for 8 epochs
|  | Sensitivity | Specificity | Accuracy | IoU | Dice |
|--|--|--|--|--|--|
| Validation |  |  | 0.9402 | 0.7948 | 0.8763 |
| Evaluation | 0.9289 | 0.9739 | 0.9723 | 0.8727 | 0.9315 |

Final Results:
  Total epochs trained: 8
  Best validation metrics:
    IoU: 0.6970
    Dice: 0.7990
    Accuracy: 0.9026
    Loss: 0.2452
Calculating metrics for each prediction:
Average Metrics:
  Sensitivity: 0.7411
  Specificity: 0.9999
  Accuracy: 0.9123
  IoU: 0.7400
  Dice: 0.8337
  Hausdorff: 57.9398

### Stage 2
- Added **Convergence** with patience of 5 with maximum of 50 **epochs**
- **Image & Mask Processing**: added random rotation and flipping both vertically and horizontally

#### Training Results
Trained for 8 epochs
|  | Sensitivity | Specificity | Accuracy | IoU | Dice |
|--|--|--|--|--|--|
| Validation |  |  | 0.9402 | 0.7948 | 0.8763 |
| Evaluation | 0.9289 | 0.9739 | 0.9723 | 0.8727 | 0.9315 |




## Old improvements

### Stage 1:

#### Training configuration additions
- Added **Convergence** with patience of 5 with maximum of 50 **epochs**
- **Image & Mask Processing**: added random rotation and flipping both vertically and horizontally

#### Training Results
Trained for 8 epochs
|  | Sensitivity | Specificity | Accuracy | IoU | Dice |
|--|--|--|--|--|--|
| Validation |  |  | 0.9402 | 0.7948 | 0.8763 |
| Evaluation | 0.9289 | 0.9739 | 0.9723 | 0.8727 | 0.9315 |

### Stage 2:

**Training change**: using Dice Coefficient to decide which model to select during validation

**Results**:
Trained for 8 epochs
|  | Sensitivity | Specificity | Accuracy | IoU | Dice |
|--|--|--|--|--|--|
| Validation |  |  | 0.9419 | 0.8049 | 0.8849 |
| Evaluation | 0.8696 | 0.9506 | 0.9369 | 0.7355 | 0.8394 |

### Stage 3:
**Image & Mask Processing change**: added colour jitter

**Training change**: increased patience to 8 and using IoU to decide which model to select during validation.

**Results**:
Trained for 33 epochs
|  | Sensitivity | Specificity | Accuracy | IoU | Dice |
|--|--|--|--|--|--|
| Validation |  |  | 0.9373 | 0.7904 | 0.8764 |
| Evaluation | 0.9202 | 0.9750 | 0.9657 | 0.8942 | 0.9433 |

### Stage 4:

**Architecture change**: added extra encoder & decoder blocks

**Results**:
Trained for 29 epochs
|  | Sensitivity | Specificity | Accuracy | IoU | Dice |
|--|--|--|--|--|--|
| Validation |  |  | 0.9369 | 0.7825 | 0.8683 |
| Evaluation | 0.9487 | 0.9906 | 0.9800 | 0.9169 | 0.9564 |


### Stage 5:

**Architecture change**: added a dropout layer on every encoder and decoder block with rate of 0.2

**Results**:
Trained for 28 epochs
|  | Sensitivity | Specificity | Accuracy | IoU | Dice |
|--|--|--|--|--|--|
| Validation |  |  | 0.9303 | 0.7574 | 0.8483 |
| Evaluation | 0.8586 | 0.9907 | 0.9574 | 0.8342 | 0.9051 |

Showed no improvement across all metrics, therefore dropout will not be used

### Stage 6:

**Training change**: using weight decay in the optimiser with rate of 0.0001

**Results**:
Trained for 25 epochs
|  | Sensitivity | Specificity | Accuracy | IoU | Dice |
|--|--|--|--|--|--|
| Validation |  |  | 0.9061 | 0.7037 | 0.8022 |
| Evaluation | 0.6733 | 0.9969 | 0.8419 | 0.6567 | 0.7744 |

adding weight decay drastically worsened model performance, therefore it will not be used

### Stage 7:
**Architecture change**: replaced BatchNorm2d layer to InstanceNorm2d in the encoder / decoder blocks

**Results**:
Trained for 33 epochs

|  | Sensitivity | Specificity | Accuracy | IoU | Dice |
|--|--|--|--|--|--|
| Validation |  |  | 0.9568 | 0.8330 | 0.9040 |
| Evaluation | 0.8776 | 0.9640 | 0.9458 | 0.7890 | 0.8705 |

### Stage 8:
**Training change**: implemented and used BCE (Binary Cross Entropy) combined with Dice metric for calculating loss.

**Results**:
Trained for 23 epochs

|  | Sensitivity | Specificity | Accuracy | IoU | Dice |
|--|--|--|--|--|--|
| Validation |  |  | 0.9286 | 0.7758 | 0.8637 |
| Evaluation | 0.7557 | 0.9837 | 0.9196 | 0.7370 | 0.8444 |


### Stage 9:
**Training change**: implemented and used BCE (Binary Cross Entropy) combined with Hausdorff metric for calculating loss.

**Results**:
Trained for 31 epochs

|  | Sensitivity | Specificity | Accuracy | IoU | Dice |
|--|--|--|--|--|--|
| Validation |  |  | 0.9397 | 0.7772 | 0.8641 |
| Evaluation | 0.9651 | 0.9232 | 0.9344 | 0.6853 | 0.7854 |


### Stage 10
**Image processing changes**: added random gaussian noise and random sharpness to the image processing

**Results**:

Trained for 50 epochs

|  | Sensitivity | Specificity | Accuracy | IoU | Dice |
|--|--|--|--|--|--|
| Validation |  |  | 0.9390 | 0.7993 | 0.8822 |
| Evaluation | 0.8352 | 0.9862 | 0.9654 | 0.7850 | 0.8656 |


## References

D. A. Gutman et al., "Skin Lesion Analysis toward Melanoma Detection: A Challenge at the International Symposium on Biomedical Imaging (ISBI) 2016, hosted by the International Skin Imaging Collaboration (ISIC)," CoRR, vol. abs/1605.01397, 2016. [Online]. Available: http://arxiv.org/abs/1605.01397

