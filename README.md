# WikiArt Art Movement Classification

A DL project that trains a convolutional neural networks to classify paintings into art movement categories using the WikiArt dataset.

## Overview

Art movement classification is a genuinely difficult image recognition problem. 
Unlike standard benchmarks where classes are visually distinct, art movements are defined by subtle and often abstract properties and the model cannot simply detect objects as it must learn finner details.

This notebook is organnized in the following manner:

1. Build a simple baseline CNN  
2. Increase capacity until the model overfits  
3. Apply regularisation to recover generalisation  
4. Use transfer learning for a substantial performance jump  
5. Fine-tune the pre-trained backbone on the target domain  

## Dataset

**Source:** [WikiArt Art Movements/Styles](https://www.kaggle.com/datasets/sivarazadi/wikiart-art-movementsstyles) via Kaggle

**10 selected art movements:**

| Movement | Approx. images |
|---|---|
| Romanticism | 6,800 |
| Renaissance | 6,200 |
| Realism | 5,400 |
| Baroque | 5,300 |
| Neoclassicism | 3,100 |
| Art_Nouveau | 3,000 |
| Expressionism | 2,600 |
| Japanese_Art | 2,200 |
| Rococo | 2,500 |
| Primitivism | 1,300 |

## Methodology

### Data splits
- **70 / 15 / 15** stratified hold-out split
- Experiments use a stratified **30% subset** of training and validation for faster iteration
- The **test set uses the full 15%** and is evaluated exactly once at the end
- A **tuning subset of 300 images per class** is used for hyperparameter search only

### Preprocessing
- Scratch CNNs: images resized to **128×128**, pixel values normalised to [0, 1]
- Transfer learning: images resized to **160×160**, passed through MobileNetV2's own preprocessing
- Pipeline built with `tf.data.Dataset` with `.cache()` and `.prefetch()` for efficiency

### Models

| Model | Val Accuracy | Notes |
|---|---|---|
| Baseline CNN | ~33% | 2 conv blocks, Flatten, no regularisation |
| Overfitting CNN | ~38% (peak) | 3 deeper blocks, deliberately over-parameterised |
| Regularised CNN | ~38% | Dropout + L2, same architecture as overfit |
| Transfer (frozen) | ~52% | MobileNetV2, frozen base, trained head |
| Fine-tuned | ~58%+ | Top 40% of MobileNetV2 unfrozen |


## Key Findings

- A simple CNN learns useful features (33% vs 10% random chance) but quickly reaches a performance ceiling on this visually complex task
- Increasing model capacity induces clear overfitting — training accuracy reaches 70% while validation collapses
- Regularisation alone was not sufficient to achieve a strong improvement. The initial combination of augmentation, `GlobalAveragePooling2D`, and dropout constrained the model too aggressively; removing augmentation and reverting to `Flatten` restored stable training
- Transfer learning produced the most significant jump in performance, confirming that pre-trained features from ImageNet generalise meaningfully to painting classification
- Fine-tuning the upper layers of MobileNetV2 on the full training set yielded the best results

**Most confused pairs:**
- Art Nouveau → Japanese Art (decorative style overlap)
- Baroque → Rococo (Rococo evolved directly from Baroque)
- Expressionism → Japanese Art (simplified, emotionally charged forms)

