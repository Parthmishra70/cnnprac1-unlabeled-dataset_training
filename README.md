# Self-Supervised Learning with SimCLR

This repository contains implementations of self-supervised learning using SimCLR (Simple Framework for Contrastive Learning of Visual Representations) on CIFAR-10 dataset using PyTorch Lightning.

## 📁 Project Structure

```
Data Science GAJAB/
├── README.md                    # This file
├── prac1.ipynb                 # SimCLR Self-Supervised Training
├── prac2.ipynb                 # Fine-tuning with Pre-trained Model
├── shared.py                   # Shared utilities and data modules
├── simclr-resnet18_model.pt    # Trained SimCLR model weights
└── logs/                       # Training logs and metrics
```

## 📋 Table of Contents

1. [Overview](#overview)
2. [File Descriptions](#file-descriptions)
3. [Implementation Details](#implementation-details)
4. [Architecture Diagrams](#architecture-diagrams)
5. [Usage Instructions](#usage-instructions)
6. [Dependencies](#dependencies)
7. [Results](#results)

## 🔍 Overview

This project demonstrates the implementation of SimCLR, a self-supervised learning framework that learns visual representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space.

### Key Concepts:
- **Self-Supervised Learning**: Learning representations without labeled data
- **Contrastive Learning**: Learning by contrasting positive and negative pairs
- **Data Augmentation**: Creating multiple views of the same image
- **InfoNCE Loss**: Noise Contrastive Estimation loss function

## 📄 File Descriptions

### 1. `prac1.ipynb` - SimCLR Self-Supervised Training

**Purpose**: Implements the complete SimCLR training pipeline for self-supervised representation learning.

**Key Components**:

#### Data Augmentation Pipeline
```python
selfsupervised_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5)),
    transforms.ToTensor()
])
```

**Logic**: Creates two different augmented views of each image to form positive pairs for contrastive learning.

#### Model Architecture
- **Base Model**: ResNet-18 (without pre-trained weights)
- **Projection Head**: Two fully connected layers (512 → 512 → 256)
- **Purpose**: Maps representations to a space where contrastive loss is applied

```python
pytorch_model.fc = torch.nn.Sequential(
    torch.nn.Linear(512, 512),  # Hidden layer
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256)   # Output embeddings
)
```

#### InfoNCE Loss Implementation
```python
def info_nce_loss(feats, temperature, mode="train"):
    # Calculate cosine similarity between all pairs
    cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
    
    # Mask out self-similarity
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    
    # Find positive pairs (batch_size//2 away)
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    
    # Compute InfoNCE loss
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()
    
    return nll, sim_argsort
```

**Logic**: 
1. Computes cosine similarity between all feature pairs
2. Identifies positive pairs (augmented versions of same image)
3. Maximizes similarity to positive pairs while minimizing similarity to negative pairs

#### Training Configuration
- **Epochs**: 50
- **Batch Size**: 256
- **Learning Rate**: 0.0005
- **Temperature**: 0.07
- **Optimizer**: Adam

### 2. `prac2.ipynb` - Fine-tuning Implementation

**Purpose**: Demonstrates how to use the pre-trained SimCLR model for downstream classification tasks.

**Key Components**:
- Loads pre-trained SimCLR weights
- Replaces projection head with classification head (10 classes for CIFAR-10)
- Fine-tunes on labeled CIFAR-10 data

```python
pytorch_model.fc = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10)  # 10 classes for CIFAR-10
)
```

### 3. `shared.py` - Utility Classes and Functions

**Purpose**: Contains reusable components for both notebooks.

#### LightningModel Class
```python
class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        # Standard supervised learning model
        # Includes training, validation, and test steps
        # Uses cross-entropy loss and accuracy metrics
```

#### Cifar10DataModule Class
```python
class Cifar10DataModule(L.LightningDataModule):
    def __init__(self, data_path="./", batch_size=64, height_width=None, 
                 num_workers=0, train_transform=None, test_transform=None):
        # Handles CIFAR-10 data loading and preprocessing
        # Supports custom transforms for different training strategies
```

#### Key Features:
- **Flexible Transforms**: Supports different augmentation strategies
- **Data Splitting**: Automatic train/validation/test splits
- **Batch Processing**: Efficient data loading with configurable batch sizes

## 🏗️ Architecture Diagrams

### SimCLR Training Pipeline

```
Input Image
     ↓
Data Augmentation → [Augmented View 1, Augmented View 2]
     ↓                        ↓
ResNet-18 Encoder    ResNet-18 Encoder
     ↓                        ↓
Projection Head      Projection Head
     ↓                        ↓
   z₁ ←------ InfoNCE Loss -----→ z₂
     ↓
Contrastive Learning
(Maximize similarity between positive pairs,
 Minimize similarity between negative pairs)
```

### Model Architecture Flow

```
Input: 3×128×128 RGB Image
         ↓
ResNet-18 Backbone (Feature Extractor)
         ↓
512-dimensional features
         ↓
Projection Head:
├── Linear(512 → 512)
├── ReLU()
└── Linear(512 → 256)
         ↓
256-dimensional embeddings
         ↓
InfoNCE Loss Computation
```

### Data Flow Diagram

```
CIFAR-10 Dataset
       ↓
   Augmentation
   ┌─────────────┐
   ↓             ↓
View 1        View 2
   ↓             ↓
Encoder       Encoder
   ↓             ↓
   └─── Batch ───┘
         ↓
   Similarity Matrix
         ↓
   InfoNCE Loss
         ↓
   Backpropagation
```

## 🚀 Usage Instructions

### Prerequisites
```bash
pip install lightning torchvision matplotlib pandas
```

### Running SimCLR Training (prac1.ipynb)

1. **Setup Environment**:
   ```python
   import lightning
   import torch
   from shared import *
   ```

2. **Configure Training**:
   ```python
   # Set random seed for reproducibility
   torch.manual_seed(123)
   
   # Initialize data module with augmentations
   dm = Cifar10DataModule(
       batch_size=256,
       num_workers=5,
       train_transform=AugmentImg(selfsupervised_transforms),
       test_transform=AugmentImg(selfsupervised_transforms)
   )
   ```

3. **Train Model**:
   ```python
   trainer = L.Trainer(
       max_epochs=50,
       accelerator="auto",
       devices="auto",
       logger=CSVLogger(save_dir="logs/", name="mine_model")
   )
   trainer.fit(LightningModel_duplicate, dm)
   ```

### Running Fine-tuning (prac2.ipynb)

1. **Load Pre-trained Model**:
   ```python
   pytorch_model.load_state_dict(torch.load('simclr-resnet18_model.pt'))
   ```

2. **Modify for Classification**:
   ```python
   pytorch_model.fc = torch.nn.Sequential(
       torch.nn.Linear(512, 512),
       torch.nn.ReLU(),
       torch.nn.Linear(512, 10)  # CIFAR-10 classes
   )
   ```

## 📊 Results and Metrics

### Training Metrics Tracked:
- **Training Loss**: InfoNCE contrastive loss
- **Training Accuracy**: Top-5 similarity ranking accuracy
- **Validation Loss**: Validation InfoNCE loss  
- **Validation Accuracy**: Validation ranking accuracy

### Visualization:
The notebooks include plotting functions to visualize:
- Loss curves over epochs
- Accuracy progression
- Training vs. validation metrics

```python
def plot_loss_and_acc(log_dir):
    # Reads CSV logs and creates matplotlib plots
    # Shows training progression and model performance
```

## 🔧 Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Batch Size | 256 | Larger batches provide more negative samples |
| Temperature | 0.07 | Controls the concentration of the distribution |
| Learning Rate | 0.0005 | Balanced learning speed |
| Epochs | 50 | Sufficient for convergence |
| Embedding Dim | 256 | Projection head output dimension |

## 🧠 Technical Details

### InfoNCE Loss Mathematics:
```
L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```
Where:
- `z_i, z_j` are positive pair embeddings
- `z_k` are all embeddings in the batch
- `τ` is the temperature parameter
- `sim()` is cosine similarity

### Augmentation Strategy:
1. **Random Resized Crop**: Spatial invariance
2. **Horizontal Flip**: Orientation invariance  
3. **Color Jitter**: Color invariance
4. **Grayscale**: Color robustness
5. **Gaussian Blur**: Texture focus

## 📈 Expected Outcomes

### Self-Supervised Training:
- **Loss**: Should decrease from ~6.0 to ~2.0
- **Accuracy**: Top-5 ranking accuracy should improve to 60-80%

### Fine-tuning Results:
- **Classification Accuracy**: Expected 85-90% on CIFAR-10
- **Convergence**: Faster than training from scratch
- **Generalization**: Better feature representations

## 🔍 Model Evaluation

### Metrics Used:
1. **Contrastive Accuracy**: Measures how often positive pairs rank in top-5
2. **Loss Convergence**: InfoNCE loss reduction over time
3. **Representation Quality**: Evaluated through downstream task performance

### Validation Strategy:
- **Train/Val Split**: 45,000/5,000 for CIFAR-10
- **Cross-Validation**: Consistent random seeds for reproducibility
- **Monitoring**: Real-time loss and accuracy tracking

## 🎯 Applications and Extensions

### Potential Use Cases:
1. **Transfer Learning**: Pre-trained features for new datasets
2. **Few-Shot Learning**: Learning with limited labeled data
3. **Domain Adaptation**: Adapting to new visual domains
4. **Feature Extraction**: Using learned representations for other tasks

### Possible Extensions:
1. **Different Architectures**: ResNet-50, Vision Transformers
2. **Advanced Augmentations**: MixUp, CutMix, AutoAugment
3. **Multi-Modal Learning**: Text-image contrastive learning
4. **Larger Datasets**: ImageNet, custom datasets

## 🐛 Troubleshooting

### Common Issues:
1. **Memory Errors**: Reduce batch size or image resolution
2. **Slow Training**: Increase num_workers or use GPU
3. **Poor Convergence**: Adjust learning rate or temperature
4. **Overfitting**: Add regularization or data augmentation

### Performance Tips:
1. **GPU Usage**: Ensure CUDA is available and used
2. **Data Loading**: Use multiple workers for faster I/O
3. **Mixed Precision**: Enable for faster training on modern GPUs
4. **Batch Size**: Larger batches generally improve contrastive learning

## 📚 References and Theory

### Key Papers:
1. **SimCLR**: "A Simple Framework for Contrastive Learning of Visual Representations"
   - https://arxiv.org/pdf/2002.05709
   - Implementation: https://github.dev/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial17/SimCLR.ipynb

2. **InfoNCE**: "Representation Learning with Contrastive Predictive Coding"
3. **ResNet**: "Deep Residual Learning for Image Recognition"

### Theoretical Foundation:
- **Contrastive Learning**: Learning by comparing similar and dissimilar examples
- **Self-Supervision**: Using data structure as supervision signal
- **Representation Learning**: Learning meaningful feature representations

## 🎓 Advanced Learning Resources

### Comprehensive Surveys:
- **A Survey on Contrastive Self-supervised Learning**: https://arxiv.org/abs/2011.00362
  - *Complete overview of contrastive learning methods and applications*

### Advanced Techniques:
- **Advances in Understanding, Improving, and Applying Contrastive Learning**: https://hazyresearch.stanford.edu/blog/2022-04-19-contrastive-1
  - *Stanford's deep dive into contrastive learning improvements*

- **Masked Autoencoders Are Scalable Vision Learners**: https://arxiv.org/abs/2111.06377
  - *Alternative self-supervised approach using masked reconstruction*
