# Data Science GAJAB - Self-Supervised Learning with SimCLR

*A comprehensive learning project for understanding self-supervised learning through SimCLR implementation*

## 🎯 Learning Objectives

By completing this project, you will:
- **Understand** the fundamental principles of contrastive learning
- **Implement** self-supervised learning from scratch using SimCLR
- **Compare** supervised vs self-supervised learning approaches
- **Evaluate** model performance using appropriate metrics
- **Analyze** the impact of data augmentation on representation learning

## 📚 Prerequisites

**Required Knowledge:**
- Basic PyTorch operations and neural networks
- Understanding of Convolutional Neural Networks (CNNs)
- Familiarity with image classification tasks
- Basic linear algebra concepts (dot products, cosine similarity)

**Technical Requirements:**
```bash
pip install lightning torchvision matplotlib pandas
```

## 🎯 Learning Progression

**Recommended Study Path:**
1. **Start Here**: Understand the problem (Why self-supervised learning?)
2. **Baseline**: Run supervised learning baseline (prac2.ipynb)
3. **Core Concept**: Learn SimCLR theory and intuition
4. **Implementation**: Build SimCLR step-by-step (prac1.ipynb)
5. **Analysis**: Compare results and understand differences
6. **Extension**: Experiment with modifications

## 📊 Data Science Methodology

### Problem Formulation
**Why use self-supervised learning?**
- Limited labeled data in real-world scenarios
- Expensive annotation costs for large datasets
- Learning general representations that transfer to multiple tasks
- Reducing dependency on human-labeled data

### Dataset Analysis: CIFAR-10
- **Size**: 60,000 32×32 color images
- **Classes**: 10 categories (airplane, automobile, bird, etc.)
- **Split**: 50,000 training + 10,000 test images
- **Challenge**: Small image size requires effective feature learning

### Model Selection Rationale
**Why ResNet-18?**
- Proven architecture for image classification
- Manageable size for educational purposes
- Good balance between complexity and performance
- Skip connections help with gradient flow

## 🔍 Conceptual Overview

### Self-Supervised vs Supervised Learning

**Traditional Supervised Learning:**
```
Image → CNN → Predictions → Compare with Labels → Loss
```

**Self-Supervised Learning (SimCLR):**
```
Image → Augment → [View1, View2] → CNN → Embeddings → Contrastive Loss
```

### SimCLR Intuition (Simple Explanation)

**Core Idea**: "Similar images should have similar representations"

1. **Create Pairs**: Take one image, create two different augmented versions
2. **Learn Similarity**: Train the model to recognize these as "similar"
3. **Learn Differences**: Make sure different images have different representations
4. **No Labels Needed**: The model learns by comparing, not by memorizing labels

### Why This Works
- **Augmentations preserve semantic content** (a rotated cat is still a cat)
- **Forces the model to learn robust features** (invariant to transformations)
- **Creates unlimited training pairs** from existing data

## 📁 Project Structure

```
Data Science GAJAB/
├── README.md                    # This comprehensive guide
├── prac1.ipynb                 # 🔥 SimCLR Self-Supervised Training
├── prac2.ipynb                 # 📊 Supervised Learning Baseline
├── shared.py                   # 🛠️ Utility functions and data loaders
├── simclr-resnet18_model.pt    # 💾 Pre-trained model weights
└── logs/                       # 📈 Training logs and metrics
```

## 🧪 Experimental Setup

### Baseline Comparisons
To understand SimCLR's effectiveness, compare these approaches:

1. **Random Initialization**: ResNet-18 trained from scratch with labels
2. **SimCLR Pre-training**: ResNet-18 pre-trained with SimCLR, then fine-tuned
3. **Expected Results**: SimCLR should achieve better performance with fewer labeled examples

### Key Experiments to Try
- **Augmentation Impact**: Remove different augmentations and observe effects
- **Temperature Sensitivity**: Try τ = 0.01, 0.07, 0.5 and compare results
- **Batch Size Effect**: Test with 64, 128, 256 batch sizes
- **Architecture Variations**: Modify projection head dimensions

## 📄 Implementation Guide

### 1. `prac1.ipynb` - SimCLR Self-Supervised Training

**Purpose**: Learn representations without labels using contrastive learning

#### Step 1: Data Augmentation (The Magic Ingredient)
```python
selfsupervised_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=128),      # Spatial invariance
    transforms.RandomHorizontalFlip(),           # Orientation invariance
    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), # Color invariance
    transforms.RandomGrayscale(p=0.2),           # Color robustness
    transforms.GaussianBlur(kernel_size=9),      # Texture focus
    transforms.ToTensor()
])
```

**Why each augmentation matters:**
- **RandomResizedCrop**: Teaches model to recognize objects at different scales
- **HorizontalFlip**: Objects look the same when flipped
- **ColorJitter**: Object identity shouldn't depend on exact colors
- **Grayscale**: Forces model to use shape, not just color
- **GaussianBlur**: Focuses on high-level features, not fine details

#### Step 2: Model Architecture
```python
# Base: ResNet-18 feature extractor
pytorch_model = torch.hub.load('pytorch/vision', 'resnet18', weights=False)

# Projection Head: Maps features to contrastive learning space
pytorch_model.fc = torch.nn.Sequential(
    torch.nn.Linear(512, 512),  # Hidden layer
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256)   # Output embeddings for contrastive loss
)
```

**Architecture Logic:**
- **ResNet-18**: Extracts visual features from images
- **Projection Head**: Creates embeddings optimized for similarity comparison
- **256 dimensions**: Good balance between expressiveness and efficiency

#### Step 3: InfoNCE Loss (Simplified Explanation)

**Intuitive Understanding:**
```python
def info_nce_loss_simple_explanation(features, temperature):
    """
    Goal: Make similar images have similar embeddings
    
    1. Calculate how similar each image is to every other image
    2. For each image, its augmented version should be MOST similar
    3. All other images should be LESS similar
    4. Temperature controls how strict this similarity requirement is
    """
```

**Mathematical Implementation:**
```python
def info_nce_loss(feats, temperature):
    # Step 1: Calculate similarity between all pairs
    cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
    
    # Step 2: Remove self-similarity (image with itself)
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool)
    cos_sim.masked_fill_(self_mask, -9e15)
    
    # Step 3: Find positive pairs (augmented versions)
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    
    # Step 4: Compute contrastive loss
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    
    return nll.mean()
```

### 2. `prac2.ipynb` - Supervised Learning Baseline

**Purpose**: Compare traditional supervised learning with SimCLR approach

```python
# Standard classification head
pytorch_model.fc = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10)  # 10 CIFAR-10 classes
)
```

### 3. `shared.py` - Utility Functions

**Key Components:**
- **LightningModel**: Standard supervised learning wrapper
- **Cifar10DataModule**: Data loading and preprocessing
- **Plotting functions**: Visualize training progress

## 🏗️ Architecture Diagrams

### SimCLR Training Flow
```
Original Image
      ↓
  Augmentation
  ┌─────────────┐
  ↓             ↓
View 1        View 2
  ↓             ↓
ResNet-18     ResNet-18
  ↓             ↓
Projection    Projection
Head          Head
  ↓             ↓
Embedding 1   Embedding 2
  ↓             ↓
  └─── Similarity ───┘
         ↓
   Contrastive Loss
```

### Model Architecture
```
Input: 3×128×128 RGB Image
         ↓
ResNet-18 Backbone
         ↓
512-dim features
         ↓
Projection Head:
├── Linear(512 → 512)
├── ReLU()
└── Linear(512 → 256)
         ↓
256-dim embeddings
         ↓
Contrastive Loss
```

## 📈 Results Analysis

### Performance Metrics
- **Training Loss**: Should decrease from ~6.0 to ~2.0
- **Contrastive Accuracy**: Measures how often positive pairs rank in top-5
- **Downstream Accuracy**: Classification performance after fine-tuning

### Training Curves Interpretation
```python
def plot_loss_and_acc(log_dir):
    # Visualizes training progress
    # Look for: Decreasing loss, increasing accuracy
    # Warning signs: Plateauing early, erratic curves
```

### Expected Results
- **SimCLR Training**: 60-80% contrastive accuracy after 50 epochs
- **Fine-tuning**: 85-90% classification accuracy on CIFAR-10
- **Comparison**: Should outperform random initialization

## 🤔 Study Questions

**Conceptual Understanding:**
1. Why does SimCLR work without labels?
2. How do augmentations create positive pairs?
3. What happens if temperature is too high/low?
4. When would you choose SimCLR over supervised learning?

**Technical Analysis:**
1. Why use cosine similarity instead of L2 distance?
2. How does batch size affect contrastive learning?
3. What role does the projection head play?
4. Why is InfoNCE loss effective for this task?

**Practical Applications:**
1. In what scenarios is labeled data expensive?
2. How would you adapt SimCLR for medical images?
3. What augmentations would work for text data?
4. How do you evaluate self-supervised models?

## 💻 Practice Exercises

### Beginner Level
1. **Modify augmentations**: Remove ColorJitter and observe impact
2. **Change temperature**: Try τ = 0.01, 0.1, 1.0 and compare results
3. **Visualize embeddings**: Use t-SNE to plot learned representations

### Intermediate Level
1. **Implement different similarity metrics**: Try L2 distance instead of cosine
2. **Modify projection head**: Test different architectures (1 layer, 3 layers)
3. **Batch size experiments**: Compare performance with 64, 128, 512 batch sizes

### Advanced Level
1. **Custom augmentations**: Design augmentations for specific domains
2. **Multi-crop strategy**: Implement multiple crops per image
3. **Momentum contrast**: Add momentum to the contrastive learning

## 🎯 Business Impact & Applications

### When to Use Self-Supervised Learning
- **Limited labeled data**: Medical imaging, satellite imagery
- **Expensive annotation**: Video analysis, 3D point clouds
- **Domain adaptation**: Transferring between different image domains
- **Representation learning**: Learning general features for multiple tasks

### Computational Considerations
- **Training time**: 2-3x longer than supervised learning
- **Memory requirements**: Larger batch sizes needed for effectiveness
- **Resource trade-offs**: More compute upfront, less labeling cost

### Real-World Applications
- **Medical imaging**: Pre-training on unlabeled scans
- **Autonomous vehicles**: Learning from unlabeled driving footage
- **Content moderation**: Understanding images without manual labeling
- **Scientific imaging**: Analyzing microscopy or astronomical data

## 🚀 Next Steps & Extensions

### Immediate Extensions
1. **Different datasets**: Try STL-10, ImageNet-100
2. **Other architectures**: ResNet-50, Vision Transformers
3. **Advanced augmentations**: MixUp, CutMix, AutoAugment

### Research Directions
1. **Multi-modal learning**: Combine images with text
2. **Video understanding**: Temporal contrastive learning
3. **Few-shot learning**: Leverage pre-trained representations
4. **Fairness analysis**: Study bias in learned representations

## 🔧 Troubleshooting Guide

### Common Issues for Students
1. **Memory errors**: Reduce batch size to 64 or 32
2. **Slow training**: Use fewer workers or smaller images
3. **Poor convergence**: Check learning rate and temperature
4. **NaN losses**: Verify data preprocessing and normalization

### Performance Tips
1. **Start small**: Use subset of data for initial experiments
2. **Monitor closely**: Watch for overfitting or underfitting
3. **Compare baselines**: Always have a simple baseline to beat
4. **Validate assumptions**: Test each component individually

## 📚 Additional Resources

### Key Papers
1. **SimCLR**: "A Simple Framework for Contrastive Learning of Visual Representations"
2. **InfoNCE**: "Representation Learning with Contrastive Predictive Coding"
3. **Data Augmentation**: "Improved Baselines with Momentum Contrastive Learning"

### Learning Materials
- **CS231n Stanford**: Computer Vision course materials
- **PyTorch tutorials**: Official self-supervised learning guides
- **Papers with Code**: Implementation comparisons and benchmarks

This comprehensive guide provides both theoretical understanding and practical implementation skills for self-supervised learning with SimCLR. Focus on understanding the concepts first, then dive into the implementation details.