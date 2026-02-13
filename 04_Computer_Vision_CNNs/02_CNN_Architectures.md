# CNN Architectures: Evolution and Modern Designs

**Author:** ML Encyclopedia Project
**Last Updated:** 2025
**Prerequisites:** CNN Fundamentals, Deep Learning Basics
**Difficulty:** PhD-level with Practical Focus

---

## Table of Contents

1. [Introduction](#introduction)
2. [Historical Evolution](#historical-evolution)
3. [LeNet-5 (1998)](#lenet-5-1998)
4. [AlexNet (2012)](#alexnet-2012)
5. [VGG (2014)](#vgg-2014)
6. [GoogLeNet/Inception (2014)](#googlenet-inception-2014)
7. [ResNet (2015)](#resnet-2015)
8. [DenseNet (2017)](#densenet-2017)
9. [MobileNet (2017)](#mobilenet-2017)
10. [EfficientNet (2019)](#efficientnet-2019)
11. [Vision Transformers (2020)](#vision-transformers-2020)
12. [Swin Transformer (2021)](#swin-transformer-2021)
13. [ConvNeXt (2022)](#convnext-2022)
14. [2025 Comparisons](#2025-comparisons)
15. [Transfer Learning](#transfer-learning)
16. [Architecture Selection Guide](#architecture-selection-guide)

---

## Introduction

CNN architectures have evolved dramatically since 1998, driven by:
- **Increased compute**: GPUs, TPUs, distributed training
- **Larger datasets**: ImageNet (14M images) --> JFT-300M (300M images)
- **Architectural innovations**: Skip connections, attention, normalization
- **AutoML**: Neural Architecture Search (NAS)

**Timeline of Breakthroughs:**
```
1998: LeNet-5 (60K params, MNIST)
2012: AlexNet (60M params, ImageNet top-5: 84.7%)
2014: VGG-16 (138M params, ImageNet top-5: 92.7%)
2014: GoogLeNet (6.8M params, ImageNet top-5: 93.3%)
2015: ResNet-152 (60M params, ImageNet top-5: 96.4%)
2017: DenseNet-264 (34M params, ImageNet top-5: 96.5%)
2017: MobileNetV1 (4.2M params, efficient)
2019: EfficientNet-B7 (66M params, ImageNet top-5: 97.1%)
2020: ViT-Huge (632M params, ImageNet top-5: 88.5% --> 90.7% pretrained)
2021: Swin-L (197M params, ImageNet top-5: 97.5%)
2022: ConvNeXt-XL (350M params, ImageNet top-5: 97.5%)
```

**This guide**: Comprehensive analysis of each architecture with production-ready PyTorch implementations.

---

## Historical Evolution

### Key Architectural Patterns

1. **Depth**: Shallow (LeNet: 5 layers) --> Deep (ResNet: 152+ layers)
2. **Width**: Narrow (64 channels) --> Wide (2048+ channels)
3. **Skip Connections**: None --> Residual --> Dense
4. **Normalization**: None --> BatchNorm --> LayerNorm
5. **Activation**: Sigmoid/Tanh --> ReLU --> GELU/Swish
6. **Efficiency**: Standard Conv --> Depthwise Separable --> Inverted Residual
7. **Attention**: None --> Squeeze-and-Excitation --> Self-Attention

---

## LeNet-5 (1998)

**Paper:** "Gradient-based learning applied to document recognition" (LeCun et al.)

**Purpose:** Handwritten digit recognition (MNIST, USPS)

**Key Innovations:**
- First successful CNN architecture
- Alternating convolution and pooling
- Backpropagation training

### Architecture

```
Input (32x32 grayscale)
--> Conv1 (6 filters, 5x5) --> Tanh --> AvgPool (2x2)
--> Conv2 (16 filters, 5x5) --> Tanh --> AvgPool (2x2)
--> Conv3 (120 filters, 5x5) --> Tanh
--> FC (84) --> Tanh
--> FC (10) --> Softmax
```

**Parameters:** ~60,000

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 for MNIST digit classification.

    Original paper used tanh activation and average pooling.
    Modern adaptation uses ReLU and max pooling.
    """
    def __init__(self, num_classes=10, use_modern=True):
        super().__init__()

        # Activation function
        self.activation = nn.ReLU() if use_modern else nn.Tanh()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)            # 14x14 -> 10x10
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)          # 5x5 -> 1x1

        # Pooling
        if use_modern:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.activation(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x

# Usage
model = LeNet5(num_classes=10, use_modern=True)
x = torch.randn(1, 1, 28, 28)  # MNIST image
output = model(x)
print(f"Output shape: {output.shape}")  # [1, 10]

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

**Legacy:**
- Proved CNNs work for visual tasks
- Established conv --> pool pattern
- Used in bank check reading systems

---

## AlexNet (2012)

**Paper:** "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al.)

**Purpose:** ImageNet classification (1000 classes, 1.2M training images)

**Key Innovations:**
- **ReLU activation**: Faster training than sigmoid/tanh
- **Dropout**: Regularization to prevent overfitting
- **Data augmentation**: Random crops, horizontal flips, color jitter
- **GPU training**: Parallelized across 2 GPUs
- **Local Response Normalization (LRN)**: Precursor to BatchNorm

### Architecture

```
Input (227x227x3)
--> Conv1 (96 filters, 11x11, stride=4) --> ReLU --> MaxPool --> LRN
--> Conv2 (256 filters, 5x5) --> ReLU --> MaxPool --> LRN
--> Conv3 (384 filters, 3x3) --> ReLU
--> Conv4 (384 filters, 3x3) --> ReLU
--> Conv5 (256 filters, 3x3) --> ReLU --> MaxPool
--> FC (4096) --> ReLU --> Dropout(0.5)
--> FC (4096) --> ReLU --> Dropout(0.5)
--> FC (1000) --> Softmax
```

**Parameters:** ~60 million

### PyTorch Implementation

```python
class AlexNet(nn.Module):
    """
    AlexNet for ImageNet classification.

    Modern adaptations:
    - Replace LRN with BatchNorm
    - Single GPU training (original used 2 GPUs)
    - Adaptive average pooling for flexibility
    """
    def __init__(self, num_classes=1000, use_batchnorm=True):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Conv1: 227x227x3 -> 55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96) if use_batchnorm else nn.Identity(),

            # Conv2: 27x27x96 -> 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256) if use_batchnorm else nn.Identity(),

            # Conv3: 13x13x256 -> 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: 13x13x384 -> 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: 13x13x384 -> 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Adaptive pooling for variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Usage
model = AlexNet(num_classes=1000)
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")
```

**Impact:**
- **Won ImageNet 2012** by large margin (top-5 error: 15.3% vs 26.2%)
- Sparked deep learning revolution
- Demonstrated GPU training effectiveness

---

## VGG (2014)

**Paper:** "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman)

**Key Innovations:**
- **Uniform architecture**: Only 3x3 convolutions and 2x2 max pooling
- **Depth**: 16-19 layers (VGG-16, VGG-19)
- **Simplicity**: Easy to understand and modify

**Insight:** Two 3x3 convolutions have same receptive field as one 5x5 but fewer parameters:
- 5x5 conv: C x C x 5 x 5 = 25C^2 parameters
- Two 3x3 convs: 2 x C x C x 3 x 3 = 18C^2 parameters
- Reduction: 28% fewer parameters, more non-linearity

### Architecture (VGG-16)

```
Input (224x224x3)
--> Conv3-64 x 2 --> MaxPool
--> Conv3-128 x 2 --> MaxPool
--> Conv3-256 x 3 --> MaxPool
--> Conv3-512 x 3 --> MaxPool
--> Conv3-512 x 3 --> MaxPool
--> FC-4096 --> Dropout
--> FC-4096 --> Dropout
--> FC-1000 --> Softmax
```

### PyTorch Implementation

```python
class VGG(nn.Module):
    """
    VGG architecture with configurable depth.

    Configurations:
    - VGG-11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    - VGG-13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    - VGG-16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    - VGG-19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    """
    def __init__(self, config, num_classes=1000, batch_norm=False):
        super().__init__()
        self.features = self._make_layers(config, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, config, batch_norm):
        layers = []
        in_channels = 3

        for x in config:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# VGG configurations
VGG_CONFIGS = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# Usage
vgg16 = VGG(VGG_CONFIGS['VGG16'], num_classes=1000, batch_norm=True)
x = torch.randn(1, 3, 224, 224)
output = vgg16(x)

print(f"VGG-16 parameters: {sum(p.numel() for p in vgg16.parameters()):,}")
```

**Strengths:**
- Simple, uniform architecture
- Good transfer learning features
- Still used as feature extractor (2025)

**Weaknesses:**
- Memory intensive (138M parameters)
- Slow to train
- Large model size

---

## GoogLeNet / Inception (2014)

**Paper:** "Going Deeper with Convolutions" (Szegedy et al.)

**Key Innovations:**
- **Inception module**: Parallel multi-scale feature extraction
- **1x1 convolutions**: Dimensionality reduction
- **Global average pooling**: Replace FC layers
- **Auxiliary classifiers**: Gradient injection for deep networks

**Motivation:** Different objects have different scales. Use multiple kernel sizes in parallel.

### Inception Module

```python
class InceptionModule(nn.Module):
    """
    Inception module with dimensionality reduction.

    Parallel branches:
    1. 1x1 conv (dimensionality reduction)
    2. 1x1 conv --> 3x3 conv
    3. 1x1 conv --> 5x5 conv
    4. 3x3 max pool --> 1x1 conv

    Concatenate outputs along channel dimension.
    """
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 --> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 --> 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        # Branch 4: 3x3 pool --> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # Concatenate along channel dimension
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) architecture.
    """
    def __init__(self, num_classes=1000, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        # Initial convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Inception blocks
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        # Auxiliary classifiers (for training only)
        if aux_logits:
            self.aux1 = self._make_aux_classifier(512, num_classes)
            self.aux2 = self._make_aux_classifier(528, num_classes)

    def _make_aux_classifier(self, in_channels, num_classes):
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # Initial convolutions
        x = self.conv1(x)
        x = self.conv2(x)

        # Inception modules
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        # Auxiliary classifier 1
        aux1 = None
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # Auxiliary classifier 2
        aux2 = None
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return x, aux1, aux2
        return x

# Usage
model = GoogLeNet(num_classes=1000)
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Evolution:**
- **Inception v2**: Batch normalization, factorized convolutions
- **Inception v3**: 7x7 factorized to 1x7 and 7x1
- **Inception v4**: Combined with ResNet (Inception-ResNet)

---

## ResNet (2015)

**Paper:** "Deep Residual Learning for Image Recognition" (He et al.)

**Key Innovation:** **Residual connections** (skip connections)

**Problem:** Deep networks (>20 layers) suffer from **degradation problem**:
- Training error increases with depth
- Not caused by overfitting
- Optimization difficulty

**Solution:** Learn residual mapping instead of direct mapping:

```
H(x) = F(x) + x

Where:
- H(x): Desired underlying mapping
- F(x): Residual function to learn
- x: Identity shortcut
```

**Insight:** It's easier to learn F(x) = 0 than to learn H(x) = x

### Residual Block

```python
class BasicBlock(nn.Module):
    """
    Basic ResNet block (used in ResNet-18, ResNet-34).

    Structure:
    x --> Conv3x3 --> BN --> ReLU --> Conv3x3 --> BN --> (+) --> ReLU
    +--------------------------------------------+
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        # Residual path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block (used in ResNet-50, ResNet-101, ResNet-152).

    Structure:
    x --> Conv1x1 --> BN --> ReLU --> Conv3x3 --> BN --> ReLU --> Conv1x1 --> BN --> (+) --> ReLU
    +------------------------------------------------------------------+

    Reduces parameters: 3x3x256x256 = 589K --> 1x1x256x64 + 3x3x64x64 + 1x1x64x256 = 70K
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv with stride for downsampling
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv to expand dimensions
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # Bottleneck path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet architecture.

    Variants:
    - ResNet-18: [2, 2, 2, 2] BasicBlocks
    - ResNet-34: [3, 4, 6, 3] BasicBlocks
    - ResNet-50: [3, 4, 6, 3] Bottlenecks
    - ResNet-101: [3, 4, 23, 3] Bottlenecks
    - ResNet-152: [3, 8, 36, 3] Bottlenecks
    """
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        # Downsample if dimensions change
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# Create ResNet variants
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)

# Usage
model = resnet50(num_classes=1000)
x = torch.randn(1, 3, 224, 224)
output = model(x)

print(f"ResNet-50 parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Impact:**
- Enabled training of 100+ layer networks
- Won ImageNet 2015 (3.57% top-5 error)
- Foundation for many modern architectures
- Still widely used in 2025

---

## DenseNet (2017)

**Paper:** "Densely Connected Convolutional Networks" (Huang et al.)

**Key Innovation:** **Dense connections** - each layer connects to all subsequent layers

**Motivation:**
- ResNet: `x_l = F(x_{l-1}) + x_{l-1}`
- DenseNet: `x_l = F([x_0, x_1, ..., x_{l-1}])`

**Benefits:**
1. **Feature reuse**: Direct access to all previous features
2. **Gradient flow**: Gradients flow directly to all layers
3. **Parameter efficiency**: Fewer parameters than ResNet
4. **Regularization**: Implicit deep supervision

### PyTorch Implementation

```python
class DenseBlock(nn.Module):
    """
    Dense block: each layer receives feature maps from all previous layers.

    Args:
        num_layers: Number of layers in the block
        in_channels: Number of input channels
        growth_rate: How many channels each layer adds (k in paper)
    """
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        """
        Bottleneck layer: BN-ReLU-Conv1x1-BN-ReLU-Conv3x3

        Reduces parameters: 1x1 conv reduces channels before 3x3 conv
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        features = [x]

        for layer in self.layers:
            # Concatenate all previous features
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)

        # Return concatenation of all features
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """
    Transition layer between dense blocks.

    Reduces spatial dimensions (2x) and channels (compression).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class DenseNet(nn.Module):
    """
    DenseNet architecture.

    Variants:
    - DenseNet-121: [6, 12, 24, 16] layers per block
    - DenseNet-169: [6, 12, 32, 32] layers per block
    - DenseNet-201: [6, 12, 48, 32] layers per block
    - DenseNet-264: [6, 12, 64, 48] layers per block
    """
    def __init__(self, block_config, growth_rate=32, num_classes=1000, compression=0.5):
        super().__init__()

        # Initial convolution
        num_init_features = 2 * growth_rate
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense blocks and transitions
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Dense block
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate

            # Transition layer (except after last block)
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, int(num_features * compression))
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# DenseNet variants
def densenet121(num_classes=1000):
    return DenseNet([6, 12, 24, 16], growth_rate=32, num_classes=num_classes)

def densenet169(num_classes=1000):
    return DenseNet([6, 12, 32, 32], growth_rate=32, num_classes=num_classes)

def densenet201(num_classes=1000):
    return DenseNet([6, 12, 48, 32], growth_rate=32, num_classes=num_classes)

# Usage
model = densenet121(num_classes=1000)
print(f"DenseNet-121 parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Comparison with ResNet:**
- **Memory**: Higher (stores all feature maps)
- **Parameters**: Lower (feature reuse)
- **Accuracy**: Slightly better
- **Training**: Requires careful implementation (gradient checkpointing for deep versions)

---

## MobileNet (2017)

**Paper:** "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (Howard et al.)

**Key Innovation:** **Depthwise Separable Convolutions**

**Motivation:** Reduce parameters and computation for mobile/embedded devices

**Standard Conv Parameters:** `C_in x C_out x k x k`
**Depthwise Separable:** `C_in x k x k + C_in x C_out`

**Reduction factor:** ~8-9x for 3x3 kernels

### PyTorch Implementation

```python
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution: Depthwise + Pointwise

    Depthwise: Apply one filter per input channel
    Pointwise: 1x1 conv to mix channels
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetV1(nn.Module):
    """
    MobileNet V1 architecture.

    Uses depthwise separable convolutions throughout.
    """
    def __init__(self, num_classes=1000, width_multiplier=1.0):
        super().__init__()

        def conv_bn_relu(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                         padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def conv_dw(in_ch, out_ch, stride):
            return DepthwiseSeparableConv(in_ch, out_ch, stride)

        # Width multiplier for model scaling
        def channels(c):
            return int(c * width_multiplier)

        # Network architecture
        self.features = nn.Sequential(
            conv_bn_relu(3, channels(32), 2),      # 224 -> 112
            conv_dw(channels(32), channels(64), 1),
            conv_dw(channels(64), channels(128), 2),   # 112 -> 56
            conv_dw(channels(128), channels(128), 1),
            conv_dw(channels(128), channels(256), 2),  # 56 -> 28
            conv_dw(channels(256), channels(256), 1),
            conv_dw(channels(256), channels(512), 2),  # 28 -> 14
            # 5 x depthwise separable conv
            conv_dw(channels(512), channels(512), 1),
            conv_dw(channels(512), channels(512), 1),
            conv_dw(channels(512), channels(512), 1),
            conv_dw(channels(512), channels(512), 1),
            conv_dw(channels(512), channels(512), 1),
            conv_dw(channels(512), channels(1024), 2), # 14 -> 7
            conv_dw(channels(1024), channels(1024), 1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels(1024), num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class InvertedResidual(nn.Module):
    """
    MobileNetV2 inverted residual block.

    Structure: 1x1 expand --> 3x3 depthwise --> 1x1 project
    Linear bottleneck: No ReLU after last 1x1

    Expansion factor: Typically 6
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []

        # Expand (if expansion factor > 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Project (linear bottleneck - no activation)
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNet V2 with inverted residuals and linear bottlenecks.
    """
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()

        # First layer
        input_channel = int(32 * width_mult)
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]

        # Inverted residual blocks
        # [expansion, channels, num_blocks, stride]
        configs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        for t, c, n, s in configs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel

        # Last layer
        last_channel = int(1280 * width_mult)
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ))

        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Usage
model_v1 = MobileNetV1(width_multiplier=1.0)
model_v2 = MobileNetV2(width_mult=1.0)

print(f"MobileNetV1 params: {sum(p.numel() for p in model_v1.parameters()):,}")
print(f"MobileNetV2 params: {sum(p.numel() for p in model_v2.parameters()):,}")
```

**MobileNet Evolution:**
- **V1 (2017)**: Depthwise separable convolutions
- **V2 (2018)**: Inverted residuals, linear bottlenecks
- **V3 (2019)**: Neural Architecture Search, Squeeze-Excite, h-swish

---

## EfficientNet (2019)

**Paper:** "EfficientNet: Rethinking Model Scaling for CNNs" (Tan & Le)

**Key Innovation:** **Compound Scaling** - simultaneously scale depth, width, and resolution

**Observation:** Previous methods scale only one dimension:
- ResNet: Increase depth (ResNet-50 --> ResNet-101)
- WideResNet: Increase width (channels)
- Higher resolution: 224x224 --> 299x299

**Compound Scaling Formula:**
```
depth: d = alpha^phi
width: w = beta^phi
resolution: r = gamma^phi

Constraint: alpha * beta^2 * gamma^2 ~= 2
```

Where phi is the compound coefficient controlling resources.

### PyTorch Implementation (Simplified)

```python
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv).

    Used in EfficientNet. Similar to MobileNetV2 but with:
    - Squeeze-and-Excitation
    - Stochastic depth (drop path)
    """
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size,
                 stride, se_ratio=0.25, drop_rate=0.0):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.drop_rate = drop_rate

        hidden_dim = in_channels * expand_ratio

        # Expansion
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()

        # Depthwise
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride,
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

        # Squeeze-and-Excitation
        squeeze_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, squeeze_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze_channels, hidden_dim, 1),
            nn.Sigmoid()
        )

        # Project
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        # Expansion
        x = self.expand_conv(x)

        # Depthwise
        x = self.dwconv(x)

        # Squeeze-and-Excitation
        se_weight = self.se(x)
        x = x * se_weight

        # Project
        x = self.project_conv(x)

        # Skip connection with stochastic depth
        if self.use_res_connect:
            if self.training and self.drop_rate > 0:
                # Stochastic depth (drop path)
                keep_prob = 1 - self.drop_rate
                mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
                mask.bernoulli_(keep_prob)
                x = x.div(keep_prob) * mask
            x = x + identity

        return x

# EfficientNet would stack these blocks with compound scaling
# Full implementation is complex; use torchvision.models.efficientnet_b0()
```

**EfficientNet Family:**
- **B0**: Baseline (phi=1), 5.3M params
- **B1-B7**: Scaled versions (phi=1.1 to 2.0)
- **B7**: 66M params, 97.1% ImageNet top-5 accuracy

**2025 Status:** Still competitive; widely used for transfer learning.

---

## Vision Transformers (2020)

**Paper:** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al.)

**Key Innovation:** Apply Transformer architecture (from NLP) directly to image patches

**Motivation:**
- CNNs have inductive biases (locality, translation equivariance)
- Transformers learn patterns from data
- With enough data, Transformers outperform CNNs

### Architecture Overview

```
Image (224x224x3)
--> Patch Embedding (16x16 patches) --> 196 patches x 768 dims
--> Add Position Embeddings
--> [CLS] Token + Patches
--> Transformer Encoder x 12 layers
--> [CLS] Output --> MLP Head --> Classification
```

### PyTorch Implementation

```python
class PatchEmbedding(nn.Module):
    """
    Split image into patches and linearly embed them.

    Args:
        img_size: Input image size
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Use Conv2d to split into patches and embed
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = self.proj(x)  # [B, 768, 14, 14]
        x = x.flatten(2)  # [B, 768, 196]
        x = x.transpose(1, 2)  # [B, 196, 768]
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.

    Variants:
    - ViT-Base: 12 layers, 768 hidden, 12 heads, 86M params
    - ViT-Large: 24 layers, 1024 hidden, 16 heads, 307M params
    - ViT-Huge: 32 layers, 1280 hidden, 16 heads, 632M params
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., dropout=0.1):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm (modern practice)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Add [CLS] token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, num_patches+1, embed_dim]

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder
        x = self.transformer(x)

        # Classification: use [CLS] token
        x = self.norm(x[:, 0])
        x = self.head(x)

        return x

# Create ViT variants
def vit_base(num_classes=1000):
    return VisionTransformer(embed_dim=768, depth=12, num_heads=12, num_classes=num_classes)

def vit_large(num_classes=1000):
    return VisionTransformer(embed_dim=1024, depth=24, num_heads=16, num_classes=num_classes)

def vit_huge(num_classes=1000):
    return VisionTransformer(embed_dim=1280, depth=32, num_heads=16, num_classes=num_classes)

# Usage
model = vit_base(num_classes=1000)
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"ViT-Base parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Key Insights:**
- **Requires large-scale pretraining**: 14M+ images (ImageNet-21k, JFT-300M)
- **No built-in inductive biases**: Learns everything from data
- **Global receptive field**: Self-attention sees entire image
- **Scales better than CNNs**: Performance improves with model/data size

---

## Swin Transformer (2021)

**Paper:** "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Liu et al.)

**Key Innovation:** **Hierarchical architecture** with **shifted window attention**

**Motivation:** ViT limitations:
- Fixed patch size (no hierarchy)
- Quadratic complexity in image size
- Not suitable for dense prediction (segmentation, detection)

**Swin Solutions:**
- **Hierarchical feature maps**: Like CNNs (progressively downsample)
- **Window-based attention**: Linear complexity
- **Shifted windows**: Enable cross-window connections

### Key Concepts

```python
class WindowAttention(nn.Module):
    """
    Window-based multi-head self attention.

    Compute attention within local windows (7x7 or 8x8).
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # Window-based attention implementation
        # (Simplified; full implementation is complex)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# Full Swin Transformer implementation is complex
# Use torchvision.models.swin_t(), swin_s(), swin_b(), swin_l()
```

**Advantages over ViT:**
- **Hierarchical features**: Suitable for dense prediction
- **Efficient**: Linear complexity in image size
- **Better accuracy**: SOTA on ImageNet, COCO, ADE20K

**2025 Status:** Preferred over ViT for computer vision tasks requiring multi-scale features.

---

## ConvNeXt (2022)

**Paper:** "A ConvNet for the 2020s" (Liu et al.)

**Key Innovation:** **Modernize CNNs** to match Transformer performance

**Question:** Can CNNs match Vision Transformers with modern training?

**Answer:** Yes! ConvNeXt = ResNet + Swin Transformer design choices

### Design Improvements

```python
class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block: Modernized ResNet block.

    Changes from ResNet:
    1. Larger kernels (7x7 instead of 3x3)
    2. Depthwise convolutions
    3. Inverted bottleneck (expand then compress)
    4. LayerNorm instead of BatchNorm
    5. GELU instead of ReLU
    6. Fewer activation functions
    7. Layer Scale for training stability
    """
    def __init__(self, dim, expansion=4, kernel_size=7, layer_scale_init=1e-6):
        super().__init__()

        # Depthwise conv (large kernel)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                               padding=kernel_size//2, groups=dim)

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Pointwise/Inverted Bottleneck (expand then compress)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)

        # Layer Scale (learnable scaling)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

        self.drop_path = nn.Identity()  # Stochastic depth can be added

    def forward(self, x):
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Permute to channel-last for LayerNorm and Linear
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        # LayerNorm
        x = self.norm(x)

        # Inverted bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer Scale
        x = self.gamma * x

        # Permute back
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # Residual connection
        x = residual + self.drop_path(x)

        return x


class ConvNeXt(nn.Module):
    """
    ConvNeXt architecture.

    Variants:
    - Tiny: [3, 3, 9, 3], dim=96
    - Small: [3, 3, 27, 3], dim=96
    - Base: [3, 3, 27, 3], dim=128
    - Large: [3, 3, 27, 3], dim=192
    - XLarge: [3, 3, 27, 3], dim=256
    """
    def __init__(self, depths, dims, num_classes=1000, drop_path_rate=0.0):
        super().__init__()

        # Stem: Aggressive downsampling (4x4 kernel, stride 4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6, data_format='channels_first')
        )

        # 4 stages
        self.stages = nn.ModuleList()
        for i in range(4):
            # Downsampling between stages
            if i > 0:
                downsample = nn.Sequential(
                    nn.LayerNorm(dims[i-1], eps=1e-6),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
                )
            else:
                downsample = nn.Identity()

            # Stack blocks
            stage = nn.Sequential(
                downsample,
                *[ConvNeXtBlock(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        # Classifier
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        # Global average pooling
        x = x.mean([-2, -1])  # [B, C, H, W] -> [B, C]
        x = self.norm(x)
        x = self.head(x)

        return x

# Create ConvNeXt variants
def convnext_tiny(num_classes=1000):
    return ConvNeXt([3, 3, 9, 3], [96, 192, 384, 768], num_classes)

def convnext_base(num_classes=1000):
    return ConvNeXt([3, 3, 27, 3], [128, 256, 512, 1024], num_classes)

# Usage
model = convnext_tiny(num_classes=1000)
print(f"ConvNeXt-Tiny params: {sum(p.numel() for p in model.parameters()):,}")
```

**Results:** ConvNeXt matches or exceeds Swin Transformer on ImageNet, COCO, ADE20K.

**Takeaway:** Architecture matters, but so do training procedures and design choices.

---

## 2025 Comparisons

### Performance Summary (ImageNet Top-1 Accuracy)

| Architecture | Params | Top-1 Acc | Inference Speed | Memory |
|--------------|--------|-----------|-----------------|--------|
| ResNet-50 | 25M | 76.1% | Fast | Low |
| EfficientNet-B0 | 5.3M | 77.1% | Fast | Low |
| ViT-Base | 86M | 81.8%* | Medium | High |
| Swin-Tiny | 28M | 81.3% | Fast | Medium |
| ConvNeXt-Tiny | 28M | 82.1% | Fast | Medium |
| ConvNeXt-XL | 350M | 87.8% | Slow | Very High |

*Requires large-scale pretraining

### When to Use Each Architecture

**ResNet (2015)**
- [x] Well-understood, reliable
- [x] Good transfer learning
- [x] Fast inference
- [ ] Not SOTA anymore

**EfficientNet (2019)**
- [x] Best accuracy/efficiency trade-off
- [x] Mobile/edge deployment
- [x] Multiple size variants
- [ ] Slower than ResNet

**Vision Transformer (2020)**
- [x] SOTA with large pretraining
- [x] Global receptive field
- [x] Scalable to huge models
- [ ] Requires massive data
- [ ] Memory intensive

**Swin Transformer (2021)**
- [x] Hierarchical features
- [x] Efficient (linear complexity)
- [x] Great for detection/segmentation
- [ ] Complex implementation

**ConvNeXt (2022)**
- [x] Matches Transformers with simpler architecture
- [x] Hierarchical features
- [x] Efficient
- [x] **2025 recommendation for most tasks**

---

## Transfer Learning

Transfer learning is the dominant paradigm in 2025. Rarely train from scratch.

### PyTorch Transfer Learning

```python
import torchvision.models as models

# Load pretrained model
model = models.resnet50(weights='IMAGENET1K_V2')  # Latest weights

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10 classes for your task

# Fine-tune only the classifier
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# Advanced: Gradual unfreezing
def unfreeze_layers(model, num_layers):
    """Unfreeze last `num_layers` layers."""
    all_params = list(model.parameters())
    for param in all_params[-num_layers:]:
        param.requires_grad = True

# After initial training, unfreeze more layers
unfreeze_layers(model, 20)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
```

### 2025 Best Practices

1. **Always start with pretrained weights**
2. **Use discriminative learning rates**: Lower LR for early layers, higher for later layers
3. **Gradual unfreezing**: Train classifier first, then unfreeze deeper layers
4. **Data augmentation**: RandAugment, MixUp, CutMix
5. **Test-time augmentation**: Average predictions over augmented versions

---

## Architecture Selection Guide

### Decision Tree

```
Is data large (>100M images)?
+--- Yes: Consider ViT or Swin Transformer
+--- No: Use CNNs or hybrid

Is deployment on mobile/edge?
+--- Yes: Use MobileNet or EfficientNet
+--- No: Continue

Need multi-scale features (detection, segmentation)?
+--- Yes: Use Swin Transformer or ConvNeXt
+--- No: Continue

Need fastest inference?
+--- Yes: Use ResNet-50 or EfficientNet-B0
+--- No: Continue

Want best accuracy regardless of cost?
+--- Yes: Use ConvNeXt-Large or Swin-Large
+--- No: Use ConvNeXt-Tiny or Swin-Tiny
```

### 2025 General Recommendation

**Default choice: ConvNeXt-Tiny or ConvNeXt-Base**
- Modern architecture
- Competitive accuracy
- Efficient inference
- Hierarchical features
- Simpler than Transformers

---

## Summary

This guide covered the evolution of CNN architectures from LeNet-5 (1998) to ConvNeXt (2022):

1. **LeNet-5**: Foundational architecture
2. **AlexNet**: Started deep learning revolution
3. **VGG**: Depth and simplicity
4. **GoogLeNet**: Multi-scale features
5. **ResNet**: Skip connections enable very deep networks
6. **DenseNet**: Dense connections for feature reuse
7. **MobileNet**: Efficient for mobile deployment
8. **EfficientNet**: Compound scaling for optimal accuracy/efficiency
9. **Vision Transformer**: Transformers for vision
10. **Swin Transformer**: Hierarchical vision transformer
11. **ConvNeXt**: Modernized CNNs match Transformers

**Key Trends:**
- Increasing depth (5 layers --> 152+ layers)
- Efficiency improvements (depthwise separable, inverted residuals)
- Skip connections (residual, dense)
- Attention mechanisms (SE, self-attention)
- Hybrid architectures (CNNs + Transformers)

**2025 Landscape:**
- ConvNeXt: Best CNN architecture
- Swin Transformer: Best hierarchical transformer
- EfficientNet: Best efficiency
- Transfer learning: Always start with pretrained weights

---

**References:**

- LeCun et al. (1998). "Gradient-based learning applied to document recognition"
- Krizhevsky et al. (2012). "ImageNet Classification with Deep CNNs"
- Simonyan & Zisserman (2014). "Very Deep Convolutional Networks"
- Szegedy et al. (2014). "Going Deeper with Convolutions"
- He et al. (2015). "Deep Residual Learning for Image Recognition"
- Huang et al. (2017). "Densely Connected Convolutional Networks"
- Howard et al. (2017). "MobileNets: Efficient CNNs"
- Tan & Le (2019). "EfficientNet: Rethinking Model Scaling"
- Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words"
- Liu et al. (2021). "Swin Transformer"
- Liu et al. (2022). "A ConvNet for the 2020s"

---

*End of CNN Architectures*
