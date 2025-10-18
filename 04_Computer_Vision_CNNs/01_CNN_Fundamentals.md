# CNN Fundamentals

**Author:** ML Encyclopedia Project
**Last Updated:** 2025
**Prerequisites:** Linear Algebra, Calculus, Deep Learning Basics
**Difficulty:** PhD-level with Practical Focus

---

## Table of Contents

1. [Introduction](#introduction)
2. [Convolutional Operations](#convolutional-operations)
3. [Feature Maps and Filters](#feature-maps-and-filters)
4. [Pooling Operations](#pooling-operations)
5. [Padding and Stride](#padding-and-stride)
6. [Parameter Sharing and Weight Tying](#parameter-sharing-and-weight-tying)
7. [Receptive Fields](#receptive-fields)
8. [CNN Architecture Design](#cnn-architecture-design)
9. [PyTorch Implementation](#pytorch-implementation)
10. [Feature Visualization](#feature-visualization)
11. [Best Practices](#best-practices)
12. [2025 State-of-the-Art](#2025-state-of-the-art)

---

## Introduction

Convolutional Neural Networks (CNNs) are specialized neural networks designed for processing grid-like data, particularly images. CNNs exploit three key architectural ideas:

1. **Local Connectivity**: Neurons connect only to a local region of the input
2. **Parameter Sharing**: Same weights used across different spatial locations
3. **Equivariance to Translation**: If input shifts, output shifts correspondingly

**Mathematical Foundation:**
CNNs implement a discrete convolution operation (technically cross-correlation in deep learning frameworks):

```
(I * K)(i,j) = Σ_m Σ_n I(i+m, j+n) K(m,n)
```

Where:
- `I` is the input (image/feature map)
- `K` is the kernel (filter/weight matrix)
- `*` denotes the convolution operation

---

## Convolutional Operations

### Cross-Correlation vs Convolution

**Important Distinction:**
Deep learning frameworks use **cross-correlation**, not mathematical convolution:

**Cross-correlation:**
```
(I ⊗ K)(i,j) = Σ_m Σ_n I(i+m, j+n) K(m,n)
```

**True convolution:**
```
(I * K)(i,j) = Σ_m Σ_n I(i-m, j-n) K(m,n)
```

The difference: true convolution flips the kernel. Since we learn the kernel weights, this distinction doesn't affect learning—we just learn the flipped version if needed.

### Discrete 2D Convolution

For a 2D image `I` of size `H × W` and kernel `K` of size `k_h × k_w`:

```
Output(i,j) = Σ_{m=0}^{k_h-1} Σ_{n=0}^{k_w-1} I(i+m, j+n) · K(m,n) + b
```

Where:
- `i ∈ [0, H-k_h]`
- `j ∈ [0, W-k_w]`
- `b` is the bias term (shared across spatial locations)

**Output dimensions (no padding, stride=1):**
```
H_out = H - k_h + 1
W_out = W - k_w + 1
```

### Multi-Channel Convolution

For RGB images (3 channels) or multi-channel feature maps:

```
Output(i,j) = Σ_{c=0}^{C_in-1} Σ_{m=0}^{k_h-1} Σ_{n=0}^{k_w-1} I(c,i+m,j+n) · K(c,m,n) + b
```

Where:
- `C_in` is the number of input channels
- Kernel has shape `[C_in, k_h, k_w]`
- Each output channel has its own kernel and bias

**For multiple output channels:**
- Use `C_out` different kernels
- Each kernel produces one output channel
- Total parameters: `C_out × C_in × k_h × k_w + C_out` (weights + biases)

### PyTorch Example: Basic Convolution

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Single convolution operation
def manual_conv2d_example():
    """
    Demonstrate manual 2D convolution calculation.
    """
    # Input: 1 image, 1 channel, 5x5 spatial dimensions
    input_tensor = torch.tensor([[
        [1., 2., 3., 4., 5.],
        [6., 7., 8., 9., 10.],
        [11., 12., 13., 14., 15.],
        [16., 17., 18., 19., 20.],
        [21., 22., 23., 24., 25.]
    ]]).unsqueeze(0)  # Shape: [1, 1, 5, 5]

    # 3x3 edge detection kernel (vertical edges)
    kernel = torch.tensor([[
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]]).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 3, 3]

    # Apply convolution
    output = F.conv2d(input_tensor, kernel, padding=0, stride=1)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: {output.shape}")  # [1, 1, 3, 3]
    print(f"Output:\n{output}")

    return output

# Using nn.Conv2d
class SimpleConvLayer(nn.Module):
    """
    Basic convolutional layer with modern best practices.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)

# Example usage
conv_layer = SimpleConvLayer(in_channels=3, out_channels=64, kernel_size=3)
input_rgb = torch.randn(1, 3, 224, 224)  # Batch of 1 RGB image
output = conv_layer(input_rgb)
print(f"Output shape: {output.shape}")  # [1, 64, 224, 224]
```

---

## Feature Maps and Filters

### Feature Maps

A **feature map** is the output of applying a filter to the input. It represents the presence of specific features at different spatial locations.

**Hierarchy of Features:**
- **Early layers**: Detect low-level features (edges, corners, colors)
- **Middle layers**: Combine low-level features (textures, patterns)
- **Deep layers**: High-level, semantic features (object parts, faces)

### Filter Design Principles

**Classic Computer Vision Filters:**

1. **Edge Detection (Sobel Filter):**
```python
sobel_x = torch.tensor([
    [-1., 0., 1.],
    [-2., 0., 2.],
    [-1., 0., 1.]
])

sobel_y = torch.tensor([
    [-1., -2., -1.],
    [0., 0., 0.],
    [1., 2., 1.]
])
```

2. **Gaussian Blur (Smoothing):**
```python
def gaussian_kernel(size=5, sigma=1.0):
    """Create Gaussian smoothing kernel."""
    ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / kernel.sum()

gaussian_5x5 = gaussian_kernel(5, 1.0)
```

3. **Sharpening Filter:**
```python
sharpen = torch.tensor([
    [0., -1., 0.],
    [-1., 5., -1.],
    [0., -1., 0.]
])
```

**Learned Filters in CNNs:**
CNNs learn optimal filters through backpropagation. Early layers often learn Gabor-like filters (oriented edge detectors).

### Multi-Scale Feature Extraction

```python
class MultiScaleFeatureExtractor(nn.Module):
    """
    Extract features at multiple scales using different kernel sizes.
    Inspired by Inception modules.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Different kernel sizes for multi-scale features
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, 5, padding=2)

        # Pooling branch
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        # Parallel branches
        branch1 = self.conv1x1(x)
        branch2 = self.conv3x3(x)
        branch3 = self.conv5x5(x)
        branch4 = self.conv_pool(self.maxpool(x))

        # Concatenate along channel dimension
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
```

---

## Pooling Operations

Pooling reduces spatial dimensions while retaining important features. This provides:
- **Translation invariance**: Small shifts in input don't change output
- **Computational efficiency**: Reduces feature map size
- **Larger receptive fields**: Each neuron sees more of the input

### Max Pooling

Takes the maximum value in each pooling window:

```
MaxPool(i,j) = max_{m,n ∈ Window} I(i+m, j+n)
```

**Properties:**
- Preserves strongest activations
- Provides exact translation invariance (within window)
- Non-differentiable (uses subgradients in backprop)
- Most commonly used: 2×2 window, stride 2

```python
import torch.nn as nn

class PoolingComparison(nn.Module):
    """
    Compare different pooling strategies.
    """
    def __init__(self):
        super().__init__()

        # Max pooling (most common)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Average pooling
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Global average pooling (reduces to 1x1)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Global max pooling
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        max_out = self.maxpool(x)
        avg_out = self.avgpool(x)
        global_avg = self.global_avgpool(x)
        global_max = self.global_maxpool(x)

        return {
            'max': max_out,
            'avg': avg_out,
            'global_avg': global_avg.squeeze(),
            'global_max': global_max.squeeze()
        }

# Example
x = torch.randn(1, 64, 28, 28)
pooling = PoolingComparison()
outputs = pooling(x)

print(f"Input shape: {x.shape}")
print(f"Max pool: {outputs['max'].shape}")      # [1, 64, 14, 14]
print(f"Avg pool: {outputs['avg'].shape}")      # [1, 64, 14, 14]
print(f"Global avg: {outputs['global_avg'].shape}")  # [1, 64]
print(f"Global max: {outputs['global_max'].shape}")  # [1, 64]
```

### Average Pooling

Computes average of values in pooling window:

```
AvgPool(i,j) = (1/k²) Σ_{m,n ∈ Window} I(i+m, j+n)
```

**Properties:**
- Smoother downsampling
- Retains more spatial information
- Fully differentiable
- Less popular than max pooling in modern architectures

### Global Pooling

**Global Average Pooling (GAP):**
```
GAP(c) = (1/HW) Σ_i Σ_j I(c,i,j)
```

**Benefits:**
- Replaces fully connected layers
- Reduces overfitting (no parameters)
- Enables variable input sizes
- Used in modern architectures (ResNet, MobileNet)

**Global Max Pooling:**
```
GMP(c) = max_{i,j} I(c,i,j)
```

### 2025 Best Practice: Strided Convolutions

**Modern trend**: Replace pooling with strided convolutions:

```python
# Old approach: Conv + Pooling
conv = nn.Conv2d(64, 128, 3, padding=1)
pool = nn.MaxPool2d(2, 2)
# output = pool(conv(x))

# Modern approach: Strided convolution
strided_conv = nn.Conv2d(64, 128, 3, stride=2, padding=1)
# output = strided_conv(x)
```

**Advantages:**
- Learnable downsampling
- Better gradient flow
- Used in modern architectures (ConvNeXt, Vision Transformers)

---

## Padding and Stride

### Padding

Padding adds zeros (or other values) around input borders.

**Types:**

1. **Valid Padding (no padding):**
```
Output size = (Input - Kernel + 1)
```

2. **Same Padding:**
```
Output size = Input size
Padding = (Kernel - 1) / 2
```

3. **Full Padding:**
```
Output size = Input + Kernel - 1
```

**Mathematical formulation:**
```
H_out = floor((H_in + 2*padding - kernel_size) / stride + 1)
W_out = floor((W_in + 2*padding - kernel_size) / stride + 1)
```

### Stride

Stride controls how far the filter moves at each step.

**Output size formula:**
```
H_out = floor((H_in + 2*padding - kernel_size) / stride + 1)
W_out = floor((W_in + 2*padding - kernel_size) / stride + 1)
```

### PyTorch Examples

```python
class PaddingStrideDemo(nn.Module):
    """
    Demonstrate different padding and stride configurations.
    """
    def __init__(self, in_channels=3):
        super().__init__()

        # Valid padding (no padding)
        self.conv_valid = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=0
        )

        # Same padding (output size = input size for stride=1)
        self.conv_same = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1
        )

        # Stride 2 for downsampling
        self.conv_stride2 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=2, padding=1
        )

        # Large kernel with appropriate padding
        self.conv_large = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3
        )

    def forward(self, x):
        # x shape: [B, 3, 224, 224]

        valid = self.conv_valid(x)      # [B, 64, 222, 222]
        same = self.conv_same(x)        # [B, 64, 224, 224]
        stride2 = self.conv_stride2(x)  # [B, 64, 112, 112]
        large = self.conv_large(x)      # [B, 64, 112, 112]

        return {
            'valid': valid,
            'same': same,
            'stride2': stride2,
            'large': large
        }

def calculate_output_size(input_size, kernel_size, stride, padding):
    """
    Calculate output spatial dimensions.

    Args:
        input_size: Input height or width
        kernel_size: Filter size
        stride: Stride value
        padding: Padding value

    Returns:
        Output size
    """
    return ((input_size + 2 * padding - kernel_size) // stride) + 1

# Examples
print("224x224 input, 3x3 kernel, stride=1, padding=1:")
print(f"Output: {calculate_output_size(224, 3, 1, 1)}x{calculate_output_size(224, 3, 1, 1)}")

print("\n224x224 input, 7x7 kernel, stride=2, padding=3:")
print(f"Output: {calculate_output_size(224, 7, 2, 3)}x{calculate_output_size(224, 7, 2, 3)}")
```

### Dilated (Atrous) Convolutions

Dilated convolutions insert spaces between kernel elements, expanding receptive field without increasing parameters:

```python
class DilatedConvolution(nn.Module):
    """
    Dilated convolutions for exponentially growing receptive fields.
    Used in semantic segmentation (DeepLab) and WaveNet.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Standard convolution
        self.conv_standard = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, dilation=1
        )

        # Dilated by 2 (receptive field: 5x5)
        self.conv_dilation2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=2, dilation=2
        )

        # Dilated by 4 (receptive field: 9x9)
        self.conv_dilation4 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=4, dilation=4
        )

    def forward(self, x):
        out1 = self.conv_standard(x)
        out2 = self.conv_dilation2(x)
        out4 = self.conv_dilation4(x)
        return torch.cat([out1, out2, out4], dim=1)

# Effective receptive field with dilation d:
# RF = kernel_size + (kernel_size - 1) * (dilation - 1)
```

---

## Parameter Sharing and Weight Tying

### Why Parameter Sharing?

**Fully Connected Layer:**
- Input: 224×224×3 = 150,528 values
- Hidden layer: 1000 neurons
- Parameters: 150,528 × 1000 = 150M parameters

**Convolutional Layer:**
- Same input
- 64 filters of 3×3×3
- Parameters: 64 × 3 × 3 × 3 = 1,728 parameters

**Benefits:**
1. **Reduces parameters**: Dramatically fewer weights to learn
2. **Translation equivariance**: Same feature detected anywhere in image
3. **Generalization**: Pattern learned at one location applies everywhere
4. **Computational efficiency**: Reuse computations across spatial locations

### Mathematical Perspective

Standard neural network:
```
h_i = σ(Σ_j W_ij × x_j + b_i)
```
Each connection has unique weight `W_ij`.

Convolutional layer:
```
h(i,j) = σ(Σ_m Σ_n W(m,n) × x(i+m, j+n) + b)
```
Same weights `W(m,n)` used at every spatial location `(i,j)`.

### Depthwise Separable Convolutions

Modern efficient approach (MobileNet, EfficientNet):

```python
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution.

    Standard conv: C_in × C_out × k × k parameters
    Depthwise separable: C_in × k × k + C_in × C_out parameters

    Reduction factor: (C_in × k × k) / (C_in × C_out × k × k)
                    ≈ 1/C_out + 1/k²

    For k=3, C_out=64: ~9x parameter reduction
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # Depthwise: Apply one filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,  # Same as input
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels  # Key: separate filter per channel
        )

        # Pointwise: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Parameter comparison
standard_conv = nn.Conv2d(64, 128, 3, padding=1)
separable_conv = DepthwiseSeparableConv(64, 128, 3, padding=1)

standard_params = sum(p.numel() for p in standard_conv.parameters())
separable_params = sum(p.numel() for p in separable_conv.parameters())

print(f"Standard conv: {standard_params:,} parameters")
print(f"Separable conv: {separable_params:,} parameters")
print(f"Reduction: {standard_params / separable_params:.2f}x")
```

---

## Receptive Fields

The **receptive field** is the region of the input that affects a particular output neuron.

### Calculating Receptive Field

For a single layer:
```
RF = kernel_size
```

For stacked layers:
```
RF_l = RF_{l-1} + (kernel_size - 1) × Π_{i=1}^{l-1} stride_i
```

**Simplified recursive formula:**
```
RF_out = RF_in + (kernel_size - 1) × jump_in
jump_out = jump_in × stride
```

### PyTorch Example

```python
def calculate_receptive_field(layers):
    """
    Calculate receptive field for a sequence of conv layers.

    Args:
        layers: List of (kernel_size, stride) tuples

    Returns:
        receptive_field, jump (stride product)
    """
    rf = 1
    jump = 1

    for kernel_size, stride in layers:
        rf += (kernel_size - 1) * jump
        jump *= stride

    return rf, jump

# Example: VGG-like architecture
vgg_layers = [
    (3, 1),  # Conv 3x3
    (3, 1),  # Conv 3x3
    (2, 2),  # MaxPool 2x2
    (3, 1),  # Conv 3x3
    (3, 1),  # Conv 3x3
    (2, 2),  # MaxPool 2x2
    (3, 1),  # Conv 3x3
    (3, 1),  # Conv 3x3
    (3, 1),  # Conv 3x3
]

rf, jump = calculate_receptive_field(vgg_layers)
print(f"Receptive field: {rf}×{rf}")
print(f"Jump (effective stride): {jump}")

# ResNet-50 first few layers
resnet_layers = [
    (7, 2),   # Conv1: 7x7, stride 2
    (3, 2),   # MaxPool: 3x3, stride 2
    (1, 1),   # Conv2_x: 1x1
    (3, 1),   # Conv2_x: 3x3
    (1, 1),   # Conv2_x: 1x1
]

rf, jump = calculate_receptive_field(resnet_layers)
print(f"\nResNet receptive field: {rf}×{rf}")
```

### Effective Receptive Field

**Important**: Theoretical vs. effective receptive field.

Research shows that the **effective receptive field** is much smaller than theoretical, with Gaussian distribution:
- Center pixels contribute most
- Edge pixels contribute little
- Practical RF ≈ 0.25 × Theoretical RF

**Implications:**
- Need deeper networks for large images
- Skip connections help propagate information
- Attention mechanisms can bypass receptive field limitations

---

## CNN Architecture Design

### Basic CNN Block

```python
class ConvBlock(nn.Module):
    """
    Modern convolutional block with best practices.

    Order: Conv -> BatchNorm -> Activation
    (Note: Some 2025 architectures use LayerNorm or no normalization)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, use_batchnorm=True):
        super().__init__()

        layers = []

        # Convolution
        layers.append(nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=not use_batchnorm
        ))

        # Batch normalization (if used)
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        # Activation
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ModernConvBlock(nn.Module):
    """
    2025-style convolutional block.

    Trends:
    - Larger kernels (7x7) making comeback
    - Depthwise convolutions
    - GELU activation
    - LayerNorm instead of BatchNorm
    """
    def __init__(self, channels, kernel_size=7):
        super().__init__()

        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size=kernel_size,
            padding=kernel_size//2, groups=channels
        )
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * channels, channels)

    def forward(self, x):
        residual = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        return residual + x
```

### Complete CNN Example

```python
class SimpleCNN(nn.Module):
    """
    Complete CNN for image classification.

    Architecture:
    - Input: 224x224x3
    - Conv layers with increasing channels
    - Pooling for downsampling
    - Global average pooling
    - Linear classifier
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            ConvBlock(3, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64),

            # Block 2: 112x112 -> 56x56
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128),

            # Block 3: 56x56 -> 28x28
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 256),
            ConvBlock(256, 256),

            # Block 4: 28x28 -> 14x14
            ConvBlock(256, 512, stride=2),
            ConvBlock(512, 512),
            ConvBlock(512, 512),

            # Block 5: 14x14 -> 7x7
            ConvBlock(512, 512, stride=2),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Usage
model = SimpleCNN(num_classes=1000)
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

---

## PyTorch Implementation

### Production-Ready CNN Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

class CNNTrainer:
    """
    Production-ready CNN training pipeline.
    """
    def __init__(self, model, device='cuda', learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        return total_loss / len(train_loader), 100. * correct / total

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / len(val_loader), 100. * correct / total

    def train(self, train_loader, val_loader, epochs=100):
        """Full training loop."""
        best_acc = 0

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step()

            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                }, 'best_model.pth')
                print(f'Saved best model with accuracy: {best_acc:.2f}%')

# Example usage
# model = SimpleCNN(num_classes=10)
# trainer = CNNTrainer(model, device='cuda')
# trainer.train(train_loader, val_loader, epochs=100)
```

---

## Feature Visualization

### Visualizing Learned Filters

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_filters(model, layer_name='features.0.block.0', num_filters=64):
    """
    Visualize learned convolutional filters.

    Args:
        model: Trained CNN model
        layer_name: Name of conv layer to visualize
        num_filters: Number of filters to display
    """
    # Get the layer
    layer = dict(model.named_modules())[layer_name]

    if not isinstance(layer, nn.Conv2d):
        raise ValueError(f"{layer_name} is not a Conv2d layer")

    # Get weights: [out_channels, in_channels, kernel_h, kernel_w]
    weights = layer.weight.data.cpu()

    # Normalize for visualization
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    # Plot filters
    num_rows = 8
    num_cols = 8
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i >= num_filters:
            ax.axis('off')
            continue

        # Get filter (average across input channels for RGB)
        filter_img = weights[i].mean(dim=0)

        ax.imshow(filter_img, cmap='viridis')
        ax.axis('off')
        ax.set_title(f'F{i}', fontsize=8)

    plt.tight_layout()
    plt.savefig('cnn_filters.png', dpi=150)
    plt.close()

def visualize_feature_maps(model, image, layer_name='features.4'):
    """
    Visualize feature maps (activations) for a given image.

    Args:
        model: Trained CNN model
        image: Input image tensor [1, C, H, W]
        layer_name: Name of layer to visualize
    """
    model.eval()

    # Hook to capture activations
    activations = {}

    def hook_fn(module, input, output):
        activations['output'] = output

    # Register hook
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(image)

    # Remove hook
    handle.remove()

    # Get activations: [1, C, H, W]
    feature_maps = activations['output'][0].cpu()
    num_features = min(64, feature_maps.shape[0])

    # Plot
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i >= num_features:
            ax.axis('off')
            continue

        feature_map = feature_maps[i]
        ax.imshow(feature_map, cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Ch{i}', fontsize=8)

    plt.tight_layout()
    plt.savefig('feature_maps.png', dpi=150)
    plt.close()

def grad_cam(model, image, target_class, target_layer='features.12'):
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Shows which regions of the image are important for prediction.
    """
    model.eval()

    # Store gradients and activations
    gradients = {}
    activations = {}

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    def forward_hook(module, input, output):
        activations['value'] = output

    # Register hooks
    layer = dict(model.named_modules())[target_layer]
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(image)

    # Backward pass for target class
    model.zero_grad()
    target = output[0, target_class]
    target.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Calculate Grad-CAM
    grads = gradients['value'][0]  # [C, H, W]
    acts = activations['value'][0]  # [C, H, W]

    # Global average pooling of gradients
    weights = grads.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]

    # Weighted combination of activation maps
    cam = (weights * acts).sum(dim=0)  # [H, W]
    cam = torch.relu(cam)  # ReLU to keep positive influence

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam.cpu().numpy()
```

---

## Best Practices

### 1. Architecture Design

```python
# ✓ GOOD: Increasing channels, decreasing spatial dimensions
channels = [64, 128, 256, 512, 512]
spatial_sizes = [112, 56, 28, 14, 7]

# ✗ BAD: Inconsistent channel scaling
# channels = [64, 100, 200, 350, 512]
```

### 2. Normalization

```python
# ✓ GOOD: BatchNorm after Conv, before activation
conv = nn.Conv2d(64, 128, 3, padding=1, bias=False)
bn = nn.BatchNorm2d(128)
relu = nn.ReLU()

# 2025 Alternative: LayerNorm for stability
ln = nn.LayerNorm([128, 28, 28])
```

### 3. Activation Functions

```python
# ✓ GOOD: ReLU or GELU
relu = nn.ReLU(inplace=True)
gelu = nn.GELU()

# ✓ GOOD: Swish/SiLU for modern architectures
silu = nn.SiLU()

# ✗ BAD: Sigmoid/Tanh in hidden layers (vanishing gradients)
```

### 4. Initialization

```python
def initialize_weights(model):
    """
    Proper weight initialization.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Kaiming (He) initialization for ReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
```

### 5. Regularization

```python
class RegularizedCNN(nn.Module):
    """CNN with multiple regularization techniques."""
    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            # ... conv layers ...
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Data augmentation (applied during training)
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
        ])

    def forward(self, x, training=True):
        x = self.features(x)
        if training:
            x = self.dropout(x)
        return x

# Optimizer with weight decay (L2 regularization)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

---

## 2025 State-of-the-Art

### Key Trends

1. **Larger Kernels Return**: 7×7 and larger (ConvNeXt)
2. **Depthwise Separable**: Efficient convolutions (MobileNet, EfficientNet)
3. **Hybrid Architectures**: CNNs + Transformers
4. **Layer Normalization**: Replacing BatchNorm in some architectures
5. **GELU Activation**: Preferred over ReLU in modern models
6. **No Pooling**: Strided convolutions instead

### Modern Conv Block (2025)

```python
class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block (2022-2025 SOTA).

    Modernizes CNNs with:
    - Large kernels (7x7)
    - Depthwise convolutions
    - Inverted bottleneck (expand-then-compress)
    - LayerNorm
    - GELU activation
    """
    def __init__(self, dim, expansion=4, kernel_size=7):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size//2, groups=dim
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)

        # Layer scale (learnable scaling factor)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(self, x):
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Channel-first to channel-last
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        # Normalization
        x = self.norm(x)

        # Inverted bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer scale
        x = self.gamma * x

        # Channel-last to channel-first
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        return residual + x
```

### When to Use CNNs (2025)

**Use CNNs when:**
- Images have strong spatial structure
- Translation equivariance is important
- Limited computational resources
- Small to medium datasets
- Real-time inference required

**Consider Vision Transformers when:**
- Large datasets available (>100M images)
- Global context is important
- Computational resources abundant
- Highest accuracy needed

**Best practice**: Hybrid architectures combining both!

---

## Summary

This guide covered CNN fundamentals from first principles to 2025 best practices:

1. **Convolution operations**: Cross-correlation, multi-channel, parameter counting
2. **Pooling**: Max, average, global pooling strategies
3. **Padding & Stride**: Spatial dimension control
4. **Parameter sharing**: Efficiency through weight tying
5. **Receptive fields**: Theoretical vs. effective
6. **Architecture design**: Modern best practices
7. **Implementation**: Production-ready PyTorch code
8. **Visualization**: Understanding learned features

**Key Takeaways:**
- CNNs are the backbone of computer vision
- Modern trends: larger kernels, depthwise separable, hybrid architectures
- Proper initialization and regularization are critical
- Understanding receptive fields guides architecture design
- 2025: CNNs remain competitive with proper modernization

**Next Steps:**
- Study specific architectures (ResNet, EfficientNet, ConvNeXt)
- Implement object detection and segmentation
- Explore Vision Transformers
- Practice on real datasets (ImageNet, COCO)

---

**References:**

- LeCun et al. (1998). "Gradient-based learning applied to document recognition"
- Krizhevsky et al. (2012). "ImageNet Classification with Deep CNNs"
- He et al. (2015). "Deep Residual Learning for Image Recognition"
- Howard et al. (2017). "MobileNets: Efficient CNNs for Mobile Vision"
- Tan & Le (2019). "EfficientNet: Rethinking Model Scaling for CNNs"
- Liu et al. (2022). "A ConvNet for the 2020s"
- Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition"

---

*End of CNN Fundamentals*
