# 27. Autoencoders

## Table of Contents
- [1. Introduction to Autoencoders](#1-introduction-to-autoencoders)
- [2. Vanilla Autoencoder](#2-vanilla-autoencoder)
- [3. Denoising Autoencoder](#3-denoising-autoencoder)
- [4. Sparse Autoencoder](#4-sparse-autoencoder)
- [5. Convolutional Autoencoder](#5-convolutional-autoencoder)
- [6. Contractive Autoencoder](#6-contractive-autoencoder)
- [7. Autoencoder Applications](#7-autoencoder-applications)
- [8. Training Best Practices](#8-training-best-practices)
- [9. Comparison with PCA](#9-comparison-with-pca)
- [10. Resources and References](#10-resources-and-references)

---

## 1. Introduction to Autoencoders

### 1.1 What are Autoencoders?

**Autoencoders** are unsupervised neural networks that learn to compress data into a lower-dimensional **latent representation** and then reconstruct the original input. Unlike supervised models, autoencoders learn useful representations without labeled data by using the input itself as the target.

**Core Architecture:**
```
Input (x) --> Encoder (f) --> Latent Code (z) --> Decoder (g) --> Reconstruction (x_hat)
```

**Mathematical Formulation:**
```
z = f(x)           # Encoding: high-dim --> low-dim
x_hat = g(z)       # Decoding: low-dim --> high-dim
x_hat = g(f(x))    # Full reconstruction
```

**Objective Function:**
```
min L(x, x_hat) = min ||x - g(f(x))||^2
```

### 1.2 Encoder-Decoder Architecture

The autoencoder consists of three components:

**Encoder Network (f):**
- Maps input x from input space R^n to latent space R^d
- Typically d << n (bottleneck forces compression)
- Can be linear (like PCA) or nonlinear (neural network)
- Learns compressed representation z = f(x; theta_enc)

**Latent Space (z):**
- Low-dimensional representation (bottleneck)
- Forces network to learn meaningful features
- Size d is a critical hyperparameter
- Too small: information loss, poor reconstruction
- Too large: network memorizes, no compression

**Decoder Network (g):**
- Maps latent code z back to input space R^n
- Mirrors encoder architecture (typically symmetric)
- Learns reconstruction x_hat = g(z; theta_dec)
- Final activation depends on data type (sigmoid for images, linear for continuous)

### 1.3 Latent Space Properties

The **latent space** is where the magic happens:

**Dimensionality Reduction:**
- Compresses high-dimensional input to low-dimensional code
- Example: 784-dim MNIST image --> 32-dim latent vector
- Compression ratio: 784/32 = 24.5x

**Feature Learning:**
- Network automatically discovers useful features
- No manual feature engineering required
- Learns hierarchical representations (deep autoencoders)

**Limitations (vs VAE):**
- Latent space is **not structured** or continuous
- Cannot sample random z to generate new data
- No probabilistic interpretation
- Points between valid codes may not decode meaningfully

### 1.4 Types of Autoencoders

**Undercomplete Autoencoder:**
- Latent dimension d < input dimension n
- Forces compression and feature learning
- Most common type

**Overcomplete Autoencoder:**
- Latent dimension d >= input dimension n
- Requires regularization to prevent identity mapping
- Uses sparsity or other constraints

**Loss Functions:**
- **MSE (Mean Squared Error):** For continuous data, L2 reconstruction
- **BCE (Binary Cross-Entropy):** For binary/normalized image data
- **Custom losses:** Perceptual loss, adversarial loss

### 1.5 History and Context

**2006:** Hinton & Salakhutdinov demonstrate deep autoencoders can outperform PCA for dimensionality reduction

**2008:** Vincent et al. introduce Denoising Autoencoders (DAE) for robust feature learning

**2010:** Rifai et al. propose Contractive Autoencoders (CAE) with Jacobian regularization

**2013:** Kingma & Welling introduce Variational Autoencoders (VAE) adding probabilistic structure

**Modern Usage:**
- Pre-training for semi-supervised learning
- Anomaly detection in production systems
- Image compression and denoising
- Feature extraction for downstream tasks
- Data augmentation (through learned manifolds)

---

## 2. Vanilla Autoencoder

### 2.1 Architecture Design

A basic **fully-connected autoencoder** for tabular or flattened image data:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaAutoencoder(nn.Module):
    """
    Basic fully-connected autoencoder.

    Args:
        input_dim: Dimension of input data (e.g., 784 for MNIST)
        latent_dim: Dimension of latent code (bottleneck size)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'tanh', 'leaky_relu')
        output_activation: Final activation ('sigmoid', 'tanh', None)
    """
    def __init__(
        self,
        input_dim=784,
        latent_dim=32,
        hidden_dims=[512, 256, 128],
        activation='relu',
        output_activation='sigmoid'
    ):
        super(VanillaAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Activation function
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(self.act)
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(self.act)
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        # Output activation
        if output_activation == 'sigmoid':
            decoder_layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            decoder_layers.append(nn.Tanh())
        # None means linear output

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent representation to reconstruction."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass: encode then decode."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
```

### 2.2 Complete Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Hyperparameters
config = {
    'input_dim': 784,           # MNIST: 28x28
    'latent_dim': 32,           # Compression to 32-dim
    'hidden_dims': [512, 256, 128],
    'batch_size': 128,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,       # L2 regularization
    'num_epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    # Normalize to [0, 1] range for sigmoid output
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=4
)

# Initialize model
model = VanillaAutoencoder(
    input_dim=config['input_dim'],
    latent_dim=config['latent_dim'],
    hidden_dims=config['hidden_dims'],
    activation='relu',
    output_activation='sigmoid'
).to(config['device'])

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

# Loss function
# MSE for general reconstruction, BCE for images in [0,1]
def reconstruction_loss(x_hat, x, loss_type='mse'):
    """
    Compute reconstruction loss.

    Args:
        x_hat: Reconstructed output
        x: Original input
        loss_type: 'mse' or 'bce'
    """
    if loss_type == 'mse':
        return F.mse_loss(x_hat, x, reduction='mean')
    elif loss_type == 'bce':
        return F.binary_cross_entropy(x_hat, x, reduction='mean')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Training function
def train_epoch(model, loader, optimizer, device, loss_type='bce'):
    model.train()
    train_loss = 0.0

    for batch_idx, (data, _) in enumerate(loader):
        # Flatten images and move to device
        data = data.view(data.size(0), -1).to(device)

        # Forward pass
        x_hat, z = model(data)

        # Compute loss
        loss = reconstruction_loss(x_hat, data, loss_type)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional, for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(loader)

# Evaluation function
def evaluate(model, loader, device, loss_type='bce'):
    model.eval()
    eval_loss = 0.0

    with torch.no_grad():
        for data, _ in loader:
            data = data.view(data.size(0), -1).to(device)
            x_hat, z = model(data)
            loss = reconstruction_loss(x_hat, data, loss_type)
            eval_loss += loss.item()

    return eval_loss / len(loader)

# Training loop
best_loss = float('inf')
train_losses = []
test_losses = []

print("Starting training...")
for epoch in range(config['num_epochs']):
    # Train
    train_loss = train_epoch(
        model, train_loader, optimizer,
        config['device'], loss_type='bce'
    )

    # Evaluate
    test_loss = evaluate(
        model, test_loader,
        config['device'], loss_type='bce'
    )

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    # Learning rate scheduling
    scheduler.step(test_loss)

    # Save best model
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': test_loss,
            'config': config
        }, 'best_autoencoder.pth')

    # Print progress
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] "
              f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

print(f"Training complete! Best test loss: {best_loss:.4f}")
```

### 2.3 Reconstruction Quality Analysis

```python
import matplotlib.pyplot as plt

def visualize_reconstructions(model, loader, device, num_samples=10):
    """Visualize original vs reconstructed images."""
    model.eval()

    # Get a batch
    data, _ = next(iter(loader))
    data = data[:num_samples]

    # Reconstruct
    with torch.no_grad():
        data_flat = data.view(num_samples, -1).to(device)
        x_hat, z = model(data_flat)
        x_hat = x_hat.cpu().view(-1, 1, 28, 28)

    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*2, 4))

    for i in range(num_samples):
        # Original
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)

        # Reconstruction
        axes[1, i].imshow(x_hat[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=12)

    plt.tight_layout()
    plt.savefig('reconstructions.png', dpi=150)
    plt.show()

visualize_reconstructions(model, test_loader, config['device'])
```

### 2.4 Latent Space Analysis

```python
def extract_latent_codes(model, loader, device):
    """Extract latent codes for entire dataset."""
    model.eval()
    latent_codes = []
    labels_list = []

    with torch.no_grad():
        for data, labels in loader:
            data = data.view(data.size(0), -1).to(device)
            z = model.encode(data)
            latent_codes.append(z.cpu().numpy())
            labels_list.append(labels.numpy())

    latent_codes = np.concatenate(latent_codes, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    return latent_codes, labels_list

# Extract codes
z_train, y_train = extract_latent_codes(model, train_loader, config['device'])

print(f"Latent codes shape: {z_train.shape}")  # (60000, 32)
print(f"Compression ratio: {config['input_dim'] / config['latent_dim']:.1f}x")

# Visualize with t-SNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
z_2d = tsne.fit_transform(z_train[:5000])  # Subset for speed

plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_train[:5000],
                     cmap='tab10', alpha=0.6, s=5)
plt.colorbar(scatter, label='Digit Class')
plt.title('t-SNE Visualization of Autoencoder Latent Space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.savefig('latent_space_tsne.png', dpi=150)
plt.show()
```

---

## 3. Denoising Autoencoder

### 3.1 Motivation and Theory

**Denoising Autoencoders (DAE)** learn robust features by reconstructing clean data from corrupted inputs. This forces the network to learn the underlying data manifold rather than just copying inputs.

**Key Idea:**
```
x_clean --> Add noise --> x_noisy --> Encoder --> z --> Decoder --> x_hat
Loss: ||x_hat - x_clean||^2
```

**Why Denoising Works:**
- Prevents identity mapping and overfitting
- Forces encoder to learn robust, invariant features
- Network must understand data structure to denoise
- Acts as implicit regularization

**Noise Types:**
- **Gaussian Noise:** Add N(0, sigma^2) to each pixel
- **Salt-and-Pepper Noise:** Randomly set pixels to 0 or 1
- **Masking Noise:** Randomly set pixels to 0 (like dropout)
- **Blur:** Apply Gaussian blur to images

### 3.2 Noise Injection Strategies

```python
import torch
import torch.nn.functional as F

class NoiseInjector:
    """Different noise injection strategies for denoising autoencoders."""

    @staticmethod
    def gaussian_noise(x, noise_factor=0.3):
        """
        Add Gaussian noise to input.

        Args:
            x: Input tensor (B, C, H, W) or (B, D)
            noise_factor: Std dev of Gaussian noise
        """
        noise = torch.randn_like(x) * noise_factor
        x_noisy = x + noise
        return torch.clamp(x_noisy, 0.0, 1.0)

    @staticmethod
    def salt_pepper_noise(x, amount=0.05):
        """
        Add salt-and-pepper noise.

        Args:
            x: Input tensor
            amount: Fraction of pixels to corrupt
        """
        x_noisy = x.clone()

        # Salt (white pixels)
        num_salt = int(amount * x.numel() * 0.5)
        salt_coords = [torch.randint(0, i, (num_salt,)) for i in x.shape]
        x_noisy[salt_coords] = 1.0

        # Pepper (black pixels)
        num_pepper = int(amount * x.numel() * 0.5)
        pepper_coords = [torch.randint(0, i, (num_pepper,)) for i in x.shape]
        x_noisy[pepper_coords] = 0.0

        return x_noisy

    @staticmethod
    def masking_noise(x, mask_prob=0.3):
        """
        Randomly mask (set to 0) input features.

        Args:
            x: Input tensor
            mask_prob: Probability of masking each feature
        """
        mask = torch.bernoulli(torch.ones_like(x) * (1 - mask_prob))
        return x * mask

    @staticmethod
    def dropout_noise(x, dropout_prob=0.5):
        """
        Apply dropout noise (different from masking by scaling).

        Args:
            x: Input tensor
            dropout_prob: Dropout probability
        """
        return F.dropout(x, p=dropout_prob, training=True)

    @staticmethod
    def blur_noise(x, kernel_size=5, sigma=1.0):
        """
        Apply Gaussian blur (for images).

        Args:
            x: Input tensor (B, C, H, W)
            kernel_size: Size of blur kernel (must be odd)
            sigma: Standard deviation of Gaussian
        """
        from torchvision.transforms import GaussianBlur
        blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        return blur(x)

# Example usage
injector = NoiseInjector()
```

### 3.3 Complete DAE Implementation

```python
class DenoisingAutoencoder(nn.Module):
    """
    Denoising autoencoder with configurable noise type.

    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension
        hidden_dims: List of hidden layer sizes
        noise_type: Type of noise ('gaussian', 'masking', 'salt_pepper')
        noise_factor: Strength of noise injection
    """
    def __init__(
        self,
        input_dim=784,
        latent_dim=64,
        hidden_dims=[512, 256],
        noise_type='gaussian',
        noise_factor=0.3
    ):
        super(DenoisingAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.noise_type = noise_type
        self.noise_factor = noise_factor
        self.injector = NoiseInjector()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    def add_noise(self, x):
        """Add noise to input based on noise_type."""
        if not self.training:
            return x  # No noise during evaluation

        if self.noise_type == 'gaussian':
            return self.injector.gaussian_noise(x, self.noise_factor)
        elif self.noise_type == 'masking':
            return self.injector.masking_noise(x, self.noise_factor)
        elif self.noise_type == 'salt_pepper':
            return self.injector.salt_pepper_noise(x, self.noise_factor)
        else:
            return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, return_noisy=False):
        """
        Forward pass with noise injection.

        Args:
            x: Clean input
            return_noisy: If True, return noisy input as well

        Returns:
            x_hat: Reconstruction
            z: Latent code
            x_noisy: Noisy input (if return_noisy=True)
        """
        x_noisy = self.add_noise(x)
        z = self.encode(x_noisy)
        x_hat = self.decode(z)

        if return_noisy:
            return x_hat, z, x_noisy
        return x_hat, z

# Training loop for DAE
def train_denoising_autoencoder(
    model, train_loader, test_loader,
    num_epochs=50, lr=1e-3, device='cuda'
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for data, _ in train_loader:
            data = data.view(data.size(0), -1).to(device)

            # Forward pass (noise added inside model)
            x_hat, z = model(data)

            # Loss: reconstruction vs CLEAN input
            loss = F.binary_cross_entropy(x_hat, data)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluation
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.view(data.size(0), -1).to(device)
                x_hat, z = model(data)
                loss = F.binary_cross_entropy(x_hat, data)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    return train_losses, test_losses

# Initialize and train
dae_model = DenoisingAutoencoder(
    input_dim=784,
    latent_dim=64,
    hidden_dims=[512, 256],
    noise_type='gaussian',
    noise_factor=0.3
).to(config['device'])

train_losses, test_losses = train_denoising_autoencoder(
    dae_model, train_loader, test_loader,
    num_epochs=50, lr=1e-3, device=config['device']
)
```

### 3.4 Denoising Visualization

```python
def visualize_denoising(model, loader, device, num_samples=10):
    """Visualize: original --> noisy --> denoised."""
    model.eval()

    data, _ = next(iter(loader))
    data = data[:num_samples]

    with torch.no_grad():
        data_flat = data.view(num_samples, -1).to(device)

        # Add noise manually for visualization
        x_noisy = model.injector.gaussian_noise(data_flat, model.noise_factor)

        # Denoise
        x_hat, z = model(data_flat)

        # Reshape for display
        x_noisy = x_noisy.cpu().view(-1, 1, 28, 28)
        x_hat = x_hat.cpu().view(-1, 1, 28, 28)

    # Plot
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples*2, 6))

    for i in range(num_samples):
        # Original
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)

        # Noisy
        axes[1, i].imshow(x_noisy[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Noisy', fontsize=12)

        # Denoised
        axes[2, i].imshow(x_hat[i].squeeze(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Denoised', fontsize=12)

    plt.tight_layout()
    plt.savefig('denoising_results.png', dpi=150)
    plt.show()

visualize_denoising(dae_model, test_loader, config['device'])
```

---

## 4. Sparse Autoencoder

### 4.1 Sparsity Motivation

**Sparse Autoencoders** encourage most neurons in the latent layer to be inactive (near zero) for any given input. This creates **sparse representations** that are:

- **Interpretable:** Each feature specializes to specific patterns
- **Efficient:** Most features inactive, reducing computation
- **Robust:** Forces network to learn independent features
- **Generalizable:** Prevents overfitting and memorization

**Biological Inspiration:** Neurons in the brain exhibit sparse firing patterns

**Mathematical Formulation:**
```
L_total = L_reconstruction + beta * L_sparsity
```

Where:
- L_reconstruction: Standard reconstruction loss (MSE or BCE)
- L_sparsity: Penalty encouraging sparsity
- beta: Sparsity weight (typically 1e-3 to 1e-1)

### 4.2 Sparsity Penalties

**L1 Regularization:**
```
L_sparsity = mean(|z|) = (1/d) * sum(|z_i|)
```
- Simplest approach
- Directly penalizes absolute values
- Encourages exact zeros

**KL Divergence Sparsity:**
```
rho_hat_j = (1/m) * sum_i sigmoid(z_ij)  # Average activation
rho = target sparsity (e.g., 0.05)
L_sparsity = sum_j KL(rho || rho_hat_j)
             = sum_j [rho*log(rho/rho_hat_j) + (1-rho)*log((1-rho)/(1-rho_hat_j))]
```
- Penalizes deviation from target activation
- More principled than L1
- Used in classic sparse autoencoder papers

**Activation Maximization:**
```
L_sparsity = -log(max(|z|)) + log(mean(|z|))
```
- Encourages few large activations
- Promotes "winner-take-all" dynamics

### 4.3 Complete Implementation

```python
class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder with L1 or KL-divergence sparsity penalty.

    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension (typically larger than vanilla AE)
        hidden_dims: Hidden layer dimensions
        sparsity_type: 'l1' or 'kl'
        sparsity_target: Target average activation (for KL)
        sparsity_weight: Beta coefficient for sparsity loss
    """
    def __init__(
        self,
        input_dim=784,
        latent_dim=128,  # Often larger than vanilla AE
        hidden_dims=[512, 256],
        sparsity_type='kl',
        sparsity_target=0.05,
        sparsity_weight=1e-3
    ):
        super(SparseAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_type = sparsity_type
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        # Latent layer WITHOUT final activation (linear)
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def l1_sparsity_loss(self, z):
        """
        L1 sparsity penalty.

        Args:
            z: Latent code (B, latent_dim)

        Returns:
            L1 norm of latent activations
        """
        return torch.mean(torch.abs(z))

    def kl_sparsity_loss(self, z):
        """
        KL divergence sparsity penalty.

        Args:
            z: Latent code (B, latent_dim)

        Returns:
            KL divergence from target sparsity
        """
        # Average activation per neuron across batch
        rho_hat = torch.sigmoid(z).mean(dim=0)  # Shape: (latent_dim,)
        rho = self.sparsity_target

        # KL divergence: KL(rho || rho_hat)
        # Avoid log(0) with epsilon
        eps = 1e-8
        kl_div = (
            rho * torch.log((rho + eps) / (rho_hat + eps)) +
            (1 - rho) * torch.log((1 - rho + eps) / (1 - rho_hat + eps))
        )

        return torch.sum(kl_div)

    def compute_loss(self, x, x_hat, z, loss_type='bce'):
        """
        Total loss = reconstruction + sparsity.

        Args:
            x: Original input
            x_hat: Reconstruction
            z: Latent code
            loss_type: 'mse' or 'bce'

        Returns:
            total_loss, recon_loss, sparsity_loss
        """
        # Reconstruction loss
        if loss_type == 'mse':
            recon_loss = F.mse_loss(x_hat, x)
        elif loss_type == 'bce':
            recon_loss = F.binary_cross_entropy(x_hat, x)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Sparsity loss
        if self.sparsity_type == 'l1':
            sparsity_loss = self.l1_sparsity_loss(z)
        elif self.sparsity_type == 'kl':
            sparsity_loss = self.kl_sparsity_loss(z)
        else:
            sparsity_loss = 0.0

        # Total loss
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        return total_loss, recon_loss, sparsity_loss

# Training function
def train_sparse_autoencoder(
    model, train_loader, test_loader,
    num_epochs=50, lr=1e-3, device='cuda'
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {
        'train_total': [], 'train_recon': [], 'train_sparse': [],
        'test_total': [], 'test_recon': [], 'test_sparse': []
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_total, train_recon, train_sparse = 0.0, 0.0, 0.0

        for data, _ in train_loader:
            data = data.view(data.size(0), -1).to(device)

            # Forward
            x_hat, z = model(data)
            total_loss, recon_loss, sparsity_loss = model.compute_loss(
                data, x_hat, z, loss_type='bce'
            )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total += total_loss.item()
            train_recon += recon_loss.item()
            train_sparse += sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss

        train_total /= len(train_loader)
        train_recon /= len(train_loader)
        train_sparse /= len(train_loader)

        # Evaluation
        model.eval()
        test_total, test_recon, test_sparse = 0.0, 0.0, 0.0

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.view(data.size(0), -1).to(device)
                x_hat, z = model(data)
                total_loss, recon_loss, sparsity_loss = model.compute_loss(
                    data, x_hat, z, loss_type='bce'
                )
                test_total += total_loss.item()
                test_recon += recon_loss.item()
                test_sparse += sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss

        test_total /= len(test_loader)
        test_recon /= len(test_loader)
        test_sparse /= len(test_loader)

        # Save history
        history['train_total'].append(train_total)
        history['train_recon'].append(train_recon)
        history['train_sparse'].append(train_sparse)
        history['test_total'].append(test_total)
        history['test_recon'].append(test_recon)
        history['test_sparse'].append(test_sparse)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train - Total: {train_total:.4f} | Recon: {train_recon:.4f} | Sparse: {train_sparse:.4f}")
            print(f"  Test  - Total: {test_total:.4f} | Recon: {test_recon:.4f} | Sparse: {test_sparse:.4f}")

    return history

# Initialize and train
sparse_model = SparseAutoencoder(
    input_dim=784,
    latent_dim=128,
    hidden_dims=[512, 256],
    sparsity_type='kl',
    sparsity_target=0.05,
    sparsity_weight=1e-3
).to(config['device'])

history = train_sparse_autoencoder(
    sparse_model, train_loader, test_loader,
    num_epochs=50, lr=1e-3, device=config['device']
)
```

### 4.4 Analyzing Sparsity

```python
def analyze_sparsity(model, loader, device, threshold=0.1):
    """Analyze sparsity of latent representations."""
    model.eval()

    all_latents = []

    with torch.no_grad():
        for data, _ in loader:
            data = data.view(data.size(0), -1).to(device)
            z = model.encode(data)
            all_latents.append(z.cpu())

    all_latents = torch.cat(all_latents, dim=0)  # (N, latent_dim)

    # Compute sparsity metrics
    # Apply sigmoid for KL-based models (activation)
    activations = torch.sigmoid(all_latents)

    # Percentage of near-zero activations
    near_zero = (activations < threshold).float().mean().item()

    # Average activation per neuron
    avg_activation = activations.mean(dim=0)  # (latent_dim,)

    # Lifetime sparsity (fraction of samples where neuron is active)
    lifetime_sparsity = (activations > threshold).float().mean(dim=0)

    print(f"Sparsity Analysis:")
    print(f"  Percentage near-zero: {near_zero*100:.2f}%")
    print(f"  Mean activation: {avg_activation.mean():.4f}")
    print(f"  Std activation: {avg_activation.std():.4f}")
    print(f"  Mean lifetime sparsity: {lifetime_sparsity.mean():.4f}")

    # Visualize activation distribution
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.hist(avg_activation.numpy(), bins=50, edgecolor='black')
    plt.xlabel('Average Activation')
    plt.ylabel('Number of Neurons')
    plt.title('Distribution of Average Activations')
    plt.axvline(model.sparsity_target, color='r', linestyle='--',
                label=f'Target: {model.sparsity_target}')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(lifetime_sparsity.numpy(), bins=50, edgecolor='black')
    plt.xlabel('Lifetime Sparsity')
    plt.ylabel('Number of Neurons')
    plt.title('Distribution of Lifetime Sparsity')

    plt.subplot(1, 3, 3)
    plt.imshow(activations[:100].T.numpy(), aspect='auto', cmap='hot')
    plt.xlabel('Sample Index')
    plt.ylabel('Neuron Index')
    plt.title('Activation Heatmap (First 100 Samples)')
    plt.colorbar(label='Activation')

    plt.tight_layout()
    plt.savefig('sparsity_analysis.png', dpi=150)
    plt.show()

    return {
        'near_zero_pct': near_zero,
        'avg_activation': avg_activation.mean().item(),
        'lifetime_sparsity': lifetime_sparsity.mean().item()
    }

sparsity_metrics = analyze_sparsity(
    sparse_model, test_loader, config['device'], threshold=0.1
)
```

---

## 5. Convolutional Autoencoder

### 5.1 Motivation for CNNs

For image data, **Convolutional Autoencoders** are more appropriate than fully-connected autoencoders because they:

- **Preserve spatial structure:** Don't flatten images, maintain 2D layout
- **Parameter efficiency:** Shared convolutional kernels vs dense connections
- **Translation invariance:** Learn local patterns that work anywhere in image
- **Hierarchical features:** Early layers detect edges, later layers detect complex patterns

**Architecture Pattern:**
```
Input Image (H, W, C)
  --> Conv layers (downsample with stride or pooling)
  --> Bottleneck (latent representation)
  --> Transposed Conv layers (upsample)
  --> Reconstructed Image (H, W, C)
```

### 5.2 Transposed Convolution (Deconvolution)

**Transposed Convolution** upsamples feature maps in the decoder. Key parameters:

```python
nn.ConvTranspose2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0  # Critical for exact size matching
)
```

**Output size formula:**
```
H_out = (H_in - 1) * stride - 2*padding + kernel_size + output_padding
```

**Common pattern:**
- Encoder: stride=2, padding=1 (halves spatial dims)
- Decoder: stride=2, padding=1, output_padding=1 (doubles spatial dims)

### 5.3 Complete Conv Autoencoder

```python
class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for image data.

    Args:
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        latent_dim: Dimension of flattened latent code
        base_channels: Base number of channels (doubled at each layer)
        image_size: Input image size (assumes square images)
    """
    def __init__(
        self,
        input_channels=1,
        latent_dim=128,
        base_channels=32,
        image_size=28
    ):
        super(ConvAutoencoder, self).__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Calculate spatial dimensions after encoding
        # With 3 stride-2 conv layers: 28 --> 14 --> 7 --> 3 (for MNIST)
        # General formula: size // (2^num_layers)
        self.encoded_spatial = image_size // 8  # 3 layers
        self.encoded_channels = base_channels * 8  # 32 * 8 = 256
        self.encoded_dim = self.encoded_channels * self.encoded_spatial * self.encoded_spatial

        # Encoder
        self.encoder_conv = nn.Sequential(
            # Layer 1: (1, 28, 28) --> (32, 14, 14)
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # Layer 2: (32, 14, 14) --> (64, 7, 7)
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),

            # Layer 3: (64, 7, 7) --> (128, 4, 4) or (256, 3, 3)
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),

            # Layer 4: (128, 4, 4) --> (256, 2, 2) or similar
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True)
        )

        # Flatten and compress to latent
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoded_dim, latent_dim)
        )

        # Decoder: expand from latent
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.encoded_dim),
            nn.ReLU(inplace=True)
        )

        # Decoder: transposed convolutions
        self.decoder_conv = nn.Sequential(
            # Unflatten
            nn.Unflatten(1, (self.encoded_channels, self.encoded_spatial, self.encoded_spatial)),

            # Layer 1: upsample
            nn.ConvTranspose2d(base_channels*8, base_channels*4,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),

            # Layer 2: upsample
            nn.ConvTranspose2d(base_channels*4, base_channels*2,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),

            # Layer 3: upsample
            nn.ConvTranspose2d(base_channels*2, base_channels,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # Layer 4: final upsample to original size
            nn.ConvTranspose2d(base_channels, input_channels,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def encode(self, x):
        """Encode image to latent code."""
        x = self.encoder_conv(x)
        z = self.encoder_fc(x)
        return z

    def decode(self, z):
        """Decode latent code to image."""
        x = self.decoder_fc(z)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        """Full forward pass."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

# Alternative: Using MaxPooling/Unpooling
class ConvAutoencoderWithPooling(nn.Module):
    """Conv autoencoder using MaxPool/Upsample instead of strided conv."""

    def __init__(self, input_channels=1, latent_dim=128):
        super(ConvAutoencoderWithPooling, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # (1, 28, 28) --> (32, 28, 28)
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # --> (32, 14, 14)

            # (32, 14, 14) --> (64, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # --> (64, 7, 7)

            # (64, 7, 7) --> (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=1),  # --> (128, 4, 4)

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),

            # (128, 4, 4) --> (128, 8, 8)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (64, 8, 8) --> (64, 14, 14)
            nn.Upsample(size=(14, 14), mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # (32, 14, 14) --> (32, 28, 28)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
```

### 5.4 Training Conv Autoencoder

```python
# Data loader for images (don't flatten!)
transform_conv = transforms.Compose([
    transforms.ToTensor(),
    # No normalization needed if using sigmoid output
])

train_dataset_conv = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_conv
)
test_dataset_conv = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_conv
)

train_loader_conv = DataLoader(
    train_dataset_conv, batch_size=128, shuffle=True, num_workers=4
)
test_loader_conv = DataLoader(
    test_dataset_conv, batch_size=128, shuffle=False, num_workers=4
)

# Initialize model
conv_model = ConvAutoencoder(
    input_channels=1,
    latent_dim=128,
    base_channels=32,
    image_size=28
).to(config['device'])

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Conv Autoencoder parameters: {count_parameters(conv_model):,}")

# Training loop
def train_conv_autoencoder(
    model, train_loader, test_loader,
    num_epochs=50, lr=1e-3, device='cuda'
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for data, _ in train_loader:
            data = data.to(device)  # Keep 4D shape: (B, C, H, W)

            x_hat, z = model(data)
            loss = F.binary_cross_entropy(x_hat, data)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluation
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                x_hat, z = model(data)
                loss = F.binary_cross_entropy(x_hat, data)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    return train_losses, test_losses

# Train
train_losses, test_losses = train_conv_autoencoder(
    conv_model, train_loader_conv, test_loader_conv,
    num_epochs=50, lr=1e-3, device=config['device']
)
```

### 5.5 Advanced: Skip Connections (UNet-style)

```python
class ConvAutoencoderWithSkips(nn.Module):
    """
    Convolutional autoencoder with skip connections.
    Inspired by U-Net architecture for better reconstruction.
    """
    def __init__(self, input_channels=1, latent_dim=128):
        super(ConvAutoencoderWithSkips, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)

        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4))
        )

        # Decoder with skip connections
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, 3, padding=1),  # +128 from skip
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up2 = nn.Upsample(size=(14, 14), mode='nearest')
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, 3, padding=1),  # +64 from skip
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = nn.Sequential(
            nn.Conv2d(32 + 32, input_channels, 3, padding=1),  # +32 from skip
            nn.Sigmoid()
        )

    def encode(self, x):
        # Forward through encoder, save skip connections
        skip1 = self.enc1(x)
        x = self.pool1(skip1)

        skip2 = self.enc2(x)
        x = self.pool2(skip2)

        skip3 = self.enc3(x)
        x = self.pool3(skip3)

        z = self.bottleneck_conv(x)

        return z, (skip1, skip2, skip3)

    def decode(self, z, skips):
        skip1, skip2, skip3 = skips

        x = self.up1(z)
        # Crop skip3 if needed to match size
        x = torch.cat([x, skip3[:, :, :x.size(2), :x.size(3)]], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec3(x)

        return x

    def forward(self, x):
        z, skips = self.encode(x)
        # For compatibility, extract latent before bottleneck
        latent = z.flatten(1) if len(z.shape) > 2 else z
        x_hat = self.decode(z, skips)
        return x_hat, latent
```

---

## 6. Contractive Autoencoder

### 6.1 Theory and Motivation

**Contractive Autoencoders (CAE)** add a regularization term that penalizes the sensitivity of learned representations to input variations. The goal is to make the latent code **robust to small perturbations** in the input.

**Key Idea:**
- Encourage the encoder to be **locally invariant** to small changes in input
- Achieved by penalizing large derivatives of the encoder w.r.t. input
- Makes learned features more robust and stable

**Mathematical Formulation:**
```
L_total = L_reconstruction + lambda * ||J_f(x)||_F^2
```

Where:
- J_f(x) is the Jacobian matrix of encoder f w.r.t. input x
- ||.||_F is the Frobenius norm (sum of squared elements)
- lambda is the contraction penalty weight (typically 1e-4 to 1e-2)

**Jacobian Matrix:**
```
J_f(x)_ij = d(z_i) / d(x_j)
```
- Shape: (latent_dim, input_dim)
- Measures how each latent dimension responds to each input dimension
- Small Jacobian --> latent code insensitive to input changes

### 6.2 Computing the Jacobian Penalty

```python
def jacobian_penalty(x, z, model):
    """
    Compute Frobenius norm of Jacobian matrix.

    Args:
        x: Input tensor (B, D) - requires grad
        z: Latent code (B, latent_dim)
        model: Autoencoder model

    Returns:
        Frobenius norm of Jacobian
    """
    # Ensure x has gradients enabled
    x.requires_grad_(True)

    # Compute Jacobian penalty for each latent dimension
    batch_size, latent_dim = z.shape
    jacobian_norm = 0.0

    for i in range(latent_dim):
        # Compute gradient of z_i w.r.t. x
        grad_outputs = torch.zeros_like(z)
        grad_outputs[:, i] = 1.0

        # Backprop to get dz_i/dx
        gradients = torch.autograd.grad(
            outputs=z,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,  # For second-order gradients
            retain_graph=True
        )[0]

        # Add squared L2 norm of gradient
        jacobian_norm += torch.sum(gradients ** 2)

    # Average over batch
    jacobian_norm = jacobian_norm / batch_size

    return jacobian_norm

# Efficient approximation using random projection
def jacobian_penalty_efficient(x, z):
    """
    Efficient approximation of Jacobian penalty using random projection.

    Args:
        x: Input tensor (B, D) - requires grad
        z: Latent code (B, latent_dim)

    Returns:
        Approximation of Frobenius norm
    """
    x.requires_grad_(True)

    # Random direction in latent space
    v = torch.randn_like(z)

    # Compute v^T * J (directional derivative)
    grad = torch.autograd.grad(
        outputs=z,
        inputs=x,
        grad_outputs=v,
        create_graph=True,
        retain_graph=True
    )[0]

    # Squared norm
    penalty = torch.sum(grad ** 2) / x.size(0)

    return penalty
```

### 6.3 Complete CAE Implementation

```python
class ContractiveAutoencoder(nn.Module):
    """
    Contractive autoencoder with Jacobian regularization.

    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension
        hidden_dims: Hidden layer dimensions
        lambda_c: Contraction penalty weight
    """
    def __init__(
        self,
        input_dim=784,
        latent_dim=64,
        hidden_dims=[512, 256],
        lambda_c=1e-4
    ):
        super(ContractiveAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lambda_c = lambda_c

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.Sigmoid())  # Bounded activation for stability

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def contractive_loss(self, x, z):
        """
        Compute contractive penalty (Jacobian Frobenius norm).

        Args:
            x: Input (must have requires_grad=True)
            z: Latent code

        Returns:
            Contractive penalty term
        """
        # Use efficient approximation
        penalty = jacobian_penalty_efficient(x, z)
        return self.lambda_c * penalty

    def compute_loss(self, x, x_hat, z, loss_type='bce'):
        """
        Total loss = reconstruction + contraction.

        Args:
            x: Original input (requires grad)
            x_hat: Reconstruction
            z: Latent code
            loss_type: 'mse' or 'bce'

        Returns:
            total_loss, recon_loss, contract_loss
        """
        # Reconstruction loss
        if loss_type == 'mse':
            recon_loss = F.mse_loss(x_hat, x)
        elif loss_type == 'bce':
            recon_loss = F.binary_cross_entropy(x_hat, x)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Contractive loss
        contract_loss = self.contractive_loss(x, z)

        # Total loss
        total_loss = recon_loss + contract_loss

        return total_loss, recon_loss, contract_loss

# Training function
def train_contractive_autoencoder(
    model, train_loader, test_loader,
    num_epochs=50, lr=1e-3, device='cuda'
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {
        'train_total': [], 'train_recon': [], 'train_contract': [],
        'test_total': [], 'test_recon': [], 'test_contract': []
    }

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_total, train_recon, train_contract = 0.0, 0.0, 0.0

        for data, _ in train_loader:
            data = data.view(data.size(0), -1).to(device)
            data.requires_grad_(True)  # CRITICAL for Jacobian computation

            # Forward
            x_hat, z = model(data)
            total_loss, recon_loss, contract_loss = model.compute_loss(
                data, x_hat, z, loss_type='bce'
            )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total += total_loss.item()
            train_recon += recon_loss.item()
            train_contract += contract_loss.item()

        train_total /= len(train_loader)
        train_recon /= len(train_loader)
        train_contract /= len(train_loader)

        # Evaluation (no contraction penalty needed, but compute for monitoring)
        model.eval()
        test_total, test_recon, test_contract = 0.0, 0.0, 0.0

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.view(data.size(0), -1).to(device)
                x_hat, z = model(data)
                recon_loss = F.binary_cross_entropy(x_hat, data)
                test_total += recon_loss.item()
                test_recon += recon_loss.item()

        test_total /= len(test_loader)
        test_recon /= len(test_loader)

        # Save history
        history['train_total'].append(train_total)
        history['train_recon'].append(train_recon)
        history['train_contract'].append(train_contract)
        history['test_total'].append(test_total)
        history['test_recon'].append(test_recon)
        history['test_contract'].append(test_contract)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train - Total: {train_total:.4f} | Recon: {train_recon:.4f} | Contract: {train_contract:.4f}")
            print(f"  Test  - Recon: {test_recon:.4f}")

    return history

# Initialize and train
cae_model = ContractiveAutoencoder(
    input_dim=784,
    latent_dim=64,
    hidden_dims=[512, 256],
    lambda_c=1e-4
).to(config['device'])

history = train_contractive_autoencoder(
    cae_model, train_loader, test_loader,
    num_epochs=50, lr=1e-3, device=config['device']
)
```

### 6.4 Analyzing Contraction

```python
def analyze_contraction(model, loader, device, num_samples=100):
    """Analyze how robust latent codes are to input perturbations."""
    model.eval()

    # Get samples
    data, _ = next(iter(loader))
    data = data[:num_samples].view(num_samples, -1).to(device)

    # Compute original latent codes
    with torch.no_grad():
        z_original = model.encode(data)

    # Add small perturbations and measure change in latent code
    perturbation_sizes = [0.01, 0.05, 0.1, 0.2, 0.5]
    latent_changes = []

    for epsilon in perturbation_sizes:
        # Add Gaussian noise
        data_perturbed = data + epsilon * torch.randn_like(data)
        data_perturbed = torch.clamp(data_perturbed, 0, 1)

        with torch.no_grad():
            z_perturbed = model.encode(data_perturbed)

        # Measure change in latent space
        latent_change = torch.mean(torch.norm(z_perturbed - z_original, dim=1)).item()
        latent_changes.append(latent_change)

        print(f"Perturbation epsilon={epsilon:.2f} --> Latent change: {latent_change:.4f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(perturbation_sizes, latent_changes, marker='o', linewidth=2)
    plt.xlabel('Input Perturbation Size (epsilon)')
    plt.ylabel('Average Latent Code Change')
    plt.title('Robustness of Latent Representation to Input Noise')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('contraction_analysis.png', dpi=150)
    plt.show()

    return perturbation_sizes, latent_changes

perturbation_sizes, latent_changes = analyze_contraction(
    cae_model, test_loader, config['device']
)
```

---

## 7. Autoencoder Applications

### 7.1 Dimensionality Reduction

Autoencoders provide **nonlinear dimensionality reduction**, superior to PCA for complex data.

```python
def dimensionality_reduction_pipeline(model, data_loader, device):
    """
    Use autoencoder for dimensionality reduction.

    Args:
        model: Trained autoencoder
        data_loader: DataLoader with high-dim data
        device: cuda or cpu

    Returns:
        latent_codes: Low-dim representations
        labels: Data labels
    """
    model.eval()
    latent_codes = []
    labels_list = []

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.view(data.size(0), -1).to(device)
            z = model.encode(data)
            latent_codes.append(z.cpu().numpy())
            labels_list.append(labels.numpy())

    latent_codes = np.concatenate(latent_codes, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    print(f"Reduced from {data.shape[1]} to {latent_codes.shape[1]} dimensions")
    print(f"Compression ratio: {data.shape[1] / latent_codes.shape[1]:.1f}x")

    return latent_codes, labels_list

# Use for downstream tasks
z_train, y_train = dimensionality_reduction_pipeline(model, train_loader, config['device'])

# Train classifier on compressed data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(z_train, y_train)

z_test, y_test = dimensionality_reduction_pipeline(model, test_loader, config['device'])
y_pred = clf.predict(z_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Classification accuracy on compressed features: {accuracy:.4f}")

# Visualize with UMAP (better than t-SNE for large datasets)
import umap

reducer = umap.UMAP(n_components=2, random_state=42)
z_2d = reducer.fit_transform(z_train[:10000])

plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_train[:10000],
                     cmap='tab10', alpha=0.6, s=3)
plt.colorbar(scatter, label='Class')
plt.title('UMAP Visualization of Autoencoder Latent Space')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.savefig('umap_latent_space.png', dpi=150)
plt.show()
```

### 7.2 Anomaly Detection

Autoencoders excel at **anomaly detection** by identifying samples with high reconstruction error.

```python
class AnomalyDetector:
    """
    Anomaly detection using autoencoder reconstruction error.

    Args:
        model: Trained autoencoder
        threshold: Reconstruction error threshold (computed from normal data)
        threshold_percentile: Percentile for automatic threshold (default: 95)
    """
    def __init__(self, model, threshold=None, threshold_percentile=95):
        self.model = model
        self.threshold = threshold
        self.threshold_percentile = threshold_percentile

    def compute_reconstruction_error(self, x, reduction='none'):
        """
        Compute reconstruction error for each sample.

        Args:
            x: Input data (B, D) or (B, C, H, W)
            reduction: 'none', 'mean', or 'sum'

        Returns:
            Reconstruction errors
        """
        self.model.eval()

        with torch.no_grad():
            x_hat, z = self.model(x)

            # Per-sample error
            if reduction == 'none':
                error = torch.mean((x - x_hat) ** 2, dim=tuple(range(1, len(x.shape))))
            elif reduction == 'mean':
                error = torch.mean((x - x_hat) ** 2)
            elif reduction == 'sum':
                error = torch.sum((x - x_hat) ** 2)
            else:
                raise ValueError(f"Unknown reduction: {reduction}")

        return error

    def fit_threshold(self, normal_loader, device):
        """
        Compute threshold from normal (non-anomalous) data.

        Args:
            normal_loader: DataLoader with only normal samples
            device: cuda or cpu
        """
        errors = []

        for data, _ in normal_loader:
            data = data.view(data.size(0), -1).to(device)
            error = self.compute_reconstruction_error(data, reduction='none')
            errors.append(error.cpu().numpy())

        errors = np.concatenate(errors)

        # Set threshold at percentile
        self.threshold = np.percentile(errors, self.threshold_percentile)

        print(f"Fitted threshold: {self.threshold:.6f}")
        print(f"Based on {self.threshold_percentile}th percentile of normal data")

        return self.threshold

    def predict(self, x, device):
        """
        Predict anomalies.

        Args:
            x: Input data
            device: cuda or cpu

        Returns:
            is_anomaly: Boolean array (True = anomaly)
            scores: Reconstruction error scores
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold() first.")

        x = x.to(device)
        scores = self.compute_reconstruction_error(x, reduction='none')
        is_anomaly = scores > self.threshold

        return is_anomaly.cpu().numpy(), scores.cpu().numpy()

# Example usage
detector = AnomalyDetector(model, threshold_percentile=95)

# Fit threshold on normal training data
detector.fit_threshold(train_loader, config['device'])

# Detect anomalies on test data
test_data, test_labels = next(iter(test_loader))
test_data = test_data.view(test_data.size(0), -1)

is_anomaly, scores = detector.predict(test_data, config['device'])

print(f"Detected {np.sum(is_anomaly)} anomalies out of {len(is_anomaly)} samples")

# Visualize anomaly scores
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(scores[~is_anomaly], bins=50, alpha=0.7, label='Normal', edgecolor='black')
plt.hist(scores[is_anomaly], bins=50, alpha=0.7, label='Anomaly', edgecolor='black')
plt.axvline(detector.threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Reconstruction Errors')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(scores)), scores, c=is_anomaly, cmap='coolwarm', alpha=0.6, s=10)
plt.axhline(detector.threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Sample Index')
plt.ylabel('Reconstruction Error')
plt.title('Anomaly Scores')
plt.legend()

plt.tight_layout()
plt.savefig('anomaly_detection.png', dpi=150)
plt.show()

# Show most anomalous samples
top_k = 10
anomaly_indices = np.argsort(scores)[-top_k:]

fig, axes = plt.subplots(2, top_k, figsize=(top_k*2, 4))
for i, idx in enumerate(anomaly_indices):
    # Original
    axes[0, i].imshow(test_data[idx].view(28, 28), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_ylabel('Original', fontsize=12)

    # Reconstruction
    with torch.no_grad():
        x_hat, _ = model(test_data[idx:idx+1].to(config['device']))
    axes[1, i].imshow(x_hat.cpu().view(28, 28), cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title(f'{scores[idx]:.4f}', fontsize=10)
    if i == 0:
        axes[1, i].set_ylabel('Reconstructed', fontsize=12)

plt.suptitle('Top Anomalies by Reconstruction Error')
plt.tight_layout()
plt.savefig('top_anomalies.png', dpi=150)
plt.show()
```

### 7.3 Image Denoising

Use trained denoising autoencoder for practical image denoising.

```python
def denoise_images(model, noisy_images, device, batch_size=32):
    """
    Denoise a batch of noisy images.

    Args:
        model: Trained denoising autoencoder
        noisy_images: Tensor of noisy images (N, C, H, W) or (N, D)
        device: cuda or cpu
        batch_size: Processing batch size

    Returns:
        Denoised images
    """
    model.eval()
    denoised = []

    with torch.no_grad():
        for i in range(0, len(noisy_images), batch_size):
            batch = noisy_images[i:i+batch_size].to(device)
            x_hat, _ = model(batch)
            denoised.append(x_hat.cpu())

    denoised = torch.cat(denoised, dim=0)
    return denoised

# Practical denoising example
from torchvision.utils import save_image

# Load clean images
clean_images, _ = next(iter(test_loader))
clean_images = clean_images[:16]

# Add noise
noise_factor = 0.4
noisy_images = clean_images + noise_factor * torch.randn_like(clean_images)
noisy_images = torch.clamp(noisy_images, 0, 1)

# Denoise
noisy_flat = noisy_images.view(16, -1)
denoised_flat = denoise_images(dae_model, noisy_flat, config['device'])
denoised_images = denoised_flat.view(-1, 1, 28, 28)

# Save comparison
comparison = torch.cat([clean_images, noisy_images, denoised_images], dim=0)
save_image(comparison, 'denoising_comparison.png', nrow=16)

# Compute metrics
mse_noisy = F.mse_loss(noisy_images, clean_images).item()
mse_denoised = F.mse_loss(denoised_images, clean_images).item()

print(f"MSE (noisy): {mse_noisy:.6f}")
print(f"MSE (denoised): {mse_denoised:.6f}")
print(f"Improvement: {(mse_noisy - mse_denoised) / mse_noisy * 100:.2f}%")
```

### 7.4 Feature Extraction and Pretraining

Use autoencoder encoder as feature extractor for supervised tasks.

```python
def pretrain_with_autoencoder(
    autoencoder, train_loader, test_loader,
    num_classes, freeze_encoder=True, device='cuda'
):
    """
    Pretrain encoder with autoencoder, then train classifier.

    Args:
        autoencoder: Trained autoencoder
        train_loader: Training data
        test_loader: Test data
        num_classes: Number of classes for classification
        freeze_encoder: Whether to freeze encoder weights
        device: cuda or cpu

    Returns:
        Trained classifier
    """
    # Extract encoder
    encoder = autoencoder.encoder

    # Build classifier
    classifier = nn.Sequential(
        encoder,
        nn.Linear(autoencoder.latent_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    ).to(device)

    # Freeze encoder if specified
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen. Training only classification layers.")
    else:
        print("Fine-tuning entire network.")

    # Training
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for data, labels in train_loader:
            data = data.view(data.size(0), -1).to(device)
            labels = labels.to(device)

            outputs = classifier(data)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total

        # Evaluation
        classifier.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.view(data.size(0), -1).to(device)
                labels = labels.to(device)
                outputs = classifier(data)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * test_correct / test_total

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    return classifier

# Train classifier with pretrained encoder
classifier = pretrain_with_autoencoder(
    model, train_loader, test_loader,
    num_classes=10, freeze_encoder=True, device=config['device']
)
```

---

## 8. Training Best Practices

### 8.1 Learning Rate Selection

```python
# Learning rate strategies
strategies = {
    'constant': {
        'optimizer': optim.Adam(model.parameters(), lr=1e-3),
        'scheduler': None
    },
    'step_decay': {
        'optimizer': optim.Adam(model.parameters(), lr=1e-3),
        'scheduler': optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )
    },
    'exponential_decay': {
        'optimizer': optim.Adam(model.parameters(), lr=1e-3),
        'scheduler': optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95
        )
    },
    'cosine_annealing': {
        'optimizer': optim.Adam(model.parameters(), lr=1e-3),
        'scheduler': optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
    },
    'reduce_on_plateau': {
        'optimizer': optim.Adam(model.parameters(), lr=1e-3),
        'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    }
}

# Recommended: Cosine annealing or ReduceLROnPlateau
# Initial LR: 1e-3 for Adam, 1e-2 for SGD
# Weight decay: 1e-5 to 1e-4 for regularization
```

### 8.2 Bottleneck Size Selection

```python
def evaluate_bottleneck_sizes(
    input_dim, bottleneck_sizes, train_loader, test_loader,
    num_epochs=30, device='cuda'
):
    """
    Evaluate different bottleneck sizes.

    Args:
        input_dim: Input dimension
        bottleneck_sizes: List of latent dimensions to try
        train_loader: Training data
        test_loader: Test data
        num_epochs: Training epochs
        device: cuda or cpu

    Returns:
        Dictionary of results
    """
    results = {}

    for latent_dim in bottleneck_sizes:
        print(f"\nTraining with latent_dim={latent_dim}...")

        # Create model
        model = VanillaAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[512, 256]
        ).to(device)

        # Train
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train_losses = []
        test_losses = []

        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for data, _ in train_loader:
                data = data.view(data.size(0), -1).to(device)
                x_hat, z = model(data)
                loss = F.binary_cross_entropy(x_hat, data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Testing
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.view(data.size(0), -1).to(device)
                    x_hat, z = model(data)
                    loss = F.binary_cross_entropy(x_hat, data)
                    test_loss += loss.item()
            test_loss /= len(test_loader)
            test_losses.append(test_loss)

        results[latent_dim] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train': train_losses[-1],
            'final_test': test_losses[-1],
            'compression_ratio': input_dim / latent_dim
        }

        print(f"  Final test loss: {test_losses[-1]:.6f}")
        print(f"  Compression ratio: {input_dim / latent_dim:.1f}x")

    # Plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for latent_dim, res in results.items():
        plt.plot(res['test_losses'], label=f'latent_dim={latent_dim}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Bottleneck Size')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    latent_dims = list(results.keys())
    final_losses = [results[ld]['final_test'] for ld in latent_dims]
    plt.plot(latent_dims, final_losses, marker='o', linewidth=2)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Final Test Loss')
    plt.title('Reconstruction Quality vs Bottleneck Size')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bottleneck_comparison.png', dpi=150)
    plt.show()

    return results

# Example: Evaluate bottleneck sizes
bottleneck_sizes = [8, 16, 32, 64, 128, 256]
results = evaluate_bottleneck_sizes(
    input_dim=784,
    bottleneck_sizes=bottleneck_sizes,
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=30,
    device=config['device']
)
```

### 8.3 Loss Function Selection

```python
def compare_loss_functions(model, data_loader, device):
    """
    Compare different reconstruction loss functions.

    Args:
        model: Autoencoder model
        data_loader: Data loader
        device: cuda or cpu

    Returns:
        Dictionary of losses
    """
    model.eval()
    losses = {
        'MSE': 0.0,
        'BCE': 0.0,
        'L1': 0.0,
        'Huber': 0.0
    }

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(data.size(0), -1).to(device)
            x_hat, z = model(data)

            # Different loss functions
            losses['MSE'] += F.mse_loss(x_hat, data).item()
            losses['BCE'] += F.binary_cross_entropy(x_hat, data).item()
            losses['L1'] += F.l1_loss(x_hat, data).item()
            losses['Huber'] += F.smooth_l1_loss(x_hat, data).item()

    # Average
    for key in losses:
        losses[key] /= len(data_loader)

    return losses

# Loss function recommendations:
# - Binary Cross-Entropy (BCE): Image data in [0,1], pixel independence
# - MSE: Continuous data, Gaussian noise assumption
# - L1: Robust to outliers, sparse reconstructions
# - Huber: Compromise between MSE and L1

# Custom perceptual loss (advanced)
from torchvision.models import vgg16

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG features.
    Better for image quality than MSE.
    """
    def __init__(self, device='cuda'):
        super(PerceptualLoss, self).__init__()

        # Load pretrained VGG16
        vgg = vgg16(pretrained=True).features[:16].to(device).eval()

        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg

    def forward(self, x, x_hat):
        """
        Compute perceptual loss.

        Args:
            x: Original image (B, C, H, W)
            x_hat: Reconstructed image (B, C, H, W)

        Returns:
            Perceptual loss
        """
        # Extract features
        features_x = self.vgg(x)
        features_x_hat = self.vgg(x_hat)

        # L2 distance in feature space
        loss = F.mse_loss(features_x_hat, features_x)

        return loss
```

### 8.4 Batch Normalization and Regularization

```python
class AutoencoderWithBatchNorm(nn.Module):
    """
    Autoencoder with batch normalization and dropout.

    Best practices:
    - BatchNorm after linear/conv, before activation
    - Dropout in decoder only (not encoder)
    - Higher dropout (0.3-0.5) for small datasets
    """
    def __init__(self, input_dim=784, latent_dim=64):
        super(AutoencoderWithBatchNorm, self).__init__()

        # Encoder with BatchNorm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, latent_dim)
        )

        # Decoder with BatchNorm and Dropout
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# Weight initialization
def init_weights(m):
    """
    Initialize weights using Xavier/Kaiming initialization.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Apply initialization
model = AutoencoderWithBatchNorm()
model.apply(init_weights)
```

### 8.5 Skip Connections and Residual Learning

```python
class ResidualBlock(nn.Module):
    """Residual block for autoencoder."""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class ResidualAutoencoder(nn.Module):
    """Autoencoder with residual connections for deeper networks."""
    def __init__(self, input_dim=784, latent_dim=64):
        super(ResidualAutoencoder, self).__init__()

        # Encoder
        self.enc_input = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.enc_res1 = ResidualBlock(512)
        self.enc_down1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.enc_res2 = ResidualBlock(256)
        self.enc_down2 = nn.Sequential(
            nn.Linear(256, latent_dim)
        )

        # Decoder
        self.dec_up1 = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.dec_res1 = ResidualBlock(256)
        self.dec_up2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.dec_res2 = ResidualBlock(512)
        self.dec_output = nn.Sequential(
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.enc_input(x)
        x = self.enc_res1(x)
        x = self.enc_down1(x)
        x = self.enc_res2(x)
        z = self.enc_down2(x)
        return z

    def decode(self, z):
        x = self.dec_up1(z)
        x = self.dec_res1(x)
        x = self.dec_up2(x)
        x = self.dec_res2(x)
        x_hat = self.dec_output(x)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
```

---

## 9. Comparison with PCA

### 9.1 Theoretical Differences

**Principal Component Analysis (PCA):**
- **Linear** transformation: z = W^T * x
- Finds orthogonal directions of maximum variance
- Closed-form solution via SVD
- Optimal for Gaussian data
- Fast, no hyperparameters

**Autoencoder:**
- **Nonlinear** transformation via neural networks
- Learns complex manifolds
- Optimized via gradient descent
- Better for non-Gaussian, complex data
- Requires hyperparameter tuning

### 9.2 Empirical Comparison

```python
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import time

def compare_pca_autoencoder(
    train_loader, test_loader,
    latent_dim=32, device='cuda'
):
    """
    Compare PCA and autoencoder for dimensionality reduction.

    Args:
        train_loader: Training data
        test_loader: Test data
        latent_dim: Number of components/latent dimensions
        device: cuda or cpu

    Returns:
        Comparison results
    """
    # Prepare data
    X_train = []
    X_test = []

    for data, _ in train_loader:
        X_train.append(data.view(data.size(0), -1).numpy())
    for data, _ in test_loader:
        X_test.append(data.view(data.size(0), -1).numpy())

    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ===== PCA =====
    print("\n=== PCA ===")
    start_time = time.time()

    pca = PCA(n_components=latent_dim, random_state=42)
    Z_train_pca = pca.fit_transform(X_train)
    X_recon_pca = pca.inverse_transform(Z_train_pca)

    Z_test_pca = pca.transform(X_test)
    X_test_recon_pca = pca.inverse_transform(Z_test_pca)

    pca_time = time.time() - start_time

    pca_train_error = mean_squared_error(X_train, X_recon_pca)
    pca_test_error = mean_squared_error(X_test, X_test_recon_pca)

    print(f"Training time: {pca_time:.2f}s")
    print(f"Train MSE: {pca_train_error:.6f}")
    print(f"Test MSE: {pca_test_error:.6f}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

    # ===== Autoencoder =====
    print("\n=== Autoencoder ===")
    start_time = time.time()

    # Train autoencoder
    ae_model = VanillaAutoencoder(
        input_dim=X_train.shape[1],
        latent_dim=latent_dim,
        hidden_dims=[512, 256]
    ).to(device)

    optimizer = optim.Adam(ae_model.parameters(), lr=1e-3)

    num_epochs = 30
    for epoch in range(num_epochs):
        ae_model.train()
        for data, _ in train_loader:
            data = data.view(data.size(0), -1).to(device)
            x_hat, z = ae_model(data)
            loss = F.mse_loss(x_hat, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ae_time = time.time() - start_time

    # Evaluate
    ae_model.eval()
    with torch.no_grad():
        X_train_torch = torch.FloatTensor(X_train).to(device)
        X_test_torch = torch.FloatTensor(X_test).to(device)

        X_recon_ae, _ = ae_model(X_train_torch)
        X_test_recon_ae, _ = ae_model(X_test_torch)

        ae_train_error = F.mse_loss(X_recon_ae, X_train_torch).item()
        ae_test_error = F.mse_loss(X_test_recon_ae, X_test_torch).item()

    print(f"Training time: {ae_time:.2f}s")
    print(f"Train MSE: {ae_train_error:.6f}")
    print(f"Test MSE: {ae_test_error:.6f}")

    # ===== Comparison =====
    print("\n=== Comparison ===")
    print(f"PCA vs AE test error: {pca_test_error:.6f} vs {ae_test_error:.6f}")
    print(f"AE improvement: {(pca_test_error - ae_test_error) / pca_test_error * 100:.2f}%")
    print(f"Speed ratio (PCA/AE): {ae_time / pca_time:.1f}x slower")

    # Visualize reconstructions
    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    indices = np.random.choice(len(X_test), 10, replace=False)

    for i, idx in enumerate(indices):
        # Original
        axes[0, i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)

        # PCA reconstruction
        axes[1, i].imshow(X_test_recon_pca[idx].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('PCA', fontsize=12)

        # AE reconstruction
        axes[2, i].imshow(X_test_recon_ae[idx].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Autoencoder', fontsize=12)

    plt.suptitle(f'PCA vs Autoencoder Reconstruction (latent_dim={latent_dim})')
    plt.tight_layout()
    plt.savefig('pca_vs_ae.png', dpi=150)
    plt.show()

    return {
        'pca': {'train_error': pca_train_error, 'test_error': pca_test_error, 'time': pca_time},
        'ae': {'train_error': ae_train_error, 'test_error': ae_test_error, 'time': ae_time}
    }

# Run comparison
results = compare_pca_autoencoder(
    train_loader, test_loader,
    latent_dim=32, device=config['device']
)
```

### 9.3 When to Use Each

**Use PCA when:**
- Data is approximately Gaussian
- Need fast, deterministic results
- Interpretability important (orthogonal components)
- Small to medium datasets
- Linear relationships dominate
- No need for reconstruction (just dimensionality reduction)

**Use Autoencoder when:**
- Data has complex, nonlinear structure
- Large datasets available for training
- Need better reconstruction quality
- Can afford training time
- Want to learn hierarchical features
- Planning to use for generation or anomaly detection

**Hybrid Approach:**
```python
# Use PCA for initial reduction, then autoencoder
pca = PCA(n_components=128)  # Reduce 784 --> 128
X_pca = pca.fit_transform(X_train)

# Train autoencoder on PCA features
ae = VanillaAutoencoder(input_dim=128, latent_dim=32)
# Train ae on X_pca...

# Benefits: Faster training, removes noise, better final compression
```

---

## 10. Resources and References

### 10.1 Seminal Papers

**Foundational Work:**
- Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the Dimensionality of Data with Neural Networks". *Science*, 313(5786), 504-507.
  - First demonstration of deep autoencoders outperforming PCA
  - Layer-wise pretraining strategy

**Denoising Autoencoders:**
- Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). "Extracting and Composing Robust Features with Denoising Autoencoders". *ICML*.
  - Introduced corruption process for robust feature learning
  - Theoretical connection to score matching

- Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., & Manzagol, P. A. (2010). "Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion". *JMLR*, 11, 3371-3408.
  - Stacking denoising autoencoders for deep architectures

**Sparse Autoencoders:**
- Ng, A. (2011). "Sparse Autoencoder". *CS294A Lecture Notes*, Stanford University.
  - KL divergence sparsity penalty
  - Visualization of learned features

**Contractive Autoencoders:**
- Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011). "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction". *ICML*.
  - Jacobian-based regularization
  - Connection to manifold learning

**Variational Autoencoders (next step):**
- Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes". *arXiv preprint arXiv:1312.6114*.
  - Probabilistic framework for autoencoders
  - Enables sampling and generation

### 10.2 Key Hyperparameters Summary

```python
hyperparameter_guide = {
    'architecture': {
        'latent_dim': {
            'typical_range': '2-512',
            'MNIST': 32,
            'CIFAR10': 128,
            'ImageNet': 512,
            'rule_of_thumb': 'input_dim / 10 to input_dim / 50'
        },
        'hidden_dims': {
            'pattern': 'Gradual reduction: [512, 256, 128] or [1024, 512, 256]',
            'depth': '2-5 hidden layers (not counting latent)',
            'symmetry': 'Decoder typically mirrors encoder'
        },
        'activation': {
            'hidden': 'ReLU (standard), LeakyReLU (avoid dead neurons)',
            'output': 'Sigmoid (images [0,1]), Tanh ([-1,1]), Linear (continuous)'
        }
    },
    'training': {
        'learning_rate': {
            'Adam': 1e-3,
            'SGD': 1e-2,
            'schedule': 'CosineAnnealing or ReduceLROnPlateau'
        },
        'batch_size': {
            'typical': 128,
            'range': '32-512 depending on GPU memory'
        },
        'epochs': {
            'vanilla_ae': 50,
            'denoising_ae': 100,
            'sparse_ae': 50,
            'contractive_ae': 50
        },
        'weight_decay': {
            'typical': 1e-5,
            'range': '1e-6 to 1e-4'
        }
    },
    'regularization': {
        'noise_factor': {
            'denoising': 0.3,
            'range': '0.1-0.5'
        },
        'sparsity_weight': {
            'l1': 1e-3,
            'kl': 1e-3,
            'range': '1e-4 to 1e-1'
        },
        'sparsity_target': {
            'typical': 0.05,
            'range': '0.01-0.1'
        },
        'contractive_lambda': {
            'typical': 1e-4,
            'range': '1e-6 to 1e-2'
        },
        'dropout': {
            'decoder': 0.2,
            'range': '0.1-0.5'
        }
    },
    'loss_functions': {
        'image_data_[0,1]': 'Binary Cross-Entropy',
        'image_data_normalized': 'MSE',
        'continuous_data': 'MSE or Huber',
        'robust_to_outliers': 'L1 or Huber'
    }
}
```

### 10.3 Common Pitfalls and Solutions

```python
troubleshooting_guide = {
    'problem': 'Model outputs all zeros or constant',
    'causes': [
        'Sigmoid activation with MSE loss',
        'Learning rate too high',
        'Vanishing gradients'
    ],
    'solutions': [
        'Use BCE loss for sigmoid output',
        'Reduce learning rate to 1e-4',
        'Add batch normalization',
        'Use LeakyReLU or ELU activations'
    ]
}

pitfall_examples = {
    'Blurry reconstructions': {
        'cause': 'MSE loss encourages averaging',
        'solution': 'Use perceptual loss, adversarial loss, or increase model capacity'
    },
    'Poor convergence': {
        'cause': 'Bad initialization, wrong learning rate',
        'solution': 'Use Kaiming/Xavier init, try different LR (1e-4, 1e-3, 1e-2)'
    },
    'Overfitting': {
        'cause': 'Bottleneck too large, no regularization',
        'solution': 'Reduce latent_dim, add weight decay, use dropout, add noise'
    },
    'Mode collapse': {
        'cause': 'Sparse autoencoder with too much sparsity',
        'solution': 'Reduce sparsity_weight, increase sparsity_target'
    },
    'Unstable training': {
        'cause': 'Contractive AE with high lambda_c',
        'solution': 'Reduce lambda_c to 1e-5 or 1e-6, clip gradients'
    }
}
```

### 10.4 Production Deployment Checklist

```python
deployment_checklist = """
1. Model Optimization
   - [ ] Convert to TorchScript for faster inference
   - [ ] Quantize weights (INT8) for deployment
   - [ ] Prune unnecessary neurons
   - [ ] Batch inputs for throughput

2. Monitoring
   - [ ] Track reconstruction error distribution
   - [ ] Set up anomaly detection thresholds
   - [ ] Monitor latent space drift
   - [ ] Log extreme reconstruction errors

3. Versioning
   - [ ] Save model architecture config
   - [ ] Track training hyperparameters
   - [ ] Version control training data
   - [ ] Document preprocessing steps

4. Inference Optimization
   - [ ] Use encoder only if just extracting features
   - [ ] Cache latent representations if possible
   - [ ] Use ONNX for cross-platform deployment
   - [ ] Consider model distillation for edge devices
"""

# Example: TorchScript conversion
def export_to_torchscript(model, input_shape, save_path):
    """Export trained autoencoder to TorchScript."""
    model.eval()
    example_input = torch.randn(1, *input_shape)

    # Trace model
    traced_model = torch.jit.trace(model, example_input)

    # Save
    traced_model.save(save_path)
    print(f"Model saved to {save_path}")

    # Load and test
    loaded_model = torch.jit.load(save_path)
    output = loaded_model(example_input)
    print(f"TorchScript model output shape: {output[0].shape}")

# Usage
export_to_torchscript(model, (784,), 'autoencoder_traced.pt')
```

### 10.5 Advanced Topics and Extensions

**Beyond Basic Autoencoders:**

1. **Variational Autoencoders (VAE):** Probabilistic framework, enables generation
2. **Vector Quantized VAE (VQ-VAE):** Discrete latent space, high-quality generation
3. **Adversarial Autoencoders (AAE):** Use GAN discriminator on latent space
4. **Beta-VAE:** Disentangled representations via modified ELBO
5. **Sequence Autoencoders:** LSTM/Transformer encoder-decoder for sequences
6. **Graph Autoencoders:** For graph-structured data

**Modern Applications:**
- Self-supervised pretraining for vision transformers
- Latent diffusion models (Stable Diffusion uses VAE)
- Neural compression (better than JPEG at high compression)
- Drug discovery (molecular autoencoder)
- Recommender systems (collaborative filtering)

### 10.6 Code Repositories and Tools

```python
useful_libraries = {
    'PyTorch': 'https://pytorch.org - Main deep learning framework',
    'Lightning': 'https://lightning.ai - High-level PyTorch wrapper',
    'Hydra': 'https://hydra.cc - Config management for experiments',
    'Weights & Biases': 'https://wandb.ai - Experiment tracking',
    'TorchVision': 'Built-in datasets and transforms',
    'ONNX': 'https://onnx.ai - Model export for deployment'
}

example_repos = {
    'PyTorch Examples': 'https://github.com/pytorch/examples/tree/main/vae',
    'AE Zoo': 'https://github.com/julianstastny/VAE-ResNet18-PyTorch',
    'Anomaly Detection': 'https://github.com/lukasruff/Deep-SAD-PyTorch'
}
```

### 10.7 Further Reading

**Books:**
- "Deep Learning" by Goodfellow, Bengio, and Courville (Chapter 14: Autoencoders)
- "Dive into Deep Learning" by Zhang et al. (Online, free)

**Online Courses:**
- Stanford CS231n (Convolutional Neural Networks)
- Fast.ai Deep Learning Course

**Blogs and Tutorials:**
- Lil'Log: "From Autoencoder to Beta-VAE"
- Distill.pub: Interactive ML visualizations
- PyTorch tutorials: Official autoencoder examples

**Research Trends (2024-2025):**
- Masked autoencoders for vision (MAE, SimMIM)
- Denoising diffusion as autoencoder generalization
- Autoencoder-based foundation models
- Efficient autoencoders for edge deployment

---

**End of Autoencoders Guide**

This comprehensive guide covers vanilla autoencoders through advanced variants, providing production-ready implementations, best practices, and practical applications. For generative modeling, proceed to Variational Autoencoders (VAE) and Diffusion Models. For anomaly detection at scale, see dedicated chapters on outlier detection and monitoring.
