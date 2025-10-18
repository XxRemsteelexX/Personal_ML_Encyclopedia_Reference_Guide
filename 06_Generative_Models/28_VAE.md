# 28. Variational Autoencoders (VAEs)

## Overview

Variational Autoencoders (VAEs) are generative models that learn a probabilistic latent representation of data. Unlike standard autoencoders, VAEs can generate new samples by sampling from the learned latent distribution.

**Key Innovation:** Structured, continuous latent space that enables generation.

---

## 28.1 Motivation

### Problem with Standard Autoencoders

- Latent space not structured
- Random z → meaningless output
- Cannot generate new samples

### VAE Solution

- Encode to distribution (not single point)
- Sample from distribution during decoding
- Regularize latent space to be continuous and complete

---

## 28.2 Mathematical Framework

### Probabilistic Model

**Generative Process:**
```
1. Sample latent: z ~ p(z) = N(0, I)
2. Generate data: x ~ p(x|z)
```

**Goal:** Learn p(x|z) and approximate posterior p(z|x)

### Evidence Lower Bound (ELBO)

**Intractable:** log p(x) = ∫ p(x|z)p(z) dz

**Solution:** Variational inference with ELBO

```
log p(x) ≥ ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
```

Where:
- q(z|x) = encoder (approximate posterior)
- p(x|z) = decoder (likelihood)
- p(z) = N(0, I) (prior)

### Loss Function

```python
loss = reconstruction_loss + β * KL_divergence

# Reconstruction: how well we reconstruct
reconstruction_loss = -E_q[log p(x|z)]

# KL divergence: how close q(z|x) to p(z)
KL_divergence = KL(q(z|x) || N(0, I))
```

---

## 28.3 Implementation

### VAE Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        # Encoder: x -> μ, σ
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: z -> x
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # z = μ + σ * ε, where ε ~ N(0,1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# Loss function
def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_div

# Training
model = VAE(input_dim=784, latent_dim=20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    for batch in dataloader:
        x = batch.view(batch.size(0), -1)
        
        x_recon, mu, logvar = model(x)
        loss = vae_loss(x_recon, x, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Reparameterization Trick

**Problem:** Cannot backprop through sampling

**Solution:** Reparameterize
```python
# Instead of: z ~ N(μ, σ²)
# Use: z = μ + σ * ε, where ε ~ N(0, 1)

# This allows gradients to flow through μ and σ
```

---

## 28.4 Convolutional VAE

```python
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28->14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 14->7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 64 * 7 * 7)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 64, 7, 7)
        return self.decoder(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

---

## 28.5 β-VAE

**Problem:** Standard VAE produces blurry images

**Solution:** Weight KL term differently

```python
loss = reconstruction_loss + β * KL_divergence

# β > 1: More disentangled representations (but blurrier)
# β < 1: Better reconstructions (but entangled)
# β = 1: Standard VAE
```

**β-VAE encourages disentangled representations:**
- Each latent dimension captures independent factor of variation
- e.g., z[0] = rotation, z[1] = color, z[2] = size

---

## 28.6 Generation and Sampling

### Generate New Samples

```python
model.eval()
with torch.no_grad():
    # Sample from prior
    z = torch.randn(num_samples, latent_dim)
    
    # Decode
    generated = model.decode(z)
    
    # Visualize
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i].view(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()
```

### Latent Space Interpolation

```python
# Encode two images
mu1, _ = model.encode(img1)
mu2, _ = model.encode(img2)

# Interpolate in latent space
alphas = torch.linspace(0, 1, steps=10)
interpolated = []

for alpha in alphas:
    z = alpha * mu1 + (1 - alpha) * mu2
    decoded = model.decode(z)
    interpolated.append(decoded)

# Visualize smooth transition
```

### Latent Space Arithmetic

```python
# "King - Man + Woman = Queen" (in latent space)
z_king = model.encode(king_image)[0]
z_man = model.encode(man_image)[0]
z_woman = model.encode(woman_image)[0]

z_result = z_king - z_man + z_woman
queen_like = model.decode(z_result)
```

---

## 28.7 Applications

### 1. Anomaly Detection

```python
def detect_anomaly(x, model, threshold):
    x_recon, mu, logvar = model(x)
    recon_error = F.mse_loss(x_recon, x, reduction='none').sum(dim=1)
    
    # High reconstruction error = anomaly
    is_anomaly = recon_error > threshold
    return is_anomaly, recon_error
```

### 2. Data Augmentation

```python
# Sample from learned distribution
for img, label in dataset:
    mu, logvar = model.encode(img)
    
    # Sample multiple versions
    for _ in range(5):
        z = model.reparameterize(mu, logvar)
        augmented = model.decode(z)
        augmented_dataset.append((augmented, label))
```

### 3. Denoising

```python
# Encode noisy image to clean latent space
noisy_img = add_noise(img)
mu, _ = model.encode(noisy_img)  # Use mean (no sampling)
denoised = model.decode(mu)
```

---

## 28.8 Conditional VAE (CVAE)

**Goal:** Control generation with class labels

```python
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super().__init__()
        
        # Encoder: x + y -> z
        self.encoder = nn.Linear(input_dim + num_classes, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: z + y -> x
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, y):
        xy = torch.cat([x, y], dim=1)
        h = F.relu(self.encoder(xy))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        return self.decoder(zy)
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# Generate specific class
y = torch.zeros(1, num_classes)
y[0, 3] = 1  # Class 3 (one-hot)
z = torch.randn(1, latent_dim)
generated = model.decode(z, y)  # Generate class 3 sample
```

---

## 28.9 Comparison: VAE vs Others

| Model | Pros | Cons |
|-------|------|------|
| **VAE** | Stable training, principled framework, structured latent space | Blurry outputs, lower sample quality |
| **GAN** | Sharp, high-quality samples | Unstable training, mode collapse |
| **Autoencoder** | Fast, simple | Cannot generate new samples |
| **Diffusion** | Highest quality (2025) | Slow sampling |

**2025 Usage:**
- VAEs: Latent space for diffusion models
- GANs: Upscaling, style transfer
- Diffusion: Main generative approach

---

## 28.10 Advanced Variants

### VQ-VAE (Vector Quantized)

- Discrete latent space (codebook)
- Used in DALL-E

### Hierarchical VAE

- Multiple levels of latent variables
- Better for complex data

### Normalizing Flow VAE

- Exact likelihood computation
- More expressive posteriors

---

## Resources

- "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (Higgins et al., 2017)
- Tutorial: https://arxiv.org/abs/1606.05908
