# 27. Autoencoders

## Overview

Autoencoders are neural networks that learn to compress data into a lower-dimensional representation (encoding) and then reconstruct it (decoding). They are unsupervised learning models used for dimensionality reduction, feature learning, and generative modeling.

**Architecture:** Encoder → Latent Space → Decoder

---

## 27.1 Basic Autoencoder

### Architecture

```
Input (x) → Encoder → Latent Code (z) → Decoder → Reconstruction (x̂)
```

**Objective:** Minimize reconstruction error

```
L = ||x - x̂||² = ||x - decoder(encoder(x))||²
```

### Implementation

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # For normalized inputs
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

# Training
model = Autoencoder(input_dim=784, latent_dim=32)  # MNIST example
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(epochs):
    for batch in dataloader:
        x = batch.view(batch.size(0), -1)  # Flatten
        
        x_recon = model(x)
        loss = criterion(x_recon, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 27.2 Convolutional Autoencoder

For image data, use convolutional layers:

```python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 → 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 14x14 → 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=7),                     # 7x7 → 1x1
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=7),            # 1x1 → 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 → 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # 14x14 → 28x28
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
```

---

## 27.3 Denoising Autoencoder

**Goal:** Learn robust features by reconstructing clean data from corrupted input.

```python
def add_noise(x, noise_factor=0.3):
    noisy_x = x + noise_factor * torch.randn_like(x)
    return torch.clamp(noisy_x, 0., 1.)

# Training
for batch in dataloader:
    x_clean = batch
    x_noisy = add_noise(x_clean)
    
    x_recon = model(x_noisy)
    loss = criterion(x_recon, x_clean)  # Reconstruct CLEAN from NOISY
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Applications:**
- Image denoising
- Anomaly detection
- Feature learning

---

## 27.4 Sparse Autoencoder

**Goal:** Encourage sparsity in latent representation (most neurons inactive).

```python
def sparse_loss(z, sparsity_target=0.05, beta=1e-3):
    # L1 sparsity penalty
    sparsity_loss = torch.mean(torch.abs(z))
    
    # Or KL divergence sparsity
    # rho = torch.mean(torch.sigmoid(z), dim=0)  # Average activation
    # kl_loss = torch.sum(sparsity_target * torch.log(sparsity_target / rho))
    
    return beta * sparsity_loss

# Training with sparsity
loss = reconstruction_loss + sparse_loss(z)
```

---

## 27.5 Contractive Autoencoder

**Goal:** Make latent representation robust to small input changes.

```python
def contractive_loss(x, z, model, lambda_c=1e-4):
    # Frobenius norm of Jacobian
    z.backward(torch.ones_like(z), retain_graph=True)
    jacobian_norm = torch.sum(x.grad ** 2)
    x.grad.zero_()
    
    return lambda_c * jacobian_norm

# Total loss
loss = reconstruction_loss + contractive_loss(x, z, model)
```

---

## 27.6 Applications

### 1. Dimensionality Reduction

```python
# Encode high-dim data to low-dim
z = model.encode(x)  # 784-dim → 32-dim

# Use for visualization
from sklearn.manifold import TSNE
z_2d = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels)
```

### 2. Anomaly Detection

```python
# Reconstruction error for normal vs anomalous
def detect_anomaly(x, model, threshold):
    x_recon = model(x)
    recon_error = torch.mean((x - x_recon) ** 2, dim=1)
    
    is_anomaly = recon_error > threshold
    return is_anomaly, recon_error

# Normal data → low reconstruction error
# Anomalous data → high reconstruction error
```

### 3. Denoising

```python
noisy_image = add_noise(clean_image)
denoised_image = model(noisy_image)
```

### 4. Feature Learning

```python
# Pre-train encoder, use for downstream task
encoder = model.encoder

# Freeze encoder, train classifier
for param in encoder.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    encoder,
    nn.Linear(latent_dim, num_classes)
)
```

---

## 27.7 Limitations

**Cannot Sample New Data:**
- Latent space not structured
- Random z doesn't produce meaningful output
- → Solution: VAE (next section)

**Blurry Reconstructions:**
- MSE loss encourages averaging
- → Solution: Perceptual loss, GANs

**No Probabilistic Interpretation:**
- Deterministic encoding
- → Solution: VAE

---

## Resources

- "Reducing the Dimensionality of Data with Neural Networks" (Hinton & Salakhutdinov, 2006)
- "Extracting and Composing Robust Features with Denoising Autoencoders" (Vincent et al., 2008)
- "Contractive Auto-Encoders" (Rifai et al., 2011)
