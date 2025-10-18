# 29. Generative Adversarial Networks (GANs)

## Overview

GANs pit two neural networks against each other: a Generator creates fake samples, and a Discriminator tries to distinguish real from fake. This adversarial training produces high-quality synthetic data.

**Key Players:**
- **Generator (G):** Creates fake samples from noise
- **Discriminator (D):** Classifies real vs fake

**2025 Status:** Still used for upscaling and style transfer, but diffusion models dominate image generation.

---

## 29.1 Core Concept

### Minimax Game

```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

**Discriminator:** Maximize ability to classify real vs fake
**Generator:** Minimize discriminator's ability (fool it)

### Training Process

1. Train D: Classify real (label=1) and fake (label=0)
2. Train G: Make D classify fake as real
3. Repeat until equilibrium

---

## 29.2 Implementation

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability of being real
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# Training
G = Generator()
D = Discriminator()

optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

for epoch in range(epochs):
    for real_imgs in dataloader:
        batch_size = real_imgs.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # ============ Train Discriminator ============
        optimizer_D.zero_grad()
        
        # Real images
        real_loss = criterion(D(real_imgs), real_labels)
        
        # Fake images
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = G(z)
        fake_loss = criterion(D(fake_imgs.detach()), fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # ============ Train Generator ============
        optimizer_G.zero_grad()
        
        # Generate and try to fool discriminator
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = G(z)
        g_loss = criterion(D(fake_imgs), real_labels)  # Want D to think it's real
        
        g_loss.backward()
        optimizer_G.step()
```

---

## 29.3 DCGAN (Deep Convolutional GAN)

**Architecture guidelines for stable training:**

```python
class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State: 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: 3 x 64 x 64
        )
    
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)

class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img).view(-1, 1)
```

**DCGAN Guidelines:**
- Replace pooling with strided convolutions
- Use BatchNorm in both G and D
- Remove fully connected hidden layers
- Use ReLU in G (except output: Tanh)
- Use LeakyReLU in D

---

## 29.4 Conditional GAN (cGAN)

**Control generation with class labels:**

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super().__init__()
        
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        label_input = self.label_embedding(labels)
        gen_input = torch.cat([z, label_input], dim=1)
        return self.model(gen_input).view(z.size(0), *img_shape)

# Generate specific digit
z = torch.randn(1, latent_dim)
label = torch.tensor([7])  # Generate digit 7
img = generator(z, label)
```

---

## 29.5 Advanced GAN Variants

### Wasserstein GAN (WGAN)

**Problem:** Vanishing gradients, mode collapse

**Solution:** Wasserstein distance instead of JS divergence

```python
# Loss function
d_loss = -torch.mean(D(real)) + torch.mean(D(fake))
g_loss = -torch.mean(D(G(z)))

# Weight clipping for Lipschitz constraint
for p in D.parameters():
    p.data.clamp_(-0.01, 0.01)
```

### WGAN-GP (Gradient Penalty)

Better than weight clipping:

```python
def gradient_penalty(D, real, fake):
    alpha = torch.rand(real.size(0), 1, 1, 1)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True
    )[0]
    
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# Training
d_loss = -torch.mean(D(real)) + torch.mean(D(fake)) + lambda_gp * gradient_penalty(D, real, fake)
```

### StyleGAN / StyleGAN2

**NVIDIA's high-quality image generation:**
- Style-based generator
- Progressive growing
- Mapping network for latent codes

**Key Features:**
- 1024×1024 resolution
- Fine control over styles
- State-of-art 2019-2020

### CycleGAN

**Unpaired image-to-image translation:**

```python
# Two generators: G (X->Y), F (Y->X)
# Two discriminators: D_X, D_Y

# Adversarial losses
loss_GAN_XY = criterion(D_Y(G(X)), real_labels)
loss_GAN_YX = criterion(D_X(F(Y)), real_labels)

# Cycle consistency losses
loss_cycle_X = L1(F(G(X)), X)  # X -> Y -> X
loss_cycle_Y = L1(G(F(Y)), Y)  # Y -> X -> Y

# Total generator loss
g_loss = loss_GAN_XY + loss_GAN_YX + lambda_cycle * (loss_cycle_X + loss_cycle_Y)
```

**Applications:**
- Photo → Painting (Monet, Van Gogh)
- Horse → Zebra
- Summer → Winter

---

## 29.6 Training Challenges

### Mode Collapse

**Problem:** Generator produces limited variety

**Solutions:**
- Minibatch discrimination
- Unrolled GANs
- Multiple discriminators

### Vanishing Gradients

**Problem:** Discriminator too good → no gradient for generator

**Solutions:**
- Wasserstein GAN
- Feature matching
- Label smoothing

### Unstable Training

**Tips:**
- Use Adam optimizer (β1=0.5)
- Learning rate: 0.0001 - 0.0002
- Batch normalization
- LeakyReLU (α=0.2)
- Avoid sparse gradients (ReLU, MaxPool)

---

## 29.7 Evaluation Metrics

### Inception Score (IS)

```python
def inception_score(images, num_splits=10):
    # Use pre-trained Inception v3
    preds = inception_model(images)
    
    # Split predictions
    split_scores = []
    for k in range(num_splits):
        part = preds[k * (N // num_splits): (k + 1) * (N // num_splits)]
        py = part.mean(axis=0)
        scores = part * (np.log(part) - np.log(py))
        split_scores.append(np.exp(scores.sum(axis=1).mean()))
    
    return np.mean(split_scores), np.std(split_scores)
```

**Higher is better** (measures quality and diversity)

### Fréchet Inception Distance (FID)

```python
def calculate_fid(real_images, generated_images):
    # Extract features from Inception
    real_features = inception_model.get_features(real_images)
    gen_features = inception_model.get_features(generated_images)
    
    # Calculate statistics
    mu_real, sigma_real = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = gen_features.mean(0), np.cov(gen_features, rowvar=False)
    
    # FID
    diff = mu_real - mu_gen
    covmean = scipy.linalg.sqrtm(sigma_real @ sigma_gen)
    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2*covmean)
    
    return fid
```

**Lower is better** (measures similarity to real distribution)

---

## 29.8 Applications (2025)

### 1. Image Super-Resolution

```python
# SRGAN, ESRGAN
# Low-res → High-res with perceptual loss
```

### 2. Style Transfer

```python
# Neural style transfer
# Content image + Style image → Stylized output
```

### 3. Data Augmentation

```python
# Generate synthetic training samples
for _ in range(1000):
    z = torch.randn(1, latent_dim)
    synthetic_img = G(z)
    augmented_dataset.append(synthetic_img)
```

### 4. Image Editing

```python
# Semantic image editing
# Modify specific attributes (age, hair, expression)
```

### 5. Deepfakes

**Ethical concerns:**
- Face swapping
- Voice cloning
- Requires regulation

---

## 29.9 2025 Status

**Where GANs Stand:**
- **Replaced by Diffusion** for most image generation
- **Still used for:**
  - Real-time applications (faster than diffusion)
  - Upscaling (ESRGAN)
  - Style transfer
  - Specific domains (faces with StyleGAN)

**Why Diffusion Won:**
- More stable training
- Better mode coverage
- Higher quality samples
- Easier to control

---

## Resources

- "Generative Adversarial Networks" (Goodfellow et al., 2014)
- "Unsupervised Representation Learning with DCGAN" (Radford et al., 2015)
- "Improved Training of Wasserstein GANs" (Gulrajani et al., 2017)
- "A Style-Based Generator Architecture" (Karras et al., 2019) - StyleGAN
