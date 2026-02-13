# 30. Diffusion Models

## Overview

Diffusion models are the dominant generative AI approach in 2025. They gradually denoise random noise into high-quality images through iterative refinement. Models like Stable Diffusion, DALL-E, and Midjourney are all based on diffusion.

**Core Idea:** Learn to reverse a gradual noising process.

---

## 30.1 How Diffusion Works

### Forward Process (Diffusion)

Gradually add Gaussian noise over T steps:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) x_{t-1}, beta_t I)

x_t = sqrt(1-beta_t) x_{t-1} + sqrtbeta_t epsilon,  where epsilon ~ N(0, I)
```

After T steps: x_T ~= pure noise

### Reverse Process (Denoising)

Learn to reverse the process:

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sum_theta(x_t, t))
```

**Key insight:** Predict noise at each step, remove it

---

## 30.2 Denoising Diffusion Probabilistic Models (DDPM)

### Training Objective

```python
L = E_t,x_0,epsilon [||epsilon - epsilon_theta(x_t, t)||^2]
```

Where:
- epsilon = actual noise added
- epsilon_theta = predicted noise by model
- x_t = noisy image at step t

### Implementation

```python
import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, model, T=1000):
        super().__init__()
        self.model = model  # UNet typically
        self.T = T
        
        # Noise schedule
        self.beta = torch.linspace(1e-4, 0.02, T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, x_0, t):
        # x_t = sqrtalpha_bar_t x_0 + sqrt(1-alpha_bar_t) epsilon
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise
    
    def forward(self, x_0):
        # Sample random timestep
        t = torch.randint(0, self.T, (x_0.shape[0],))
        
        # Add noise
        x_t, noise = self.add_noise(x_0, t)
        
        # Predict noise
        noise_pred = self.model(x_t, t)
        
        # Loss
        loss = nn.MSELoss()(noise_pred, noise)
        return loss

# Training
diffusion = DiffusionModel(unet_model, T=1000)
optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

for epoch in range(epochs):
    for images in dataloader:
        loss = diffusion(images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Sampling (Generation)

```python
@torch.no_grad()
def sample(model, shape, T=1000):
    # Start with pure noise
    x = torch.randn(shape)
    
    for t in reversed(range(T)):
        # Predict noise
        t_tensor = torch.full((shape[0],), t, dtype=torch.long)
        noise_pred = model(x, t_tensor)
        
        # Remove predicted noise
        alpha_t = model.alpha[t]
        alpha_bar_t = model.alpha_bar[t]
        beta_t = model.beta[t]
        
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = 0
        
        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        ) + torch.sqrt(beta_t) * z
    
    return x

# Generate images
images = sample(diffusion.model, shape=(16, 3, 64, 64))
```

---

## 30.3 UNet Architecture

**Standard backbone for diffusion models:**

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Downsampling
        self.down1 = DownBlock(in_channels, 64, time_emb_dim)
        self.down2 = DownBlock(64, 128, time_emb_dim)
        self.down3 = DownBlock(128, 256, time_emb_dim)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(512, 256, 3, padding=1)
        )
        
        # Upsampling
        self.up1 = UpBlock(256 + 256, 128, time_emb_dim)  # +skip connection
        self.up2 = UpBlock(128 + 128, 64, time_emb_dim)
        self.up3 = UpBlock(64 + 64, out_channels, time_emb_dim)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        
        # Downsample
        d1 = self.down1(x, t_emb)
        d2 = self.down2(d1, t_emb)
        d3 = self.down3(d2, t_emb)
        
        # Bottleneck
        b = self.bottleneck(d3)
        
        # Upsample with skip connections
        u1 = self.up1(torch.cat([b, d3], dim=1), t_emb)
        u2 = self.up2(torch.cat([u1, d2], dim=1), t_emb)
        u3 = self.up3(torch.cat([u2, d1], dim=1), t_emb)
        
        return u3
```

---

## 30.4 Latent Diffusion Models (LDM)

**Problem:** Diffusion in pixel space is slow

**Solution:** Diffusion in latent space (Stable Diffusion approach)

```python
# Architecture
VAE Encoder: image --> latent z
Diffusion: denoise z
VAE Decoder: z --> image

# Advantages:
# - 4-8x faster
# - Lower memory
# - Better quality
```

---

## 30.5 Text-to-Image (Stable Diffusion)

### Conditioning with Text

```python
class ConditionalDiffusion(nn.Module):
    def __init__(self, unet, text_encoder):
        super().__init__()
        self.unet = unet
        self.text_encoder = text_encoder  # CLIP or T5
    
    def forward(self, images, text):
        # Encode text
        text_emb = self.text_encoder(text)
        
        # Add noise to images
        t = torch.randint(0, self.T, (images.shape[0],))
        x_t, noise = self.add_noise(images, t)
        
        # Predict noise conditioned on text
        noise_pred = self.unet(x_t, t, context=text_emb)
        
        loss = nn.MSELoss()(noise_pred, noise)
        return loss

# Generation
@torch.no_grad()
def text_to_image(prompt, model):
    text_emb = model.text_encoder(prompt)
    
    x = torch.randn(1, 3, 64, 64)
    for t in reversed(range(model.T)):
        noise_pred = model.unet(x, t, context=text_emb)
        x = denoise_step(x, noise_pred, t)
    
    return x
```

### Classifier-Free Guidance

**Improves adherence to prompt:**

```python
# During training: randomly drop conditioning (10-20%)
if random.random() < 0.15:
    text_emb = null_embedding  # Unconditional

# During sampling: combine conditional and unconditional
noise_pred = (1 + w) * noise_pred_cond - w * noise_pred_uncond

# w > 0: stronger conditioning
# w = 0: no guidance
# w typical: 7.5
```

---

## 30.6 Stable Diffusion Pipeline

```python
from diffusers import StableDiffusionPipeline

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

# Generate
prompt = "A serene lake surrounded by mountains at sunset, photorealistic"
image = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

image.save("generated.png")
```

### ControlNet (Precise Control)

```python
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

# Load ControlNet (e.g., for edge maps)
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet
)

# Generate with edge guidance
image = pipe(
    prompt="A beautiful castle",
    image=canny_edge_map,  # Control condition
    num_inference_steps=20
).images[0]
```

---

## 30.7 Advanced Techniques (2025)

### SDXL (Stable Diffusion XL)

- Higher resolution (1024x1024)
- Better prompt understanding
- Dual text encoders (CLIP + OpenCLIP)

### Cascaded Diffusion

- Low-res --> High-res in stages
- Used in DALL-E 2, Imagen

### Video Diffusion

- Extend to temporal dimension
- AnimateDiff, Stable Video Diffusion

### 3D Diffusion

- DreamFusion, Point-E
- Text --> 3D models

---

## 30.8 Comparison: 2025 Landscape

| Model | Quality | Speed | Control | Use Case |
|-------|---------|-------|---------|----------|
| **Diffusion** | Excellent | Slow (50+ steps) | Good (ControlNet) | Main approach |
| **GAN** | Good | Fast (1 step) | Limited | Real-time, upscaling |
| **VAE** | Fair (blurry) | Fast | Good | Latent space |

**2025 Winners:**
- **Image Generation:** Diffusion (Stable Diffusion, DALL-E, Midjourney)
- **Video:** Diffusion (Sora, Pika, Runway)
- **3D:** Diffusion + NeRF

---

## 30.9 Applications

### 1. Text-to-Image

```python
prompt = "A futuristic city with flying cars, cyberpunk style"
image = generate(prompt)
```

### 2. Image-to-Image

```python
# Img2Img: modify existing image
strength = 0.75  # How much to change
image = img2img(init_image, prompt, strength)
```

### 3. Inpainting

```python
# Fill masked regions
image = inpaint(image, mask, prompt="a red apple")
```

### 4. Super-Resolution

```python
# Upscale 512x512 --> 2048x2048
high_res = super_resolve(low_res_image)
```

---

## 30.10 Key Innovations Timeline

**2020:** DDPM (Denoising Diffusion Probabilistic Models)
**2021:** DDIM (faster sampling), Classifier Guidance
**2022:** DALL-E 2, Imagen, Stable Diffusion, LDM
**2023:** SDXL, ControlNet, LCM (ultra-fast)
**2024:** Stable Diffusion 3, SDXL Turbo, Cascade models
**2025:** Multimodal diffusion (text+image+video+audio)

---

## Resources

- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "High-Resolution Image Synthesis with Latent Diffusion" (Rombach et al., 2022) - Stable Diffusion
- "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
- Hugging Face Diffusers: https://github.com/huggingface/diffusers
