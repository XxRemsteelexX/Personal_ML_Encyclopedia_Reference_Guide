# 11. GANs and Generative Models: Complete Guide

## Overview

Generative models create new data samples similar to training data. This guide covers when to use GANs, VAEs, Diffusion models, and other generative approaches for images, audio, and other data types.

**Models Covered:**
- Generative Adversarial Networks (GANs)
- Variational Autoencoders (VAEs)
- Diffusion Models (Stable Diffusion, DALL-E)
- Autoencoders
- Hybrid Approaches

---

## 11.1 Generative Models Evolution

### Timeline

```
2006: Autoencoders
  ↓
2013: Variational Autoencoders (VAEs)
  ↓
2014: GANs (Generative Adversarial Networks)
  ↓
2015-2020: GAN variants (StyleGAN, CycleGAN, Pix2Pix)
  ↓
2020: Diffusion Models emerge
  ↓
2022-2025: Diffusion dominates (Stable Diffusion, DALL-E, Midjourney)
```

### 2025 Landscape

**Current State:**
- **Diffusion Models = Standard** for image generation (Stable Diffusion, DALL-E, Midjourney)
- **GANs = Niche uses** (real-time generation, style transfer, upscaling)
- **VAEs = Preprocessing** for diffusion models and feature learning
- **Transformers = Text-to-X** generation (GPT, Claude, Gemini)

---

## 11.2 Quick Decision Guide

```
What do you want to generate?

├─ Images (high quality, text-to-image)?
│  └─ Use: Diffusion Models (Stable Diffusion, DALL-E)
│     Status: Best quality, industry standard (2025)
│
├─ Images (real-time, fast inference)?
│  └─ Use: GANs (StyleGAN, conditional GAN)
│     Status: Faster than diffusion
│
├─ Need diversity > quality?
│  └─ Use: VAE
│     Status: Broader coverage, less sharp
│
├─ Feature learning / anomaly detection?
│  └─ Use: VAE or Autoencoder
│     Status: Good for representation learning
│
├─ Style transfer / image-to-image?
│  └─ Use: CycleGAN or Diffusion
│     Status: Both work well
│
├─ Data augmentation (small dataset)?
│  └─ Use: GAN or simple augmentations
│     Status: GAN if need realism
│
├─ Video generation?
│  └─ Use: Diffusion-based (Runway, Sora)
│     Status: Rapidly evolving
│
└─ Text/audio generation?
   └─ Use: Transformers (GPT) or Diffusion
      Status: Transformers dominant for text
```

---

## 11.3 Model Comparison Table

| Model | Fidelity | Diversity | Training Difficulty | Speed | 2025 Status |
|-------|----------|-----------|---------------------|-------|-------------|
| **GAN** | ⭐⭐⭐⭐⭐ High | ⭐⭐⭐ Medium (mode collapse) | ⚠️⚠️⚠️ Very Hard | ⚡⚡⚡ Fast | Niche uses |
| **VAE** | ⭐⭐⭐ Medium (blurry) | ⭐⭐⭐⭐⭐ High | ✅✅✅ Easy | ⚡⚡⚡ Fast | Feature learning |
| **Diffusion** | ⭐⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ High | ✅✅ Moderate | ⚡ Slow | **Standard** |
| **Autoencoder** | ⭐⭐⭐ Medium | N/A (reconstruction) | ✅✅✅ Easy | ⚡⚡⚡ Fast | Compression only |

---

## 11.4 Generative Adversarial Networks (GANs)

### What They Are

GANs consist of two neural networks:
1. **Generator:** Creates fake data
2. **Discriminator:** Distinguishes real from fake

They compete: Generator tries to fool discriminator, discriminator tries to catch fakes.

### How They Work

```
Real Data ──┐
            ├──→ Discriminator ──→ Real or Fake?
Noise ──→ Generator ──→ Fake Data ──┘
            ↑
            └── Feedback to improve
```

**Training Loop:**
1. Generator creates fake samples from random noise
2. Discriminator sees real + fake, classifies each
3. Discriminator loss: Did it correctly identify real/fake?
4. Generator loss: Did it fool discriminator?
5. Both networks improve through backpropagation

### ✅ When to Use GANs

1. **Need highest quality images**
   - GANs produce sharp, realistic images
   - Better than VAE (less blurry)

2. **Fast inference required**
   - Generate image in one forward pass (~0.1s)
   - **vs Diffusion:** 50+ steps (~5-10s)
   - **Use case:** Real-time applications

3. **Image-to-image translation**
   - Pix2Pix: Sketch → photo
   - CycleGAN: Horse → zebra (unpaired)
   - Super-resolution (upscaling)

4. **Style transfer**
   - Change artistic style
   - Face editing (age, expression)
   - StyleGAN for face generation

5. **Data augmentation**
   - Generate synthetic training samples
   - When dataset is small (<10K images)

6. **Specific GAN applications**
   - Face generation (StyleGAN)
   - Medical imaging (data augmentation)
   - Video game textures
   - Art and creative applications

### ❌ When NOT to Use GANs

1. **No experience with GANs**
   - Training is notoriously difficult
   - Mode collapse, vanishing gradients
   - **Better:** Diffusion models (more stable)

2. **Need high diversity**
   - GANs suffer from mode collapse
   - May generate limited variety
   - **Better:** VAE or Diffusion

3. **Text-to-image generation**
   - Diffusion models are state-of-the-art
   - **Better:** Stable Diffusion, DALL-E, Midjourney

4. **Limited compute/time**
   - GANs require extensive tuning
   - **Better:** Use pretrained models or diffusion

5. **Need interpretability**
   - GAN latent space is complex
   - **Better:** VAE (more interpretable latent space)

6. **2025 general use**
   - Diffusion models have mostly replaced GANs
   - **Use GANs only for:** Speed or specific niches

---

## 11.5 GAN Variants and Use Cases

### Popular GAN Architectures

| GAN Type | Use Case | When to Use | Example |
|----------|----------|-------------|---------|
| **DCGAN** | Basic image generation | Learning GANs, simple tasks | Face generation |
| **StyleGAN** | High-quality faces | Face generation, editing | thispersondoesnotexist.com |
| **CycleGAN** | Unpaired image-to-image | Style transfer, domain adaptation | Horse ↔ Zebra |
| **Pix2Pix** | Paired image-to-image | Sketch to photo, colorization | Edges → Photo |
| **ProGAN** | High-resolution images | Need 1024x1024+ images | Large-scale image generation |
| **StyleGAN2** | State-of-art faces | Best quality faces | Portrait generation |
| **BigGAN** | ImageNet-scale generation | Large diverse datasets | Multi-class generation |
| **SRGAN** | Super-resolution | Upscaling images | Image enhancement |

### Quick Selection

```
What's your GAN task?

├─ Generate realistic faces?
│  └─ Use: StyleGAN2 or StyleGAN3
│
├─ Image-to-image (have paired data)?
│  └─ Use: Pix2Pix
│
├─ Image-to-image (NO paired data)?
│  └─ Use: CycleGAN
│
├─ Upscale low-res images?
│  └─ Use: SRGAN or ESRGAN
│
├─ Learning GANs / simple task?
│  └─ Use: DCGAN (vanilla)
│
└─ Multi-class image generation?
   └─ Use: BigGAN or conditional GAN
```

---

## 11.6 GAN Training: Problems and Solutions

### Problem 1: Mode Collapse

**What it is:** Generator produces limited variety (e.g., only generates blonde faces)

**Symptoms:**
- Low diversity in generated samples
- Generator ignores parts of data distribution
- Same outputs for different noise inputs

**Solutions:**
```python
# 1. Minibatch discrimination
# Add layer that looks at entire batch

# 2. Wasserstein loss (WGAN)
# More stable than vanilla GAN loss

# 3. Spectral normalization
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = spectral_norm(nn.Linear(100, 256))
        # Prevents discriminator from getting too strong

# 4. Unrolled GANs
# Generator considers future discriminator updates

# 5. Progressive training
# Start low-res (4x4), gradually increase to 1024x1024
```

---

### Problem 2: Training Instability

**Symptoms:**
- Loss oscillates wildly
- Generator/discriminator don't converge
- Gradients vanish or explode

**Solutions:**
```python
# 1. Use batch size ≤ 64
batch_size = 32  # Recommended

# 2. Label smoothing
# Real labels: 0.9-1.0 (not 1.0)
# Fake labels: 0.0-0.1 (not 0.0)
real_labels = torch.rand(batch_size) * 0.1 + 0.9  # [0.9, 1.0]
fake_labels = torch.rand(batch_size) * 0.1        # [0.0, 0.1]

# 3. Add noise to inputs
noise = torch.randn_like(real_images) * 0.1
noisy_real = real_images + noise

# 4. Two time-scale update rule (TTUR)
# Discriminator learns faster than generator
optimizer_G = Adam(generator.parameters(), lr=0.0001)
optimizer_D = Adam(discriminator.parameters(), lr=0.0004)  # 4x faster

# 5. Use WGAN-GP (Gradient Penalty)
# More stable loss function
```

---

### Problem 3: Discriminator Too Strong

**Symptoms:**
- Discriminator achieves 100% accuracy
- Generator loss stops improving
- Generator gradients vanish

**Solutions:**
```python
# 1. Train discriminator less often
for epoch in range(num_epochs):
    # Train discriminator every 5 iterations
    if iteration % 5 == 0:
        train_discriminator()

    # Train generator every iteration
    train_generator()

# 2. Add dropout to discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.3)  # Weaken discriminator

# 3. Lower discriminator learning rate
optimizer_D = Adam(discriminator.parameters(), lr=0.0001)  # Low

# 4. Use LeakyReLU (not ReLU)
self.activation = nn.LeakyReLU(0.2)  # Prevents dead neurons
```

---

### Best Practices Checklist

**Architecture:**
- [ ] Use LeakyReLU (0.2) in discriminator
- [ ] Use ReLU in generator
- [ ] Use batch normalization (NOT in first/last layers)
- [ ] Use spectral normalization in discriminator
- [ ] Use transposed convolutions (generator) and strided convs (discriminator)

**Training:**
- [ ] Batch size ≤ 64
- [ ] Label smoothing (0.9-1.0 for real)
- [ ] Add noise to inputs (0.1 std)
- [ ] Use Adam optimizer (β1=0.5, β2=0.999)
- [ ] Learning rate: 0.0002 (or TTUR: D=4x G)
- [ ] Train discriminator 1-5 times per generator update

**Loss Function:**
- [ ] Start with WGAN-GP (most stable)
- [ ] Alternative: Hinge loss
- [ ] Avoid vanilla GAN loss (unstable)

**Monitoring:**
- [ ] Save generated samples every N iterations
- [ ] Monitor discriminator accuracy (should be ~50-80%)
- [ ] Check for mode collapse visually
- [ ] Track both G and D losses

---

## 11.7 Variational Autoencoders (VAEs)

### What They Are

VAEs learn a probabilistic latent space representation of data. They encode data into a distribution (not a point), then decode from that distribution.

**Architecture:**
```
Input → Encoder → μ (mean), σ (std) → Sample z ~ N(μ, σ) → Decoder → Output
```

**Key difference from regular autoencoder:**
- Autoencoder: Input → Code (fixed point) → Output
- VAE: Input → Distribution → Sample → Output

### ✅ When to Use VAEs

1. **Need high diversity**
   - VAE generates more diverse samples than GAN
   - Less prone to mode collapse
   - Covers full data distribution

2. **Easier training required**
   - More stable than GANs
   - No adversarial training
   - Single loss function (reconstruction + KL divergence)

3. **Interpretable latent space**
   - Smooth interpolation between samples
   - Latent dimensions have meaning
   - Good for exploring data

4. **Anomaly detection**
   - High reconstruction error = anomaly
   - **Use case:** Manufacturing defect detection, fraud detection

5. **Feature learning**
   - Learn compressed representations
   - Use encoder for downstream tasks
   - Better than PCA for non-linear data

6. **Data compression**
   - Lossy compression
   - Better than JPEG for specific domains

7. **Generative modeling for tabular data**
   - Works better than GANs for tabular data
   - Less mode collapse issues

### ❌ When NOT to Use VAEs

1. **Need sharp, high-fidelity images**
   - VAE outputs are blurry
   - Reconstruction loss encourages "averaging"
   - **Better:** GAN or Diffusion

2. **Text-to-image generation**
   - VAEs not good at conditioning on text
   - **Better:** Diffusion models

3. **Just need compression**
   - Regular autoencoder is simpler
   - VAE adds complexity (KL loss)
   - **Better:** Standard autoencoder

4. **Real-time high-quality generation**
   - VAE quality lower than GAN
   - **Better:** GAN for speed + quality

---

## 11.8 VAE vs GAN vs Autoencoder

### Comparison

| Feature | Autoencoder | VAE | GAN |
|---------|-------------|-----|-----|
| **Purpose** | Compression, denoising | Generation, feature learning | Generation |
| **Output Quality** | Reconstruction | Blurry generation | Sharp generation |
| **Diversity** | N/A | High | Medium (mode collapse) |
| **Training** | Easy | Easy | Very Hard |
| **Latent Space** | Discrete | Continuous, probabilistic | Continuous |
| **Loss** | Reconstruction (MSE) | Reconstruction + KL divergence | Adversarial |
| **Inference Speed** | ⚡⚡⚡ Fast | ⚡⚡⚡ Fast | ⚡⚡⚡ Fast |

### When to Use Each

```
What's your goal?

├─ Just reconstruct/compress data?
│  └─ Use: Autoencoder
│
├─ Generate diverse samples (OK if blurry)?
│  └─ Use: VAE
│
├─ Generate high-quality images (OK if difficult)?
│  └─ Use: GAN (or better: Diffusion)
│
├─ Anomaly detection?
│  └─ Use: Autoencoder or VAE
│
├─ Learn features for downstream task?
│  └─ Use: VAE or Autoencoder
│
└─ Need interpretable latent space?
   └─ Use: VAE
```

---

## 11.9 Diffusion Models (2025 Standard)

### What They Are

Diffusion models learn to reverse a noise process:
1. **Forward process:** Gradually add noise to data until it's pure noise
2. **Reverse process:** Learn to denoise, recovering original data

**Example:**
```
Clear Image → Add noise → Noisy → More noise → Pure noise
              ← Denoise  ← Denoise ← Denoise   ← Start here
```

### Why Diffusion Won (2025)

**Advantages over GANs:**
- ✅ More stable training (no adversarial game)
- ✅ Higher diversity (no mode collapse)
- ✅ Better quality (DALL-E, Midjourney, Stable Diffusion)
- ✅ Easier to condition on text

**Disadvantages:**
- ❌ Slower inference (50-1000 denoising steps)
- ❌ More compute during generation

### ✅ When to Use Diffusion Models

1. **Text-to-image generation (BEST CHOICE)**
   - State-of-the-art: Stable Diffusion, DALL-E, Midjourney
   - Handles text conditioning naturally

2. **Need highest quality + diversity**
   - Best of both worlds (vs GAN/VAE)
   - Realistic, diverse, high-resolution

3. **Image editing**
   - Inpainting (fill in missing parts)
   - Outpainting (extend image)
   - Image-to-image with text guidance

4. **2025 production applications**
   - Industry standard
   - Extensive tooling and pretrained models

5. **Creative applications**
   - Art generation
   - Design prototyping
   - Concept art

### ❌ When NOT to Use Diffusion Models

1. **Need real-time generation**
   - 50+ denoising steps = slow (~5-10s)
   - **Better:** GAN (0.1s)

2. **Limited compute**
   - Requires GPU
   - Memory-intensive
   - **Better:** Smaller models or API

3. **Simple tasks**
   - Overkill for basic augmentation
   - **Better:** Traditional augmentations

4. **Tabular or sequential data**
   - Designed for images/audio
   - **Better:** VAE, GAN, or specialized models

---

## 11.10 Diffusion Model Variants

### Popular Models (2024-2025)

| Model | Provider | Best For | Cost |
|-------|----------|----------|------|
| **Stable Diffusion** | Stability AI | Open-source, self-host | Free |
| **DALL-E 3** | OpenAI | Highest quality text-to-image | $$$ API |
| **Midjourney** | Midjourney | Artistic images | $$ Subscription |
| **Imagen** | Google | Photorealistic images | Not public |
| **Runway** | Runway | Video generation | $$ Subscription |
| **Sora** | OpenAI | Long-form video | Not released |

### Stable Diffusion Versions

```python
# 2025 Recommendations:

# 1. SDXL (Stable Diffusion XL) - Best quality
model = "stabilityai/stable-diffusion-xl-base-1.0"
# 1024x1024, best quality, slower

# 2. SD 2.1 - Balanced
model = "stabilityai/stable-diffusion-2-1"
# 768x768, good quality, faster

# 3. SD 1.5 - Fast
model = "runwayml/stable-diffusion-v1-5"
# 512x512, fastest, lowest quality

# 4. Turbo models (2024+) - Real-time
model = "stabilityai/sdxl-turbo"
# 1-4 steps instead of 50! Much faster
```

---

## 11.11 Practical: Using Stable Diffusion

### Basic Text-to-Image

```python
from diffusers import StableDiffusionPipeline
import torch

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16  # Faster, less memory
)
pipe = pipe.to("cuda")

# Generate image
prompt = "A serene mountain landscape at sunset, oil painting style"
image = pipe(
    prompt,
    num_inference_steps=50,     # More steps = better quality
    guidance_scale=7.5,          # How closely to follow prompt (7-9 typical)
    negative_prompt="blurry, low quality, distorted"  # Avoid these
).images[0]

image.save("output.png")
```

### Key Parameters

```python
# num_inference_steps: 20-50 typical
# - More steps = better quality, slower
# - 20 = fast preview, 50 = high quality

# guidance_scale: 7-9 typical
# - Higher = follow prompt more closely
# - Too high (>15) = oversaturated, artifacts

# negative_prompt: What to avoid
negative_prompt = "blurry, low quality, distorted, ugly, malformed"

# Seed: Reproducibility
generator = torch.Generator("cuda").manual_seed(42)
image = pipe(prompt, generator=generator).images[0]
```

---

### Image-to-Image

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load initial image
init_image = Image.open("input.jpg").resize((1024, 1024))

# Transform image
prompt = "Turn this photo into a watercolor painting"
image = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75,  # 0.0 = keep original, 1.0 = completely new
    guidance_scale=7.5
).images[0]
```

---

### Inpainting (Fill in Missing Parts)

```python
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load image and mask
image = Image.open("photo.jpg")
mask = Image.open("mask.jpg")  # White = inpaint, Black = keep

# Inpaint
result = pipe(
    prompt="A red apple on the table",
    image=image,
    mask_image=mask
).images[0]
```

---

## 11.12 Hybrid Approaches

### VAE-GAN

**Concept:** Combine VAE's diversity with GAN's quality

**When to use:**
- Need both quality and diversity
- VAE alone too blurry
- GAN alone too unstable

**Architecture:**
```
Input → VAE Encoder → Latent → VAE Decoder → Output
                                      ↓
                               GAN Discriminator
                                      ↓
                            Adversarial loss
```

**Benefits:**
- VAE provides stable training
- GAN sharpens outputs
- Best of both worlds

---

### Diffusion + LLM (Current Trend)

**Example:** DALL-E 3 = GPT-4 (text understanding) + Diffusion (image generation)

**How it works:**
1. User prompt → GPT-4 refines prompt
2. Refined prompt → Diffusion model
3. Generate image

**Why it's powerful:**
- LLM understands complex prompts
- Diffusion generates high-quality images
- Better prompt adherence

---

## 11.13 Model Selection Decision Tree

```
Start: What are you generating?

├─ IMAGES
│  ├─ Text-to-image?
│  │  └─ Use: Stable Diffusion / DALL-E (Diffusion models)
│  │
│  ├─ Real-time generation (<1s)?
│  │  └─ Use: GAN (StyleGAN, conditional GAN)
│  │
│  ├─ Need diversity > quality?
│  │  └─ Use: VAE
│  │
│  ├─ Style transfer / image-to-image?
│  │  ├─ Have time: Diffusion
│  │  └─ Need speed: CycleGAN or Pix2Pix
│  │
│  └─ Upscaling?
│     └─ Use: SRGAN or Diffusion upscaler
│
├─ FEATURES / COMPRESSION
│  ├─ Just compress/reconstruct?
│  │  └─ Use: Autoencoder
│  │
│  ├─ Anomaly detection?
│  │  └─ Use: VAE or Autoencoder
│  │
│  └─ Learn features for downstream task?
│     └─ Use: VAE
│
├─ TEXT
│  └─ Use: Transformer (GPT, Claude, etc.) - NOT covered here
│
├─ AUDIO
│  ├─ Music generation?
│  │  └─ Use: Diffusion (Riffusion) or Transformer (MusicLM)
│  │
│  └─ Speech synthesis?
│     └─ Use: Diffusion (WaveGrad) or GAN (WaveGAN)
│
└─ VIDEO
   └─ Use: Diffusion (Runway, Sora) - Rapidly evolving
```

---

## 11.14 Common Pitfalls & Solutions

### Pitfall 1: Training GANs Like Regular NNs

❌ **Wrong:**
```python
# Same hyperparameters as regular NN
optimizer = Adam(lr=0.001)  # Too high!
batch_size = 128            # Too large!
```

✅ **Correct:**
```python
# GAN-specific hyperparameters
optimizer_G = Adam(lr=0.0002, betas=(0.5, 0.999))  # Low LR, β1=0.5
optimizer_D = Adam(lr=0.0002, betas=(0.5, 0.999))
batch_size = 32  # Small batch

# Label smoothing
real_labels = 0.9
fake_labels = 0.0
```

---

### Pitfall 2: Not Checking for Mode Collapse

**How to detect:**
```python
# Generate many samples, check diversity
samples = [generator(noise) for _ in range(100)]

# Visual inspection: Do they all look similar?

# Quantitative: Check variance
import numpy as np
samples_np = np.array([s.cpu().numpy() for s in samples])
variance = samples_np.std()  # Low variance = mode collapse
```

---

### Pitfall 3: Using Wrong Loss for VAE

❌ **Wrong:**
```python
# Only reconstruction loss
loss = F.mse_loss(output, input)
```

✅ **Correct:**
```python
# Reconstruction + KL divergence
reconstruction_loss = F.mse_loss(output, input)
kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

# Both losses are essential!
total_loss = reconstruction_loss + beta * kl_divergence
# beta = 0.1-1.0 (higher = more regularization)
```

---

### Pitfall 4: Not Using Negative Prompts (Diffusion)

❌ **Wrong:**
```python
image = pipe(prompt="A beautiful landscape")
# May get low-quality, blurry results
```

✅ **Correct:**
```python
image = pipe(
    prompt="A beautiful landscape",
    negative_prompt="blurry, low quality, distorted, ugly, bad anatomy"
)
# Much better quality!
```

---

### Pitfall 5: Forgetting to Normalize Inputs

❌ **Wrong:**
```python
# Images in [0, 255]
images = load_images()  # [0-255]
discriminator(images)   # Wrong range!
```

✅ **Correct:**
```python
# Normalize to [-1, 1] for GANs
images = (images / 255.0) * 2 - 1  # [0, 255] → [-1, 1]

# Or [0, 1] for VAEs
images = images / 255.0  # [0, 255] → [0, 1]
```

---

## 11.15 Summary Checklist

### Choosing a Model:
- [ ] 2025 default: Diffusion models (Stable Diffusion)
- [ ] Need speed: GAN
- [ ] Need diversity: VAE or Diffusion
- [ ] Need quality: Diffusion or GAN

### Before Training GANs:
- [ ] Understand mode collapse
- [ ] Use small batch size (≤64)
- [ ] Use label smoothing
- [ ] Use LeakyReLU in discriminator
- [ ] Use spectral normalization
- [ ] Monitor generated samples frequently

### VAE Training:
- [ ] Use both reconstruction and KL losses
- [ ] Tune beta parameter (0.1-1.0)
- [ ] Check latent space interpolation
- [ ] Monitor reconstruction quality

### Using Diffusion Models:
- [ ] Use negative prompts
- [ ] Set num_inference_steps (20-50)
- [ ] Set guidance_scale (7-9)
- [ ] Use GPU (CPU too slow)
- [ ] Consider SDXL for best quality

### Evaluation:
- [ ] Visual inspection (diversity, quality)
- [ ] FID score (Fréchet Inception Distance) for GANs
- [ ] Reconstruction error for VAEs
- [ ] User studies for subjective quality

---

## 11.16 Quick Reference: Generation Tasks

| Task | Model | Alternative | Quality | Speed |
|------|-------|-------------|---------|-------|
| Text-to-image | Stable Diffusion | DALL-E 3 | ⭐⭐⭐⭐⭐ | ⚡⚡ |
| Face generation | StyleGAN3 | Diffusion | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ |
| Style transfer | Diffusion | CycleGAN | ⭐⭐⭐⭐ | ⚡⚡ |
| Image upscaling | ESRGAN | Diffusion upscaler | ⭐⭐⭐⭐ | ⚡⚡ |
| Inpainting | Diffusion | GAN | ⭐⭐⭐⭐⭐ | ⚡⚡ |
| Data augmentation | Simple transforms | GAN | ⭐⭐⭐ | ⚡⚡⚡ |
| Anomaly detection | VAE | Autoencoder | ⭐⭐⭐⭐ | ⚡⚡⚡ |
| Feature learning | VAE | Autoencoder | ⭐⭐⭐⭐ | ⚡⚡⚡ |

---

## 11.17 Resources & Further Reading

**Stable Diffusion:**
- HuggingFace Diffusers: https://huggingface.co/docs/diffusers
- Stable Diffusion: https://stability.ai/

**GANs:**
- GAN Lab (interactive): https://poloclub.github.io/ganlab/
- StyleGAN: https://github.com/NVlabs/stylegan3
- GAN Zoo: https://github.com/hindupuravinash/the-gan-zoo

**Papers:**
- GANs (Goodfellow, 2014): https://arxiv.org/abs/1406.2661
- VAE (Kingma, 2013): https://arxiv.org/abs/1312.6114
- Diffusion Models (Ho, 2020): https://arxiv.org/abs/2006.11239
- Stable Diffusion: https://arxiv.org/abs/2112.10752

**Tutorials:**
- PyTorch GAN Tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- Fast.ai: https://www.fast.ai/

---

**Last Updated:** 2025-10-12
**Next Section:** Master Model Selection Decision Tree (Phase 8)
