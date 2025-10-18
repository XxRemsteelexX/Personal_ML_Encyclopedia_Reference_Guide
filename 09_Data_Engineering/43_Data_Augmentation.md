# Data Augmentation - Complete Guide (2025)

## Overview

**Data augmentation artificially expands training datasets** by creating modified versions of existing data, improving model generalization and reducing overfitting.

**2025 State-of-the-Art:**
- AutoAugment & learned policies (AutoML-based)
- TrivialAugment (parameter-free SOTA)
- AugMax (adversarial composition)
- CTGAN for tabular data
- SpecAugment for audio

---

## Image Augmentation

### Basic Geometric Transformations

```python
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class BasicImageAugmentation:
    """Standard geometric augmentations"""

    def __init__(self):
        self.transform = transforms.Compose([
            # Geometric transformations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # 10% translation
                scale=(0.9, 1.1),       # 90-110% scale
                shear=10                 # Shear angle
            ),

            # Cropping
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),

            # Color augmentations
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),

            # Convert to tensor
            transforms.ToTensor(),

            # Normalization
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, image):
        return self.transform(image)

# Usage
augmenter = BasicImageAugmentation()
augmented_image = augmenter(original_image)
```

---

## Advanced Image Augmentation

### Mixup (2017 - Still Effective 2025)

**Blend images and labels for better generalization.**

```python
import torch
import numpy as np

def mixup_data(x, y, alpha=1.0):
    """
    Mixup: blend two samples

    Args:
        x: batch of images
        y: batch of labels
        alpha: mixup hyperparameter (controls distribution)

    Returns:
        mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    # Mix images
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Return both labels for loss calculation
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Modified loss for mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Training loop with Mixup
for images, labels in train_loader:
    # Apply Mixup
    images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=1.0)

    # Forward
    outputs = model(images)

    # Mixup loss
    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Benefits:**
- Improved generalization
- Better calibration (more confident predictions)
- Regularization effect

---

### CutMix (2019 - SOTA for CNNs)

**Cut and paste image patches, preserving spatial structure.**

```python
def cutmix_data(x, y, alpha=1.0):
    """
    CutMix: cut and paste patches between images
    """
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    # Get bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    # Mix images by cutting patch
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Generate random bounding box"""
    W = size[2]
    H = size[3]

    # Calculate cut size
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling of center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
```

**Advantages over Mixup:**
- Preserves spatial structure
- Better localization ability
- Efficient use of pixels

---

### Cutout / Random Erasing

**Simulate occlusion by masking random patches.**

```python
import torchvision.transforms as T

# Method 1: Using torchvision
transform_with_cutout = T.Compose([
    T.RandomErasing(
        p=0.5,              # Probability
        scale=(0.02, 0.33),  # Area range
        ratio=(0.3, 3.3),    # Aspect ratio
        value='random'       # Fill with random values
    )
])

# Method 2: Custom Cutout
class Cutout:
    """Randomly mask square regions"""

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img: Tensor image of size (C, H, W)
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
```

---

## Automated Augmentation Policies

### AutoAugment (2019 - Search-Based)

**Learn optimal augmentation policies via reinforcement learning.**

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# Pre-learned policies for different datasets
transform_imagenet = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)
transform_cifar = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
transform_svhn = AutoAugment(policy=AutoAugmentPolicy.SVHN)

# Use in pipeline
train_transform = transforms.Compose([
    AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

**How it works:**
1. Search space: 16 operations (rotate, shear, color, etc.)
2. Each operation has magnitude and probability
3. RL controller searches for best policy
4. Policy = sequence of sub-policies

**Cost:** Expensive search (15,000 GPU hours for ImageNet)

---

### RandAugment (2020 - Simplified)

**Single magnitude parameter controls all augmentations.**

```python
from torchvision.transforms import RandAugment

# Simple: only 2 hyperparameters!
transform = RandAugment(
    num_ops=2,      # Number of augmentations per image
    magnitude=9     # Magnitude (0-30, typically 9-15)
)

# Full pipeline
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

**Advantages:**
- **Much cheaper** than AutoAugment (no search needed)
- Competitive performance
- Only 2 hyperparameters vs. 32+ in AutoAugment

---

### TrivialAugment (2021 - SOTA, Parameter-Free)

**Randomly sample augmentation AND magnitude - no tuning needed!**

```python
from torchvision.transforms import TrivialAugmentWide

# ZERO hyperparameters to tune!
transform = TrivialAugmentWide()

# That's it - works out of the box
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    TrivialAugmentWide(),  # Parameter-free!
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

**How it works:**
1. Randomly select ONE augmentation operation
2. Randomly sample magnitude for that operation
3. Apply to image

**Performance (2025):**
- Outperforms RandAugment
- Matches AutoAugment (no expensive search!)
- **Recommended default for most use cases**

---

### AugMax (2021 - Adversarial Composition)

**Combine diversity and hardness through adversarial mixing.**

```python
import torch
import torch.nn.functional as F

class AugMax:
    """Adversarial composition of augmentations"""

    def __init__(self, num_ops=3, temperature=1.0):
        self.num_ops = num_ops
        self.temperature = temperature
        self.augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # ... more augmentations
        ]

    def __call__(self, image, model):
        """Apply adversarially hard augmentation"""

        # Sample random augmentations
        selected_augs = np.random.choice(
            self.augmentations,
            size=self.num_ops,
            replace=False
        )

        # Generate augmented versions
        aug_images = []
        for aug in selected_augs:
            aug_img = aug(image)
            aug_images.append(aug_img)

        aug_images = torch.stack(aug_images)

        # Get model predictions (without gradient)
        with torch.no_grad():
            logits = model(aug_images)
            losses = F.cross_entropy(logits, target, reduction='none')

        # Learn mixing weights adversarially
        weights = F.softmax(losses / self.temperature, dim=0)

        # Mix augmented images
        mixed_image = (aug_images * weights.view(-1, 1, 1, 1)).sum(dim=0)

        return mixed_image
```

**Key Innovation:**
- Samples multiple augmentations
- Learns **adversarial mixture** (harder samples get higher weight)
- Balances diversity + difficulty

---

## Text/NLP Augmentation

### Synonym Replacement

```python
from nltk.corpus import wordnet
import random

def get_synonyms(word):
    """Get synonyms using WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def synonym_replacement(sentence, n=2):
    """Replace n random words with synonyms"""
    words = sentence.split()

    # Get replaceable words (exclude stopwords)
    replaceable = [w for w in words if w.lower() not in stopwords]

    # Randomly select n words
    random.shuffle(replaceable)

    for word in replaceable[:n]:
        synonyms = get_synonyms(word)
        if synonyms:
            synonym = random.choice(synonyms)
            words = [synonym if w == word else w for w in words]

    return ' '.join(words)

# Example
original = "The quick brown fox jumps over the lazy dog"
augmented = synonym_replacement(original, n=2)
print(augmented)
# Output: "The fast brown fox leaps over the lazy dog"
```

---

### Back-Translation

```python
from transformers import MarianMTModel, MarianTokenizer

class BackTranslation:
    """Augment by translating to another language and back"""

    def __init__(self, source_lang='en', pivot_lang='fr'):
        # English -> French
        self.model_forward = MarianMTModel.from_pretrained(
            f'Helsinki-NLP/opus-mt-{source_lang}-{pivot_lang}'
        )
        self.tokenizer_forward = MarianTokenizer.from_pretrained(
            f'Helsinki-NLP/opus-mt-{source_lang}-{pivot_lang}'
        )

        # French -> English
        self.model_backward = MarianMTModel.from_pretrained(
            f'Helsinki-NLP/opus-mt-{pivot_lang}-{source_lang}'
        )
        self.tokenizer_backward = MarianTokenizer.from_pretrained(
            f'Helsinki-NLP/opus-mt-{pivot_lang}-{source_lang}'
        )

    def augment(self, text):
        # Translate to pivot language
        inputs = self.tokenizer_forward(text, return_tensors="pt", padding=True)
        translated = self.model_forward.generate(**inputs)
        pivot_text = self.tokenizer_forward.decode(translated[0], skip_special_tokens=True)

        # Translate back
        inputs = self.tokenizer_backward(pivot_text, return_tensors="pt", padding=True)
        back_translated = self.model_backward.generate(**inputs)
        augmented_text = self.tokenizer_backward.decode(back_translated[0], skip_special_tokens=True)

        return augmented_text

# Usage
bt = BackTranslation(source_lang='en', pivot_lang='de')
augmented = bt.augment("Machine learning is fascinating")
print(augmented)
```

---

### Contextual Word Embeddings (BERT-based)

```python
from transformers import pipeline

class ContextualAugmentation:
    """Use BERT to predict masked words"""

    def __init__(self):
        self.unmasker = pipeline('fill-mask', model='bert-base-uncased')

    def augment(self, sentence, n_words=1):
        words = sentence.split()

        # Randomly mask n words
        mask_indices = random.sample(range(len(words)), min(n_words, len(words)))

        augmented_sentences = []

        for idx in mask_indices:
            masked_sentence = words.copy()
            masked_sentence[idx] = '[MASK]'
            masked_text = ' '.join(masked_sentence)

            # Get predictions
            predictions = self.unmasker(masked_text)

            # Use top prediction
            new_word = predictions[0]['token_str'].strip()
            words[idx] = new_word

        return ' '.join(words)

# Usage
aug = ContextualAugmentation()
augmented = aug.augment("The model achieved excellent results", n_words=2)
```

---

### Random Operations (EDA - Easy Data Augmentation)

```python
import random

class EDA:
    """Easy Data Augmentation (4 operations)"""

    def __init__(self, alpha=0.1):
        self.alpha = alpha  # % of words to change

    def random_insertion(self, words):
        """Insert random synonym"""
        n = max(1, int(self.alpha * len(words)))
        for _ in range(n):
            add_word = random.choice(words)
            synonyms = get_synonyms(add_word)
            if synonyms:
                words.insert(random.randint(0, len(words)), random.choice(synonyms))
        return words

    def random_swap(self, words):
        """Randomly swap two words"""
        n = max(1, int(self.alpha * len(words)))
        for _ in range(n):
            if len(words) >= 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        return words

    def random_deletion(self, words):
        """Randomly delete words"""
        if len(words) == 1:
            return words

        new_words = []
        for word in words:
            if random.random() > self.alpha:
                new_words.append(word)

        return new_words if new_words else [random.choice(words)]

    def augment(self, sentence):
        words = sentence.split()

        # Randomly choose operation
        operation = random.choice([
            self.random_insertion,
            self.random_swap,
            self.random_deletion
        ])

        augmented_words = operation(words.copy())
        return ' '.join(augmented_words)
```

---

## Audio Augmentation

### SpecAugment (2019 - SOTA for Speech)

**Augment spectrogram (time-frequency representation) directly.**

```python
import torchaudio
import torch

class SpecAugment:
    """SpecAugment for speech recognition"""

    def __init__(self, freq_mask_param=15, time_mask_param=35, n_freq_masks=2, n_time_masks=2):
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, spectrogram):
        """
        Args:
            spectrogram: (freq, time) mel spectrogram
        """
        # Frequency masking
        for _ in range(self.n_freq_masks):
            spectrogram = self.freq_masking(spectrogram)

        # Time masking
        for _ in range(self.n_time_masks):
            spectrogram = self.time_masking(spectrogram)

        return spectrogram

# Full audio augmentation pipeline
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=80
)

spec_aug = SpecAugment(
    freq_mask_param=15,
    time_mask_param=35,
    n_freq_masks=2,
    n_time_masks=2
)

# Usage
waveform, sr = torchaudio.load('audio.wav')
spec = mel_spectrogram(waveform)
augmented_spec = spec_aug(spec)
```

---

### Time-Domain Audio Augmentation

```python
import numpy as np

class AudioAugmentation:
    """Time-domain audio augmentation"""

    def add_noise(self, audio, noise_factor=0.005):
        """Add white noise"""
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        return augmented

    def time_shift(self, audio, shift_max=0.2):
        """Shift audio in time"""
        shift = np.random.randint(int(len(audio) * shift_max))
        direction = np.random.choice(['left', 'right'])

        if direction == 'right':
            augmented = np.pad(audio, (shift, 0), mode='constant')[:-shift]
        else:
            augmented = np.pad(audio, (0, shift), mode='constant')[shift:]

        return augmented

    def pitch_shift(self, audio, sr, n_steps=2):
        """Shift pitch (requires librosa)"""
        import librosa
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    def time_stretch(self, audio, rate=1.2):
        """Speed up or slow down"""
        import librosa
        return librosa.effects.time_stretch(audio, rate=rate)
```

---

## Tabular Data Augmentation

### SMOTE (Synthetic Minority Oversampling)

```python
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

# Standard SMOTE
smote = SMOTE(sampling_strategy='minority', k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ADASYN (Adaptive Synthetic Sampling)
# Generates more samples in harder-to-learn regions
adasyn = ADASYN(sampling_strategy='minority', n_neighbors=5)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# Borderline-SMOTE
# Only augment samples near decision boundary
borderline_smote = BorderlineSMOTE(sampling_strategy='minority', k_neighbors=5)
X_resampled, y_resampled = borderline_smote.fit_resample(X_train, y_train)
```

---

### CTGAN (Conditional Tabular GAN) - 2024 SOTA

```python
from ctgan import CTGAN
import pandas as pd

# Load tabular data
train_data = pd.read_csv('train.csv')

# Initialize CTGAN
ctgan = CTGAN(
    epochs=300,
    batch_size=500,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256)
)

# Fit on real data
ctgan.fit(train_data, discrete_columns=['category', 'type'])

# Generate synthetic samples
synthetic_data = ctgan.sample(1000)

# Combine with original
augmented_data = pd.concat([train_data, synthetic_data], ignore_index=True)
```

**CTGAN Benefits:**
- Handles mixed data types (continuous + categorical)
- Learns complex distributions
- Better than SMOTE for high-dimensional data
- 2024 research shows effectiveness for imbalanced datasets

---

### Gaussian Noise (Simple Baseline)

```python
import numpy as np

class TabularAugmentation:
    """Simple tabular augmentation"""

    def add_gaussian_noise(self, X, noise_level=0.01):
        """Add Gaussian noise to continuous features"""
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise

    def mixup_tabular(self, X, y, alpha=0.2):
        """Mixup for tabular data"""
        lam = np.random.beta(alpha, alpha, size=len(X))

        # Random permutation
        index = np.random.permutation(len(X))

        # Mix features
        X_mixed = lam.reshape(-1, 1) * X + (1 - lam.reshape(-1, 1)) * X[index]

        # For classification: soft labels
        # For regression: mixed targets
        y_mixed = lam * y + (1 - lam) * y[index]

        return X_mixed, y_mixed
```

---

## Best Practices

### 1. Augmentation Strategy by Data Type

| Data Type | Recommended Methods | Avoid |
|-----------|-------------------|-------|
| **Images (2025)** | TrivialAugment, CutMix, Mixup | Over-aggressive rotation/distortion |
| **Text** | Back-translation, BERT masking, EDA | Simple synonym replacement alone |
| **Audio** | SpecAugment, time stretch, pitch shift | Excessive noise |
| **Tabular** | CTGAN, SMOTE variants, Mixup | Plain Gaussian noise (weak) |

### 2. Implementation Guidelines

```python
class ProductionAugmentation:
    """Production-ready augmentation pipeline"""

    def __init__(self, mode='train'):
        self.mode = mode

        if mode == 'train':
            # Strong augmentation during training
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                TrivialAugmentWide(),  # SOTA, parameter-free
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # No augmentation for validation/test
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, image):
        return self.transform(image)

# Usage
train_aug = ProductionAugmentation(mode='train')
val_aug = ProductionAugmentation(mode='val')
```

### 3. Progressive Augmentation

```python
class ProgressiveAugmentation:
    """Increase augmentation strength over training"""

    def __init__(self, initial_magnitude=5, final_magnitude=15, total_epochs=100):
        self.initial_magnitude = initial_magnitude
        self.final_magnitude = final_magnitude
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def get_magnitude(self):
        # Linear schedule
        progress = self.current_epoch / self.total_epochs
        magnitude = self.initial_magnitude + progress * (self.final_magnitude - self.initial_magnitude)
        return int(magnitude)

    def __call__(self, image):
        magnitude = self.get_magnitude()
        transform = RandAugment(num_ops=2, magnitude=magnitude)
        return transform(image)

    def step_epoch(self):
        self.current_epoch += 1
```

### 4. Consistency Regularization

```python
def consistency_loss(model, original, augmented, temperature=1.0):
    """Encourage consistent predictions on augmented versions"""

    with torch.no_grad():
        pred_original = F.softmax(model(original) / temperature, dim=1)

    pred_augmented = F.log_softmax(model(augmented) / temperature, dim=1)

    # KL divergence
    consistency = F.kl_div(pred_augmented, pred_original, reduction='batchmean')

    return consistency

# In training loop
for images, labels in train_loader:
    # Create augmented version
    augmented = strong_augmentation(images)

    # Supervised loss
    outputs = model(images)
    supervised_loss = criterion(outputs, labels)

    # Consistency loss
    cons_loss = consistency_loss(model, images, augmented)

    # Total loss
    total_loss = supervised_loss + 0.1 * cons_loss

    total_loss.backward()
    optimizer.step()
```

---

## Key Takeaways

1. **Images (2025 SOTA):** TrivialAugment (parameter-free), CutMix, Mixup
2. **Text:** Back-translation, BERT masking, EDA for quick baseline
3. **Audio:** SpecAugment is standard (mask time-frequency)
4. **Tabular:** CTGAN for complex data, SMOTE variants for simple cases
5. **AutoAugment family:** AutoAugment → RandAugment → TrivialAugment (simpler, better)
6. **Always:** No augmentation on validation/test sets
7. **Progressive:** Start weak, increase augmentation strength during training
8. **Consistency:** Use augmented views for semi-supervised learning

**2025 Recommendation:** Start with TrivialAugment (images) or back-translation (text) - simple, effective, no tuning needed!
