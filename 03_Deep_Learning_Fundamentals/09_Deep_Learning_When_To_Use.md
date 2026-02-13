# 9. Deep Learning: When to Use & Best Practices

## Overview

Deep learning (DL) is powerful but not always necessary. This guide helps you decide when to use deep learning vs traditional ML, which architecture to choose, and how to train effectively.

**Key Truth:** Deep learning is NOT always better. For tabular data, XGBoost often outperforms neural networks.

---

## 9.1 Deep Learning vs Traditional ML Decision Flow

```
Should I use Deep Learning?

+--- Data Type?
|  +--- TABULAR/STRUCTURED
|  |  +---  Use Traditional ML (XGBoost, Random Forest)
|  |     Exception: Very large datasets (1M+ rows) with complex interactions
|  |
|  +--- IMAGES
|  |  +---  Use CNNs (Convolutional Neural Networks)
|  |
|  +--- TEXT
|  |  +---  Use Transformers or RNNs/LSTMs
|  |
|  +--- TIME SERIES (Sequential)
|  |  +---  Use RNNs/LSTMs or Transformers
|  |
|  +--- AUDIO/VIDEO
|     +---  Use CNNs + RNNs or specialized architectures
|
+--- Dataset Size?
|  +--- < 10,000 samples
|  |  +---  Too small for DL, use Traditional ML
|  |
|  +--- 10K - 100K samples
|  |  +--- [WARNING] Consider transfer learning or traditional ML
|  |
|  +--- > 100K samples
|     +---  Deep Learning viable
|
+--- Need Interpretability?
|  +--- YES -->  Use Traditional ML (easier to explain)
|
+--- Computational Resources?
|  +--- Limited (no GPU) -->  Use Traditional ML (faster)
|
+--- Automatic Feature Learning?
   +--- YES (raw pixels, text) -->  Use Deep Learning
```

---

## 9.2 When to Use Deep Learning

###  Use Deep Learning When:

#### 1. **Unstructured Data (Images, Audio, Video)**

**Why:** DL automatically learns hierarchical features from raw data

**Examples:**
- Image classification (cat vs dog, medical scans)
- Object detection (self-driving cars)
- Facial recognition
- Medical image analysis (X-rays, MRIs)

**Traditional ML would require:**
- Manual feature engineering (edges, textures, SIFT, HOG)
- Domain expertise to extract features
- Often inferior results

---

#### 2. **Natural Language Processing (Text)**

**Why:** DL captures context and semantics better

**Examples:**
- Sentiment analysis
- Machine translation
- Text generation
- Question answering
- Named entity recognition

**Use transformers (2024+)** for most NLP tasks

---

#### 3. **Large Datasets (100K+ samples)**

**Why:** DL needs lots of data to learn complex patterns

**Rule of thumb:**
- < 10K samples: Traditional ML
- 10K-100K: Transfer learning or traditional ML
- > 100K: Deep learning viable
- > 1M: Deep learning often wins

---

#### 4. **Complex Non-Linear Patterns**

**Why:** Multiple layers can model very complex relationships

**Examples:**
- Speech recognition (waveforms --> words)
- Video understanding (frame sequences --> actions)
- Game playing (AlphaGo, DeepMind)

---

#### 5. **Automatic Feature Learning Needed**

**Why:** Don't need manual feature engineering

**Examples:**
- Raw pixel data (no need to manually extract edges)
- Raw audio waveforms
- End-to-end learning (input --> output directly)

---

#### 6. **Have GPU Resources**

**Why:** DL training is 10-100x faster on GPUs

**Reality check:**
- Training ResNet on ImageNet: 1 week on GPU vs months on CPU
- Transformers: Essentially require GPUs

---

###  When NOT to Use Deep Learning

#### 1. **Tabular/Structured Data**

**Problem:** XGBoost/LightGBM often outperform neural nets on tables

**Why:**
- Decision trees naturally handle:
  - Missing values
  - Mixed data types
  - Non-linear relationships
  - Feature interactions
- Neural nets need heavy preprocessing

**Exception:** Very large tabular datasets (1M+ rows) with complex interactions

**Real-world example (Kaggle):**
- 90% of tabular competitions won by XGBoost
- Only 10% by neural networks

---

#### 2. **Small Datasets (< 10K samples)**

**Problem:** Deep learning overfits on small data

**Why:**
- Neural nets have millions of parameters
- Need lots of data to avoid memorization

**Solutions if stuck with DL:**
- Use transfer learning (pretrained models)
- Aggressive regularization (dropout, L2)
- Data augmentation
- Or just use traditional ML

---

#### 3. **Need Interpretability**

**Problem:** Neural nets are "black boxes"

**Why:**
- Millions of parameters
- Non-linear transformations
- Hard to explain predictions

**Use cases requiring interpretability:**
- Healthcare (must explain diagnosis)
- Finance (regulatory compliance)
- Legal (decisions must be justifiable)

**Better alternatives:**
- Linear models (clear coefficients)
- Decision trees (can visualize rules)
- Rule-based systems

---

#### 4. **Limited Computational Resources**

**Problem:** DL training is slow and expensive

**Reality:**
- Training deep models requires GPUs ($1000s)
- Cloud GPU costs: $1-3/hour
- Days to weeks of training
- High electricity costs

**Traditional ML:**
- Trains on CPU in minutes to hours
- Much cheaper
- Lower carbon footprint

---

#### 5. **Quick Prototyping Needed**

**Problem:** DL takes time to tune

**Why:**
- Many hyperparameters (learning rate, batch size, architecture)
- Long training times
- Requires experimentation

**Traditional ML:**
- XGBoost with defaults works well
- Fast to train and iterate
- Better for quick MVPs

---

#### 6. **Production Constraints**

**Problem:** DL models are large and slow

**Deployment challenges:**
- Model size: 100s of MBs to GBs
- Inference latency: 10-100ms
- Need GPU for real-time (expensive)
- Power consumption (mobile/edge devices)

**Traditional ML:**
- Model size: KBs to MBs
- Inference: < 1ms
- Runs on CPU
- Mobile-friendly

---

## 9.3 Neural Network Architectures

### Architecture Selection Guide

| Architecture | Data Type | Use Cases | Key Strength |
|--------------|-----------|-----------|--------------|
| **MLP (Fully Connected)** | Tabular, Structured | Classification, Regression on structured data | Simple, baseline |
| **CNN (Convolutional)** | Images, Spatial Data | Image classification, Object detection | Spatial relationships |
| **RNN/LSTM** | Sequential, Time Series | Text, Time series, Speech | Temporal patterns |
| **Transformer** | Text, Sequences | NLP, Translation, GPT | Long-range dependencies |
| **Autoencoder** | Any | Dimensionality reduction, Denoising | Unsupervised learning |
| **GAN** | Images | Image generation, Style transfer | Generative modeling |

---

### 9.3.1 Multi-Layer Perceptron (MLP)

**What:** Fully connected layers (each neuron connects to all in next layer)

###  When to Use MLP

1. **Tabular/structured data**
   - Loan applications, fraud detection
   - Customer churn prediction
   - As baseline before trying XGBoost

2. **Simple classification/regression**
   - When data doesn't have spatial or temporal structure

3. **As final layers in hybrid architectures**
   - CNN --> MLP (extract features, then classify)
   - RNN --> MLP (process sequence, then predict)

###  When NOT to Use MLP

1. **Images** (use CNN instead)
   - MLP treats each pixel independently
   - Ignores spatial relationships
   - Requires far more parameters

2. **Sequences** (use RNN/Transformer)
   - No memory of previous inputs
   - Can't handle variable-length sequences

3. **High-dimensional sparse data**
   - Millions of parameters
   - Overfits easily

---

### 9.3.2 CNN (Convolutional Neural Network)

**What:** Uses convolutional layers to detect spatial patterns (edges --> shapes --> objects)

###  When to Use CNN

1. **Image classification**
   - Cat vs dog, digit recognition
   - Medical image diagnosis

2. **Object detection**
   - Bounding boxes around objects
   - Face detection

3. **Image segmentation**
   - Pixel-level classification
   - Self-driving cars (lane detection)

4. **Video analysis**
   - 3D convolutions over time
   - Action recognition

5. **Document classification**
   - Text as 2D images
   - Surprisingly effective for NLP

###  When NOT to Use CNN

1. **Tabular data**
   - No spatial structure
   - Use MLP or XGBoost

2. **Pure sequential data** (time series without spatial component)
   - Use RNN/LSTM
   - Exception: 1D CNNs work for some time series

---

### 9.3.3 RNN/LSTM (Recurrent Neural Network)

**What:** Processes sequences one element at a time, maintains hidden state (memory)

###  When to Use RNN/LSTM

1. **Sequential text**
   - Sentiment analysis
   - Text generation
   - Machine translation (though Transformers now better)

2. **Time series prediction**
   - Stock prices, weather forecasting
   - Sensor data

3. **Speech recognition**
   - Waveform --> text

4. **Video captioning**
   - Sequence of frames --> description

5. **Music generation**
   - Sequence of notes

###  When NOT to Use RNN

1. **Long sequences (>100 steps)**
   - Vanishing gradient problem
   - **Use:** LSTM/GRU or Transformers

2. **Modern NLP tasks**
   - **Use:** Transformers (2024+ standard)
   - Better parallelization
   - Superior performance

3. **Images**
   - No sequential structure
   - **Use:** CNN

**2024 Reality:** Transformers have largely replaced RNNs for NLP

---

## 9.4 Training Neural Networks: Best Practices

### 9.4.1 Batch Size

**What:** Number of samples processed before updating weights

| Batch Size | Pros | Cons | When to Use |
|------------|------|------|-------------|
| **Small (32-64)** | Better generalization, Less memory | Slower, Noisy gradients | Limited GPU memory, Want better generalization |
| **Medium (128-256)** | Good balance | - | Default choice |
| **Large (512+)** | Faster training, Stable gradients | Worse generalization, Needs more memory | Large datasets, Multiple GPUs |

**Guidelines:**
- **Start with 32 or 64**
- Use powers of 2 (32, 64, 128, 256) for GPU efficiency
- Larger batch size --> increase learning rate proportionally

**Code:**
```python
# Good starting points
batch_size = 32  # Small GPU, better generalization
batch_size = 64  # Typical default
batch_size = 128  # Large GPU, faster training
```

---

### 9.4.2 Learning Rate

**What:** Step size for weight updates

**Most important hyperparameter!**

| Learning Rate | Effect | Risk |
|---------------|--------|------|
| **Too high (> 0.1)** | Training unstable, diverges | Loss explodes |
| **Good (0.001-0.01)** | Steady improvement | - |
| **Too low (< 0.0001)** | Very slow training | Gets stuck in local minima |

**Finding optimal learning rate:**

```python
# Learning Rate Finder (fastai technique)
from torch.optim import Adam
import matplotlib.pyplot as plt

def find_lr(model, train_loader, init_lr=1e-8, final_lr=10, beta=0.98):
    """
    Find optimal learning rate by gradually increasing
    """
    num_batches = len(train_loader)
    lr_mult = (final_lr / init_lr) ** (1 / num_batches)

    optimizer = Adam(model.parameters(), lr=init_lr)
    lrs, losses = [], []

    for batch_idx, (data, target) in enumerate(train_loader):
        # Training step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Record
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        # Update learning rate
        optimizer.param_groups[0]['lr'] *= lr_mult

        if loss.item() > 4 * min(losses):  # Stop if loss explodes
            break

    # Plot
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()

    # Pick LR where loss decreases fastest (steepest slope)
    # Typically 10x smaller than minimum loss point
```

**Common strategies:**

```python
# 1. Fixed learning rate
optimizer = Adam(model.parameters(), lr=0.001)

# 2. Learning rate decay (reduce over time)
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce by 10x every 10 epochs

# 3. ReduceLROnPlateau (reduce when stuck)
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# 4. Cosine Annealing (smooth decay)
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

**Best practice 2024:**
- Start with `lr=0.001` for Adam optimizer
- Use learning rate scheduler
- Monitor training loss; if stuck, reduce LR

---

### 9.4.3 Optimizers

| Optimizer | Best For | Learning Rate | Notes |
|-----------|----------|---------------|-------|
| **SGD** | Computer vision | 0.01-0.1 | Needs momentum (0.9) |
| **Adam** | Most tasks | 0.001-0.01 | Default choice 2024 |
| **AdamW** | Transformers, NLP | 0.0001-0.001 | Adam + weight decay |
| **RMSprop** | RNNs | 0.001 | Handles non-stationary |

**Code:**
```python
from torch.optim import Adam, SGD, AdamW

# Adam (default choice)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# SGD with momentum (for CNNs)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# AdamW (for Transformers)
optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
```

---

### 9.4.4 Regularization Techniques

#### Dropout

**What:** Randomly drop neurons during training

**Code:**
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # Drop 50% of neurons
    nn.Linear(256, 10)
)
```

**Guidelines:**
- Start with 0.5 for fully connected layers
- Use 0.2-0.3 for convolutional layers
- Don't use in final layer

---

#### Batch Normalization

**What:** Normalize activations between layers

**Benefits:**
- Faster training
- Allows higher learning rates
- Acts as regularization

**Code:**
```python
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.BatchNorm1d(256),  # Add BatchNorm
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

---

#### Data Augmentation (Images)

**What:** Artificially expand dataset with transformations

**Code:**
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

---

#### Early Stopping

**What:** Stop training when validation loss stops improving

**Code:**
```python
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model, 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping!")
        break
```

---

## 9.5 Common Pitfalls & Solutions

### Pitfall 1: Not Enough Data

 **Wrong:**
```python
# Training ResNet50 from scratch on 1000 images
model = resnet50(pretrained=False)
```

 **Correct:**
```python
# Use transfer learning
model = resnet50(pretrained=True)
# Freeze early layers
for param in model.parameters():
    param.requires_grad = False
# Only train final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

---

### Pitfall 2: Not Normalizing Inputs

 **Wrong:**
```python
# Raw pixel values [0, 255]
X = images
```

 **Correct:**
```python
# Normalize to [0, 1] or standardize
X = images / 255.0
# Or use ImageNet statistics for transfer learning
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

---

### Pitfall 3: Overfitting Without Noticing

 **Wrong:**
```python
# Only checking training loss
train_loss = 0.01  # Great!
# But validation loss = 2.5 (overfitting!)
```

 **Correct:**
```python
# Always monitor both
print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
if val_loss > train_loss * 1.5:
    print("[WARNING] Overfitting detected!")
    # Add regularization, reduce model complexity
```

---

### Pitfall 4: Using CPU When GPU Available

 **Wrong:**
```python
# Slow training on CPU
model = Model()
output = model(data)
```

 **Correct:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
data = data.to(device)
output = model(data)
```

---

### Pitfall 5: Forgetting to Set model.eval()

 **Wrong:**
```python
# Dropout/BatchNorm still active during testing!
with torch.no_grad():
    predictions = model(test_data)
```

 **Correct:**
```python
model.eval()  # Disable dropout, use BN running stats
with torch.no_grad():
    predictions = model(test_data)
model.train()  # Re-enable for training
```

---

## 9.6 Quick Reference: Model Selection

```
What's your data type?

+--- TABULAR
|  +--- Use XGBoost (not deep learning)
|     Exception: 1M+ rows with complex interactions --> try MLP
|
+--- IMAGES
|  +--- Use CNN
|     +--- < 10K images? --> Transfer learning (ResNet, EfficientNet)
|     +--- > 100K images? --> Train from scratch or fine-tune
|
+--- TEXT
|  +--- Use Transformers (BERT, GPT)
|     +--- < 10K samples? --> Use pretrained models
|     +--- Simple task? --> Try simpler models first (logistic, LSTM)
|
+--- TIME SERIES
|  +--- Try traditional methods first (ARIMA, Prophet)
|     If complex: --> LSTM or Transformer
|
+--- MIXED (tabular + images/text)
   +--- Hybrid: CNN/Transformer --> embeddings --> concatenate --> MLP
```

---

## 9.7 Summary Checklist

### Before Using Deep Learning:
- [ ] Have > 10K samples (preferably 100K+)
- [ ] Data is unstructured (images, text, audio)
- [ ] Have GPU available
- [ ] Don't need real-time inference (< 10ms)
- [ ] Don't need interpretability
- [ ] Traditional ML has been tried and failed

### Training Checklist:
- [ ] Normalize/standardize inputs
- [ ] Use appropriate architecture (CNN for images, etc.)
- [ ] Start with batch_size=32-64
- [ ] Start with lr=0.001 (Adam) or 0.01 (SGD)
- [ ] Use learning rate scheduler
- [ ] Add regularization (dropout, weight decay, data augmentation)
- [ ] Monitor both training and validation loss
- [ ] Use early stopping
- [ ] Save best model (based on validation loss)
- [ ] Use GPU if available

### After Training:
- [ ] Evaluate on separate test set
- [ ] Check for overfitting (train vs test performance)
- [ ] Verify inference speed meets requirements
- [ ] Check model size for deployment constraints

---

## Resources & Further Reading

**Courses:**
- fast.ai Practical Deep Learning
- Stanford CS231n (Computer Vision)
- DeepLearning.AI specialization

**Papers:**
- ResNet: "Deep Residual Learning" (He et al., 2015)
- LSTM: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- Attention: "Attention Is All You Need" (Vaswani et al., 2017)

**Libraries:**
- PyTorch: https://pytorch.org/
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/

**Books:**
- "Deep Learning" by Goodfellow, Bengio, Courville
- "Hands-On Machine Learning" by Geron

---

**Last Updated:** 2025-10-12
**Next Section:** NLP & Transformers (Phase 6)
