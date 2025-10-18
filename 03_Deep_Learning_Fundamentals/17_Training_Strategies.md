# Training Strategies for Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Learning Rate Schedules (Advanced)](#learning-rate-schedules-advanced)
3. [Mixed Precision Training](#mixed-precision-training)
4. [Gradient Accumulation](#gradient-accumulation)
5. [Curriculum Learning](#curriculum-learning)
6. [Transfer Learning and Fine-Tuning](#transfer-learning-and-fine-tuning)
7. [Model Checkpointing](#model-checkpointing)
8. [Distributed Training](#distributed-training)
9. [Hyperparameter Tuning](#hyperparameter-tuning)
10. [Complete Production Training Pipeline](#complete-production-training-pipeline)
11. [2025 Best Practices](#2025-best-practices)

---

## Introduction

Training deep neural networks is an engineering challenge that requires:
- Efficient use of computational resources (GPUs, memory)
- Stable and fast convergence
- Reproducibility
- Scalability to larger models and datasets

**Modern training (2025) requires:**
1. Mixed precision training (FP16/BF16)
2. Gradient accumulation for large effective batch sizes
3. Distributed training across multiple GPUs
4. Proper learning rate scheduling
5. Robust checkpointing and recovery
6. Systematic hyperparameter tuning

---

## Learning Rate Schedules (Advanced)

### Learning Rate Warm-Up (Deep Dive)

**Problem:** Large learning rates at initialization can cause:
- Exploding gradients
- Training instability
- Divergence

**Solution:** Gradually increase learning rate from small value to target value.

**Linear warm-up:**
```
α_t = α_target * min(1, t / T_warmup)

where:
  t: current step
  T_warmup: total warm-up steps
```

**Example:**
```
T_warmup = 1000 steps
α_target = 1e-3

Step 0:    α = 0
Step 250:  α = 2.5e-4
Step 500:  α = 5.0e-4
Step 1000: α = 1e-3
```

**Why warm-up works:**

1. **BatchNorm statistics:** Need time to stabilize
2. **Adaptive optimizers (Adam):** Need time to estimate moments
3. **Large models:** Parameters initialized randomly, large updates harmful

**PyTorch implementation:**
```python
def get_warmup_cosine_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01
):
    """
    Learning rate schedule with linear warm-up followed by cosine annealing.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warm-up steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial LR
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warm-up
            return step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Usage
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = get_warmup_cosine_scheduler(
    optimizer,
    warmup_steps=1000,
    total_steps=10000,
    min_lr_ratio=0.01
)

for step in range(10000):
    # Training step
    ...
    optimizer.step()
    scheduler.step()  # Update LR after each step
```

### Cyclical Learning Rates

**Idea:** Learning rate cycles between bounds rather than monotonically decreasing.

**Benefits:**
1. Escapes saddle points
2. Explores loss landscape
3. Can find better minima

**Triangular policy:**
```
# Cycle through: min_lr → max_lr → min_lr

step_size = 2000  # Half cycle length
cycle = floor(1 + step / (2 * step_size))
x = abs(step / step_size - 2 * cycle + 1)
lr = min_lr + (max_lr - min_lr) * max(0, 1 - x)
```

**PyTorch implementation:**
```python
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=1e-4,
    max_lr=1e-3,
    step_size_up=2000,
    mode='triangular2',  # Halve amplitude each cycle
    cycle_momentum=True
)
```

### One-Cycle Policy (Deep Dive)

**Comprehensive schedule combining:**
1. Learning rate warm-up and annealing
2. Momentum annealing (inverse to LR)
3. Single cycle over entire training

**Philosophy:** Aggressively increase LR to speedup training, then decrease for convergence.

**Schedule phases:**

**Phase 1 (30% of training):** Warm-up
```
LR: min_lr → max_lr (increase)
Momentum: max_momentum → min_momentum (decrease)
```

**Phase 2 (70% of training):** Annealing
```
LR: max_lr → min_lr (decrease)
Momentum: min_momentum → max_momentum (increase)
```

**Hyperparameters:**
- **max_lr:** Most important! Use LR range test
- **pct_start:** Fraction of training for phase 1 (default 0.3)
- **div_factor:** Initial LR = max_lr / div_factor (default 25)
- **final_div_factor:** Final LR = max_lr / final_div_factor (default 1e4)

**PyTorch implementation:**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=100,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1e4
)

for epoch in range(100):
    for batch in train_loader:
        # Training
        optimizer.step()
        scheduler.step()  # Call after each batch!
```

**When to use:**
- Time-constrained training
- When you can afford to tune max_lr
- Often achieves best results fastest

### ReduceLROnPlateau

**Idea:** Reduce learning rate when validation metric plateaus.

**Algorithm:**
```
1. Monitor validation metric (loss or accuracy)
2. If metric doesn't improve for `patience` epochs:
   - Reduce LR by factor
3. Repeat until min_lr reached
```

**PyTorch implementation:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # 'min' for loss, 'max' for accuracy
    factor=0.1,      # LR = LR * 0.1
    patience=10,     # Wait 10 epochs
    verbose=True,
    min_lr=1e-6
)

for epoch in range(epochs):
    train_loss = train_epoch(...)
    val_loss = evaluate(...)

    scheduler.step(val_loss)  # Pass metric to scheduler
```

**Advantages:**
- Adaptive to training dynamics
- No need to specify schedule in advance
- Works well when optimal training length unknown

**Disadvantages:**
- Requires validation set
- Can be slow (waits for plateau)
- May reduce LR too early or late

---

## Mixed Precision Training

### Motivation

**Problem:** FP32 (32-bit float) is default, but:
- High memory usage (4 bytes per parameter)
- Slower computation
- Limits batch size and model size

**Solution:** Use lower precision (FP16, BF16) where possible.

### Floating Point Formats

**FP32 (32-bit float):**
```
1 sign bit | 8 exponent bits | 23 mantissa bits
Range: ~1e-38 to 1e38
Precision: ~7 decimal digits
```

**FP16 (16-bit float):**
```
1 sign bit | 5 exponent bits | 10 mantissa bits
Range: ~6e-8 to 6e4
Precision: ~3 decimal digits
```

**BF16 (Brain Float 16):**
```
1 sign bit | 8 exponent bits | 7 mantissa bits
Range: ~1e-38 to 1e38 (same as FP32!)
Precision: ~2 decimal digits
```

**Comparison:**

| Format | Memory | Speed | Range | Precision | Hardware Support |
|--------|--------|-------|-------|-----------|------------------|
| FP32 | 4 bytes | 1× (baseline) | Wide | High | All GPUs |
| FP16 | 2 bytes | 2-3× | Narrow | Medium | Volta+ (2017+) |
| BF16 | 2 bytes | 2-3× | Wide | Medium | Ampere+ (2020+) |

**2025 Recommendation:** Use BF16 if available (RTX 3000+, A100+), otherwise FP16.

### Challenges with FP16

**1. Underflow:** Small gradients → 0
```
Gradient = 1e-8 in FP32
→ Underflow to 0 in FP16
→ No weight update
```

**2. Overflow:** Large activations/gradients → Inf
```
Activation = 70000 in FP32
→ Overflow to Inf in FP16
→ NaN in subsequent operations
```

**3. Imprecise weight updates:**
```
Weight = 1.0
Gradient = 1e-4
Update: weight + gradient * lr

In FP16: 1.0 + 1e-4 * 0.01 = 1.0 (no change due to precision limits)
```

### Mixed Precision Training Algorithm

**Solution:** Combine FP16 and FP32 strategically.

**Algorithm:**
```
1. Maintain FP32 master copy of weights
2. Forward pass in FP16 (faster)
3. Loss computation in FP16
4. Backward pass in FP16 (faster)
5. Gradient scaling to prevent underflow
6. Unscale gradients
7. Update FP32 master weights
8. Convert weights to FP16 for next iteration
```

**Gradient scaling:**
```
# Scale loss before backward pass
scaled_loss = loss * scale_factor  # e.g., scale=2^16

# Backward pass
scaled_loss.backward()  # Gradients are also scaled

# Unscale gradients
unscaled_gradients = scaled_gradients / scale_factor

# Update weights (FP32)
optimizer.step()
```

**Why scaling works:**
- Small gradients scaled up → avoid underflow
- After backward, unscale before optimizer → correct magnitudes

### PyTorch Automatic Mixed Precision (AMP)

**Modern approach:** Let PyTorch handle complexity automatically.

**Basic usage:**
```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

# Training loop
for data, target in train_loader:
    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Backward pass with scaling
    scaler.scale(loss).backward()

    # Gradient clipping (must unscale first!)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step with scaling
    scaler.step(optimizer)
    scaler.update()
```

**What `autocast()` does:**
- Casts operations to FP16 where safe
- Keeps some operations in FP32 (e.g., softmax, layer norm)
- Automatically determines which operations to cast

**Operations kept in FP32:**
- Softmax, log_softmax
- Cross entropy loss
- Layer normalization
- Batch normalization (usually)

**Benefits:**
- 2-3× faster training
- 50% memory reduction
- Larger batch sizes possible
- Minimal code changes

### BF16 Training

**BF16 advantages over FP16:**
- Same range as FP32 (fewer overflow/underflow issues)
- No gradient scaling needed
- More stable training

**PyTorch implementation (BF16):**
```python
# Check if BF16 available
if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16

# Training loop
for data, target in train_loader:
    with autocast(dtype=dtype):
        output = model(data)
        loss = criterion(output, target)

    loss.backward()
    optimizer.step()
```

**Note:** BF16 doesn't require GradScaler (no gradient scaling needed).

### Memory Savings Example

**Model:** ResNet-50 (25M parameters)

**Memory usage:**

| Precision | Model Weights | Activations (batch=32) | Optimizer State (Adam) | Total |
|-----------|---------------|------------------------|------------------------|-------|
| FP32 | 100 MB | ~8 GB | 200 MB | ~8.3 GB |
| FP16/BF16 | 50 MB | ~4 GB | 200 MB | ~4.25 GB |

**Effective batch size increase:** 2× (from 32 to 64) with same GPU memory.

---

## Gradient Accumulation

### Motivation

**Problem:** Large batch sizes don't fit in GPU memory.

**Desired:** Batch size 512
**Actual:** Can only fit batch size 64

**Solution:** Accumulate gradients over multiple forward/backward passes.

### Algorithm

**Standard training (batch size B):**
```
1. Load batch of size B
2. Forward pass
3. Backward pass
4. Optimizer step (update weights)
```

**Gradient accumulation (effective batch size B × K):**
```
1. For k = 1 to K:
   a. Load mini-batch of size B
   b. Forward pass
   c. Backward pass (accumulate gradients)
2. Optimizer step (update weights with accumulated gradients)
3. Zero gradients
```

### PyTorch Implementation

```python
def train_with_gradient_accumulation(
    model,
    train_loader,
    optimizer,
    criterion,
    accumulation_steps=4,
    device='cuda'
):
    """
    Training with gradient accumulation.

    Effective batch size = train_loader.batch_size * accumulation_steps
    """
    model.train()
    optimizer.zero_grad()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Scale loss (important!)
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()  # Gradients accumulate in .grad

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Handle remaining batches (if total batches not divisible by accumulation_steps)
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Key points:**

1. **Scale loss by accumulation_steps:** Ensures gradients have correct magnitude
2. **Accumulate gradients:** Don't call `optimizer.zero_grad()` until after update
3. **Update every K steps:** Call `optimizer.step()` periodically

### With Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer.zero_grad()

for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)

    # Forward with autocast
    with autocast():
        output = model(data)
        loss = criterion(output, target) / accumulation_steps

    # Scaled backward
    scaler.scale(loss).backward()

    # Update every accumulation_steps
    if (batch_idx + 1) % accumulation_steps == 0:
        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Batch Size vs Gradient Accumulation

**Trade-offs:**

| Aspect | Large Batch | Gradient Accumulation |
|--------|-------------|----------------------|
| **Speed** | Faster (more parallel) | Slower (sequential) |
| **Memory** | High | Low |
| **Generalization** | Can be worse | Same as large batch |
| **Batch statistics** | More accurate | Less accurate per step |

**When to use gradient accumulation:**
- GPU memory limited
- Want large effective batch size
- Training very large models (GPT, BERT)

**2025 Best Practice:**
- Start with largest batch size that fits in memory
- Use gradient accumulation for effective batch sizes > 256

---

## Curriculum Learning

### Core Idea

**Train on easy examples first, gradually introduce harder examples.**

**Inspiration:** How humans learn (elementary → advanced).

**Benefits:**
1. Faster convergence
2. Better generalization
3. Escape local minima
4. More stable training

### Example Curricula

**1. Sample difficulty:**
- Easy: Clear, high-quality examples
- Hard: Noisy, ambiguous, outliers

**2. Task complexity:**
- Easy: Simple sub-tasks
- Hard: Full complex task

**3. Data ordering:**
- Easy: Short sequences, simple images
- Hard: Long sequences, complex images

### Implementation Strategies

**Strategy 1: Hard example mining**
```python
def curriculum_sampling(dataset, model, epoch, total_epochs):
    """
    Sample from dataset based on current difficulty level.

    Gradually include harder examples as training progresses.
    """
    # Compute difficulty scores (e.g., loss on each example)
    difficulties = compute_difficulties(dataset, model)

    # Determine difficulty threshold (increases with epoch)
    difficulty_threshold = (epoch / total_epochs) * max(difficulties)

    # Filter dataset
    easy_indices = [i for i, d in enumerate(difficulties) if d <= difficulty_threshold]

    return Subset(dataset, easy_indices)
```

**Strategy 2: Loss-based curriculum**
```python
class CurriculumSampler(torch.utils.data.Sampler):
    """
    Sample based on loss (start with low-loss examples).
    """

    def __init__(self, losses, epoch, total_epochs):
        self.losses = losses
        self.progress = epoch / total_epochs

    def __iter__(self):
        # Sort by loss
        sorted_indices = np.argsort(self.losses)

        # Take easier examples based on progress
        n_samples = int(len(sorted_indices) * (0.5 + 0.5 * self.progress))
        selected = sorted_indices[:n_samples]

        # Shuffle selected examples
        np.random.shuffle(selected)
        return iter(selected)

    def __len__(self):
        return len(self.losses)
```

**Strategy 3: Sequence length curriculum (for NLP/sequences)**
```python
def get_curriculum_dataloader(dataset, epoch, total_epochs, batch_size):
    """
    Start with short sequences, gradually increase length.
    """
    # Determine max sequence length for this epoch
    min_length = 16
    max_length = 512
    current_max = int(min_length + (max_length - min_length) * (epoch / total_epochs))

    # Filter by length
    filtered_data = [(x, y) for x, y in dataset if len(x) <= current_max]

    return DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
```

### When to Use Curriculum Learning

**Good for:**
- Noisy datasets
- Very deep networks (helps convergence)
- Reinforcement learning
- Tasks with clear difficulty hierarchy

**Not necessary for:**
- Clean, well-curated datasets
- Standard image classification
- When data already well-ordered

**2025 Note:** Less commonly used than in past; modern techniques (batch normalization, better optimizers) reduce need.

---

## Transfer Learning and Fine-Tuning

### Transfer Learning Paradigm

**Core idea:** Leverage knowledge from pre-trained models.

**Workflow:**
```
1. Pre-trained model (trained on large dataset, e.g., ImageNet)
2. Remove final layer(s)
3. Add new layer(s) for target task
4. Fine-tune on target dataset
```

**Why transfer learning works:**
- Early layers learn general features (edges, textures)
- Later layers learn task-specific features
- Pre-trained weights are better initialization than random

### Fine-Tuning Strategies

**Strategy 1: Feature extraction (freeze backbone)**
```python
# Load pre-trained model
model = torchvision.models.resnet50(pretrained=True)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)  # Only this layer trainable

# Train only the new layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

**Strategy 2: Fine-tune all layers**
```python
# Load pre-trained model
model = torchvision.models.resnet50(pretrained=True)

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Train all parameters (but use small learning rate for pre-trained layers)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

**Strategy 3: Discriminative learning rates (recommended for fine-tuning)**
```python
# Load pre-trained model
model = torchvision.models.resnet50(pretrained=True)

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Different learning rates for different parts
optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},  # Early layers: smallest LR
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},  # Middle layers: medium LR
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}       # New layer: largest LR
])
```

**Strategy 4: Gradual unfreezing**
```python
def gradual_unfreezing(model, epoch, unfreeze_schedule):
    """
    Gradually unfreeze layers during training.

    Example schedule:
    Epoch 0-5: Only train final layer
    Epoch 5-10: Unfreeze layer4
    Epoch 10+: Unfreeze all
    """
    if epoch >= unfreeze_schedule['layer4']:
        for param in model.layer4.parameters():
            param.requires_grad = True

    if epoch >= unfreeze_schedule['layer3']:
        for param in model.layer3.parameters():
            param.requires_grad = True

    # etc.
```

### Learning Rate Guidelines for Fine-Tuning

| Dataset Size | Similarity to Pre-training | Strategy | Learning Rate |
|--------------|---------------------------|----------|---------------|
| Small | High | Feature extraction | N/A (freeze backbone) |
| Small | Low | Fine-tune top layers | 1e-5 to 1e-4 |
| Large | High | Fine-tune all layers | 1e-5 to 1e-4 |
| Large | Low | Fine-tune all layers | 1e-4 to 1e-3 |

**Rule of thumb (2025):**
- New layers: LR = 1e-3
- Pre-trained layers: LR = 1e-5 to 1e-4
- Use discriminative learning rates

### Modern Transfer Learning (2025)

**Vision Transformers (ViT):**
```python
import timm

# Load pre-trained ViT
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

# Fine-tune with very small learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Use warm-up + cosine decay
scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps=500, total_steps=10000)
```

**Language Models (BERT, GPT):**
```python
from transformers import BertForSequenceClassification, AdamW

# Load pre-trained BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Fine-tune all parameters
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Warm-up + linear decay (standard for transformers)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000
)
```

---

## Model Checkpointing

### Why Checkpointing?

**Essential for:**
1. Resume training after interruptions
2. Save best model during training
3. Model selection (try multiple checkpoints)
4. Debugging (inspect model at different stages)

### What to Save

**Minimum:**
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, checkpoint_path)
```

**Comprehensive (recommended):**
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'scaler_state_dict': scaler.state_dict() if scaler else None,  # For mixed precision
    'train_loss': train_loss,
    'val_loss': val_loss,
    'best_val_loss': best_val_loss,
    'hyperparameters': {
        'learning_rate': lr,
        'batch_size': batch_size,
        'model_architecture': model_arch,
        # etc.
    },
    'random_state': {
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }
}, checkpoint_path)
```

### Loading Checkpoints

```python
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """
    Load checkpoint and restore training state.
    """
    checkpoint = torch.load(checkpoint_path)

    # Restore model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore scheduler
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Restore scaler
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Restore random state (for reproducibility)
    if 'random_state' in checkpoint:
        torch.set_rng_state(checkpoint['random_state']['torch_rng_state'])
        np.random.set_state(checkpoint['random_state']['numpy_rng_state'])
        random.setstate(checkpoint['random_state']['python_rng_state'])

    return checkpoint['epoch'], checkpoint.get('best_val_loss', float('inf'))
```

### Checkpointing Strategies

**Strategy 1: Periodic checkpoints**
```python
# Save every N epochs
if epoch % save_every == 0:
    save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
```

**Strategy 2: Best model checkpoint**
```python
# Save only when validation improves
if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_checkpoint('best_model.pth')
```

**Strategy 3: Top-K checkpoints**
```python
# Keep only top K checkpoints by validation metric
checkpoint_queue = []  # (val_loss, path)

if len(checkpoint_queue) < K or val_loss < checkpoint_queue[-1][0]:
    # Save new checkpoint
    path = f'checkpoint_epoch_{epoch}.pth'
    save_checkpoint(path)
    checkpoint_queue.append((val_loss, path))
    checkpoint_queue.sort()

    # Remove worst checkpoint if queue full
    if len(checkpoint_queue) > K:
        _, path_to_remove = checkpoint_queue.pop()
        os.remove(path_to_remove)
```

**Strategy 4: Exponentially spaced checkpoints**
```python
# Save more frequently at start, less frequently later
def should_save(epoch):
    # Save at epochs: 1, 2, 4, 8, 16, 32, ...
    return epoch > 0 and (epoch & (epoch - 1)) == 0  # Check if power of 2
```

**2025 Best Practice:**
- Save best model (by validation metric)
- Save every N epochs for long training runs
- Keep last K checkpoints
- Include full training state for reproducibility

---

## Distributed Training

### Why Distributed Training?

**Benefits:**
1. **Faster training:** Utilize multiple GPUs
2. **Larger models:** Split across multiple GPUs
3. **Larger batches:** Scale to thousands of examples per batch

**Types:**
1. **Data parallelism:** Replicate model, split data
2. **Model parallelism:** Split model across GPUs
3. **Pipeline parallelism:** Split model into stages
4. **Hybrid:** Combine approaches

### Data Parallel (DP) - Simple but Limited

**Idea:** Replicate model on each GPU, split batch across GPUs.

```python
# Wrap model
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# Training loop (no changes needed)
for data, target in train_loader:
    output = model(data)  # Automatically split across GPUs
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**How it works:**
```
1. Split batch across GPUs
2. Forward pass on each GPU independently
3. Gather outputs to GPU 0
4. Compute loss on GPU 0
5. Broadcast loss to all GPUs
6. Backward pass on each GPU
7. Gather gradients to GPU 0
8. Update weights on GPU 0
9. Broadcast updated weights to all GPUs
```

**Limitations:**
- GPU 0 bottleneck (gathers all outputs and gradients)
- Inefficient communication
- Single-machine only
- **Not recommended for 2025** (use DDP instead)

### Distributed Data Parallel (DDP) - Recommended

**Idea:** Each GPU has its own process with model replica.

**Advantages over DP:**
- No GPU 0 bottleneck
- More efficient communication (ring-allreduce)
- Multi-machine support
- Better scaling

**Setup:**
```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    """Training function for each GPU."""
    setup(rank, world_size)

    # Create model and move to GPU
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # Create data sampler (ensures each GPU sees different data)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important! Ensures different shuffling each epoch

        for data, target in dataloader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
```

**Key concepts:**

**world_size:** Total number of processes (typically = number of GPUs)
**rank:** Process ID (0 to world_size - 1)
**DistributedSampler:** Ensures each GPU sees different subset of data

**Launching:**
```bash
# Single machine, 4 GPUs
python train.py

# Multiple machines (e.g., 2 machines, 4 GPUs each)
# On machine 0:
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=12355 train.py

# On machine 1:
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=12355 train.py
```

### Fully Sharded Data Parallel (FSDP) - For Very Large Models

**Problem:** DDP replicates entire model on each GPU (memory intensive).

**FSDP solution:** Shard model parameters, gradients, and optimizer states across GPUs.

**Use cases:**
- Models too large to fit on single GPU
- GPT-scale models (billions of parameters)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Wrap model with FSDP
model = FSDP(model)

# Training (same as DDP)
for data, target in dataloader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**2025 Note:** FSDP is becoming standard for training very large models.

---

## Hyperparameter Tuning

### Key Hyperparameters (Priority Order)

**Priority 1 (tune first):**
1. **Learning rate:** Most important!
2. **Batch size:** Affects training dynamics

**Priority 2:**
3. **Weight decay:** Regularization strength
4. **Number of layers / hidden units:** Model capacity
5. **Learning rate schedule:** Warm-up, decay

**Priority 3:**
6. **Dropout rate**
7. **Optimizer choice (Adam vs SGD vs AdamW)**
8. **Activation functions**

### Search Strategies

**1. Manual search:**
- Start with defaults
- Adjust one at a time
- Time-consuming but educational

**2. Grid search:**
```python
learning_rates = [1e-4, 1e-3, 1e-2]
weight_decays = [0, 1e-4, 1e-3]
batch_sizes = [32, 64, 128]

for lr in learning_rates:
    for wd in weight_decays:
        for bs in batch_sizes:
            model = train(lr=lr, weight_decay=wd, batch_size=bs)
            evaluate(model)
```

**Problems:**
- Exponential in number of hyperparameters
- Wastes compute on bad combinations

**3. Random search (recommended for initial exploration):**
```python
import random

def random_search(n_trials):
    for _ in range(n_trials):
        # Sample hyperparameters
        lr = 10 ** random.uniform(-5, -2)  # Log-uniform in [1e-5, 1e-2]
        wd = 10 ** random.uniform(-5, -2)
        batch_size = random.choice([32, 64, 128, 256])
        dropout = random.uniform(0.1, 0.5)

        # Train and evaluate
        model = train(lr=lr, weight_decay=wd, batch_size=batch_size, dropout=dropout)
        score = evaluate(model)

        # Track best
        if score > best_score:
            best_score = score
            best_hyperparameters = {...}

    return best_hyperparameters
```

**4. Bayesian optimization (recommended for expensive tuning):**
```python
from ax.service.managed_loop import optimize

def train_evaluate(parameters):
    """Objective function for Bayesian optimization."""
    model = train(
        lr=parameters['lr'],
        weight_decay=parameters['wd'],
        batch_size=parameters['batch_size']
    )
    return evaluate(model)

# Bayesian optimization
best_parameters, best_values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True},
        {"name": "wd", "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True},
        {"name": "batch_size", "type": "choice", "values": [32, 64, 128, 256]},
    ],
    evaluation_function=train_evaluate,
    objective_name="val_accuracy",
    total_trials=50
)
```

**2025 Best Practice:**
- Start with random search (20-50 trials)
- Use Bayesian optimization for refinement
- Focus on learning rate first
- Use learning rate finder before extensive tuning

### Hyperparameter Ranges (2025 Defaults)

| Hyperparameter | Typical Range | Search Space |
|----------------|---------------|--------------|
| Learning rate | 1e-5 to 1e-1 | Log-uniform |
| Weight decay | 0 to 1e-2 | Log-uniform |
| Batch size | 16 to 512 | Powers of 2 |
| Dropout | 0 to 0.5 | Uniform |
| Number of layers | 2 to 10 | Integer |
| Hidden units | 64 to 1024 | Powers of 2 |

---

## Complete Production Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from pathlib import Path
import logging
import json
from typing import Optional, Dict
import wandb  # For experiment tracking

class ProductionTrainer:
    """
    Production-ready training pipeline with all modern best practices:

    - Mixed precision training (FP16/BF16)
    - Distributed data parallel (multi-GPU)
    - Gradient accumulation
    - Learning rate scheduling with warm-up
    - Gradient clipping
    - Model checkpointing
    - Early stopping
    - Experiment tracking (W&B)
    - Reproducibility (seed setting)
    - Logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        rank: int = 0,
        world_size: int = 1
    ):
        self.config = config or self.get_default_config()
        self.rank = rank
        self.world_size = world_size

        # Device setup
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        # Distributed training setup
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Mixed precision
        self.use_amp = self.config['use_amp']
        self.scaler = GradScaler(enabled=self.use_amp)

        # Optimizer
        self.optimizer = self.create_optimizer()

        # Learning rate scheduler
        self.scheduler = self.create_scheduler()

        # Checkpointing
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Experiment tracking (only on rank 0)
        if rank == 0 and self.config['use_wandb']:
            wandb.init(
                project=self.config['project_name'],
                config=self.config
            )

        # Logging
        self.setup_logging()

        # Training state
        self.epoch = 0
        self.global_step = 0

    @staticmethod
    def get_default_config():
        """Default configuration."""
        return {
            # Optimization
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'betas': (0.9, 0.999),

            # Learning rate schedule
            'scheduler': 'cosine',
            'warmup_steps': 1000,
            'total_steps': 10000,
            'min_lr_ratio': 0.01,

            # Training
            'epochs': 100,
            'accumulation_steps': 1,
            'grad_clip': 1.0,
            'use_amp': True,

            # Regularization
            'dropout': 0.1,
            'label_smoothing': 0.1,

            # Checkpointing
            'checkpoint_dir': './checkpoints',
            'save_every': 10,
            'keep_last_k': 5,

            # Early stopping
            'early_stopping': True,
            'patience': 10,

            # Experiment tracking
            'use_wandb': True,
            'project_name': 'my-project',

            # Reproducibility
            'seed': 42,
        }

    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'train_rank_{self.rank}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_optimizer(self):
        """Create optimizer."""
        if self.config['optimizer'] == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                betas=self.config['betas']
            )
        elif self.config['optimizer'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay'],
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")

    def create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config['scheduler'] == 'cosine':
            return get_warmup_cosine_scheduler(
                self.optimizer,
                warmup_steps=self.config['warmup_steps'],
                total_steps=self.config['total_steps'],
                min_lr_ratio=self.config['min_lr_ratio']
            )
        elif self.config['scheduler'] == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['learning_rate'],
                epochs=self.config['epochs'],
                steps_per_epoch=len(self.train_loader) // self.config['accumulation_steps']
            )
        else:
            return None

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Mixed precision forward pass
            with autocast(enabled=self.use_amp):
                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss = loss / self.config['accumulation_steps']

            # Backward pass
            self.scaler.scale(loss).backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Learning rate scheduling
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            total_loss += loss.item() * self.config['accumulation_steps']
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = nn.functional.cross_entropy(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100. * correct / total
        }

    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save training checkpoint."""
        if self.rank != 0:
            return  # Only save on rank 0

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")

        for epoch in range(self.epoch, self.config['epochs']):
            self.epoch = epoch

            # Set epoch for distributed sampler (ensures different shuffling each epoch)
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            train_loss = self.train_epoch()

            # Evaluate
            val_metrics = self.evaluate()

            # Logging (only rank 0)
            if self.rank == 0:
                log_message = f"Epoch {epoch}/{self.config['epochs']} - Train Loss: {train_loss:.4f}"
                if val_metrics:
                    log_message += f" - Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['accuracy']:.2f}%"
                self.logger.info(log_message)

                # W&B logging
                if self.config['use_wandb']:
                    log_dict = {
                        'train/loss': train_loss,
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch
                    }
                    if val_metrics:
                        log_dict['val/loss'] = val_metrics['loss']
                        log_dict['val/accuracy'] = val_metrics['accuracy']
                    wandb.log(log_dict, step=self.global_step)

            # Checkpointing
            if epoch % self.config['save_every'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            # Early stopping
            if val_metrics and self.config['early_stopping']:
                val_loss = val_metrics['loss']

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pth', is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config['patience']:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break

        self.logger.info("Training complete!")

        # Cleanup
        if self.rank == 0 and self.config['use_wandb']:
            wandb.finish()


# Helper function for warm-up + cosine schedule
def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    """Warm-up followed by cosine annealing."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

---

## 2025 Best Practices

### Essential Checklist

**Before training:**
- [ ] Set random seeds for reproducibility
- [ ] Use mixed precision training (BF16 if available)
- [ ] Enable gradient clipping (max_norm=1.0)
- [ ] Set up proper logging and experiment tracking
- [ ] Validate data loading pipeline

**During training:**
- [ ] Monitor training and validation losses
- [ ] Watch for NaN/Inf values
- [ ] Check gradient norms (detect exploding/vanishing)
- [ ] Verify GPU utilization (should be 90%+)
- [ ] Save checkpoints regularly

**After training:**
- [ ] Evaluate on test set (never used during training!)
- [ ] Compare to baselines
- [ ] Analyze failure cases
- [ ] Save best model and configuration

### Typical Training Time Estimates (2025)

**On single RTX 4090 (24GB):**

| Task | Model | Dataset | Batch Size | Time |
|------|-------|---------|------------|------|
| Image Classification | ResNet-50 | ImageNet | 256 | ~24 hours |
| Image Classification | ViT-Base | ImageNet | 512 | ~48 hours |
| Language Modeling | GPT-2 (124M) | OpenWebText | 32 | ~7 days |
| Fine-tuning | BERT-Base | GLUE | 32 | ~2 hours |

**With 8x A100 (80GB):**
- 5-10× faster (depending on communication overhead)

### Common Training Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Exploding gradients** | Loss → NaN, weights → Inf | Gradient clipping, lower LR |
| **Vanishing gradients** | No learning, loss plateaus | Better initialization, residual connections |
| **Slow convergence** | Loss decreases very slowly | Higher LR, better optimizer (AdamW) |
| **Overfitting** | Train loss << Val loss | Regularization, data augmentation |
| **Underfitting** | High train and val loss | Larger model, train longer |
| **OOM (Out of Memory)** | CUDA out of memory error | Smaller batch, gradient accumulation, mixed precision |

---

## Summary

**Key takeaways for production deep learning (2025):**

1. **Mixed precision is standard:** Always use FP16/BF16
2. **AdamW is default optimizer:** With cosine annealing + warm-up
3. **Gradient accumulation for large batches:** When batch doesn't fit in memory
4. **DDP for multi-GPU:** Not DataParallel
5. **Comprehensive checkpointing:** Save everything needed to resume
6. **Experiment tracking:** Use W&B or similar
7. **Reproducibility:** Set seeds, save configs

**Modern training pipeline:**
```
Mixed Precision + Gradient Accumulation + DDP +
AdamW + Cosine Annealing + Warm-up +
Gradient Clipping + Checkpointing +
Experiment Tracking
```

**This is the 2025 standard for production deep learning.**

---

## References

1. Smith (2017). "Cyclical Learning Rates for Training Neural Networks"
2. Smith (2018). "A disciplined approach to neural network hyper-parameters"
3. Micikevicius et al. (2018). "Mixed Precision Training"
4. Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization"
5. Li et al. (2020). "PyTorch Distributed: Experiences on Accelerating Data Parallel Training"
6. Bengio et al. (2009). "Curriculum Learning"
7. Yosinski et al. (2014). "How transferable are features in deep neural networks?"
