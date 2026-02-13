# Transfer Learning and Domain Adaptation

Transfer learning leverages knowledge gained from solving one task (source) to improve performance on a related but different task (target). This is one of the most impactful techniques in modern machine learning, enabling state-of-the-art results even with limited labeled data.

## Table of Contents
1. [Fundamentals of Transfer Learning](#fundamentals)
2. [Computer Vision Transfer Learning](#computer-vision)
3. [NLP Transfer Learning](#nlp-transfer-learning)
4. [Domain Adaptation](#domain-adaptation)
5. [Few-Shot Learning](#few-shot-learning)
6. [Zero-Shot Learning](#zero-shot-learning)
7. [Production Pipelines](#production-pipelines)
8. [When to Use Transfer Learning](#when-to-use)

---

## Fundamentals of Transfer Learning {#fundamentals}

### Why Transfer Learning Works

**Intuition**: Early layers of neural networks learn general features (edges, textures, basic patterns) that transfer across tasks, while later layers learn task-specific features.

**Mathematical Framework**: Given source domain DS with task TS and target domain DT with task TT:

- **Domain**: D = {X, P(X)} where X is feature space and P(X) is marginal distribution
- **Task**: T = {Y, f(.)} where Y is label space and f(.) is objective function

**Types of Transfer**:
1. **Inductive Transfer**: Different tasks (TS != TT), source data available
2. **Transductive Transfer**: Same task (TS = TT), different domains (DS != DT)
3. **Unsupervised Transfer**: No labeled data in source or target

### Transfer Learning Taxonomy

```
Transfer Learning
+---- Feature Extraction (Frozen pretrained features)
+---- Fine-Tuning (Update pretrained weights)
|   +---- Full fine-tuning (all parameters)
|   +---- Partial fine-tuning (selected layers)
|   +---- Parameter-efficient fine-tuning (LoRA, adapters)
+---- Domain Adaptation (Align distributions)
|   +---- Feature-based (MMD, CORAL)
|   +---- Adversarial (DANN, ADDA)
+---- Multi-task Learning (Joint training)
```

### Mathematical Foundation: Inductive Bias Transfer

The success of transfer learning relies on shared inductive biases:

**Hypothesis**: Let h*_S be optimal hypothesis for source task and h*_T for target task. Transfer learning assumes:

```
d(h*_S, h*_T) << d(h_random, h*_T)
```

Where d(.,.) is distance in hypothesis space. Starting from h*_S requires less optimization than starting from random initialization.

**Generalization Bound**: For target domain with n samples:

```
Error_T(h) <= Error_S(h) + d(DS, DT) + O(sqrt(1/n))
```

Where d(DS, DT) is distribution discrepancy (e.g., H-divergence).

---

## Computer Vision Transfer Learning {#computer-vision}

### ImageNet Pretraining

ImageNet (14M images, 1000 classes) provides powerful pretrained features for vision tasks.

**Why ImageNet Features Transfer**:
- Layer 1: Gabor filters, edge detectors (universal)
- Layer 2-3: Textures, patterns (broadly applicable)
- Layer 4-5: Object parts (domain-dependent)
- Final layers: ImageNet-specific (replace for new tasks)

### Feature Extraction (Frozen Backbone)

**Approach**: Use pretrained CNN as fixed feature extractor, train only new classifier head.

**When to Use**:
- Target dataset is small (< 10K images)
- Target domain similar to ImageNet
- Limited computational resources

```python
import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    """Feature extraction with frozen pretrained ResNet."""
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)

        # Freeze all parameters
        for param in resnet.parameters():
            param.requires_grad = False

        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Add new classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features (no gradients)
        with torch.no_grad():
            features = self.features(x)
        # Classify (with gradients)
        return self.classifier(features)

# Usage
model = FeatureExtractor(num_classes=10)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
```

### Fine-Tuning Strategies

**Approach**: Update pretrained weights with small learning rate on target task.

**When to Use**:
- Target dataset is medium-large (> 10K images)
- Target domain differs from ImageNet
- Want to adapt low-level features

#### Discriminative Fine-Tuning (Layer-wise Learning Rates)

**Idea**: Use different learning rates for different layers (smaller for early layers, larger for later layers).

```python
class FineTunedModel(nn.Module):
    """Fine-tuning with layer-wise learning rates."""
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)

        # Replace final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Layer-wise learning rate setup
def get_optimizer_grouped(model, base_lr=1e-3):
    """Create optimizer with discriminative learning rates."""
    # Group parameters by layer depth
    layer_groups = [
        list(model.resnet.conv1.parameters()) +
        list(model.resnet.bn1.parameters()),  # Early layers
        list(model.resnet.layer1.parameters()),  # Mid layers
        list(model.resnet.layer2.parameters()),
        list(model.resnet.layer3.parameters()),
        list(model.resnet.layer4.parameters()),  # Deep layers
        list(model.resnet.fc.parameters())  # New head
    ]

    # Learning rates: 0.1x, 0.2x, 0.4x, 0.6x, 0.8x, 1x of base_lr
    lr_multipliers = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    param_groups = [
        {'params': params, 'lr': base_lr * mult}
        for params, mult in zip(layer_groups, lr_multipliers)
    ]

    return torch.optim.Adam(param_groups)

# Usage
model = FineTunedModel(num_classes=10)
optimizer = get_optimizer_grouped(model, base_lr=1e-4)
```

#### Gradual Unfreezing

**Approach**: Progressively unfreeze layers from top to bottom during training.

```python
def gradual_unfreezing_schedule(model, optimizer, epoch, unfreeze_epochs=[5, 10, 15, 20]):
    """Unfreeze layers progressively during training."""
    resnet_layers = [
        model.resnet.layer4,
        model.resnet.layer3,
        model.resnet.layer2,
        model.resnet.layer1
    ]

    for i, unfreeze_epoch in enumerate(unfreeze_epochs):
        if epoch == unfreeze_epoch:
            # Unfreeze this layer
            for param in resnet_layers[i].parameters():
                param.requires_grad = True

            # Add to optimizer
            optimizer.add_param_group({
                'params': resnet_layers[i].parameters(),
                'lr': 1e-4 * (0.5 ** i)  # Smaller LR for earlier layers
            })

            print(f"Epoch {epoch}: Unfroze layer {4-i}")

# Usage during training
for epoch in range(num_epochs):
    gradual_unfreezing_schedule(model, optimizer, epoch)
    train_epoch(model, train_loader, optimizer)
```

### Advanced: Stochastic Weight Averaging (SWA) for Fine-Tuning

**Idea**: Average weights from multiple points along training trajectory for better generalization.

```python
from torch.optim.swa_utils import AveragedModel, SWALR

# Wrap model for SWA
swa_model = AveragedModel(model)

# SWA learning rate scheduler
swa_scheduler = SWALR(optimizer, swa_lr=1e-5)

# Training loop
for epoch in range(num_epochs):
    train_epoch(model, train_loader, optimizer)

    if epoch > swa_start_epoch:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

# Update batch norm statistics for SWA model
torch.optim.swa_utils.update_bn(train_loader, swa_model)

# Use swa_model for inference
```

---

## NLP Transfer Learning {#nlp-transfer-learning}

### Pretrained Language Models

**Evolution**:
- 2018: BERT, GPT (transformer-based pretraining)
- 2019: RoBERTa, ALBERT, T5 (improved training)
- 2020: GPT-3 (175B parameters, few-shot learning)
- 2022: ChatGPT, GPT-4 (RLHF, instruction tuning)
- 2025: Multimodal foundation models (GPT-4V, Gemini)

### Fine-Tuning BERT for Classification

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class BERTClassifier(nn.Module):
    """BERT fine-tuning for text classification."""
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation
        pooled_output = outputs.pooler_output  # [batch, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Training setup
model = BERTClassifier(num_classes=3)

# Different learning rates for BERT and classifier
optimizer = torch.optim.AdamW([
    {'params': model.bert.parameters(), 'lr': 2e-5},  # Small LR for BERT
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # Larger LR for new head
], weight_decay=0.01)

# Training loop
def train_step(model, batch, optimizer):
    model.train()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    optimizer.zero_grad()
    logits = model(input_ids, attention_mask)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    # Gradient clipping (important for transformers)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    return loss.item()
```

### Parameter-Efficient Fine-Tuning (PEFT) - 2025 Best Practice

**Problem**: Fine-tuning large language models (LLMs) with billions of parameters is expensive.

**Solution**: Update only a small subset of parameters while keeping most frozen.

#### LoRA (Low-Rank Adaptation)

**Idea**: Instead of updating full weight matrix W in R^(dxk), learn low-rank decomposition:

```
W' = W + DeltaW = W + BA
```

Where B in R^(dxr), A in R^(rxk), and r << min(d, k).

**Mathematical Derivation**:

Original forward pass: h = Wx

LoRA forward pass:
```
h = Wx + BAx = Wx + B(Ax)
```

Number of trainable parameters:
- Original: d x k
- LoRA: (d + k) x r
- Reduction factor: (d x k) / ((d + k) x r) ~= k/r for d ~= k

For r = 8, d = k = 4096: Reduction = 512x fewer parameters!

**PyTorch Implementation**:

```python
class LoRALayer(nn.Module):
    """LoRA: Low-Rank Adaptation layer."""
    def __init__(self, in_features, out_features, rank=8, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # Low-rank matrices (trainable)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Scaling factor
        self.scaling = alpha / rank

    def forward(self, x):
        # x: [batch, seq_len, in_features]
        # Low-rank adaptation
        return (x @ self.lora_A @ self.lora_B) * self.scaling

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    def __init__(self, linear_layer, rank=8, alpha=32):
        super().__init__()
        # Original layer (frozen)
        self.linear = linear_layer
        for param in self.linear.parameters():
            param.requires_grad = False

        # LoRA adaptation
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

# Apply LoRA to model
def add_lora_to_model(model, target_modules=['query', 'value'], rank=8):
    """Add LoRA layers to target modules in transformer."""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA version
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, LoRALinear(module, rank=rank))

    return model

# Usage with Hugging Face PEFT library (recommended)
from peft import get_peft_model, LoraConfig, TaskType

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification
    r=8,  # Rank
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,
    target_modules=["query", "value"],  # Apply to attention Q,V
    bias="none"
)

# Wrap model
model = AutoModel.from_pretrained("bert-base-uncased")
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 109,483,778 || trainable%: 0.27%
```

#### QLoRA (Quantized LoRA) - 2025 Efficiency

**Idea**: Combine LoRA with 4-bit quantization for extreme memory efficiency.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Nested quantization
    bnb_4bit_quant_type="nf4",  # 4-bit NormalFloat
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute in bf16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Add LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Result: Fine-tune 7B model on single GPU with 16GB VRAM!
```

#### Adapters

**Idea**: Insert small trainable bottleneck layers between frozen transformer layers.

```python
class AdapterLayer(nn.Module):
    """Adapter layer for parameter-efficient fine-tuning."""
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_size, hidden_size)

        # Initialize near identity
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual  # Residual connection

# Insert adapters into transformer block
class TransformerBlockWithAdapter(nn.Module):
    def __init__(self, transformer_block, adapter_size=64):
        super().__init__()
        self.attention = transformer_block.attention
        self.adapter_1 = AdapterLayer(transformer_block.config.hidden_size, adapter_size)
        self.feed_forward = transformer_block.feed_forward
        self.adapter_2 = AdapterLayer(transformer_block.config.hidden_size, adapter_size)

        # Freeze original parameters
        for param in self.attention.parameters():
            param.requires_grad = False
        for param in self.feed_forward.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Attention + adapter
        x = self.attention(x)
        x = self.adapter_1(x)

        # FFN + adapter
        x = self.feed_forward(x)
        x = self.adapter_2(x)
        return x
```

#### Prefix Tuning and Prompt Tuning

**Prefix Tuning**: Prepend trainable continuous vectors to each layer's input.

```python
class PrefixTuning(nn.Module):
    """Prefix tuning for transformer models."""
    def __init__(self, num_layers, hidden_size, prefix_length=20):
        super().__init__()
        self.prefix_length = prefix_length

        # Trainable prefix tokens for each layer
        self.prefix_tokens = nn.Parameter(
            torch.randn(num_layers, prefix_length, hidden_size) * 0.01
        )

    def get_prefix(self, layer_idx, batch_size):
        # Expand prefix for batch
        prefix = self.prefix_tokens[layer_idx].unsqueeze(0)
        return prefix.expand(batch_size, -1, -1)

# Usage in transformer forward pass
def forward_with_prefix(transformer, x, prefix_tuning):
    batch_size = x.size(0)

    for layer_idx, layer in enumerate(transformer.layers):
        # Get prefix for this layer
        prefix = prefix_tuning.get_prefix(layer_idx, batch_size)

        # Concatenate prefix with input
        x_with_prefix = torch.cat([prefix, x], dim=1)

        # Forward pass
        x = layer(x_with_prefix)

        # Remove prefix from output
        x = x[:, prefix_tuning.prefix_length:, :]

    return x
```

**Prompt Tuning**: Learn task-specific soft prompts (simpler than prefix tuning).

```python
class PromptTuning(nn.Module):
    """Soft prompt tuning."""
    def __init__(self, vocab_size, hidden_size, num_prompt_tokens=10):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens

        # Trainable soft prompt embeddings
        self.soft_prompt = nn.Parameter(
            torch.randn(num_prompt_tokens, hidden_size) * 0.01
        )

    def forward(self, input_embeds):
        # input_embeds: [batch, seq_len, hidden_size]
        batch_size = input_embeds.size(0)

        # Expand soft prompt for batch
        prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # Prepend to input
        return torch.cat([prompt, input_embeds], dim=1)

# Usage
prompt_tuning = PromptTuning(vocab_size=50000, hidden_size=768, num_prompt_tokens=10)
embeddings = model.get_input_embeddings()(input_ids)
embeddings_with_prompt = prompt_tuning(embeddings)
outputs = model(inputs_embeds=embeddings_with_prompt)
```

---

## Domain Adaptation {#domain-adaptation}

**Problem**: Source and target domains have different distributions: P_S(X) != P_T(X).

**Goal**: Learn representations that work well on target domain using labeled source data and unlabeled target data.

### Maximum Mean Discrepancy (MMD)

**Idea**: Minimize distance between source and target feature distributions.

**Mathematical Formulation**:

```
MMD^2(P_S, P_T) = ||E_S[phi(x)] - E_T[phi(x)]||^2_H
```

Where phi(.) maps to reproducing kernel Hilbert space (RKHS), and ||.||_H is RKHS norm.

**Empirical Estimator** (using Gaussian kernel):

```
MMD^2(S, T) = (1/n_s^2) sumsum k(x_i, x_j) + (1/n_t^2) sumsum k(x'_i, x'_j) - (2/n_s n_t) sumsum k(x_i, x'_j)
```

Where k(x, y) = exp(-||x - y||^2/(2sigma^2)) is RBF kernel.

```python
def mmd_loss(source_features, target_features, kernel='rbf', gamma=1.0):
    """
    Compute Maximum Mean Discrepancy between source and target.

    Args:
        source_features: [n_s, d] tensor
        target_features: [n_t, d] tensor
        kernel: 'rbf' or 'linear'
        gamma: kernel bandwidth
    """
    n_s = source_features.size(0)
    n_t = target_features.size(0)

    if kernel == 'rbf':
        # Compute pairwise distances
        def rbf_kernel(x, y, gamma):
            dist = torch.cdist(x, y, p=2)  # Euclidean distance
            return torch.exp(-gamma * dist ** 2)

        # K_ss: source-source kernel
        K_ss = rbf_kernel(source_features, source_features, gamma)
        # K_tt: target-target kernel
        K_tt = rbf_kernel(target_features, target_features, gamma)
        # K_st: source-target kernel
        K_st = rbf_kernel(source_features, target_features, gamma)
    else:  # linear
        K_ss = source_features @ source_features.T
        K_tt = target_features @ target_features.T
        K_st = source_features @ target_features.T

    # MMD^2 = E_ss + E_tt - 2*E_st
    mmd_squared = (K_ss.sum() / (n_s ** 2) +
                   K_tt.sum() / (n_t ** 2) -
                   2 * K_st.sum() / (n_s * n_t))

    return mmd_squared

# Usage in training
def train_with_mmd(model, source_loader, target_loader, mmd_weight=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
        # Forward pass
        source_features = model.feature_extractor(source_data)
        source_logits = model.classifier(source_features)

        target_features = model.feature_extractor(target_data)

        # Classification loss (supervised on source)
        cls_loss = nn.CrossEntropyLoss()(source_logits, source_labels)

        # Domain adaptation loss (MMD)
        mmd = mmd_loss(source_features, target_features)

        # Combined loss
        loss = cls_loss + mmd_weight * mmd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### CORAL (Correlation Alignment)

**Idea**: Align second-order statistics (covariance) of source and target features.

**Loss**:
```
L_CORAL = (1/4d^2) ||C_S - C_T||^2_F
```

Where C_S, C_T are covariance matrices and ||.||_F is Frobenius norm.

```python
def coral_loss(source_features, target_features):
    """
    CORAL loss: align covariance matrices.

    Args:
        source_features: [n_s, d]
        target_features: [n_t, d]
    """
    d = source_features.size(1)

    # Center features
    source_mean = source_features.mean(0, keepdim=True)
    target_mean = target_features.mean(0, keepdim=True)
    source_centered = source_features - source_mean
    target_centered = target_features - target_mean

    # Covariance matrices
    cov_source = (source_centered.T @ source_centered) / (source_features.size(0) - 1)
    cov_target = (target_centered.T @ target_centered) / (target_features.size(0) - 1)

    # Frobenius norm of difference
    loss = ((cov_source - cov_target) ** 2).sum() / (4 * d ** 2)

    return loss
```

### Adversarial Domain Adaptation (DANN)

**Idea**: Learn features that are discriminative for task but invariant to domain.

**Architecture**:
```
Input --> Feature Extractor --> Task Classifier (maximize task accuracy)
                         \ Domain Classifier (maximize domain confusion)
```

**Gradient Reversal Layer** (GRL): Reverses gradients during backpropagation.

```python
from torch.autograd import Function

class GradientReversalFunction(Function):
    """Gradient Reversal Layer for adversarial domain adaptation."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class DANN(nn.Module):
    """Domain-Adversarial Neural Network."""
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()

        # Feature extractor (shared)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )

        # Domain classifier with gradient reversal
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=1.0),
            nn.Linear(hidden_dim, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 2)  # Binary: source vs target
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        task_output = self.task_classifier(features)

        # Update GRL lambda (gradually increase during training)
        self.domain_classifier[0].lambda_ = alpha
        domain_output = self.domain_classifier(features)

        return task_output, domain_output

# Training loop
def train_dann(model, source_loader, target_loader, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        # Gradually increase domain adaptation strength
        p = epoch / num_epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # Ramp from 0 to 1

        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
            batch_size = source_data.size(0)

            # Combine source and target
            combined_data = torch.cat([source_data, target_data], dim=0)

            # Domain labels: 0 for source, 1 for target
            domain_labels = torch.cat([
                torch.zeros(batch_size),
                torch.ones(batch_size)
            ]).long().to(source_data.device)

            # Forward pass
            task_output, domain_output = model(combined_data, alpha)

            # Task loss (only on source)
            source_task_output = task_output[:batch_size]
            task_loss = nn.CrossEntropyLoss()(source_task_output, source_labels)

            # Domain loss (on both source and target)
            domain_loss = nn.CrossEntropyLoss()(domain_output, domain_labels)

            # Combined loss
            loss = task_loss + domain_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## Few-Shot Learning {#few-shot-learning}

**Problem**: Learn from very few examples per class (e.g., 1-shot, 5-shot).

**Notation**: N-way K-shot learning = classify N classes with K examples each.

### Prototypical Networks

**Idea**: Represent each class by prototype (mean of support examples) in embedding space. Classify query points by nearest prototype.

**Algorithm**:
1. Embed support set: {f_theta(x_i)}
2. Compute class prototypes: c_k = (1/K) sum_{iinS_k} f_theta(x_i)
3. Classify query by distance to prototypes: y = argmin_k d(f_theta(x_q), c_k)

```python
class PrototypicalNetwork(nn.Module):
    """Prototypical Networks for few-shot learning."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder  # Feature encoder

    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        """
        Compute class prototypes from support set.

        Args:
            support_embeddings: [n_way * k_shot, embed_dim]
            support_labels: [n_way * k_shot]
            n_way: number of classes
        Returns:
            prototypes: [n_way, embed_dim]
        """
        prototypes = []
        for class_idx in range(n_way):
            # Get embeddings for this class
            class_mask = (support_labels == class_idx)
            class_embeddings = support_embeddings[class_mask]
            # Compute mean (prototype)
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)

    def forward(self, support_images, support_labels, query_images, n_way, k_shot):
        """
        Args:
            support_images: [n_way * k_shot, C, H, W]
            support_labels: [n_way * k_shot]
            query_images: [n_query, C, H, W]
        Returns:
            logits: [n_query, n_way]
        """
        # Encode support and query
        support_embeddings = self.encoder(support_images)  # [n_way*k_shot, embed_dim]
        query_embeddings = self.encoder(query_images)      # [n_query, embed_dim]

        # Compute prototypes
        prototypes = self.compute_prototypes(
            support_embeddings, support_labels, n_way
        )  # [n_way, embed_dim]

        # Compute distances (negative squared Euclidean)
        distances = torch.cdist(query_embeddings, prototypes) ** 2  # [n_query, n_way]
        logits = -distances  # Higher score = closer to prototype

        return logits

# Simple CNN encoder
class ConvEncoder(nn.Module):
    """4-layer CNN encoder for few-shot learning."""
    def __init__(self, in_channels=3, hidden_dim=64, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, embed_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.encoder(x)

# Training episode
def train_episode(model, data_loader, n_way=5, k_shot=5, n_query=15):
    """Train on one episode (task)."""
    # Sample episode
    support_images, support_labels, query_images, query_labels = sample_episode(
        data_loader, n_way, k_shot, n_query
    )

    # Forward pass
    logits = model(support_images, support_labels, query_images, n_way, k_shot)

    # Compute loss
    loss = nn.CrossEntropyLoss()(logits, query_labels)

    return loss
```

### Matching Networks

**Idea**: Use attention mechanism over support set for classification.

**Classifier**:
```
P(y=k | x_q, S) = sum_{i=1}^{|S|} a(x_q, x_i) * y_i
```

Where a(x_q, x_i) is attention weight (cosine similarity).

```python
class MatchingNetwork(nn.Module):
    """Matching Networks with full context embeddings."""
    def __init__(self, encoder, lstm_hidden=128):
        super().__init__()
        self.encoder = encoder
        self.lstm = nn.LSTM(
            encoder.embed_dim,
            lstm_hidden,
            bidirectional=True,
            batch_first=True
        )

    def full_context_embeddings(self, embeddings):
        """Enhance embeddings with LSTM over support set."""
        lstm_out, _ = self.lstm(embeddings.unsqueeze(0))
        return lstm_out.squeeze(0)

    def forward(self, support_images, support_labels, query_images, n_way):
        # Encode
        support_embeds = self.encoder(support_images)
        query_embeds = self.encoder(query_images)

        # Full context embeddings (attend over support set)
        support_embeds = self.full_context_embeddings(support_embeds)

        # Cosine similarity attention
        support_norm = F.normalize(support_embeds, p=2, dim=1)
        query_norm = F.normalize(query_embeds, p=2, dim=1)

        # Attention: [n_query, n_support]
        attention = query_norm @ support_norm.T
        attention = F.softmax(attention, dim=1)

        # Weighted combination of support labels
        support_labels_onehot = F.one_hot(support_labels, n_way).float()
        logits = attention @ support_labels_onehot  # [n_query, n_way]

        return logits
```

### Siamese Networks

**Idea**: Learn similarity metric between pairs of examples.

```python
class SiameseNetwork(nn.Module):
    """Siamese network for one-shot learning."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        # Similarity predictor
        self.similarity = nn.Sequential(
            nn.Linear(encoder.embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        """Predict if x1 and x2 are from same class."""
        # Encode both inputs
        embed1 = self.encoder(x1)
        embed2 = self.encoder(x2)

        # Concatenate embeddings
        combined = torch.cat([embed1, embed2], dim=1)

        # Predict similarity
        similarity_score = self.similarity(combined)
        return similarity_score

    def predict_class(self, query, support_images, support_labels):
        """Classify query by comparing to support set."""
        n_support = support_images.size(0)

        # Repeat query for each support example
        query_repeated = query.unsqueeze(0).repeat(n_support, 1, 1, 1)

        # Compute similarity with each support example
        similarities = self.forward(query_repeated, support_images).squeeze()

        # Find most similar support example
        best_match = similarities.argmax()
        predicted_label = support_labels[best_match]

        return predicted_label

# Contrastive loss for training
def contrastive_loss(similarity, label, margin=1.0):
    """
    Args:
        similarity: predicted similarity [0, 1]
        label: 1 if same class, 0 if different
        margin: margin for dissimilar pairs
    """
    loss_same = label * (1 - similarity) ** 2
    loss_diff = (1 - label) * F.relu(similarity - margin) ** 2
    return (loss_same + loss_diff).mean()
```

---

## Zero-Shot Learning {#zero-shot-learning}

**Problem**: Classify classes never seen during training.

**Key**: Use semantic embeddings (attributes, word vectors) to transfer knowledge.

### CLIP (Contrastive Language-Image Pre-training) - 2025 Standard

**Idea**: Learn joint embedding space for images and text. Zero-shot classification via text prompts.

**Architecture**:
```
Image Encoder (Vision Transformer) --> Image Embedding
Text Encoder (Transformer)          --> Text Embedding

Loss: Contrastive (match paired image-text, separate unpaired)
```

**Mathematical Formulation**:

Contrastive loss (InfoNCE):
```
L = -log(exp(sim(I_i, T_i) / tau) / sum_j exp(sim(I_i, T_j) / tau))
```

Where sim(I, T) = cosine(f_I(I), f_T(T)) and tau is temperature.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class CLIPZeroShot:
    """Zero-shot classification with CLIP."""
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def classify(self, image, candidate_labels):
        """
        Zero-shot classification.

        Args:
            image: PIL Image
            candidate_labels: List of class names
        Returns:
            probabilities: [num_classes]
        """
        # Create text prompts
        text_prompts = [f"a photo of a {label}" for label in candidate_labels]

        # Process inputs
        inputs = self.processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Get embeddings
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds  # [1, embed_dim]
        text_embeds = outputs.text_embeds    # [num_classes, embed_dim]

        # Compute similarity (cosine)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        similarity = (image_embeds @ text_embeds.T).squeeze()  # [num_classes]

        # Convert to probabilities
        probabilities = F.softmax(similarity / 0.01, dim=0)  # Temperature=0.01

        return probabilities

    @torch.no_grad()
    def retrieve_images(self, text_query, image_dataset, top_k=5):
        """Text-to-image retrieval."""
        # Encode text
        text_inputs = self.processor(text=[text_query], return_tensors="pt")
        text_embeds = self.model.get_text_features(**text_inputs)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        # Encode all images
        all_image_embeds = []
        for image in image_dataset:
            image_inputs = self.processor(images=image, return_tensors="pt")
            image_embeds = self.model.get_image_features(**image_inputs)
            all_image_embeds.append(image_embeds)

        all_image_embeds = torch.cat(all_image_embeds, dim=0)
        all_image_embeds = F.normalize(all_image_embeds, p=2, dim=-1)

        # Compute similarities
        similarities = (text_embeds @ all_image_embeds.T).squeeze()

        # Get top-k
        top_indices = similarities.topk(top_k).indices
        return top_indices, similarities[top_indices]

# Training CLIP from scratch (simplified)
class SimpleCLIP(nn.Module):
    """Simplified CLIP implementation."""
    def __init__(self, image_encoder, text_encoder, embed_dim=512, temperature=0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Projection heads
        self.image_projection = nn.Linear(image_encoder.output_dim, embed_dim)
        self.text_projection = nn.Linear(text_encoder.output_dim, embed_dim)

    def encode_image(self, images):
        image_features = self.image_encoder(images)
        image_embeds = self.image_projection(image_features)
        return F.normalize(image_embeds, p=2, dim=-1)

    def encode_text(self, text_tokens):
        text_features = self.text_encoder(text_tokens)
        text_embeds = self.text_projection(text_features)
        return F.normalize(text_embeds, p=2, dim=-1)

    def forward(self, images, text_tokens):
        # Encode
        image_embeds = self.encode_image(images)  # [batch, embed_dim]
        text_embeds = self.encode_text(text_tokens)  # [batch, embed_dim]

        # Cosine similarity
        logits = (image_embeds @ text_embeds.T) / self.temperature

        # Symmetric cross-entropy loss
        batch_size = images.size(0)
        labels = torch.arange(batch_size, device=images.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        return loss

# Usage
clip_zeroshot = CLIPZeroShot()
probs = clip_zeroshot.classify(
    image=pil_image,
    candidate_labels=["dog", "cat", "bird", "car", "airplane"]
)
print(f"Predicted class: {candidate_labels[probs.argmax()]}")
```

---

## Production Pipelines {#production-pipelines}

### Complete Fine-Tuning Pipeline

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig
import wandb

class ProductionFineTuner:
    """Production-ready fine-tuning pipeline."""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = self._init_model()

        # Initialize optimizer and scheduler
        self.optimizer = self._init_optimizer()
        self.scheduler = None

        # Metrics tracking
        self.best_val_metric = float('-inf')

    def _init_model(self):
        """Initialize model with optional PEFT."""
        base_model = AutoModel.from_pretrained(self.config.model_name)

        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout
            )
            model = get_peft_model(base_model, lora_config)
            print("Using LoRA fine-tuning")
            model.print_trainable_parameters()
        else:
            model = base_model

        # Add task head
        model.classifier = nn.Linear(
            model.config.hidden_size,
            self.config.num_classes
        )

        return model.to(self.device)

    def _init_optimizer(self):
        """Initialize optimizer with layer-wise learning rates."""
        if self.config.use_discriminative_lr:
            param_groups = self._get_layer_wise_params()
        else:
            param_groups = [{'params': self.model.parameters()}]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        return optimizer

    def _get_layer_wise_params(self):
        """Group parameters by layer for discriminative learning rates."""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if 'classifier' in n and not any(nd in n for nd in no_decay)],
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if 'classifier' in n and any(nd in n for nd in no_decay)],
                'lr': self.config.learning_rate,
                'weight_decay': 0.0
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if 'classifier' not in n and not any(nd in n for nd in no_decay)],
                'lr': self.config.learning_rate * 0.1,  # Smaller for pretrained
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if 'classifier' not in n and any(nd in n for nd in no_decay)],
                'lr': self.config.learning_rate * 0.1,
                'weight_decay': 0.0
            }
        ]
        return optimizer_grouped_parameters

    def train(self, train_loader, val_loader):
        """Complete training loop."""
        # Initialize scheduler
        num_training_steps = len(train_loader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Initialize experiment tracking
        if self.config.use_wandb:
            wandb.init(project=self.config.project_name, config=vars(self.config))

        for epoch in range(self.config.num_epochs):
            # Training
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validation
            val_metrics = self._validate(val_loader)

            # Logging
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Save best model
            if val_metrics['accuracy'] > self.best_val_metric:
                self.best_val_metric = val_metrics['accuracy']
                self._save_checkpoint(epoch, val_metrics)

        if self.config.use_wandb:
            wandb.finish()

    def _train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = self.model.classifier(outputs.pooler_output)

            # Compute loss
            loss = F.cross_entropy(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Logging
            if batch_idx % self.config.log_interval == 0:
                print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100.*correct/total:.2f}%')

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }

    def _validate(self, val_loader):
        """Validation loop."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = self.model.classifier(outputs.pooler_output)
                loss = F.cross_entropy(logits, labels)

                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }

    def _save_checkpoint(self, epoch, metrics):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': vars(self.config)
        }

        path = f"{self.config.save_dir}/best_model.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def _log_metrics(self, epoch, train_metrics, val_metrics):
        """Log metrics to console and W&B."""
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}")

        if self.config.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'learning_rate': self.scheduler.get_last_lr()[0]
            })
```

---

## When to Use Transfer Learning {#when-to-use}

### Decision Framework

**Use Transfer Learning When:**
1. **Limited labeled data** (< 10K examples)
2. **Similar source/target domains** (e.g., both natural images)
3. **Pretrained models available** for your domain
4. **Computational constraints** (training from scratch expensive)
5. **Quick deployment** needed

**Use Fine-Tuning When:**
1. **Medium-large dataset** (> 10K examples)
2. **Target differs from pretraining** (e.g., medical images vs ImageNet)
3. **Domain-specific features** needed

**Use Feature Extraction When:**
1. **Small dataset** (< 5K examples)
2. **Very similar to pretraining** domain
3. **Limited compute/time**

**Use PEFT (LoRA/QLoRA) When:**
1. **Large models** (> 1B parameters)
2. **Multiple tasks** (can swap adapters)
3. **Limited GPU memory**
4. **Need to share base model** across tasks

**Use Domain Adaptation When:**
1. **Distribution shift** between source and target
2. **Unlabeled target data** available
3. **Related but different domains** (e.g., synthetic --> real images)

**Use Few-Shot Learning When:**
1. **Very limited data** (1-20 examples per class)
2. **Many related tasks** available for meta-training
3. **Rapid adaptation** to new classes needed

**Use Zero-Shot Learning When:**
1. **No labeled examples** for target classes
2. **Semantic descriptions** available (text, attributes)
3. **Vision-language tasks** (CLIP-based)

---

## Best Practices (2025)

### Transfer Learning
1. **Always start with pretrained models** (ImageNet for vision, BERT/GPT for NLP)
2. **Use parameter-efficient fine-tuning** for large models (LoRA, QLoRA)
3. **Apply discriminative learning rates** (smaller for early layers)
4. **Use strong regularization** (dropout, weight decay, early stopping)
5. **Monitor for catastrophic forgetting** (validate on source task)

### Fine-Tuning Strategy
1. **Phase 1**: Freeze backbone, train head (5-10 epochs)
2. **Phase 2**: Unfreeze top layers, fine-tune (10-20 epochs)
3. **Phase 3** (optional): Full fine-tuning with small LR (5-10 epochs)

### Hyperparameters
- **Learning rate**: 1e-5 to 1e-4 for backbone, 1e-3 to 1e-2 for head
- **Batch size**: As large as GPU allows (use gradient accumulation)
- **Warmup**: 5-10% of total steps
- **Weight decay**: 0.01 to 0.1
- **Gradient clipping**: Max norm 1.0

### Production Considerations
1. **Version control** pretrained models and checkpoints
2. **Experiment tracking** (Weights & Biases, MLflow)
3. **Reproducibility** (fix random seeds, log all hyperparameters)
4. **Efficient storage** (save only LoRA adapters, not full model)
5. **A/B testing** (compare transfer vs from-scratch when possible)
6. **Monitoring** for distribution drift in production

---

## Summary

Transfer learning is the most practical approach for most real-world ML applications in 2025:

- **Computer Vision**: Start with ImageNet-pretrained models, fine-tune or use as features
- **NLP**: Fine-tune BERT/GPT with LoRA/QLoRA for efficiency
- **Domain Adaptation**: Use DANN or MMD when source/target distributions differ
- **Few-Shot**: Prototypical networks or meta-learning for limited data
- **Zero-Shot**: CLIP for vision-language tasks without labeled data

The key insight: **Don't train from scratch unless you have exceptional reasons** (novel domain, massive dataset, unique architecture requirements).
