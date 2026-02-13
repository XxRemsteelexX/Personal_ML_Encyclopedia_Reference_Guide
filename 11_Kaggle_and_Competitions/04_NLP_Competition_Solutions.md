# NLP Competition Solutions

## Table of Contents

- [Introduction](#introduction)
- [Model Selection](#model-selection)
  - [DeBERTa-v3](#deberta-v3)
  - [RoBERTa](#roberta)
  - [Longformer and BigBird](#longformer-and-bigbird)
  - [ELECTRA](#electra)
  - [Multilingual Models](#multilingual-models)
  - [Model Comparison Table](#model-comparison-table)
  - [Code: Loading Models with HuggingFace](#code-loading-models-with-huggingface)
- [Fine-Tuning Recipes](#fine-tuning-recipes)
  - [Layerwise Learning Rate Decay](#layerwise-learning-rate-decay)
  - [Reinitializing Last N Layers](#reinitializing-last-n-layers)
  - [Warmup and Weight Decay](#warmup-and-weight-decay)
  - [Gradient Accumulation and Mixed Precision](#gradient-accumulation-and-mixed-precision)
  - [Max Sequence Length Selection](#max-sequence-length-selection)
  - [Complete Fine-Tuning Code](#complete-fine-tuning-code)
- [Adversarial Training AWP and FGM](#adversarial-training-awp-and-fgm)
  - [FGM Fast Gradient Method](#fgm-fast-gradient-method)
  - [AWP Adversarial Weight Perturbation](#awp-adversarial-weight-perturbation)
  - [PGD Projected Gradient Descent](#pgd-projected-gradient-descent)
  - [When to Use Adversarial Training](#when-to-use-adversarial-training)
- [Winning Solution Breakdowns](#winning-solution-breakdowns)
  - [Feedback Prize 2022](#feedback-prize-2022)
  - [CommonLit Readability 2021](#commonlit-readability-2021)
  - [Google QUEST QA 2020](#google-quest-qa-2020)
  - [Jigsaw Multilingual Toxicity 2020](#jigsaw-multilingual-toxicity-2020)
  - [US Patent Phrase Matching 2022](#us-patent-phrase-matching-2022)
  - [NBME Clinical Notes 2022](#nbme-clinical-notes-2022)
- [Text Classification Tricks](#text-classification-tricks)
  - [Pooling Strategies](#pooling-strategies)
  - [Multi-Sample Dropout](#multi-sample-dropout)
  - [Concatenating Hidden States](#concatenating-hidden-states)
  - [Adding RNN Heads](#adding-rnn-heads)
  - [Mixout Regularization](#mixout-regularization)
  - [Code: Custom Pooling Heads](#code-custom-pooling-heads)
- [Token Classification and NER](#token-classification-and-ner)
  - [Tagging Schemes BIO and BIOES](#tagging-schemes-bio-and-bioes)
  - [Span-Based vs Token-Based Approaches](#span-based-vs-token-based-approaches)
  - [Sliding Window for Long Documents](#sliding-window-for-long-documents)
  - [Word-Piece Tokenization Alignment](#word-piece-tokenization-alignment)
  - [Post-Processing](#post-processing)
  - [Code: Token Classification with Sliding Window](#code-token-classification-with-sliding-window)
- [Pseudo-Labeling for NLP](#pseudo-labeling-for-nlp)
  - [Basic Pipeline](#basic-pipeline)
  - [Soft Labels vs Hard Labels](#soft-labels-vs-hard-labels)
  - [Progressive Pseudo-Labeling](#progressive-pseudo-labeling)
  - [Teacher-Student Framework](#teacher-student-framework)
  - [TTA for Text](#tta-for-text)
  - [Code: NLP Pseudo-Labeling Pipeline](#code-nlp-pseudo-labeling-pipeline)
- [Ensemble Strategies for NLP](#ensemble-strategies-for-nlp)
  - [Cross-Model Ensembles](#cross-model-ensembles)
  - [Multi-Fold Ensembles](#multi-fold-ensembles)
  - [Preprocessing Diversity](#preprocessing-diversity)
  - [Stacking with Meta-Learner](#stacking-with-meta-learner)
  - [Code: NLP Ensemble with Rank Averaging](#code-nlp-ensemble-with-rank-averaging)
- [Data Augmentation for NLP](#data-augmentation-for-nlp)
  - [Back-Translation](#back-translation)
  - [Token-Level Augmentations](#token-level-augmentations)
  - [Contextual Augmentation with MLM](#contextual-augmentation-with-mlm)
  - [When NOT to Augment](#when-not-to-augment)
- [Common Mistakes](#common-mistakes)
- [Resources](#resources)

---

## Introduction

Natural language processing competitions on Kaggle and similar platforms have undergone a dramatic shift since 2018. The release of BERT (`bert-base-uncased`, 110M parameters) marked the transition from feature-engineered pipelines (TF-IDF + LightGBM, word2vec + LSTM) to transfer learning with pretrained transformers. By 2020, nearly every top NLP competition solution used some variant of a transformer encoder. By 2022, **DeBERTa-v3-large** had become the single most dominant model in text-based competitions, appearing in the majority of gold-medal solutions.

The typical NLP competition pipeline in the transformer era follows a consistent pattern: (1) select a pretrained encoder, (2) add a task-specific head (linear layer, span classifier, or regression head), (3) fine-tune end-to-end with layerwise learning rate decay, adversarial training, and mixed precision, (4) ensemble across folds and model families. The difference between a top-50 solution and a gold medal often comes down to careful hyperparameter tuning, adversarial training methods like AWP or FGM, and intelligent ensembling across DeBERTa, RoBERTa, and ELECTRA backbones.

Competition NLP differs from production NLP in several important ways. Inference latency is rarely a constraint -- submissions typically run on Kaggle notebooks with a generous time budget (9 hours for most competitions). This means competitors routinely ensemble 20+ models. Training is done on single or dual GPU setups (typically T4 or P100 on Kaggle, A100 on Colab Pro or private servers). The focus is entirely on maximizing a single metric (e.g., F1, log-loss, MCRMSE, QWK) rather than balancing accuracy with latency or memory constraints.

Key libraries used across nearly all NLP competition solutions include `transformers` (HuggingFace, version 4.x+), `tokenizers` for fast tokenization, `datasets` for data loading, `torch` (PyTorch 1.10+), `numpy`, `pandas`, `scikit-learn` for CV splitting and metrics, and `wandb` or `neptune` for experiment tracking. The `accelerate` library from HuggingFace simplifies multi-GPU and mixed-precision training, though many competitors still write custom training loops for maximum control.

---

## Model Selection

### DeBERTa-v3

**DeBERTa-v3-large** (`microsoft/deberta-v3-large`, 304M parameters, 24 layers, hidden size 1024) is the undisputed king of NLP competitions as of 2022-2024. It dominates across text classification, NER, question answering, and regression tasks. The model uses **disentangled attention** which separately encodes content and position information, then computes attention using disentangled matrices for content-to-content, content-to-position, and position-to-content. The v3 variant uses **ELECTRA-style replaced token detection** (RTD) pretraining instead of masked language modeling, which is more sample-efficient.

**DeBERTa-v3-base** (`microsoft/deberta-v3-base`, 86M parameters, 12 layers, hidden size 768) is the go-to when GPU memory is constrained or when training speed matters. It still outperforms RoBERTa-base on most benchmarks and is an excellent choice for fold-based ensembles where you need to train many models quickly. On Kaggle T4 GPUs (16GB VRAM), you can typically fit DeBERTa-v3-base with max_length=512 and batch_size=8, whereas DeBERTa-v3-large requires batch_size=2-4 with gradient accumulation.

Key advantages of DeBERTa-v3 over alternatives: (1) enhanced mask decoder with absolute position information in the decoding layer, (2) virtual adversarial training during pretraining improves generalization, (3) it consistently scores 0.5-2% higher than RoBERTa-large on downstream tasks without any tricks. One quirk: DeBERTa-v3 uses a SentencePiece tokenizer rather than a BPE tokenizer, so its vocabulary and tokenization behavior differ from BERT/RoBERTa.

**DeBERTa-v2-xlarge** (`microsoft/deberta-v2-xlarge`, 900M parameters, 24 layers, hidden size 1536) and **DeBERTa-v2-xxlarge** (1.5B parameters, 48 layers, hidden size 1536) offer even higher performance but require significant GPU memory. These models are used selectively in competitions, often fine-tuned on a single fold with batch_size=1 and gradient accumulation steps of 16-32. They can provide a meaningful boost in ensemble diversity.

### RoBERTa

**RoBERTa-large** (`roberta-large`, 355M parameters, 24 layers, hidden size 1024) remains a strong baseline. It was pretrained with dynamic masking, larger mini-batches (8K sequences), longer training, and no next-sentence-prediction objective -- improvements over the original BERT pretraining. RoBERTa uses a byte-level BPE tokenizer with a vocabulary of 50,265 tokens.

**RoBERTa-base** (`roberta-base`, 125M parameters, 12 layers, hidden size 768) trains roughly 3x faster than RoBERTa-large and serves as a reliable fold model. In ensemble settings, mixing RoBERTa with DeBERTa improves diversity because the models were pretrained differently (MLM vs RTD) and use different tokenizers (BPE vs SentencePiece).

The **FunNel Transformer** (`funnel-transformer/large`) is worth mentioning as a RoBERTa variant that progressively reduces sequence length through pooling, making it more efficient for long sequences. It has been used in several top solutions as an ensemble member.

### Longformer and BigBird

For tasks with documents exceeding 512 tokens (legal text, scientific papers, student essays), standard transformers truncate content and lose information. **Longformer** (`allenai/longformer-base-4096`) uses a combination of sliding window local attention (window_size=512) and global attention on selected tokens (typically CLS and task-specific tokens). It supports sequences up to **4096 tokens** out of the box.

**BigBird** (`google/bigbird-roberta-base`) uses a combination of random attention, window attention, and global attention to achieve O(n) complexity instead of O(n^2). It also supports sequences up to 4096 tokens. BigBird uses `block_size=64` for its sparse attention pattern.

In practice, competitors often find that simply using DeBERTa-v3-large with max_length=1536 or 2048 (by modifying position embeddings) can outperform Longformer/BigBird, because DeBERTa's base model quality is higher. The trick is to gradually extend position embeddings during fine-tuning:

```python
from transformers import AutoModel, AutoConfig

config = AutoConfig.from_pretrained("microsoft/deberta-v3-large")
config.max_position_embeddings = 2048  # extend from default 512
model = AutoModel.from_pretrained("microsoft/deberta-v3-large", config=config)
```

However, for truly long documents (4096+ tokens), Longformer remains the practical choice due to memory constraints.

### ELECTRA

**ELECTRA** (`google/electra-large-discriminator`, 335M parameters) uses a generator-discriminator pretraining approach. A small generator (similar to a masked LM) produces corrupted tokens, and the discriminator (the model you fine-tune) learns to detect which tokens were replaced. This is more sample-efficient than MLM because every token provides a training signal, not just the 15% masked ones.

ELECTRA-large is competitive with RoBERTa-large but offers valuable ensemble diversity because of its fundamentally different pretraining objective. **ELECTRA-base** (`google/electra-base-discriminator`, 110M parameters) is a fast, efficient option. Note that ELECTRA uses a `[CLS]` token but its representation requires a different head than BERT -- the discriminator head outputs binary predictions per token, so for classification you still use the `[CLS]` representation but with a fresh linear head.

### Multilingual Models

**XLM-RoBERTa-large** (`xlm-roberta-large`, 550M parameters, 24 layers, vocabulary 250,002 tokens covering 100 languages) is the default choice for multilingual NLP competitions. It was pretrained on 2.5TB of CommonCrawl data in 100 languages. Despite its multilingual focus, it is surprisingly competitive on English-only tasks and provides ensemble diversity.

**mDeBERTa-v3-base** (`microsoft/mdeberta-v3-base`, 86M parameters) applies DeBERTa-v3's improvements to a multilingual setting. It covers 100+ languages and often outperforms XLM-R-base on cross-lingual benchmarks. This model has been a key component in competitions like Jigsaw Multilingual Toxicity.

**RemBERT** (`google/rembert`, 575M parameters) uses a larger embedding dimension (256 for input, 1152 for hidden) and has shown strong cross-lingual transfer. It is an underused model that can add diversity to multilingual ensembles.

### Model Comparison Table

| Model | Params | Layers | Hidden | Max Len | Best For | Typical LR |
|---|---|---|---|---|---|---|
| deberta-v3-large | 304M | 24 | 1024 | 512 | Classification, NER, QA | 1e-5 |
| deberta-v3-base | 86M | 12 | 768 | 512 | Fast training, ensembles | 2e-5 |
| roberta-large | 355M | 24 | 1024 | 512 | Baseline, ensemble diversity | 1e-5 |
| roberta-base | 125M | 12 | 768 | 512 | Quick experiments | 2e-5 |
| electra-large | 335M | 24 | 1024 | 512 | Ensemble diversity | 1e-5 |
| longformer-base | 149M | 12 | 768 | 4096 | Long documents | 3e-5 |
| bigbird-roberta-base | 128M | 12 | 768 | 4096 | Long documents | 3e-5 |
| xlm-roberta-large | 550M | 24 | 1024 | 512 | Multilingual tasks | 1e-5 |
| mdeberta-v3-base | 86M | 12 | 768 | 512 | Multilingual tasks | 2e-5 |
| deberta-v2-xlarge | 900M | 24 | 1536 | 512 | Max accuracy, single fold | 5e-6 |

### Code: Loading Models with HuggingFace

```python
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoConfig,
)

# --- Loading DeBERTa-v3-large for classification ---
model_name = "microsoft/deberta-v3-large"
num_labels = 6  # adjust per competition

config = AutoConfig.from_pretrained(model_name)
config.num_labels = num_labels
config.hidden_dropout_prob = 0.1
config.attention_probs_dropout_prob = 0.1

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
)

# --- Loading for custom head (no built-in classifier) ---
backbone = AutoModel.from_pretrained(model_name, config=config)
# backbone.last_hidden_state shape: (batch, seq_len, 1024)

# --- Loading RoBERTa-large for regression ---
model_name_roberta = "roberta-large"
config_roberta = AutoConfig.from_pretrained(model_name_roberta)
config_roberta.num_labels = 1  # regression
config_roberta.problem_type = "regression"

model_roberta = AutoModelForSequenceClassification.from_pretrained(
    model_name_roberta,
    config=config_roberta,
)

# --- Loading Longformer with global attention ---
from transformers import LongformerTokenizer, LongformerModel

tokenizer_lf = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model_lf = LongformerModel.from_pretrained("allenai/longformer-base-4096")

text = "This is a long document..." * 200
inputs = tokenizer_lf(text, return_tensors="pt", max_length=4096, truncation=True)
# Set global attention on CLS token
global_attention_mask = [0] * inputs["input_ids"].shape[1]
global_attention_mask[0] = 1  # CLS token gets global attention
inputs["global_attention_mask"] = torch.tensor([global_attention_mask])

outputs = model_lf(**inputs)

# --- Tokenizer sanity check ---
sample_text = "The quick brown fox"
tokens = tokenizer.tokenize(sample_text)
input_ids = tokenizer.encode(sample_text)
decoded = tokenizer.decode(input_ids)
print(f"Tokens: {tokens}")
print(f"Input IDs: {input_ids}")
print(f"Decoded: {decoded}")
# Always verify your tokenizer handles special cases correctly
```

---

## Fine-Tuning Recipes

### Layerwise Learning Rate Decay

**Layerwise learning rate decay** (also called discriminative learning rates or LLRD) assigns progressively smaller learning rates to lower transformer layers. The intuition is that lower layers capture general linguistic features that should change less during fine-tuning, while upper layers capture task-specific patterns and need more adaptation. A typical decay rate is **0.9**, meaning each layer below gets `0.9x` the learning rate of the layer above.

For a 24-layer model like DeBERTa-v3-large with a base learning rate of `1e-5` and decay_rate of `0.9`, the top layer (layer 23) uses `1e-5`, layer 22 uses `0.9e-5 = 9e-6`, layer 21 uses `0.81e-5 = 8.1e-6`, and so on down to layer 0 which uses `1e-5 * 0.9^23 = 8.9e-7`. The embedding layer typically uses the same learning rate as layer 0 or even smaller.

```python
def get_optimizer_grouped_parameters(
    model,
    model_type="deberta-v3-large",
    learning_rate=1e-5,
    weight_decay=0.01,
    layerwise_lr_decay=0.9,
    no_decay=("bias", "LayerNorm.weight", "LayerNorm.bias"),
):
    """
    Create parameter groups with layerwise learning rate decay.
    Supports DeBERTa, RoBERTa, and BERT-style models.
    """
    # Identify the layers based on model type
    if "deberta" in model_type.lower():
        layers = [model.deberta.embeddings] + list(model.deberta.encoder.layer)
    elif "roberta" in model_type.lower():
        layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
    elif "electra" in model_type.lower():
        layers = [model.electra.embeddings] + list(model.electra.encoder.layer)
    else:
        layers = [model.bert.embeddings] + list(model.bert.encoder.layer)

    num_layers = len(layers)
    optimizer_grouped_parameters = []

    for i, layer in enumerate(layers):
        lr = learning_rate * (layerwise_lr_decay ** (num_layers - 1 - i))
        optimizer_grouped_parameters.extend([
            {
                "params": [
                    p for n, p in layer.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "lr": lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in layer.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "lr": lr,
                "weight_decay": 0.0,
            },
        ])

    # Classifier head gets the highest learning rate (or even higher)
    head_lr = learning_rate * 5  # 5x the base LR for the head
    # Gather all head/classifier parameters not part of the backbone
    backbone_params = set()
    for layer in layers:
        for p in layer.parameters():
            backbone_params.add(id(p))

    head_params_decay = []
    head_params_no_decay = []
    for n, p in model.named_parameters():
        if id(p) not in backbone_params and p.requires_grad:
            if any(nd in n for nd in no_decay):
                head_params_no_decay.append(p)
            else:
                head_params_decay.append(p)

    optimizer_grouped_parameters.extend([
        {"params": head_params_decay, "lr": head_lr, "weight_decay": weight_decay},
        {"params": head_params_no_decay, "lr": head_lr, "weight_decay": 0.0},
    ])

    return optimizer_grouped_parameters
```

### Reinitializing Last N Layers

**Reinitializing the last N transformer layers** before fine-tuning forces those layers to learn task-specific representations from scratch. The pretrained weights of the final layers are most task-specific to the pretraining objective (MLM or RTD) and may interfere with the downstream task. Typical values: `n_reinit=1` for most tasks, `n_reinit=2` for tasks that differ significantly from pretraining (e.g., regression, span detection), and `n_reinit=3` rarely but sometimes beneficial for very different tasks.

```python
import torch.nn as nn

def reinit_last_n_layers(model, n_reinit=1, model_type="deberta"):
    """Reinitialize the last n transformer layers."""
    if "deberta" in model_type.lower():
        encoder = model.deberta.encoder
    elif "roberta" in model_type.lower():
        encoder = model.roberta.encoder
    else:
        encoder = model.bert.encoder

    for layer in encoder.layer[-n_reinit:]:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

    print(f"Reinitialized last {n_reinit} layers of {model_type}")

# Usage:
# reinit_last_n_layers(model, n_reinit=2, model_type="deberta")
```

### Warmup and Weight Decay

The standard warmup schedule in NLP competitions uses a **linear warmup** for the first 5-10% of training steps, followed by either linear decay or cosine decay to zero. A **warmup_ratio of 0.06** (6% of total steps) is a common default. Cosine annealing tends to outperform linear decay slightly.

**Weight decay of 0.01** is the standard for AdamW. Higher values (0.05-0.1) can help with overfitting on small datasets. Critically, weight decay should NOT be applied to bias terms or LayerNorm parameters -- this is handled in the optimizer parameter groups shown above.

The **AdamW optimizer** with `betas=(0.9, 0.999)` and `eps=1e-6` (note: 1e-6 rather than the PyTorch default of 1e-8, following the original BERT setup) is standard. Some competitors use **AdaFactor** to reduce memory usage, but AdamW remains the dominant choice.

```python
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

# Cosine schedule with warmup (most common in top solutions)
num_training_steps = len(train_dataloader) * num_epochs
num_warmup_steps = int(num_training_steps * 0.06)  # 6% warmup

optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=1e-5,
    betas=(0.9, 0.999),
    eps=1e-6,
    weight_decay=0.01,
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)
```

### Gradient Accumulation and Mixed Precision

**Gradient accumulation** allows effective batch sizes larger than what fits in GPU memory. If your physical batch_size is 4 and you set `gradient_accumulation_steps=8`, the effective batch size is 32. This is essential for large models like DeBERTa-v3-large on 16GB GPUs.

**Mixed precision training** (FP16 or BF16) roughly halves memory usage and increases training speed by 1.5-2x. Use FP16 on V100/T4/P100 and BF16 on A100/A10. BF16 has a larger dynamic range and avoids some NaN issues that FP16 can encounter with large models.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # For FP16 only; BF16 does not need scaler

for step, batch in enumerate(train_dataloader):
    with autocast(dtype=torch.float16):  # or torch.bfloat16 on A100
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps

    scaler.scale(loss).backward()

    if (step + 1) % gradient_accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
```

### Max Sequence Length Selection

Choosing the right **max_length** is a critical but often overlooked decision. Too short and you truncate meaningful text, hurting performance. Too long and you waste memory and training time on padding tokens.

The standard approach: compute the token-length distribution of your training data and set max_length to the **95th percentile**. This captures the vast majority of the text while avoiding extreme outliers.

```python
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

# Compute token lengths for your dataset
token_lengths = []
for text in train_df["text"].values:
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_lengths.append(len(tokens))

token_lengths = np.array(token_lengths)
print(f"Mean: {token_lengths.mean():.0f}")
print(f"Median: {np.median(token_lengths):.0f}")
print(f"90th percentile: {np.percentile(token_lengths, 90):.0f}")
print(f"95th percentile: {np.percentile(token_lengths, 95):.0f}")
print(f"99th percentile: {np.percentile(token_lengths, 99):.0f}")
print(f"Max: {token_lengths.max()}")

# Set max_length to 95th percentile, rounded up to nearest 64
p95 = int(np.percentile(token_lengths, 95))
max_length = ((p95 + 63) // 64) * 64  # round up to nearest 64
max_length = min(max_length, 512)  # cap at model's maximum
print(f"Selected max_length: {max_length}")
```

For competitions where long documents are important (essay scoring, legal analysis), consider a **dynamic padding** approach where each batch is padded to the length of its longest sample rather than a global max_length. This is implemented via a custom collate function.

```python
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union
import torch

@dataclass
class DynamicPaddingCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None

    def __call__(self, features):
        label_name = "labels" if "labels" in features[0] else "label"
        labels = [f.pop(label_name, None) for f in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,  # "longest" pads to the longest in the batch
            max_length=self.max_length,
            return_tensors="pt",
        )

        if labels[0] is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.float)

        return batch
```

### Complete Fine-Tuning Code

```python
import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings("ignore")


class CFG:
    model_name = "microsoft/deberta-v3-large"
    max_length = 512
    batch_size = 4
    gradient_accumulation_steps = 8  # effective batch = 32
    epochs = 4
    learning_rate = 1e-5
    head_lr_factor = 5.0
    weight_decay = 0.01
    warmup_ratio = 0.06
    layerwise_lr_decay = 0.9
    n_reinit = 1
    max_grad_norm = 1.0
    num_labels = 6
    n_folds = 5
    seed = 42
    fp16 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NLPDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class CustomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model_name)
        self.config.hidden_dropout_prob = 0.1
        self.config.attention_probs_dropout_prob = 0.1
        self.backbone = AutoModel.from_pretrained(cfg.model_name, config=self.config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, cfg.num_labels)
        self._init_weights(self.classifier)

        # Reinitialize last N layers
        if cfg.n_reinit > 0:
            for layer in self.backbone.encoder.layer[-cfg.n_reinit:]:
                for module in layer.modules():
                    self._init_weights(module)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**kwargs)
        # Mean pooling
        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / count

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def train_one_fold(fold, train_df, val_df, cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = CustomModel(cfg).to(cfg.device)

    train_dataset = NLPDataset(
        train_df["text"].values, train_df["label"].values, tokenizer, cfg.max_length
    )
    val_dataset = NLPDataset(
        val_df["text"].values, val_df["label"].values, tokenizer, cfg.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size * 2, shuffle=False)

    optimizer_params = get_optimizer_grouped_parameters(
        model, "deberta-v3-large", cfg.learning_rate, cfg.weight_decay, cfg.layerwise_lr_decay
    )
    optimizer = torch.optim.AdamW(optimizer_params, eps=1e-6)

    total_steps = len(train_loader) * cfg.epochs // cfg.gradient_accumulation_steps
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler() if cfg.fp16 else None
    best_score = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            if cfg.fp16:
                with autocast(dtype=torch.float16):
                    outputs = model(**batch)
                    loss = outputs["loss"] / cfg.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss = outputs["loss"] / cfg.gradient_accumulation_steps
                loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if cfg.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(cfg.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                with autocast(dtype=torch.float16) if cfg.fp16 else nullcontext():
                    outputs = model(**batch)
                val_preds.append(outputs["logits"].softmax(dim=-1).cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        score = log_loss(val_labels, val_preds)
        print(f"Fold {fold}, Epoch {epoch}: log_loss = {score:.5f}")

        if score < best_score:
            best_score = score
            torch.save(model.state_dict(), f"model_fold{fold}.pth")

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return best_score
```

---

## Adversarial Training AWP and FGM

### FGM Fast Gradient Method

**FGM (Fast Gradient Method)** is the simplest adversarial training technique used in NLP competitions. It works by adding a small perturbation to the word embedding layer in the direction of the gradient, forcing the model to be robust to small input changes. The perturbation magnitude is controlled by **epsilon** (typically `epsilon=1.0` for NLP tasks). FGM typically yields a **+0.002-0.003** improvement on NLP metrics.

The algorithm: (1) compute the loss on the clean input and backpropagate to get gradients, (2) compute the adversarial perturbation `r = epsilon * grad / ||grad||` for the embedding parameters, (3) add the perturbation to embeddings, (4) compute the loss on the perturbed input and backpropagate again, (5) restore the original embeddings, (6) update model parameters using the accumulated gradients.

```python
class FGM:
    """Fast Gradient Method for adversarial training."""

    def __init__(self, model, epsilon=1.0, emb_name="word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        """Add adversarial perturbation to embedding layer."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        """Restore original embedding weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# --- Usage in training loop ---
fgm = FGM(model, epsilon=1.0, emb_name="word_embeddings")

for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}

    # Step 1: Normal forward + backward
    outputs = model(**batch)
    loss = outputs["loss"]
    loss.backward()

    # Step 2: Adversarial forward + backward
    fgm.attack()
    adv_outputs = model(**batch)
    adv_loss = adv_outputs["loss"]
    adv_loss.backward()
    fgm.restore()

    # Step 3: Update
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### AWP Adversarial Weight Perturbation

**AWP (Adversarial Weight Perturbation)** is a more powerful adversarial training technique that perturbs all model weights, not just embeddings. It was introduced in the paper "Adversarial Weight Perturbation Helps Robust Generalization" (Wu et al., 2020). AWP typically yields a **+0.003-0.005** improvement over baseline and **+0.001-0.002** over FGM.

Key hyperparameters: `adv_lr=1.0` (step size for computing the perturbation), `adv_eps=0.01` (maximum perturbation magnitude), and `adv_param="weight"` (perturb parameters containing "weight" in their name). The `start_epoch` parameter controls when to begin AWP -- starting too early can destabilize training.

```python
class AWP:
    """
    Adversarial Weight Perturbation.
    Perturbs model weights to improve generalization.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1.0,
        adv_eps=0.01,
        start_epoch=1,
        adv_step=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}

    def _attack_step(self):
        """Compute and apply adversarial perturbation to weights."""
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data)
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def _save(self):
        """Save original weights and compute perturbation bounds."""
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, batch, epoch):
        """Full AWP attack: save -> perturb -> forward -> backward -> restore."""
        if epoch < self.start_epoch:
            return None

        self._save()
        self._attack_step()

        # Forward pass with perturbed weights
        outputs = self.model(**batch)
        adv_loss = outputs["loss"]
        self.optimizer.zero_grad()
        adv_loss.backward()

        self._restore()
        return adv_loss.item()


# --- Usage in training loop ---
awp = AWP(
    model=model,
    optimizer=optimizer,
    adv_lr=1.0,
    adv_eps=0.01,
    start_epoch=1,  # start AWP from epoch 1 (skip epoch 0)
)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # Normal forward + backward
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()

        # AWP attack (only after start_epoch)
        awp.attack_backward(batch, epoch)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### PGD Projected Gradient Descent

**PGD (Projected Gradient Descent)** is a multi-step variant of FGM that iteratively perturbs embeddings, projecting back onto an epsilon-ball after each step. It is stronger than FGM but takes `K` times longer (typically `K=3` attack steps). The perturbation at each step uses `alpha=0.3` (step size) and is clipped to remain within `epsilon=1.0` of the original embeddings.

```python
class PGD:
    """Projected Gradient Descent adversarial training."""

    def __init__(self, model, emb_name="word_embeddings", epsilon=1.0, alpha=0.3, K=3):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.K = K
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    # Project back to epsilon ball
                    param.data = self._project(name, param.data)

    def _project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.grad_backup:
                    param.grad = self.grad_backup[name]
        self.grad_backup = {}


# --- Usage ---
pgd = PGD(model, epsilon=1.0, alpha=0.3, K=3)

for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(**batch)
    loss = outputs["loss"]
    loss.backward()
    pgd.backup_grad()

    for k in range(pgd.K):
        pgd.attack(is_first_attack=(k == 0))
        if k != pgd.K - 1:
            model.zero_grad()
        adv_outputs = model(**batch)
        adv_loss = adv_outputs["loss"]
        adv_loss.backward()

    pgd.restore()
    pgd.restore_grad()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### When to Use Adversarial Training

Adversarial training is most beneficial when: (1) the dataset is small to medium (fewer than 50K training samples), (2) the task involves nuanced text understanding (toxicity detection, argument quality, readability), (3) the metric is sensitive to small improvements. On very large datasets (100K+ samples), the benefit diminishes because the model has enough data to regularize naturally.

Typical improvements by method: FGM yields +0.002-0.003, AWP yields +0.003-0.005, PGD yields +0.003-0.004 but takes 3x longer. In practice, AWP is preferred in competitions because it provides the best accuracy/time tradeoff. Always start AWP after epoch 0 or 1 to let the model converge on the clean data first -- starting AWP from the beginning can cause training instability.

---

## Winning Solution Breakdowns

### Feedback Prize 2022

The **Feedback Prize - Predicting Effective Arguments** competition (Kaggle, 2022) asked participants to classify argumentative elements in student essays into three effectiveness levels (Adequate, Effective, Ineffective) across seven discourse types (Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, Concluding Statement). The metric was **MCRMSE** (Mean Column-wise Root Mean Squared Error).

The 1st place solution used an ensemble of **DeBERTa-v3-large** models with the following key components: (1) multi-task learning -- predicting both the effectiveness label and the discourse type simultaneously, (2) span-level classification -- encoding the full essay with special tokens marking the target span boundaries (e.g., `[START_CLAIM]` ... `[END_CLAIM]`), (3) AWP adversarial training with `adv_lr=1.0, adv_eps=0.01`, (4) layerwise learning rate decay with `decay_rate=0.85`, (5) 5-fold CV with different seeds.

Architecture: the essay text was prepended with the discourse type as a special prefix. The target span was surrounded by special boundary tokens. Mean pooling was applied over the target span tokens only (not the whole sequence). A 2-layer MLP head (hidden_size=256, dropout=0.1) mapped the pooled representation to 3 output scores.

Training config: `lr=1e-5, batch_size=4, grad_accum=8, epochs=5, max_length=1536, warmup_ratio=0.1, weight_decay=0.01`. The extended max_length of 1536 was critical because student essays frequently exceeded 512 tokens.

### CommonLit Readability 2021

The **CommonLit Readability Prize** (Kaggle, 2021) was a regression task predicting the readability score of literary passages. The metric was **RMSE**. This competition was notable for having only ~2,800 training samples, making overfitting a major challenge.

The top solutions combined transformer features with traditional ML models. The winning approach: (1) extract features from the `[CLS]` token and mean-pooled last hidden states of **RoBERTa-large** and **DeBERTa-large**, (2) extract handcrafted features (Flesch-Kincaid score, Dale-Chall score, word count, sentence count, average word length, syllable count, type-token ratio), (3) feed all features into an **SVR** (Support Vector Regression) with `kernel="rbf", C=10.0, epsilon=0.1` or a **Ridge regression** as a meta-learner.

Key tricks: (1) **Multi-sample dropout** with 5 dropout masks averaged during training to reduce variance, (2) **Stochastic Weight Averaging (SWA)** over the last 3 epochs to smooth the loss landscape, (3) pseudo-labeling on the test set using a cross-validated model with soft labels (predicted scores rather than rounded labels), (4) ensembling across RoBERTa-large, DeBERTa-large, ALBERT-xxlarge-v2, and ELECTRA-large.

### Google QUEST QA 2020

The **Google QUEST Q&A Labeling** competition (Kaggle, 2020) required predicting 30 target labels (21 for questions, 9 for answers) as continuous values between 0 and 1 for question-answer pairs from StackExchange. The metric was **Spearman correlation** averaged across targets.

The winning solution used **BERT-large-uncased** and **BERT-large-cased** as backbones (this was pre-DeBERTa era). The input format was `[CLS] question_title [SEP] question_body [SEP] answer [SEP]`. Key innovations: (1) separate heads for question targets and answer targets with shared backbone, (2) post-processing with **threshold optimization** -- discretizing predictions to match the known label distribution (labels were derived from median aggregation of 3-5 annotators, so many targets clustered at 0.0, 0.33, 0.67, 1.0), (3) **pseudo-labeling** on external StackExchange data to augment the small training set (~6,000 samples).

Training: `lr=3e-5, epochs=5, max_length=512, batch_size=8`. The multi-target regression used **BCEWithLogitsLoss** on each target independently, with the final predictions passed through a sigmoid.

### Jigsaw Multilingual Toxicity 2020

The **Jigsaw Multilingual Toxic Comment Classification** competition (Kaggle, 2020) asked competitors to classify toxic comments in 6 non-English languages (Turkish, Spanish, Italian, Portuguese, Russian, French) using only English training data. The metric was **AUC-ROC**.

The winning approach centered on **XLM-RoBERTa-large** (`xlm-roberta-large`) with several critical techniques: (1) **pseudo-labeling** -- train on English data, predict on the non-English validation/test data, use high-confidence predictions as additional training data, retrain, and iterate. Threshold for pseudo-labels: confidence > 0.95 for positive class, confidence < 0.05 for negative class. (2) **TTA for text** -- translate each test comment into English using multiple translation APIs (Google Translate, DeepL), predict on both the original and translated versions, and average the predictions. This gave +0.002-0.003 AUC. (3) **Back-translation augmentation** -- translate English training data to each target language and back to English to create paraphrased training samples.

Architecture: `XLM-RoBERTa-large -> mean pooling -> dropout(0.1) -> linear(1024, 1) -> sigmoid`. Training: `lr=1e-5, epochs=3, max_length=256, batch_size=16`. The ensemble included XLM-R-large, mBERT, and models fine-tuned on translated data.

### US Patent Phrase Matching 2022

The **U.S. Patent Phrase to Phrase Matching** competition (Kaggle, 2022) required predicting the semantic similarity (0.0 to 1.0) between patent phrases, given a CPC (Cooperative Patent Classification) context code. The metric was **Pearson correlation**.

The top solutions treated this as a **cross-encoder** regression task. The input format: `[CLS] anchor_phrase [SEP] target_phrase [SEP] CPC_title [SEP]`. Using **DeBERTa-v3-large** as the backbone, the 1st place solution achieved a Pearson of 0.8870.

Key techniques: (1) **sentence-transformers** approach -- using the cross-encoder architecture from the `sentence-transformers` library for pair-wise scoring, (2) **CPC context injection** -- appending the CPC category title (e.g., "PERFORMING OPERATIONS; TRANSPORTING") to provide domain context, (3) data augmentation by swapping anchor and target phrases (symmetric similarity assumption), (4) training separate models for different CPC sections and ensembling.

Training config: `lr=2e-5, batch_size=16, epochs=5, max_length=256, n_reinit=1, layerwise_lr_decay=0.95`. Loss function: **MSELoss** for regression, with predictions clipped to `[0, 1]` during inference.

### NBME Clinical Notes 2022

The **NBME - Score Clinical Patient Notes** competition (Kaggle, 2022) was a token classification task requiring extraction of clinical entity spans from patient notes. Given a clinical case description (the "feature") and a patient note, the model needed to identify which character spans in the note corresponded to each clinical feature. The metric was micro-averaged **F1** on character-level spans.

The winning solution used **DeBERTa-v3-large** for token classification with these innovations: (1) **sliding window approach** -- patient notes exceeded 512 tokens, so a stride-based sliding window (stride=128) was used with overlap, and per-token predictions were averaged across overlapping windows, (2) **multi-feature encoding** -- the clinical feature text was prepended to the patient note as `[CLS] feature_text [SEP] patient_note [SEP]`, creating a separate prediction for each feature, (3) **character-to-token alignment** -- BIO labels were assigned at the token level, then mapped back to character spans using the tokenizer's offset mappings.

Post-processing: (1) merge adjacent predicted spans separated by fewer than 2 characters, (2) apply per-feature threshold optimization using validation data (thresholds ranging from 0.3 to 0.7 depending on the feature), (3) filter out spans shorter than 2 characters.

Training: `lr=1e-5, epochs=5, max_length=512, batch_size=4, grad_accum=4, stride=128`. Token classification head: `nn.Linear(1024, 2)` applied to each token (binary: entity vs non-entity).

---

## Text Classification Tricks

### Pooling Strategies

The choice of pooling strategy for extracting a fixed-size representation from the transformer output has a significant impact on performance. The three main options are **CLS pooling**, **mean pooling**, and **attention pooling**.

**CLS pooling** takes the hidden state of the `[CLS]` (or `<s>`) token. This is the simplest approach and is the default in most HuggingFace classification models. However, the CLS token representation can be suboptimal because it is a single token's representation that must compress all sequence information.

**Mean pooling** averages all token hidden states, weighted by the attention mask to exclude padding tokens. This is consistently the best default choice in competitions and typically outperforms CLS by 0.001-0.005. The key is to properly mask padding tokens so they do not dilute the average.

**Attention pooling** (also called weighted-layer pooling or attention-based pooling) learns a weighted combination of token representations using a small attention mechanism. This allows the model to focus on the most task-relevant tokens rather than treating all tokens equally.

### Multi-Sample Dropout

**Multi-sample dropout** (Inoue 2019) applies multiple different dropout masks during training and averages the resulting logits before computing the loss. This acts as a regularizer and variance reducer, particularly effective for small datasets. Using `n_drops=5` with `dropout_rate=0.1-0.3` is typical. The computational overhead is minimal because you only rerun the dropout + classification head, not the entire transformer.

The technique is especially powerful for regression tasks (like CommonLit Readability) where the output distribution is continuous and small perturbations in the representation can cause large prediction swings. Multi-sample dropout smooths these fluctuations.

### Concatenating Hidden States

Instead of using only the last hidden state, concatenating the outputs of the **last 4 hidden layers** (layers -1, -2, -3, -4) captures information at different levels of abstraction. Lower layers capture syntactic patterns while higher layers capture semantic patterns. The concatenated representation has dimension `4 * hidden_size` (e.g., 4096 for a model with hidden_size=1024), which is then projected down through a linear layer.

An alternative is **weighted-layer pooling**, where a learnable scalar weight is assigned to each layer's output and the weighted sum is used. This avoids the 4x dimensionality increase while still leveraging multiple layers.

### Adding RNN Heads

Adding an **LSTM** or **GRU** layer on top of the transformer output before the classification head can capture sequential dependencies that the linear head misses. This is particularly useful for token classification and span detection tasks. A bidirectional LSTM with `hidden_size=256` and `num_layers=1` adds minimal parameters but can improve performance by 0.001-0.003.

Architecture: `transformer -> BiLSTM(input=hidden_size, hidden=256, bidirectional=True) -> linear(512, num_labels)`. The BiLSTM output dimension is `2 * 256 = 512` because of bidirectionality.

### Mixout Regularization

**Mixout** (Lee et al., 2020) is a regularization technique that randomly replaces model parameters with their pretrained values during training, similar to dropout but for weights instead of activations. With probability `p=0.1`, each weight is replaced with its pretrained value. This prevents the model from drifting too far from the pretrained initialization and is particularly effective on small datasets.

Implementation: replace `nn.Linear` layers with a custom `MixLinear` layer that stochastically mixes current and pretrained weights. Mixout is less commonly used than multi-sample dropout in competitions because it is harder to implement and tune, but it can provide meaningful gains on datasets with fewer than 5,000 training samples.

### Code: Custom Pooling Heads

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPooling(nn.Module):
    """Mean pooling over token embeddings, properly handling attention mask."""

    def forward(self, last_hidden_state, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class AttentionPooling(nn.Module):
    """Attention-based pooling with learnable query."""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        # attention_mask: (batch, seq_len)
        weights = self.attention(last_hidden_state).squeeze(-1)  # (batch, seq_len)
        weights = weights.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)  # (batch, seq_len)
        pooled = torch.bmm(weights.unsqueeze(1), last_hidden_state).squeeze(1)
        return pooled


class WeightedLayerPooling(nn.Module):
    """Learnable weighted combination of all hidden layers."""

    def __init__(self, num_hidden_layers, layer_start=9, layer_weights=None):
        super().__init__()
        self.layer_start = layer_start
        self.num_layers = num_hidden_layers - layer_start + 1
        self.layer_weights = nn.Parameter(
            torch.tensor([1.0] * self.num_layers, dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        # all_hidden_states: tuple of (num_layers+1, batch, seq_len, hidden)
        selected = torch.stack(all_hidden_states[self.layer_start:], dim=0)
        weights = F.softmax(self.layer_weights, dim=0)
        weighted = (selected * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted


class ConcatLastFourPooling(nn.Module):
    """Concatenate last 4 hidden layers then project down."""

    def __init__(self, hidden_size, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.projection = nn.Linear(hidden_size * num_layers, hidden_size)

    def forward(self, all_hidden_states, attention_mask):
        # Concatenate last N layers
        concat = torch.cat(
            [all_hidden_states[-i] for i in range(1, self.num_layers + 1)],
            dim=-1,
        )
        # Mean pool the concatenated representation
        mask_expanded = attention_mask.unsqueeze(-1).expand(concat.size()).float()
        pooled = (concat * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        return self.projection(pooled)


class MultiSampleDropout(nn.Module):
    """Apply multiple dropout masks and average logits."""

    def __init__(self, hidden_size, num_labels, n_drops=5, dropout_rate=0.2):
        super().__init__()
        self.n_drops = n_drops
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_drops)])
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_state, labels=None):
        logits_list = [self.classifier(dropout(hidden_state)) for dropout in self.dropouts]
        logits = torch.stack(logits_list, dim=0).mean(dim=0)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # Compute loss on each dropout sample and average
            losses = [loss_fn(l, labels) for l in logits_list]
            loss = torch.stack(losses).mean()

        return {"loss": loss, "logits": logits}


class LSTMHead(nn.Module):
    """BiLSTM head on top of transformer for sequence labeling."""

    def __init__(self, hidden_size, lstm_hidden=256, num_labels=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_size,
            lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)

    def forward(self, hidden_states):
        lstm_out, _ = self.lstm(hidden_states)
        logits = self.classifier(self.dropout(lstm_out))
        return logits


# --- Complete model using custom pooling ---
class CompetitionModel(nn.Module):
    def __init__(self, model_name, num_labels, pooling_type="mean"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True  # needed for layer pooling
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        if pooling_type == "mean":
            self.pooler = MeanPooling()
            pool_dim = self.config.hidden_size
        elif pooling_type == "attention":
            self.pooler = AttentionPooling(self.config.hidden_size)
            pool_dim = self.config.hidden_size
        elif pooling_type == "weighted_layer":
            self.pooler = WeightedLayerPooling(self.config.num_hidden_layers)
            pool_dim = self.config.hidden_size
        elif pooling_type == "concat_last4":
            self.pooler = ConcatLastFourPooling(self.config.hidden_size)
            pool_dim = self.config.hidden_size

        self.head = MultiSampleDropout(pool_dim, num_labels, n_drops=5, dropout_rate=0.2)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(self.pooler, MeanPooling):
            pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        elif isinstance(self.pooler, AttentionPooling):
            pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        elif isinstance(self.pooler, (WeightedLayerPooling,)):
            weighted = self.pooler(outputs.hidden_states)
            # Mean pool the weighted layer output
            mask_expanded = attention_mask.unsqueeze(-1).expand(weighted.size()).float()
            pooled = (weighted * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        elif isinstance(self.pooler, ConcatLastFourPooling):
            pooled = self.pooler(outputs.hidden_states, attention_mask)

        return self.head(pooled, labels)
```

---

## Token Classification and NER

### Tagging Schemes BIO and BIOES

**BIO (Begin-Inside-Outside)** is the most common tagging scheme for NER in competitions. Each token is labeled as `B-TYPE` (beginning of an entity), `I-TYPE` (inside/continuation of an entity), or `O` (not part of any entity). For example, in "New York City is large", the tags would be `B-LOC I-LOC I-LOC O O`.

**BIOES (Begin-Inside-Outside-End-Single)** adds two extra tags: `E-TYPE` for the last token of a multi-token entity and `S-TYPE` for single-token entities. The same example becomes `B-LOC I-LOC E-LOC O O`. BIOES provides more precise boundary information and typically yields +0.5-1.0 F1 over BIO in NER tasks, though it increases the label space.

For competition NER tasks, BIO is the default choice unless the evaluation metric heavily penalizes boundary errors, in which case BIOES can help. When converting predictions back to spans, BIO requires careful handling of consecutive `B` tags (which indicate the start of a new entity immediately after another entity).

### Span-Based vs Token-Based Approaches

**Token-based** approaches assign a label to each token independently (or with a CRF layer). They are straightforward to implement with `AutoModelForTokenClassification` and work well when entities do not overlap.

**Span-based** approaches enumerate all possible spans up to a maximum length and classify each span as an entity type or "no entity". The span representation is typically `[start_token_repr; end_token_repr; span_width_embedding]`. This naturally handles overlapping entities and avoids the BIO boundary ambiguity.

In competitions like NBME Clinical Notes, span-based approaches were among the top solutions. A span classifier takes all `O(n * L)` candidate spans (where `n` is the sequence length and `L` is the maximum span width, e.g., `L=25`) and scores each one. The span representation is computed as:

```python
span_repr = torch.cat([
    hidden_states[start_indices],   # (num_spans, hidden_size)
    hidden_states[end_indices],     # (num_spans, hidden_size)
    span_width_embedding,           # (num_spans, width_emb_size)
], dim=-1)
```

### Sliding Window for Long Documents

When documents exceed the model's max_length (typically 512 tokens), a **sliding window** approach splits the document into overlapping chunks. Each chunk is processed independently, and per-token predictions are aggregated across overlapping regions.

The key parameters are: `max_length=512` (chunk size), `doc_stride=128` (overlap between consecutive chunks, matching SQuAD conventions). For tokens that appear in multiple chunks, the final prediction is the **average** of predictions across all chunks, or alternatively the prediction from the chunk where the token is farthest from the edges (least affected by truncation).

### Word-Piece Tokenization Alignment

Transformer tokenizers split words into subword tokens (word-pieces). The word "unhappiness" might be tokenized as `["un", "##happiness"]` by BERT or `["_un", "happiness"]` by DeBERTa. For token classification, you need to align subword tokens with word-level labels.

The standard approach: assign the label to the first subword token of each word and use `-100` (the ignore index for CrossEntropyLoss) for subsequent subword tokens. The tokenizer's `return_offsets_mapping=True` parameter provides character-to-token mappings, and `word_ids()` maps subword tokens back to their original word indices.

```python
def align_labels_with_tokens(labels, word_ids):
    """
    Align word-level labels with subword tokens.
    labels: list of label ids, one per word.
    word_ids: output of tokenizer.word_ids(), maps each token to its word index.
    Returns: list of label ids, one per subword token.
    """
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens ([CLS], [SEP], [PAD])
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            # First subword of a new word: use the actual label
            aligned_labels.append(labels[word_idx])
        else:
            # Subsequent subword: use -100 to ignore, or use I- tag
            aligned_labels.append(-100)  # or labels[word_idx] for I- continuation
        previous_word_idx = word_idx
    return aligned_labels
```

### Post-Processing

Post-processing is critical for NER competition scores. Common steps include:

1. **Threshold tuning**: instead of argmax, use a per-class probability threshold optimized on validation data. For binary entity detection, thresholds between 0.3 and 0.7 are typical, optimized per entity type.

2. **Entity merging**: merge adjacent predicted spans of the same type that are separated by 1-2 tokens (often punctuation or whitespace tokens that the model missed).

3. **Minimum span length**: filter out predicted spans shorter than a minimum length (e.g., 2 characters for clinical entities, 1 word for named entities).

4. **Character-level alignment**: convert token-level predictions back to character-level spans using the tokenizer's offset mappings. This is critical when the evaluation metric is character-based (as in NBME).

5. **Constraint enforcement**: ensure predictions satisfy structural constraints (e.g., no overlapping spans, entities must be complete words not partial words).

### Code: Token Classification with Sliding Window

```python
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig


class TokenClassificationModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(outputs.last_hidden_state))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}


def tokenize_with_sliding_window(
    text,
    tokenizer,
    max_length=512,
    doc_stride=128,
):
    """
    Tokenize a long text using sliding window with overlap.
    Returns list of chunks, each a dict with input_ids, attention_mask,
    offset_mapping, and token_to_original indices.
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=doc_stride,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    chunks = []
    num_chunks = encoding["input_ids"].shape[0]

    for i in range(num_chunks):
        chunk = {
            "input_ids": encoding["input_ids"][i],
            "attention_mask": encoding["attention_mask"][i],
            "offset_mapping": encoding["offset_mapping"][i],
        }
        # overflow_to_sample_mapping tells which original sample each chunk belongs to
        if "overflow_to_sample_mapping" in encoding:
            chunk["sample_idx"] = encoding["overflow_to_sample_mapping"][i].item()
        chunks.append(chunk)

    return chunks


def predict_with_sliding_window(
    text,
    model,
    tokenizer,
    max_length=512,
    doc_stride=128,
    device="cuda",
):
    """
    Predict token labels for a long text using sliding window.
    Aggregates predictions from overlapping chunks by averaging.
    """
    model.eval()
    chunks = tokenize_with_sliding_window(text, tokenizer, max_length, doc_stride)

    # Get full tokenization for the text (without truncation) to know total length
    full_encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    num_tokens = len(full_encoding["input_ids"])

    # Accumulate predictions for each original token position
    pred_accumulator = np.zeros((num_tokens, model.num_labels), dtype=np.float32)
    pred_counts = np.zeros(num_tokens, dtype=np.float32)

    for chunk in chunks:
        input_ids = chunk["input_ids"].unsqueeze(0).to(device)
        attention_mask = chunk["attention_mask"].unsqueeze(0).to(device)
        offset_mapping = chunk["offset_mapping"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()[0]

        # Map chunk token positions back to original text positions
        for token_idx in range(max_length):
            if attention_mask[0, token_idx] == 0:
                continue
            start_char, end_char = offset_mapping[token_idx].tolist()
            if start_char == 0 and end_char == 0:
                continue  # special token

            # Find which original token this corresponds to
            for orig_idx, (orig_start, orig_end) in enumerate(
                full_encoding["offset_mapping"]
            ):
                if orig_start == start_char and orig_end == end_char:
                    pred_accumulator[orig_idx] += probs[token_idx]
                    pred_counts[orig_idx] += 1
                    break

    # Average predictions where tokens appeared in multiple chunks
    pred_counts = np.maximum(pred_counts, 1)
    avg_preds = pred_accumulator / pred_counts[:, None]

    # Convert to label predictions
    predicted_labels = np.argmax(avg_preds, axis=-1)

    # Convert to character-level spans
    spans = []
    current_entity = None
    for idx, label in enumerate(predicted_labels):
        start_char, end_char = full_encoding["offset_mapping"][idx]
        if label > 0:  # Non-O label
            if current_entity is None:
                current_entity = {"start": start_char, "end": end_char, "label": label}
            elif label == current_entity["label"]:
                current_entity["end"] = end_char
            else:
                spans.append(current_entity)
                current_entity = {"start": start_char, "end": end_char, "label": label}
        else:
            if current_entity is not None:
                spans.append(current_entity)
                current_entity = None

    if current_entity is not None:
        spans.append(current_entity)

    return predicted_labels, spans, avg_preds


def merge_adjacent_spans(spans, max_gap=2):
    """
    Merge adjacent spans of the same label type separated by max_gap characters.
    """
    if not spans:
        return spans

    merged = [spans[0].copy()]
    for span in spans[1:]:
        prev = merged[-1]
        if span["label"] == prev["label"] and span["start"] - prev["end"] <= max_gap:
            prev["end"] = span["end"]
        else:
            merged.append(span.copy())

    return merged
```

---

## Pseudo-Labeling for NLP

### Basic Pipeline

**Pseudo-labeling** (also called self-training) is one of the most powerful techniques in NLP competitions, particularly when there is abundant unlabeled data (e.g., the test set, external corpora). The basic pipeline: (1) train a model on the labeled training data with k-fold CV, (2) predict labels for unlabeled data using the trained model, (3) filter predictions by confidence (keep only samples with max probability > threshold), (4) retrain on the combined labeled + pseudo-labeled data.

The confidence threshold is critical: too low and you inject noise, too high and you add too few samples. A typical starting threshold is **0.9** for classification tasks (softmax probability of the predicted class > 0.9) and within **0.5 standard deviations** of the predicted mean for regression tasks.

### Soft Labels vs Hard Labels

**Hard labels** convert model predictions to discrete categories (e.g., argmax for classification). **Soft labels** use the full probability distribution (e.g., `[0.05, 0.85, 0.10]` for a 3-class problem). Soft labels preserve the model's uncertainty and are generally superior because they provide richer training signal and reduce confirmation bias.

For regression tasks, the predicted continuous value is used directly (no discretization). For multi-label classification, per-label probabilities are used as soft targets with BCEWithLogitsLoss.

When using soft labels, the loss function changes from CrossEntropyLoss to **KL divergence** or a **soft cross-entropy**:

```python
def soft_cross_entropy(logits, soft_targets, reduction="mean"):
    """Cross-entropy loss with soft targets (probability distributions)."""
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    return loss
```

### Progressive Pseudo-Labeling

**Progressive pseudo-labeling** starts with a high confidence threshold and gradually lowers it across rounds. This ensures early rounds add only high-quality pseudo-labels, and subsequent rounds can be more aggressive because the model has already improved.

A typical schedule: Round 1 uses threshold=0.95 (adds ~10% of unlabeled data), Round 2 uses threshold=0.90 (adds ~20%), Round 3 uses threshold=0.85 (adds ~30%). Each round retrains from scratch (not from the previous model) to avoid error accumulation.

### Teacher-Student Framework

The **teacher-student** variant of pseudo-labeling uses a separate teacher model to generate pseudo-labels while the student model is trained. The teacher can be: (1) the same architecture trained on a different fold, (2) a different model family (e.g., DeBERTa teacher, RoBERTa student), (3) an ensemble of multiple models (most robust). Using a teacher ensemble reduces the noise in pseudo-labels significantly.

**Knowledge distillation** extends this by training the student to match the teacher's full output distribution (soft labels) with a temperature parameter `T=3-5`:

```python
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    """Combined distillation + hard label loss."""
    soft_loss = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
    ) * (T * T)

    hard_loss = nn.CrossEntropyLoss()(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### TTA for Text

**Test-Time Augmentation (TTA)** for text is less straightforward than for images, but several approaches work in practice:

1. **Different max_length values**: predict with max_length=384, 512, and 640, then average. Different truncation points capture different parts of long documents.

2. **Different tokenizer preprocessing**: predict with original text and with lowercased text, then average.

3. **Translation-based TTA**: translate text to another language and back (back-translation), predict on both original and paraphrased versions, average.

4. **Segment shuffling**: for multi-segment inputs (question + context), try different orderings and average predictions.

TTA typically provides +0.001-0.003 improvement and stacks with ensembling.

### Code: NLP Pseudo-Labeling Pipeline

```python
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy


class PseudoLabelPipeline:
    """
    Complete pseudo-labeling pipeline for NLP competitions.
    Supports soft labels, progressive thresholds, and teacher ensembles.
    """

    def __init__(
        self,
        model_cls,
        model_kwargs,
        tokenizer,
        dataset_cls,
        train_fn,
        predict_fn,
        n_folds=5,
        confidence_threshold=0.9,
        use_soft_labels=True,
        progressive_rounds=3,
        threshold_decay=0.05,
    ):
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.tokenizer = tokenizer
        self.dataset_cls = dataset_cls
        self.train_fn = train_fn
        self.predict_fn = predict_fn
        self.n_folds = n_folds
        self.confidence_threshold = confidence_threshold
        self.use_soft_labels = use_soft_labels
        self.progressive_rounds = progressive_rounds
        self.threshold_decay = threshold_decay

    def generate_pseudo_labels(
        self, fold_models, unlabeled_texts, device="cuda", batch_size=32
    ):
        """Generate pseudo-labels using an ensemble of fold models."""
        all_preds = []

        for model in fold_models:
            model.eval()
            dataset = self.dataset_cls(
                unlabeled_texts, labels=None, tokenizer=self.tokenizer, max_length=512
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            preds = []
            with torch.no_grad():
                for batch in loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    probs = torch.softmax(outputs["logits"], dim=-1)
                    preds.append(probs.cpu().numpy())

            all_preds.append(np.concatenate(preds))

        # Ensemble predictions (average across folds)
        ensemble_preds = np.mean(all_preds, axis=0)  # (num_samples, num_classes)
        return ensemble_preds

    def filter_by_confidence(self, predictions, threshold):
        """
        Filter pseudo-labeled samples by confidence threshold.
        Returns indices of samples that pass the threshold.
        """
        max_probs = predictions.max(axis=1)  # (num_samples,)
        confident_mask = max_probs >= threshold
        confident_indices = np.where(confident_mask)[0]

        print(
            f"Threshold {threshold:.2f}: {confident_mask.sum()} / {len(predictions)} "
            f"samples pass ({100 * confident_mask.mean():.1f}%)"
        )
        return confident_indices

    def run(self, train_df, unlabeled_texts, device="cuda"):
        """
        Execute the full progressive pseudo-labeling pipeline.
        """
        current_train_df = train_df.copy()

        for round_idx in range(self.progressive_rounds):
            threshold = self.confidence_threshold - round_idx * self.threshold_decay
            print(f"\n=== Pseudo-Label Round {round_idx + 1} (threshold={threshold:.2f}) ===")

            # Step 1: Train fold models on current training data
            fold_models = []
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(
                skf.split(current_train_df, current_train_df["label"])
            ):
                fold_train = current_train_df.iloc[train_idx]
                fold_val = current_train_df.iloc[val_idx]

                model = self.model_cls(**self.model_kwargs).to(device)
                model = self.train_fn(model, fold_train, fold_val, fold=fold)
                fold_models.append(model)

            # Step 2: Generate pseudo-labels for unlabeled data
            pseudo_preds = self.generate_pseudo_labels(
                fold_models, unlabeled_texts, device
            )

            # Step 3: Filter by confidence
            confident_idx = self.filter_by_confidence(pseudo_preds, threshold)

            if len(confident_idx) == 0:
                print("No samples pass threshold. Stopping.")
                break

            # Step 4: Create pseudo-labeled dataframe
            if self.use_soft_labels:
                pseudo_df = pd.DataFrame({
                    "text": [unlabeled_texts[i] for i in confident_idx],
                    "label": pseudo_preds[confident_idx].argmax(axis=1),
                    "soft_label": [
                        pseudo_preds[i].tolist() for i in confident_idx
                    ],
                    "is_pseudo": True,
                })
            else:
                pseudo_df = pd.DataFrame({
                    "text": [unlabeled_texts[i] for i in confident_idx],
                    "label": pseudo_preds[confident_idx].argmax(axis=1),
                    "is_pseudo": True,
                })

            # Step 5: Combine original + pseudo-labeled data
            current_train_df = pd.concat(
                [train_df, pseudo_df], ignore_index=True
            )
            print(
                f"Training set size: {len(train_df)} original + "
                f"{len(pseudo_df)} pseudo = {len(current_train_df)} total"
            )

            # Clean up fold models
            del fold_models
            torch.cuda.empty_cache()

        return current_train_df


# --- TTA for text ---
def predict_with_tta(
    model, texts, tokenizer, max_lengths=[384, 512, 640], device="cuda", batch_size=32
):
    """
    Test-Time Augmentation: average predictions across different max_lengths.
    """
    model.eval()
    all_tta_preds = []

    for max_len in max_lengths:
        dataset = NLPDataset(texts, labels=None, tokenizer=tokenizer, max_length=max_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                probs = torch.softmax(outputs["logits"], dim=-1)
                preds.append(probs.cpu().numpy())

        all_tta_preds.append(np.concatenate(preds))

    # Average across TTA variants
    tta_ensemble = np.mean(all_tta_preds, axis=0)
    return tta_ensemble
```

---

## Ensemble Strategies for NLP

### Cross-Model Ensembles

The most impactful ensembling strategy in NLP competitions is combining predictions from **different model families**. A typical top-3 solution ensembles DeBERTa-v3-large + RoBERTa-large + ELECTRA-large. These models differ in pretraining objective (RTD vs MLM), tokenizer (SentencePiece vs BPE), and architecture details, ensuring low correlation between their errors.

Ensemble weights are either **uniform** (simple average) or **optimized** on validation data using `scipy.optimize.minimize` with the competition metric as the objective. For 3 models, a grid search over weights in 0.05 increments (e.g., `[0.5, 0.3, 0.2]`) is feasible and often slightly better than learned weights.

### Multi-Fold Ensembles

Standard 5-fold cross-validation produces 5 models per architecture. Averaging predictions from all 5 folds of a single model family typically gives +0.002-0.005 over a single fold. This is the simplest and most reliable form of ensembling.

For maximum performance, use **different random seeds** for each fold split (e.g., fold with seed=42, fold with seed=2022, fold with seed=1337). This creates even more prediction diversity. A 5-fold x 3-seed setup gives 15 models per architecture.

### Preprocessing Diversity

Using different preprocessing strategies for each ensemble member adds diversity at zero architecture cost:

1. **Different max_length**: model A uses max_length=384, model B uses max_length=512, model C uses max_length=640.
2. **Lowercase vs original**: train one model on original-case text and another on lowercased text.
3. **Different truncation**: truncate from the beginning (keep the end) vs truncate from the end (keep the beginning) vs truncate from both ends (keep the middle).
4. **Text cleaning variants**: one model trained on raw text, another on text with URLs/emails removed, another with numbers normalized.

### Stacking with Meta-Learner

**Stacking** uses first-level model predictions as features for a second-level meta-learner. The procedure: (1) generate out-of-fold predictions from each first-level model using k-fold CV, (2) train a meta-learner (typically Ridge regression, logistic regression, or LightGBM) on these predictions, (3) at test time, generate predictions from all first-level models and feed them to the meta-learner.

For NLP competitions, a **linear meta-learner** (Ridge or logistic regression) is preferred over tree-based models because: (1) the features (model probabilities) are already well-calibrated, (2) linear models are less prone to overfitting with few features (typically 3-10 model predictions), (3) they are faster to optimize.

```python
from sklearn.linear_model import Ridge, LogisticRegression
import numpy as np

# Assuming we have OOF predictions from 3 models
# oof_deberta shape: (n_train, num_classes)
# oof_roberta shape: (n_train, num_classes)
# oof_electra shape: (n_train, num_classes)

# Stack features
X_meta = np.concatenate([oof_deberta, oof_roberta, oof_electra], axis=1)
# shape: (n_train, num_classes * 3)

# For regression: use Ridge
meta_model = Ridge(alpha=1.0)
meta_model.fit(X_meta, y_train)

# For classification: use LogisticRegression
meta_model = LogisticRegression(C=1.0, max_iter=1000)
meta_model.fit(X_meta, y_train)

# Test-time stacking
X_test_meta = np.concatenate([test_deberta, test_roberta, test_electra], axis=1)
final_preds = meta_model.predict(X_test_meta)
```

### Code: NLP Ensemble with Rank Averaging

**Rank averaging** converts predictions to ranks (percentiles) before averaging, which handles different prediction scales across models. This is particularly useful when ensembling models that produce differently calibrated probabilities.

```python
import numpy as np
from scipy.stats import rankdata
from scipy.optimize import minimize


def rank_average(predictions_list, weights=None):
    """
    Rank-average ensemble: convert each model's predictions to ranks,
    then average ranks, then convert back to probabilities.

    predictions_list: list of arrays, each shape (n_samples,) or (n_samples, n_classes)
    weights: optional array of weights per model (default: uniform)
    """
    n_models = len(predictions_list)
    if weights is None:
        weights = np.ones(n_models) / n_models
    else:
        weights = np.array(weights) / np.sum(weights)

    if predictions_list[0].ndim == 1:
        # Single-column predictions (regression or binary)
        n_samples = len(predictions_list[0])
        ranked = np.zeros((n_models, n_samples))
        for i, preds in enumerate(predictions_list):
            ranked[i] = rankdata(preds) / len(preds)  # normalize to [0, 1]

        avg_ranks = np.average(ranked, axis=0, weights=weights)
        return avg_ranks

    else:
        # Multi-class predictions: rank-average each class column
        n_samples, n_classes = predictions_list[0].shape
        result = np.zeros((n_samples, n_classes))

        for c in range(n_classes):
            class_preds = [preds[:, c] for preds in predictions_list]
            result[:, c] = rank_average(class_preds, weights)

        # Re-normalize rows to sum to 1
        row_sums = result.sum(axis=1, keepdims=True)
        result = result / np.maximum(row_sums, 1e-9)
        return result


def simple_average(predictions_list, weights=None):
    """Weighted average ensemble."""
    if weights is None:
        return np.mean(predictions_list, axis=0)
    weights = np.array(weights) / np.sum(weights)
    return np.average(predictions_list, axis=0, weights=weights)


def optimize_ensemble_weights(oof_predictions_list, true_labels, metric_fn, n_classes=None):
    """
    Optimize ensemble weights to maximize a given metric.

    oof_predictions_list: list of OOF predictions, each (n_samples,) or (n_samples, n_classes)
    true_labels: ground truth labels
    metric_fn: function(y_true, y_pred) -> float (higher is better)
    """
    n_models = len(oof_predictions_list)

    def objective(weights):
        weights = np.abs(weights)
        weights = weights / weights.sum()
        blended = np.average(oof_predictions_list, axis=0, weights=weights)
        score = metric_fn(true_labels, blended)
        return -score  # minimize negative metric

    # Initial uniform weights
    x0 = np.ones(n_models) / n_models

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-6},
    )

    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()
    best_score = -result.fun

    print(f"Optimal weights: {optimal_weights}")
    print(f"Best score: {best_score:.5f}")

    return optimal_weights, best_score


# --- Full ensemble pipeline ---
def ensemble_pipeline(
    model_predictions,
    model_names,
    true_labels=None,
    metric_fn=None,
    method="rank_average",
):
    """
    Complete ensemble pipeline supporting multiple methods.

    model_predictions: dict of {model_name: {"oof": oof_preds, "test": test_preds}}
    """
    oof_list = [model_predictions[name]["oof"] for name in model_names]
    test_list = [model_predictions[name]["test"] for name in model_names]

    if method == "simple_average":
        oof_ensemble = simple_average(oof_list)
        test_ensemble = simple_average(test_list)

    elif method == "rank_average":
        oof_ensemble = rank_average(oof_list)
        test_ensemble = rank_average(test_list)

    elif method == "optimized":
        assert true_labels is not None and metric_fn is not None
        weights, score = optimize_ensemble_weights(oof_list, true_labels, metric_fn)
        oof_ensemble = simple_average(oof_list, weights)
        test_ensemble = simple_average(test_list, weights)

    else:
        raise ValueError(f"Unknown method: {method}")

    return oof_ensemble, test_ensemble


# --- Example usage ---
# predictions = {
#     "deberta_v3_large": {
#         "oof": deberta_oof,    # shape: (n_train, n_classes)
#         "test": deberta_test,  # shape: (n_test, n_classes)
#     },
#     "roberta_large": {
#         "oof": roberta_oof,
#         "test": roberta_test,
#     },
#     "electra_large": {
#         "oof": electra_oof,
#         "test": electra_test,
#     },
# }
#
# oof_final, test_final = ensemble_pipeline(
#     predictions,
#     model_names=["deberta_v3_large", "roberta_large", "electra_large"],
#     true_labels=train_labels,
#     metric_fn=lambda y_true, y_pred: -log_loss(y_true, y_pred),
#     method="optimized",
# )
```

---

## Data Augmentation for NLP

### Back-Translation

**Back-translation** translates text from the source language to a pivot language and then back. For example, English -> French -> English produces a paraphrase that preserves meaning but varies phrasing. The quality depends on the translation model; using high-quality models (MarianMT, Helsinki-NLP, or Google Translate API) produces better augmentations.

```python
from transformers import MarianMTModel, MarianTokenizer


def back_translate(texts, src_lang="en", pivot_lang="fr", batch_size=16, device="cuda"):
    """
    Back-translation augmentation using MarianMT.
    Translates en->pivot->en to generate paraphrases.
    """
    # Forward model: en -> pivot
    fwd_name = f"Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}"
    fwd_tokenizer = MarianTokenizer.from_pretrained(fwd_name)
    fwd_model = MarianMTModel.from_pretrained(fwd_name).to(device)

    # Backward model: pivot -> en
    bwd_name = f"Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}"
    bwd_tokenizer = MarianTokenizer.from_pretrained(bwd_name)
    bwd_model = MarianMTModel.from_pretrained(bwd_name).to(device)

    augmented = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Forward translation
        fwd_inputs = fwd_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            fwd_outputs = fwd_model.generate(**fwd_inputs, max_length=512, num_beams=4)
        pivot_texts = fwd_tokenizer.batch_decode(fwd_outputs, skip_special_tokens=True)

        # Backward translation
        bwd_inputs = bwd_tokenizer(
            pivot_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            bwd_outputs = bwd_model.generate(**bwd_inputs, max_length=512, num_beams=4)
        back_texts = bwd_tokenizer.batch_decode(bwd_outputs, skip_special_tokens=True)

        augmented.extend(back_texts)

    return augmented


# Usage:
# original = ["This movie was fantastic and well-directed."]
# paraphrased = back_translate(original, pivot_lang="de")
# -> ["This film was great and well directed."]
```

Common pivot languages: French (`fr`), German (`de`), Russian (`ru`), Chinese (`zh`). Using multiple pivot languages and keeping all augmented versions further increases diversity. However, augmented samples should typically receive a lower weight (e.g., `sample_weight=0.5`) during training to prevent the model from learning translation artifacts.

### Token-Level Augmentations

Simple token-level augmentations from the **EDA (Easy Data Augmentation)** paper (Wei and Zou, 2019):

1. **Synonym replacement**: replace `n` words with their WordNet synonyms. Parameter: `alpha_sr=0.1` (fraction of words to replace).

2. **Random insertion**: insert a random synonym of a random word at a random position. Parameter: `alpha_ri=0.1`.

3. **Random swap**: swap the positions of two random words. Parameter: `alpha_rs=0.1`.

4. **Random deletion**: delete each word with probability `p_rd=0.1`.

```python
import random
import nltk
from nltk.corpus import wordnet

# nltk.download("wordnet")
# nltk.download("omw-1.4")


def get_synonyms(word):
    """Get synonyms from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)


def synonym_replacement(words, n=1):
    """Replace n words with their synonyms."""
    new_words = words.copy()
    random_word_indices = list(range(len(words)))
    random.shuffle(random_word_indices)
    num_replaced = 0
    for idx in random_word_indices:
        synonyms = get_synonyms(words[idx])
        if synonyms:
            synonym = random.choice(synonyms)
            new_words[idx] = synonym
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words


def random_insertion(words, n=1):
    """Insert n random synonyms at random positions."""
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(words):
    synonyms = []
    counter = 0
    while not synonyms:
        random_word = random.choice(words)
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(words))
    words.insert(random_idx, synonym)


def random_swap(words, n=1):
    """Swap n pairs of words."""
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) < 2:
            break
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words


def random_deletion(words, p=0.1):
    """Delete each word with probability p."""
    if len(words) <= 1:
        return words
    new_words = [w for w in words if random.random() > p]
    if not new_words:
        return [random.choice(words)]
    return new_words


def eda_augment(text, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=4):
    """Apply EDA augmentation to generate multiple augmented versions."""
    words = text.split()
    n = max(1, int(alpha_sr * len(words)))

    augmented = []
    for _ in range(num_aug):
        choice = random.random()
        if choice < 0.25:
            new_words = synonym_replacement(words, n)
        elif choice < 0.5:
            new_words = random_insertion(words, n)
        elif choice < 0.75:
            new_words = random_swap(words, n)
        else:
            new_words = random_deletion(words, p_rd)
        augmented.append(" ".join(new_words))

    return augmented
```

### Contextual Augmentation with MLM

**Contextual augmentation** uses a masked language model (MLM) to replace words with contextually appropriate alternatives. This produces more coherent augmentations than random synonym replacement because the replacements are conditioned on the surrounding context.

```python
from transformers import pipeline
import random

# Use a masked LM for contextual augmentation
mlm = pipeline("fill-mask", model="roberta-base", top_k=5, device=0)


def contextual_augment(text, n_replacements=3, mask_token="<mask>"):
    """
    Replace n random words with MLM predictions.
    """
    words = text.split()
    if len(words) <= n_replacements:
        return text

    # Choose random positions to mask (skip very short words)
    candidates = [i for i, w in enumerate(words) if len(w) > 3]
    if not candidates:
        return text

    positions = random.sample(candidates, min(n_replacements, len(candidates)))

    for pos in positions:
        masked_text = words.copy()
        masked_text[pos] = mask_token
        masked_sentence = " ".join(masked_text)

        try:
            predictions = mlm(masked_sentence)
            if predictions:
                # Pick a random prediction from top-5
                replacement = random.choice(predictions)["token_str"].strip()
                words[pos] = replacement
        except Exception:
            continue

    return " ".join(words)
```

### When NOT to Augment

Data augmentation in NLP competitions often **hurts** performance, unlike in computer vision where it almost always helps. Reasons:

1. **Transformer pretraining already provides massive augmentation**: the model has seen billions of text examples during pretraining, so additional augmented data provides diminishing returns.

2. **Semantic corruption**: augmentations can change the meaning of text. Replacing "not good" with "not excellent" via synonym replacement changes sentiment. Back-translation can lose nuances crucial for fine-grained tasks.

3. **Label noise amplification**: if the original label depends on specific phrasing (e.g., readability scoring, argument effectiveness), augmentation can create samples whose labels no longer match the text.

4. **Overfitting to augmented patterns**: the model may learn that augmented text (which has subtle translation artifacts or unnatural phrasing) corresponds to certain labels, creating a distribution mismatch with the test set.

**When augmentation helps**: (1) very small datasets (fewer than 1,000 training samples), (2) multilingual tasks where back-translation creates natural cross-lingual bridges, (3) token-level tasks (NER) where insertion/deletion/swap augmentations create more diverse entity boundary patterns, (4) when used in pseudo-labeling contexts where augmented text is used for TTA rather than as training data.

**When augmentation hurts**: (1) medium to large datasets (5K+ samples), (2) fine-grained regression tasks (readability, quality scoring), (3) tasks where specific words or phrases are the key signal, (4) when the augmentation is low-quality (poor translation, random replacement of key words).

---

## Common Mistakes

**Using the wrong tokenizer for the model.** Every pretrained model has a paired tokenizer. Using `BertTokenizer` with a RoBERTa model or using `AutoTokenizer.from_pretrained("bert-base-uncased")` with `AutoModel.from_pretrained("roberta-base")` produces garbage. Always load the tokenizer from the same checkpoint: `AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")` for DeBERTa-v3-large. The special tokens differ (`[CLS]`/`[SEP]` for BERT, `<s>`/`</s>` for RoBERTa, `[CLS]`/`[SEP]` for DeBERTa), and vocabulary sizes vary.

**Not handling special tokens properly.** When computing mean pooling, you must exclude `[CLS]`, `[SEP]`, and `[PAD]` tokens. When doing token classification, special tokens should receive label `-100` (the ignore index). When concatenating two texts for pair classification, the separator token(s) must match the model's expected format (BERT uses `[SEP]`, RoBERTa uses `</s></s>`). Verify by checking `tokenizer.encode("text A", "text B")` and inspecting the output.

**Sequence length too short.** Setting `max_length=128` to save memory when the text distribution has a median of 300 tokens throws away critical information. Always analyze the token-length distribution first (see the sequence length selection code above). For competitions with long texts (essays, legal documents, clinical notes), use at least the 95th percentile of the distribution, even if it requires smaller batch sizes and more gradient accumulation.

**Not using mixed precision.** Training DeBERTa-v3-large in FP32 on a 16GB GPU limits you to batch_size=1 with max_length=512. Switching to FP16 (or BF16 on A100) roughly halves memory usage and increases throughput by 50-100%. There is no measurable accuracy loss. Use `torch.cuda.amp.autocast` and `GradScaler` for FP16, or the HuggingFace `Trainer` with `fp16=True`. On A100, always prefer BF16 over FP16 because it has better numerical stability (no need for loss scaling).

**Overfitting to small datasets without regularization.** On datasets with fewer than 5,000 samples, it is easy to overfit within 2-3 epochs. Mitigations: (1) use multi-sample dropout (n_drops=5, p=0.2-0.3), (2) apply adversarial training (FGM or AWP), (3) reinitialize the last 1-2 transformer layers, (4) use aggressive weight decay (0.05-0.1), (5) reduce the number of epochs (2-3 instead of 5), (6) use Stochastic Weight Averaging (SWA) to smooth the loss landscape, (7) consider freezing lower transformer layers and only fine-tuning the top 6-12 layers.

**Ignoring cross-validation.** Training on the full dataset without cross-validation gives no reliable estimate of generalization. Use stratified k-fold CV (k=5 is standard) for classification, GroupKFold when data has natural groups (e.g., essays from the same student), and MultilabelStratifiedKFold (from `iterstrat` library) for multi-label tasks. Never tune hyperparameters on the test set or on the public leaderboard -- this leads to overfitting the leaderboard (known as "LB probing").

**Not freezing embeddings early in training.** For very small datasets, freezing the embedding layer and bottom transformer layers for the first 1-2 epochs can reduce overfitting. Then unfreeze all layers with a smaller learning rate for the remaining epochs. This is a form of **gradual unfreezing** (Howard and Ruder, 2018).

**Forgetting gradient clipping.** Training transformers without gradient clipping (`max_norm=1.0`) can lead to training instability, especially with FP16. Gradient explosions are particularly common when using adversarial training (AWP/FGM) or when fine-tuning very large models. Always use `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

**Not saving the tokenizer with the model.** When saving model checkpoints, always save the tokenizer alongside: `tokenizer.save_pretrained(save_dir)` and `model.save_pretrained(save_dir)`. For Kaggle submissions, this ensures reproducibility and avoids downloading from HuggingFace Hub during inference (which may fail due to network restrictions).

**Using the default pooler output.** BERT-family models have a `pooler_output` that applies an additional dense layer + tanh activation to the CLS token. This pooler was trained for the next-sentence prediction task during pretraining and is often suboptimal for downstream tasks. Always use the raw `last_hidden_state[:, 0, :]` (CLS token) or mean pooling instead of `outputs.pooler_output`.

---

## Resources

**Competition solution repositories and write-ups:**

- Kaggle Discussion Forums: every completed competition has a dedicated discussion section where top teams post their solution write-ups. Search for "[competition name] 1st place solution" on Kaggle.
- GitHub repositories: most gold-medal solutions are open-sourced. Search "[competition name] kaggle gold" on GitHub.
- The Kaggle "Competitions" tab under each user's profile shows their medals and links to their solution code.

**Libraries:**

- `transformers` (HuggingFace): the core library for loading pretrained models and tokenizers. Version 4.20+ recommended for DeBERTa-v3 support. Install: `pip install transformers>=4.20.0`.
- `tokenizers`: fast tokenizer backend. Install: `pip install tokenizers`.
- `datasets` (HuggingFace): efficient data loading and preprocessing. Install: `pip install datasets`.
- `sentence-transformers`: for cross-encoder and bi-encoder architectures. Install: `pip install sentence-transformers`.
- `accelerate` (HuggingFace): simplified multi-GPU and mixed-precision training. Install: `pip install accelerate`.
- `iterstrat`: MultilabelStratifiedKFold for multi-label CV. Install: `pip install iterative-stratification`.
- `wandb`: experiment tracking. Install: `pip install wandb`.

**Key papers:**

- DeBERTa: "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" (He et al., 2020). Introduces disentangled attention and enhanced mask decoder.
- DeBERTa-v3: "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing" (He et al., 2021).
- ELECTRA: "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators" (Clark et al., 2020).
- RoBERTa: "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (Liu et al., 2019).
- Longformer: "Longformer: The Long-Document Transformer" (Beltagy et al., 2020).
- AWP: "Adversarial Weight Perturbation Helps Robust Generalization" (Wu et al., 2020).
- Multi-sample Dropout: "Multi-Sample Dropout for Accelerated Training and Better Generalization" (Inoue, 2019).
- EDA: "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" (Wei and Zou, 2019).
- Mixout: "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models" (Lee et al., 2020).

**Practical guides:**

- HuggingFace documentation: `https://huggingface.co/docs/transformers/` -- comprehensive API reference and task-specific tutorials.
- The Kaggle NLP Getting Started competitions ("Natural Language Processing with Disaster Tweets", "Contradictory, My Dear Watson") are excellent for practicing competition NLP from scratch.
- Chris Deotte's Kaggle notebooks: consistently high-quality, well-documented competition notebooks that demonstrate best practices for NLP fine-tuning, ensembling, and inference optimization.
