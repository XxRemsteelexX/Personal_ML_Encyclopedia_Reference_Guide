# 25. Transformer Architecture

## Overview

The Transformer, introduced in "Attention Is All You Need" (Vaswani et al., 2017), revolutionized NLP by relying entirely on attention mechanisms without recurrence or convolutions. It became the foundation for BERT, GPT, and all modern large language models.

**Key Innovation:** Parallel processing of sequences using self-attention, enabling efficient training on GPUs and capturing long-range dependencies.

---

## 25.1 Core Architecture

### High-Level Structure

```
Input Sequence
    v
[Encoder Stack] --> [Decoder Stack]
                      v
                Output Sequence
```

**Encoder:**
- Processes input sequence
- N identical layers (typically 6 or 12)
- Each layer: Multi-Head Attention + Feed-Forward

**Decoder:**
- Generates output sequence auto-regressively
- N identical layers
- Each layer: Masked Multi-Head Attention + Cross-Attention + Feed-Forward

---

## 25.2 Input Embeddings and Positional Encoding

### Token Embeddings

Convert tokens to dense vectors:

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # Scale embeddings by sqrtd_model (from paper)
        return self.embedding(x) * math.sqrt(self.d_model)
```

### Positional Encoding

**Problem:** Attention has no notion of position/order.

**Solution:** Add positional information to embeddings.

**Formula:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- pos = position in sequence
- i = dimension index
- d_model = embedding dimension

**Implementation:**

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x
```

**Why Sinusoidal?**
- Allows extrapolation to longer sequences (not seen during training)
- Relative positions can be represented as linear functions
- `PE(pos+k)` can be represented as linear function of `PE(pos)`

**Alternative:** Learned positional embeddings (used in BERT):

```python
self.pos_embedding = nn.Embedding(max_len, d_model)
```

---

## 25.3 Encoder Layer

### Components

Each encoder layer contains:
1. Multi-Head Self-Attention
2. Layer Normalization + Residual Connection
3. Position-wise Feed-Forward Network
4. Layer Normalization + Residual Connection

### Architecture

```
Input
  v
Multi-Head Attention
  v
Add & Norm (Residual + LayerNorm)
  v
Feed-Forward Network
  v
Add & Norm
  v
Output
```

### Implementation

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # Position-wise feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
```

### Full Encoder Stack

```python
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
```

---

## 25.4 Decoder Layer

### Components

Each decoder layer contains:
1. Masked Multi-Head Self-Attention
2. Add & Norm
3. Cross-Attention (attend to encoder output)
4. Add & Norm
5. Feed-Forward Network
6. Add & Norm

### Architecture

```
Target Input
  v
Masked Multi-Head Attention (self)
  v
Add & Norm
  v
Cross-Attention (to encoder)
  v
Add & Norm
  v
Feed-Forward Network
  v
Add & Norm
  v
Output
```

### Implementation

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention to encoder
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
```

### Causal Mask for Decoder

```python
def create_causal_mask(seq_len, device):
    """
    Create mask to prevent attending to future positions.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
```

---

## 25.5 Complete Transformer Model

### Full Architecture

```python
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_len=5000,
        dropout=0.1
    ):
        super().__init__()

        # Embeddings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Encoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout
        )

        # Decoder
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_heads, d_ff, dropout
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source
        src_emb = self.dropout(self.positional_encoding(self.src_embedding(src)))
        encoder_output = self.encoder(src_emb, src_mask)

        # Decode target
        tgt_emb = self.dropout(self.positional_encoding(self.tgt_embedding(tgt)))
        decoder_output = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        output = self.output_projection(decoder_output)

        return output

    def encode(self, src, src_mask=None):
        src_emb = self.dropout(self.positional_encoding(self.src_embedding(src)))
        return self.encoder(src_emb, src_mask)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        tgt_emb = self.dropout(self.positional_encoding(self.tgt_embedding(tgt)))
        return self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
```

---

## 25.6 Training the Transformer

### Loss Function

**Label Smoothing Cross-Entropy:**

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, pad_idx=0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, predictions, targets):
        # predictions: (batch * seq_len, vocab_size)
        # targets: (batch * seq_len)

        # Create smoothed distribution
        true_dist = predictions.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0

        # Mask padding
        mask = (targets != self.pad_idx).unsqueeze(1)
        true_dist = true_dist * mask

        return self.criterion(predictions, true_dist.detach())
```

### Optimizer: Warmup + Decay

**Learning Rate Schedule from paper:**

```
lr = d_model^(-0.5) x min(step^(-0.5), step x warmup_steps^(-1.5))
```

**Implementation:**

```python
class NoamOpt:
    def __init__(self, d_model, warmup_steps, optimizer):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.get_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def get_rate(self):
        return self.d_model ** (-0.5) * min(
            self._step ** (-0.5),
            self._step * self.warmup_steps ** (-1.5)
        )

# Usage
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = NoamOpt(d_model=512, warmup_steps=4000, optimizer=optimizer)
```

### Training Loop

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)

        # Create masks
        src_mask = create_padding_mask(src, pad_idx=0)
        tgt_mask = create_causal_mask(tgt.size(1), device) & create_padding_mask(tgt, pad_idx=0)

        # Forward pass
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)  # Shift target

        # Compute loss
        output = output.reshape(-1, output.size(-1))
        targets = tgt[:, 1:].reshape(-1)  # Shift target
        loss = criterion(output, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

---

## 25.7 Inference

### Greedy Decoding

```python
def greedy_decode(model, src, src_mask, max_len, start_token, end_token):
    model.eval()

    # Encode source
    encoder_output = model.encode(src, src_mask)

    # Initialize decoder input with start token
    tgt = torch.ones(1, 1).fill_(start_token).type_as(src)

    for i in range(max_len - 1):
        tgt_mask = create_causal_mask(tgt.size(1), src.device)

        # Decode
        output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        output = model.output_projection(output[:, -1, :])

        # Get next token
        next_token = output.argmax(dim=-1).unsqueeze(0)

        # Append to target
        tgt = torch.cat([tgt, next_token], dim=1)

        # Stop if end token generated
        if next_token.item() == end_token:
            break

    return tgt
```

### Beam Search

```python
def beam_search(model, src, src_mask, max_len, start_token, end_token, beam_size=5):
    model.eval()

    # Encode source
    encoder_output = model.encode(src, src_mask)

    # Initialize beam
    sequences = [[start_token]]
    scores = [0.0]

    for _ in range(max_len):
        all_candidates = []

        for seq, score in zip(sequences, scores):
            if seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue

            # Prepare input
            tgt = torch.tensor([seq]).to(src.device)
            tgt_mask = create_causal_mask(tgt.size(1), src.device)

            # Decode
            output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            output = model.output_projection(output[:, -1, :])
            log_probs = F.log_softmax(output, dim=-1)

            # Get top-k candidates
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                candidate = seq + [topk_indices[0, i].item()]
                candidate_score = score + topk_log_probs[0, i].item()
                all_candidates.append((candidate, candidate_score))

        # Select top beam_size candidates
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = [seq for seq, score in ordered[:beam_size]]
        scores = [score for seq, score in ordered[:beam_size]]

        # Stop if all sequences end with end_token
        if all(seq[-1] == end_token for seq in sequences):
            break

    return sequences[0]  # Return best sequence
```

---

## 25.8 Key Hyperparameters

### Original Paper (Base Model)

```python
d_model = 512          # Model dimension
num_heads = 8          # Attention heads
num_layers = 6         # Encoder/decoder layers
d_ff = 2048           # Feed-forward dimension
dropout = 0.1
warmup_steps = 4000
```

### Large Model (Big)

```python
d_model = 1024
num_heads = 16
num_layers = 6
d_ff = 4096
```

### Modern LLMs (e.g., GPT-3)

```python
d_model = 12288        # 96 x 128
num_heads = 96
num_layers = 96
d_ff = 49152           # 4 x d_model
```

---

## 25.9 Variants and Extensions

### 1. Pre-Layer Normalization

**Original (Post-LN):**
```
x = LayerNorm(x + Sublayer(x))
```

**Pre-LN (more stable):**
```
x = x + Sublayer(LayerNorm(x))
```

**Benefits:**
- More stable training (especially for deep models)
- Less sensitive to learning rate
- Used in GPT-2, GPT-3

### 2. Relative Positional Encoding

**Problem:** Absolute position less important than relative distance.

**Solution (Transformer-XL):**
```
Attention(Q, K, V) = softmax((QK^T + Q R^T) / sqrtd_k) V
```

Where R encodes relative positions.

### 3. Rotary Position Embedding (RoPE)

**Used in modern LLMs (LLaMA, GPT-NeoX):**
- Encodes absolute position with rotation matrices
- Naturally captures relative position
- Better extrapolation to longer sequences

```python
# Simplified RoPE
def apply_rotary_pos_emb(x, cos, sin):
    # x: (batch, seq_len, num_heads, head_dim)
    x1, x2 = x[..., ::2], x[..., 1::2]
    # Apply rotation
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)
    return x_rotated
```

### 4. Flash Attention

**Memory-efficient attention (2022-2025):**
- Fused attention kernel
- O(N) memory instead of O(N^2)
- 2-4x faster training
- Standard in 2025 implementations

---

## 25.10 Encoder-Only vs Decoder-Only

### Encoder-Only (BERT)

```
Architecture: Just encoder stack
Task: Bidirectional context
Use: Classification, NER, QA
```

**Advantages:**
- See full context (past + future)
- Better for understanding tasks

**Examples:** BERT, RoBERTa, ELECTRA

### Decoder-Only (GPT)

```
Architecture: Just decoder stack (self-attention only, no cross-attention)
Task: Autoregressive generation
Use: Text generation, few-shot learning
```

**Advantages:**
- Simpler architecture
- Easy to scale
- Natural for generation

**Examples:** GPT-2, GPT-3, GPT-4, LLaMA

### Encoder-Decoder (T5)

```
Architecture: Full transformer
Task: Seq2seq
Use: Translation, summarization, any task as text-to-text
```

**Examples:** T5, BART, mT5

---

## 25.11 Computational Complexity

### Attention Complexity

**Self-Attention:** O(n^2 x d)
- n = sequence length
- d = model dimension

**Feed-Forward:** O(n x d^2)

**Total per layer:** O(n^2 x d + n x d^2)

### Scaling Bottleneck

For long sequences (n large):
- Attention dominates: O(n^2)
- Memory: O(n^2) for attention matrix

**Solutions:**
- Sparse attention patterns
- Local attention windows
- Linear attention approximations
- Flash Attention (memory optimization)

---

## 25.12 Complete Working Example

```python
# Full transformer for translation
class TranslationTransformer:
    def __init__(self, src_vocab_size, tgt_vocab_size):
        self.model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            dropout=0.1
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        self.scheduler = NoamOpt(d_model=512, warmup_steps=4000, optimizer=self.optimizer)
        self.criterion = LabelSmoothingLoss(vocab_size=tgt_vocab_size, smoothing=0.1)

    def train_step(self, src, tgt):
        self.model.train()

        # Create masks
        src_mask = create_padding_mask(src)
        tgt_mask = create_causal_mask(tgt.size(1)) & create_padding_mask(tgt)

        # Forward
        output = self.model(src, tgt[:, :-1], src_mask, tgt_mask)

        # Loss
        loss = self.criterion(
            output.reshape(-1, output.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scheduler.step()

        return loss.item()

    def translate(self, src_sentence, src_vocab, tgt_vocab, max_len=100):
        self.model.eval()

        # Tokenize and encode source
        src_tokens = src_vocab.encode(src_sentence)
        src = torch.tensor([src_tokens])

        # Greedy decode
        tgt = greedy_decode(
            self.model,
            src,
            src_mask=None,
            max_len=max_len,
            start_token=tgt_vocab.start_idx,
            end_token=tgt_vocab.end_idx
        )

        # Decode to text
        translation = tgt_vocab.decode(tgt[0].tolist())
        return translation
```

---

## Summary

**Key Contributions:**
1. **Self-Attention:** Relate all positions in parallel
2. **Positional Encoding:** Add position information
3. **Multi-Head Attention:** Multiple representation subspaces
4. **Residual Connections:** Enable deep networks
5. **Layer Normalization:** Stable training

**Why Transformers Won:**
- Parallelizable (unlike RNNs)
- Long-range dependencies (unlike CNNs)
- Scalable (enabled GPT-3, GPT-4)
- Flexible (works for vision, audio, multimodal)

**2025 Impact:**
- Foundation of all modern LLMs
- Scaled to 100B+ parameters
- Adapted to vision (ViT), audio, multimodal
- Still evolving (efficient attention, better positional encodings)

---

## Resources

### Original Paper
- "Attention Is All You Need" (Vaswani et al., 2017)

### Tutorials
- The Annotated Transformer: http://nlp.seas.harvard.edu/annotated-transformer/
- The Illustrated Transformer: http://jalammar.github.io/illustrated-transformer/

### Implementations
- Hugging Face Transformers: https://github.com/huggingface/transformers
- fairseq: https://github.com/facebookresearch/fairseq
- Tensor2Tensor: https://github.com/tensorflow/tensor2tensor
