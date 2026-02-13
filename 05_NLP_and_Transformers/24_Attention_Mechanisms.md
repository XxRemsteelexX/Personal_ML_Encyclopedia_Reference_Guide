# 24. Attention Mechanisms

## Overview

Attention mechanisms allow neural networks to focus on specific parts of the input when producing each output. Introduced to solve the bottleneck problem in sequence-to-sequence models, attention became the foundation for the Transformer architecture that dominates NLP in 2025.

**Core Idea:** Instead of compressing entire input into fixed-size vector, dynamically attend to relevant parts.

---

## 24.1 The Bottleneck Problem

### Seq2Seq Without Attention

```
Encoder: x_1, x_2, ..., x_n --> single context vector (c)
Decoder: c --> y_1, y_2, ..., y_m
```

**Problem:**
- Single context vector must encode entire input sequence
- Information bottleneck, especially for long sequences
- Early tokens often forgotten by the time decoding starts

**Example (Translation):**
```
Input:  "The cat sat on the mat"
Context: [0.2, -0.5, 0.8, ...]  # All info compressed here!
Output: "Le chat s'est assis sur le tapis"
```

The context vector struggles to remember "cat", "sat", and "mat" simultaneously.

---

## 24.2 Bahdanau Attention (Additive Attention)

**Introduced by:** Bahdanau et al. (2014)

**Key Innovation:** Decoder attends to all encoder hidden states, not just final one.

### Mechanism

```python
# For each decoder step t:

# 1. Compute alignment scores
e_tj = score(s_{t-1}, h_j)  # How well decoder state s_{t-1} matches encoder state h_j

# 2. Normalize scores to get attention weights
alpha_tj = exp(e_tj) / sum_k exp(e_tk)

# 3. Compute context vector as weighted sum
c_t = sum_j alpha_tj * h_j

# 4. Use context vector in decoder
s_t = f(s_{t-1}, y_{t-1}, c_t)
```

### Score Function (Additive)

```python
score(s, h) = v^T tanh(W_1 s + W_2 h)
```

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)

        seq_len = encoder_outputs.size(1)

        # Repeat decoder hidden state for each encoder output
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # Shape: (batch, seq_len, hidden_size)

        # Compute alignment scores
        energy = torch.tanh(self.W_s(decoder_hidden) + self.W_h(encoder_outputs))
        # Shape: (batch, seq_len, hidden_size)

        attention_scores = self.v(energy).squeeze(2)
        # Shape: (batch, seq_len)

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        # Shape: (batch, seq_len)

        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # Shape: (batch, 1, hidden_size)

        context = context.squeeze(1)
        # Shape: (batch, hidden_size)

        return context, attention_weights
```

---

## 24.3 Luong Attention (Multiplicative Attention)

**Introduced by:** Luong et al. (2015)

**Simpler than Bahdanau, often performs similarly.**

### Score Functions

**1. Dot Product (simplest):**
```python
score(s, h) = s^T h
```

**2. General (with weight matrix):**
```python
score(s, h) = s^T W h
```

**3. Concat (similar to Bahdanau):**
```python
score(s, h) = v^T tanh(W [s; h])
```

### Implementation

```python
class LuongAttention(nn.Module):
    def __init__(self, hidden_size, method='dot'):
        super(LuongAttention, self).__init__()
        self.method = method

        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == 'concat':
            self.W = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)

        if self.method == 'dot':
            # Dot product: s^T H
            attention_scores = torch.bmm(
                encoder_outputs,
                decoder_hidden.unsqueeze(2)
            ).squeeze(2)

        elif self.method == 'general':
            # General: s^T W H
            transformed = self.W(encoder_outputs)
            attention_scores = torch.bmm(
                transformed,
                decoder_hidden.unsqueeze(2)
            ).squeeze(2)

        elif self.method == 'concat':
            # Concat: v^T tanh(W [s; h])
            seq_len = encoder_outputs.size(1)
            decoder_repeated = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            concat = torch.cat([decoder_repeated, encoder_outputs], dim=2)
            energy = torch.tanh(self.W(concat))
            attention_scores = self.v(energy).squeeze(2)

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights
```

---

## 24.4 Self-Attention (Intra-Attention)

**Key Idea:** Attend to different positions within the same sequence.

**Use Case:** Relate different words in a sentence to each other.

### Example

```
Sentence: "The animal didn't cross the street because it was too tired"

Query: "it"
Attention: Should focus on "animal" (not "street")
```

### Mechanism

For each position i in sequence:
1. Query (Q_i): What I'm looking for
2. Key (K_j): What I offer (for all positions j)
3. Value (V_j): What I actually provide (for all positions j)

```python
# For position i:
attention_weights = softmax(Q_i * K / sqrtd_k)
output_i = sum_j attention_weights_j * V_j
```

### Implementation

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask=None):
        N = queries.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention: Q*K^T / sqrtd_k
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
```

---

## 24.5 Scaled Dot-Product Attention

**The attention used in Transformers.**

### Formula

```
Attention(Q, K, V) = softmax(QK^T / sqrtd_k) V
```

Where:
- Q = Queries matrix (n x d_k)
- K = Keys matrix (m x d_k)
- V = Values matrix (m x d_v)
- d_k = dimension of keys
- sqrtd_k = scaling factor (prevents softmax saturation)

### Why Scaling?

Without scaling, for large d_k:
- Dot products grow large in magnitude
- Softmax saturates (gradients vanish)
- Model harder to train

**Example:**
```python
# Without scaling
Q*K^T = [100, 95, 5, 3]  # One value dominates
softmax([100, 95, 5, 3]) ~= [0.88, 0.12, 0, 0]  # Gradient ~= 0 for small values

# With scaling (d_k = 64)
Q*K^T / sqrt64 = [12.5, 11.9, 0.63, 0.38]
softmax(...) ~= [0.55, 0.40, 0.02, 0.01]  # Better gradient flow
```

### Implementation

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, num_heads, seq_len_q, d_k)
    K: (batch, num_heads, seq_len_k, d_k)
    V: (batch, num_heads, seq_len_v, d_v)
    """
    d_k = Q.size(-1)

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores shape: (batch, num_heads, seq_len_q, seq_len_k)

    # Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attention_weights, V)
    # output shape: (batch, num_heads, seq_len_q, d_v)

    return output, attention_weights
```

---

## 24.6 Multi-Head Attention

**Key Idea:** Run attention multiple times in parallel with different learned projections.

### Why Multi-Head?

Single attention head focuses on one aspect. Multiple heads can attend to:
- Syntactic relationships (subject-verb)
- Semantic similarities (synonyms)
- Positional patterns (nearby words)
- Long-range dependencies

### Architecture

```
Input --> Linear projections (Q, K, V) x h heads
      --> Scaled dot-product attention x h heads
      --> Concatenate heads
      --> Linear projection
      --> Output
```

### Formula

```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

### Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """Split last dimension into (num_heads, d_k)"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        # Output projection
        output = self.W_o(context)

        return output, attention
```

---

## 24.7 Types of Attention in Transformers

### 1. Encoder Self-Attention

```
Input: Source sentence
Q = K = V = encoder input
Purpose: Relate words in source sentence to each other
```

**Example:**
```
"The animal didn't cross the street because it was too tired"
"it" attends to "animal"
```

### 2. Decoder Self-Attention (Masked)

```
Input: Target sentence
Q = K = V = decoder input
Mask: Prevent attending to future positions
Purpose: Relate words in target sentence (auto-regressive)
```

**Causal Masking:**
```python
def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # Lower triangular matrix

# Example for seq_len=4:
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

Position i can only attend to positions <= i.

### 3. Cross-Attention (Encoder-Decoder)

```
Q = decoder state
K = V = encoder outputs
Purpose: Decoder attends to source sentence
```

**Example (Translation):**
```
Encoder output: "The cat sat" --> [h1, h2, h3]
Decoder generating: "Le" --> attends to [h1, h2, h3]
Next: "chat" --> attends to [h1, h2, h3] (focuses on "cat")
```

---

## 24.8 Attention Visualization

### Visualizing Attention Weights

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_weights, input_tokens, output_tokens):
    """
    attention_weights: (output_len, input_len)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=input_tokens,
        yticklabels=output_tokens,
        cmap='Blues',
        annot=True,
        fmt='.2f'
    )
    plt.xlabel('Input Tokens')
    plt.ylabel('Output Tokens')
    plt.title('Attention Weights')
    plt.show()

# Example usage
input_tokens = ['The', 'cat', 'sat', 'on', 'mat']
output_tokens = ['Le', 'chat', 'assis']
# attention_weights shape: (3, 5)
plot_attention(attention_weights, input_tokens, output_tokens)
```

### Interpreting Attention

**Strong attention (high weight):**
- Model finds input position relevant for current output
- Often aligns related concepts (e.g., "chat" --> "cat")

**Weak attention (low weight):**
- Input position not relevant for current output

**Patterns to look for:**
- Diagonal patterns: Monotonic alignment (translation)
- Sparse patterns: Few relevant inputs
- Distributed patterns: Many inputs contribute

---

## 24.9 Advanced Attention Variants

### 1. Local Attention

**Problem:** Full attention is O(n^2) in sequence length.

**Solution:** Only attend to local window.

```python
def local_attention(Q, K, V, window_size=5):
    seq_len = Q.size(1)
    attention_scores = []

    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)

        local_K = K[:, start:end, :]
        local_V = V[:, start:end, :]

        scores = torch.matmul(Q[:, i:i+1, :], local_K.transpose(-2, -1))
        scores = scores / math.sqrt(Q.size(-1))
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, local_V)
        attention_scores.append(output)

    return torch.cat(attention_scores, dim=1)
```

### 2. Sparse Attention

**Patterns:**
- Fixed patterns (e.g., attend to every k-th position)
- Learned patterns (learn which positions to attend to)

**Benefits:**
- O(nsqrtn) or O(n log n) complexity
- Can handle longer sequences

### 3. Linear Attention

**Reformulate attention to avoid explicit softmax:**

```python
# Standard: O(n^2)
Attention(Q, K, V) = softmax(QK^T) V

# Linear: O(n)
Attention(Q, K, V) = phi(Q) (phi(K)^T V)
```

Where phi is a feature map (e.g., ELU + 1).

---

## 24.10 Practical Tips

### When to Use Attention

**Use attention when:**
- Sequence-to-sequence tasks (translation, summarization)
- Need interpretability (see what model focuses on)
- Variable-length inputs/outputs
- Long-range dependencies important

### Hyperparameter Choices

**Number of heads:**
- Common: 8, 12, 16
- More heads = more diverse attention patterns
- Diminishing returns beyond 16

**Head dimension (d_k):**
- Common: 64
- Smaller = less expressive, faster
- Larger = more expressive, slower

**Attention dropout:**
- Apply dropout to attention weights
- Typical: 0.1
- Prevents overfitting to specific positions

### Common Issues

**1. Attention Collapse**
- All positions get similar attention weights
- Solution: Increase model capacity, better initialization

**2. Over-Attention to [CLS] or [SEP] tokens**
- Model uses special tokens as "no-op" attention
- Solution: May be expected behavior, or mask special tokens

**3. Memory Issues**
- Attention is O(n^2) in memory
- Solution: Gradient checkpointing, local/sparse attention

---

## 24.11 Code Example: Complete Transformer Block

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)

        # Feed-forward network
        self.ffn = nn.Sequential(
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
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x

# Usage
d_model = 512
num_heads = 8
d_ff = 2048

block = TransformerBlock(d_model, num_heads, d_ff)
x = torch.randn(32, 100, 512)  # (batch, seq_len, d_model)
output = block(x)  # (32, 100, 512)
```

---

## Summary

**Evolution:**
1. No attention --> Information bottleneck
2. Bahdanau/Luong --> Attend to encoder states
3. Self-attention --> Relate positions within sequence
4. Multi-head --> Multiple attention patterns
5. Transformers --> Attention is all you need

**Key Concepts:**
- Query, Key, Value paradigm
- Scaled dot-product for stability
- Multi-head for diversity
- Masking for causality

**2025 Status:**
- Foundation of all modern LLMs
- Still active research (efficient attention variants)
- Essential for understanding transformers

---

## Resources

### Papers
- "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2014)
- "Effective Approaches to Attention-based Neural Machine Translation" (Luong et al., 2015)
- "Attention Is All You Need" (Vaswani et al., 2017)

### Visualization Tools
- BertViz: https://github.com/jessevig/bertviz
- Tensor2Tensor: https://github.com/tensorflow/tensor2tensor

### Tutorials
- The Illustrated Transformer: http://jalammar.github.io/illustrated-transformer/
- Attention? Attention!: https://lilianweng.github.io/posts/2018-06-24-attention/
