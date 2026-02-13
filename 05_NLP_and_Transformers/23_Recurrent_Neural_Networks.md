# 23. Recurrent Neural Networks (RNNs)

## Overview

Recurrent Neural Networks (RNNs) are neural networks designed to handle sequential data by maintaining an internal state (memory) that captures information from previous time steps. They were the dominant architecture for sequence modeling before transformers.

**Key Idea:** Process sequences one element at a time, maintaining hidden state that captures context from previous elements.

---

## 23.1 Basic RNN Architecture

### Vanilla RNN

**Structure:**
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

Where:
- `h_t` = hidden state at time t
- `x_t` = input at time t
- `y_t` = output at time t
- `W_hh` = hidden-to-hidden weights
- `W_xh` = input-to-hidden weights
- `W_hy` = hidden-to-output weights

**Visual:**
```
Input:  x_1 --> x_2 --> x_3 --> ... --> x_T
         v     v     v           v
Hidden: h_1 --> h_2 --> h_3 --> ... --> h_T
         v     v     v           v
Output: y_1   y_2   y_3         y_T
```

### Implementation

```python
import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size

        # Weight matrices
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        # Concatenate input and hidden state
        combined = torch.cat((input, hidden), 1)

        # Compute new hidden state
        hidden = torch.tanh(self.i2h(combined))

        # Compute output
        output = self.i2o(combined)

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# Usage
rnn = VanillaRNN(input_size=10, hidden_size=20, output_size=5)
input = torch.randn(1, 10)  # batch_size=1, input_size=10
hidden = rnn.init_hidden(1)

output, next_hidden = rnn(input, hidden)
```

---

## 23.2 Problems with Vanilla RNNs

### Vanishing Gradient Problem

**Issue:** Gradients decay exponentially through time, making it hard to learn long-term dependencies.

**Math:**
```
dL/dh_1 = dL/dh_T * dh_T/dh_{T-1} * ... * dh_2/dh_1

If dh_t/dh_{t-1} < 1, gradient vanishes
If dh_t/dh_{t-1} > 1, gradient explodes
```

**Consequences:**
- Cannot learn dependencies beyond 10-15 time steps
- Difficult to train on long sequences
- Early time steps get almost no gradient signal

### Exploding Gradient Problem

**Issue:** Gradients grow exponentially, causing unstable training.

**Solutions:**
- Gradient clipping: `grad = min(grad, threshold)`
- Better architectures: LSTM, GRU

---

## 23.3 Long Short-Term Memory (LSTM)

**Developed by:** Hochreiter & Schmidhuber (1997)

**Key Innovation:** Cell state acts as "memory highway" with gated control.

### Architecture

**Components:**
1. **Cell State (c_t):** Long-term memory
2. **Hidden State (h_t):** Short-term output
3. **Forget Gate (f_t):** What to remove from cell state
4. **Input Gate (i_t):** What to add to cell state
5. **Output Gate (o_t):** What to output from cell state

**Equations:**
```python
# Forget gate: what to forget from cell state
f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)

# Input gate: what to add to cell state
i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)
c_tilde_t = tanh(W_c * [h_{t-1}, x_t] + b_c)

# Update cell state
c_t = f_t (o) c_{t-1} + i_t (o) c_tilde_t

# Output gate: what to output
o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)
h_t = o_t (o) tanh(c_t)
```

Where:
- sigma = sigmoid function (0 to 1, acts as gate)
- (o) = element-wise multiplication
- tanh = hyperbolic tangent (-1 to 1)

### PyTorch Implementation

```python
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # out shape: (batch, seq_len, hidden_size)
        # hn shape: (num_layers, batch, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out

# Example usage
model = LSTMModel(
    input_size=10,
    hidden_size=128,
    num_layers=2,
    output_size=5
)

# Input: batch_size=32, sequence_length=100, input_size=10
x = torch.randn(32, 100, 10)
output = model(x)  # Shape: (32, 5)
```

### Why LSTM Works

**Gradient Flow:**
- Cell state provides direct path for gradients
- Additive update (`c_t = f_t (o) c_{t-1} + i_t (o) c_tilde_t`) preserves gradients
- Can learn dependencies over 100+ time steps

**Gating Mechanism:**
- Gates learn what to remember/forget
- Sigmoid gates (0-1) allow partial updates
- Protects against vanishing/exploding gradients

---

## 23.4 Gated Recurrent Unit (GRU)

**Developed by:** Cho et al. (2014)

**Key Idea:** Simplified LSTM with fewer parameters, often similar performance.

### Architecture

**Gates:**
1. **Reset Gate (r_t):** How much past info to forget
2. **Update Gate (z_t):** How much to update hidden state

**Equations:**
```python
# Reset gate
r_t = sigma(W_r * [h_{t-1}, x_t])

# Update gate
z_t = sigma(W_z * [h_{t-1}, x_t])

# Candidate hidden state
h_tilde_t = tanh(W * [r_t (o) h_{t-1}, x_t])

# Final hidden state
h_t = (1 - z_t) (o) h_{t-1} + z_t (o) h_tilde_t
```

### PyTorch Implementation

```python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
```

### LSTM vs GRU

| Feature | LSTM | GRU |
|---------|------|-----|
| Parameters | More (4 gates) | Fewer (2 gates) |
| Training Speed | Slower | Faster |
| Memory | Cell + Hidden state | Only Hidden state |
| Performance | Slightly better on complex tasks | Similar on most tasks |
| Use Case | Long sequences, complex patterns | Shorter sequences, faster training |

**Rule of Thumb (2025):**
- Start with GRU (faster, fewer parameters)
- Use LSTM if GRU doesn't work well
- Consider Transformers for most NLP tasks

---

## 23.5 Bidirectional RNNs

**Concept:** Process sequence in both forward and backward directions.

### Architecture

```
Forward:  h_vec_1 --> h_vec_2 --> h_vec_3 --> ... --> h_vec_T
Backward: h_vec_1 <-- h_vec_2 <-- h_vec_3 <-- ... <-- h_vec_T

Output: y_t = f([h_vec_t; h_vec_t])  # Concatenate both directions
```

### Implementation

```python
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # Key parameter
            dropout=0.2 if num_layers > 1 else 0
        )

        # Output layer (note: hidden_size * 2 due to bidirectional)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # h0 and c0 have shape: (num_layers * 2, batch, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # out shape: (batch, seq_len, hidden_size * 2)

        out = self.fc(out[:, -1, :])
        return out
```

**Use Cases:**
- Named Entity Recognition (NER)
- Part-of-Speech (POS) tagging
- Machine Translation
- Any task where future context helps

**Limitation:** Cannot be used for real-time prediction (need full sequence).

---

## 23.6 RNN Variants and Architectures

### Sequence-to-Sequence (Seq2Seq)

**Architecture:**
```
Encoder RNN: x_1, x_2, ..., x_n --> context vector (c)
Decoder RNN: c --> y_1, y_2, ..., y_m
```

**Applications:**
- Machine translation
- Text summarization
- Question answering

```python
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(output_size, hidden_size, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, trg):
        # Encode
        _, (hidden, cell) = self.encoder(src)

        # Decode
        outputs = []
        input = trg[:, 0:1, :]  # Start token

        for t in range(1, trg.size(1)):
            output, (hidden, cell) = self.decoder(input, (hidden, cell))
            output = self.fc(output)
            outputs.append(output)
            input = trg[:, t:t+1, :]  # Teacher forcing

        return torch.cat(outputs, dim=1)
```

### Many-to-One (Sentiment Analysis)

```python
# Last hidden state used for classification
output = model(input_sequence)
sentiment = classifier(output[:, -1, :])  # Use last time step
```

### Many-to-Many (POS Tagging)

```python
# Output at each time step
outputs = model(input_sequence)  # (batch, seq_len, hidden)
pos_tags = classifier(outputs)    # (batch, seq_len, num_tags)
```

### One-to-Many (Image Captioning)

```python
# Image --> CNN --> feature vector --> RNN --> caption
image_features = cnn(image)
hidden = image_features.unsqueeze(0)
caption = rnn.generate(hidden, max_length=20)
```

---

## 23.7 Training Techniques

### Teacher Forcing

**Concept:** Use ground truth as input to decoder instead of previous prediction.

```python
# With teacher forcing (training)
for t in range(target_length):
    output, hidden = decoder(target[:, t], hidden)
    loss += criterion(output, target[:, t+1])

# Without teacher forcing (inference)
input = start_token
for t in range(max_length):
    output, hidden = decoder(input, hidden)
    input = output.argmax(dim=-1)
```

**Pros:** Faster convergence, more stable training
**Cons:** Exposure bias (model never sees its own mistakes during training)

**Solution:** Scheduled sampling (gradually reduce teacher forcing ratio).

### Gradient Clipping

```python
# Clip gradients to prevent explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# Training loop
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Clip
optimizer.step()
```

### Dropout for RNNs

```python
# Apply dropout between layers (not within recurrent connections)
lstm = nn.LSTM(
    input_size=100,
    hidden_size=256,
    num_layers=3,
    dropout=0.5  # Applied between layers 1-2 and 2-3
)

# For input/output dropout, apply manually
dropout = nn.Dropout(0.5)
x = dropout(x)
output, _ = lstm(x)
output = dropout(output)
```

---

## 23.8 Practical Applications

### Text Classification

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        # text shape: (seq_len, batch)
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.rnn(embedded)

        # Concatenate final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)

        return self.fc(hidden)
```

### Time Series Forecasting

```python
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Predict next time step
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

# Multi-step forecasting
def forecast_multistep(model, initial_sequence, steps):
    predictions = []
    current_seq = initial_sequence

    for _ in range(steps):
        pred = model(current_seq)
        predictions.append(pred)
        # Update sequence with prediction
        current_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(1)], dim=1)

    return torch.cat(predictions, dim=0)
```

### Named Entity Recognition (NER)

```python
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

# Usage for sequence labeling
model = BiLSTM_CRF(vocab_size=10000, tagset_size=9, embedding_dim=100, hidden_dim=256)
sentence = torch.tensor([[1, 5, 3, 8, 9, 2]])  # Token IDs
tag_scores = model(sentence)  # (batch, seq_len, num_tags)
```

---

## 23.9 Limitations and When Not to Use RNNs

### Limitations

1. **Sequential Processing:** Cannot parallelize across time steps
2. **Long Sequences:** Still struggle with very long dependencies (500+ tokens)
3. **Speed:** Slower than transformers on modern hardware (GPUs)
4. **Context Length:** Limited by memory constraints

### When to Use RNNs (2025)

**Still Good For:**
- Online/streaming predictions (process one token at a time)
- Very long sequences where transformers are too memory-intensive
- Edge devices with limited memory
- Time series with irregular sampling

**Use Transformers Instead For:**
- Most NLP tasks (2025 standard)
- Tasks requiring long-range dependencies
- When you have sufficient GPU memory
- Batch processing (not real-time)

---

## 23.10 Evolution Timeline

**1990s:** Vanilla RNNs introduced
- Problem: Vanishing gradients

**1997:** LSTM invented
- Solution: Gating mechanism for long-term memory

**2014:** GRU introduced
- Improvement: Simpler, fewer parameters

**2014:** Seq2Seq with attention
- Breakthrough: Machine translation

**2017:** Transformers introduced
- Revolution: Parallel processing, better performance

**2025 Status:**
- RNNs mostly replaced by Transformers for NLP
- Still used for time series, streaming, edge deployment
- Educational value: Understanding sequential processing

---

## Code Example: Complete Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Define model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# 2. Training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        texts, labels = batch
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(texts).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# 3. Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# 4. Main training loop
def main():
    # Hyperparameters
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    DROPOUT = 0.5
    LEARNING_RATE = 0.001
    EPOCHS = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = SentimentLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM,
                         OUTPUT_DIM, N_LAYERS, DROPOUT).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tVal Loss: {val_loss:.3f}')

if __name__ == '__main__':
    main()
```

---

## Resources

### Key Papers
- "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- "Learning Phrase Representations using RNN Encoder-Decoder" (Cho et al., 2014) - GRU
- "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014)

### Tutorials
- PyTorch RNN Tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
- Understanding LSTM Networks: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

### Modern Alternatives
- Transformers (most NLP tasks)
- Temporal Convolutional Networks (TCN) for time series
- State Space Models (S4, Mamba) for long sequences
