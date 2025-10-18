# 10. NLP and Transformers: Complete Guide

## Overview

Natural Language Processing (NLP) has evolved dramatically from traditional statistical methods to modern transformer-based models. This guide covers when to use each approach, from simple word embeddings to state-of-the-art transformers.

**Models/Approaches Covered:**
- Traditional NLP (TF-IDF, Bag-of-Words)
- Word Embeddings (Word2Vec, GloVe, FastText)
- RNN/LSTM
- Transformers (BERT, GPT, T5)
- Modern LLMs

---

## 10.1 Traditional NLP vs Deep Learning NLP

### The Evolution

```
Traditional NLP → Word Embeddings → RNN/LSTM → Transformers → LLMs
(2000s)          (2013-2015)       (2015-2018)  (2017-now)    (2023-now)
```

### Comparison Table

| Approach | When Introduced | Best For | Limitations | 2025 Status |
|----------|----------------|----------|-------------|-------------|
| **TF-IDF / BoW** | 1970s-2000s | Simple classification, small datasets | No semantic meaning, high dimensionality | Still used for baselines |
| **Word2Vec/GloVe** | 2013-2014 | When need pre-trained embeddings, limited compute | Static embeddings (one vector per word) | Mostly replaced |
| **LSTM/GRU** | 2015-2017 | Sequential data, time series | Sequential processing (slow), gradient issues | Niche use cases only |
| **Transformers** | 2017-now | Most NLP tasks, state-of-the-art | Requires more data/compute | **Current standard** |
| **LLMs (GPT-4, etc)** | 2023-now | Complex reasoning, generation | Expensive, needs API or huge compute | **Cutting edge** |

---

## 10.2 Decision Flow: Which NLP Approach?

```
Start: What's your NLP task?

├─ Simple text classification with <10K samples?
│  └─ Use: TF-IDF + Logistic Regression / Naive Bayes
│
├─ Need embeddings for downstream task?
│  ├─ Have GPU + large dataset?
│  │  └─ Use: Fine-tune BERT embeddings
│  └─ Limited compute?
│     └─ Use: Pre-trained Word2Vec/GloVe
│
├─ Text generation (stories, code, etc)?
│  ├─ Need best quality?
│  │  └─ Use: GPT-based models (GPT-4, GPT-3.5)
│  └─ Need to self-host?
│     └─ Use: Open-source LLMs (Llama, Mistral)
│
├─ Text understanding (classification, NER, Q&A)?
│  └─ Use: BERT-based models
│
├─ Multi-task (translation, summarization, Q&A)?
│  └─ Use: T5 or Flan-T5
│
└─ Time-series text data (sensor logs, monitoring)?
   └─ Consider: LSTM (specialized use case)
```

---

## 10.3 Traditional NLP: TF-IDF & Bag-of-Words

### What They Are

- **Bag-of-Words (BoW):** Count word frequencies
- **TF-IDF:** Weight words by importance (term frequency × inverse document frequency)

### ✅ When to Use Traditional NLP

1. **Small datasets (<10,000 samples)**
   - Deep learning needs more data
   - TF-IDF + Logistic Regression works well

2. **Need fast baseline**
   - Train in seconds
   - Easy to interpret

3. **Simple classification tasks**
   - Spam detection
   - Topic classification
   - Sentiment analysis (basic)

4. **Limited computational resources**
   - No GPU needed
   - Runs on laptop

5. **Explainability required**
   - Can see exact words/features used
   - Easy to debug

### ❌ When NOT to Use Traditional NLP

1. **Need semantic understanding**
   - TF-IDF: "king" and "monarch" are different words
   - **Better:** Word embeddings or transformers

2. **Word order matters**
   - "dog bites man" vs "man bites dog" = same BoW
   - **Better:** LSTM or transformers

3. **Large datasets (100K+ samples)**
   - Deep learning will outperform
   - **Better:** Transformers

4. **Complex tasks (generation, translation)**
   - TF-IDF only for classification
   - **Better:** Transformers

---

## 10.4 Word Embeddings: Word2Vec, GloVe, FastText

### What They Are

Convert words to dense vectors where similar words have similar vectors.

- **Word2Vec:** Predicts context words from target word (or vice versa)
- **GloVe:** Matrix factorization on word co-occurrence
- **FastText:** Word2Vec + subword information (handles typos, rare words)

### Comparison

| Feature | Word2Vec | GloVe | FastText |
|---------|----------|-------|----------|
| **Training Method** | Predictive (neural net) | Count-based (matrix factorization) | Predictive + subword |
| **Handles OOV** | ❌ No | ❌ No | ✅ Yes (subword vectors) |
| **Speed** | Fast | Faster | Slower |
| **Pretrained Available** | ✅ Yes (Google News 300d) | ✅ Yes (Wikipedia, Common Crawl) | ✅ Yes (157 languages) |
| **Best For** | General use | Global statistics important | Morphologically rich languages, typos |

### ✅ When to Use Static Word Embeddings

1. **Limited compute (no GPU)**
   - Pretrained embeddings + simple model
   - Example: Word2Vec + CNN for classification

2. **Need consistent word representation**
   - Same word = same vector always
   - Good for word similarity tasks

3. **Lightweight deployment**
   - Small model size (few MB)
   - Fast inference

4. **Morphologically rich languages**
   - FastText handles word variations well
   - Example: German, Turkish, Finnish

### ❌ When NOT to Use Static Embeddings (Use Transformers Instead)

1. **Need context-dependent meanings**
   - Word2Vec: "bank" always same vector
   - **Problem:** "river bank" vs "savings bank"
   - **Solution:** BERT (contextual embeddings)

2. **Have GPU + large dataset**
   - Transformers will outperform
   - **Better:** Fine-tune BERT

3. **State-of-the-art accuracy needed**
   - Word2Vec is outdated (2013)
   - **Better:** Transformer-based models

4. **Long-range dependencies**
   - Static embeddings don't capture document context
   - **Better:** Transformers

---

## 10.5 RNN/LSTM vs Transformers

### What They Are

- **RNN/LSTM:** Process sequences one token at a time (sequential)
- **Transformers:** Process entire sequence at once (parallel) using attention

### Key Differences

| Feature | LSTM | Transformer |
|---------|------|-------------|
| **Processing** | Sequential (slow) | Parallel (fast) |
| **Long-range dependencies** | ❌ Vanishing gradient issues | ✅ Attention captures all positions |
| **Training Speed** | Slow (no parallelization) | Fast (GPU-optimized) |
| **Context** | Unidirectional (or bidirectional LSTM) | Bidirectional by default |
| **Memory** | Hidden state (limited) | Full attention matrix |
| **2025 Status** | ⚠️ Mostly obsolete | ✅ Standard |

### ✅ When to Use LSTM (Rare in 2025)

1. **Time-series data (NOT text)**
   - Sensor readings, stock prices
   - Sequential dependencies critical

2. **Extremely small datasets (<1000 samples)**
   - LSTM has fewer parameters
   - Transformers might overfit

3. **Real-time streaming data**
   - Process one token at a time
   - Can't wait for full sequence

4. **Very long sequences (>10K tokens)**
   - Transformer attention = O(n²) memory
   - LSTM = O(1) memory per step
   - **Note:** Long-context transformers (e.g., ModernBERT 8K) are replacing this use case

### ✅ When to Use Transformers (Default Choice)

1. **Any standard NLP task**
   - Text classification, NER, Q&A, summarization
   - Transformers are state-of-the-art

2. **Have GPU**
   - Transformers optimize well on GPUs
   - Training/inference much faster

3. **Need bidirectional context**
   - "The bank by the river" – needs both directions
   - Transformers capture this naturally

4. **Have 1K+ samples or can use pretrained models**
   - Fine-tuning BERT works with small data
   - Transfer learning is powerful

5. **Production deployment (2025)**
   - Industry standard
   - Extensive tooling (HuggingFace)

### ❌ When NOT to Use LSTM

**In 2025, LSTMs are mostly obsolete for NLP.** Transformers have replaced them for almost all use cases.

**Only exceptions:**
- Extremely resource-constrained edge devices
- Real-time streaming where buffering is impossible
- Time-series forecasting (non-text)

---

## 10.6 BERT vs GPT vs T5: Which Transformer?

### Architecture Differences

| Model | Architecture | Direction | Pretraining Task | Best For |
|-------|-------------|-----------|------------------|----------|
| **BERT** | Encoder-only | Bidirectional | Masked Language Modeling | Understanding text, classification |
| **GPT** | Decoder-only | Unidirectional (left-to-right) | Next token prediction | Generating text |
| **T5** | Encoder-Decoder | Both | Text-to-text (all tasks unified) | Versatile: translation, summarization, Q&A |

### Quick Decision Guide

```
What's your task?

├─ Text Classification / NER / Q&A (understanding)?
│  └─ Use: BERT
│     Models: bert-base-uncased, roberta-base, distilbert
│
├─ Text Generation (stories, code, chat)?
│  └─ Use: GPT
│     Models: gpt-3.5-turbo, gpt-4, llama, mistral
│
├─ Translation / Summarization / Multi-task?
│  └─ Use: T5 or Flan-T5
│
├─ Sentiment Analysis?
│  └─ Use: RoBERTa or DistilBERT (fine-tuned)
│
├─ Named Entity Recognition?
│  └─ Use: BERT or RoBERTa
│
└─ Question Answering?
   └─ Use: BERT or T5
```

---

## 10.7 BERT (Bidirectional Encoder Representations)

### What It Is

BERT reads text bidirectionally (left AND right context) to understand meaning.

**Example:**
- Input: "The bank by the river"
- BERT sees: ["The", "bank", "by", "the", "river"] all at once
- Understands "bank" = river bank (not financial bank)

### ✅ When to Use BERT

1. **Text classification**
   - Sentiment analysis, topic classification
   - Binary or multi-class

2. **Named Entity Recognition (NER)**
   - Extract names, locations, dates
   - Bidirectional context helps

3. **Question Answering**
   - Extractive Q&A (find answer in passage)
   - **Example:** SQuAD dataset

4. **Semantic similarity**
   - Compare sentence meanings
   - Find duplicate questions

5. **Token classification**
   - POS tagging, chunking
   - Each token gets a label

6. **Need fine-tuning on domain data**
   - Medical, legal, scientific text
   - BioBERT, SciBERT, LegalBERT

### ❌ When NOT to Use BERT

1. **Text generation**
   - BERT is encoder-only (no decoder)
   - **Better:** GPT, T5

2. **Open-ended creative tasks**
   - Cannot generate long-form text
   - **Better:** GPT-4, Llama

3. **Translation**
   - No decoder for target language
   - **Better:** T5, MarianMT

### BERT Variants

```python
# Popular BERT variants (2025)
models = {
    'bert-base-uncased': 'Original BERT, 110M params',
    'roberta-base': 'Improved BERT, better performance, 125M params',
    'distilbert-base': 'Smaller/faster BERT, 66M params, 95% accuracy',
    'albert-base-v2': 'Parameter-efficient BERT, 12M params',
    'deberta-v3-base': 'State-of-the-art BERT successor, 184M params',
    'modernbert-base': '2024: Long context (8K tokens), 149M params'
}
```

**Recommendation:** Use `roberta-base` for most tasks (best balance). Use `distilbert` for speed.

---

## 10.8 GPT (Generative Pre-trained Transformer)

### What It Is

GPT reads text left-to-right and generates next tokens. Optimized for text generation.

### ✅ When to Use GPT

1. **Text generation**
   - Creative writing, stories
   - Code generation
   - Email drafts, blog posts

2. **Conversational AI**
   - Chatbots
   - Customer support

3. **Text completion**
   - Autocomplete, suggestions
   - "Given X, write Y"

4. **Few-shot learning**
   - GPT-3/4 can learn from examples in prompt
   - No fine-tuning needed

5. **Complex reasoning tasks**
   - Chain-of-thought prompting
   - Multi-step problem solving

### ❌ When NOT to Use GPT

1. **Simple classification**
   - Overkill and expensive
   - **Better:** Fine-tuned BERT (faster, cheaper)

2. **Need bidirectional context**
   - GPT only sees left context
   - **Better:** BERT, T5

3. **Budget constraints**
   - GPT-4 API is expensive ($0.03/1K tokens)
   - **Better:** Open-source models, or fine-tuned BERT

4. **Need full control/transparency**
   - API black box
   - **Better:** Self-hosted open models

### GPT Options (2025)

| Model | Provider | Size | Cost | Best For |
|-------|----------|------|------|----------|
| **GPT-4-turbo** | OpenAI | ~1.7T params | $$$ High | Best quality, complex reasoning |
| **GPT-3.5-turbo** | OpenAI | ~175B params | $ Low | Cost-effective generation |
| **Llama 3 (70B)** | Meta | 70B params | Free (self-host) | Open-source alternative |
| **Mistral (7B/8x7B)** | Mistral AI | 7B-47B params | Free/$ | Fast, efficient |
| **CodeLlama** | Meta | 7B-34B params | Free | Code generation |

---

## 10.9 T5 (Text-to-Text Transfer Transformer)

### What It Is

T5 treats ALL NLP tasks as text-to-text:
- Translation: "translate English to French: Hello" → "Bonjour"
- Summarization: "summarize: [long text]" → "[summary]"
- Classification: "sentiment: I love this!" → "positive"

### ✅ When to Use T5

1. **Need one model for multiple tasks**
   - Translation, summarization, Q&A
   - T5 can do all with same model

2. **Summarization**
   - T5 particularly strong here
   - Better than BERT (which can't summarize)

3. **Translation**
   - Especially with Flan-T5
   - Good for low-resource languages

4. **Question Answering (generative)**
   - Generates answer (not just extracts)
   - More flexible than BERT

5. **Need instruction-following**
   - Flan-T5 fine-tuned on instructions
   - Good zero-shot performance

### ❌ When NOT to Use T5

1. **Simple classification**
   - Slower than BERT encoder
   - **Better:** BERT (encoder-only is faster)

2. **Long-form generation**
   - GPT-style models better
   - **Better:** GPT, Llama

3. **Conversational AI**
   - Not designed for chat
   - **Better:** GPT, Claude, dialog models

### T5 Variants

```python
models = {
    't5-small': '60M params, fast baseline',
    't5-base': '220M params, good balance',
    't5-large': '770M params, better quality',
    'flan-t5-base': '250M params, instruction-tuned, RECOMMENDED',
    'flan-t5-xl': '3B params, high quality',
}
```

**Recommendation:** Use `flan-t5-base` for versatility.

---

## 10.10 Text Preprocessing for Different Models

### Critical Insight: Transformers Need MINIMAL Preprocessing

**Old way (Word2Vec, LSTM):**
- Lowercase, remove punctuation, stem, lemmatize
- Tokenize manually

**New way (BERT, GPT, T5):**
- Use raw text with model's tokenizer
- Model handles everything

### Preprocessing Comparison

| Model Type | Lowercase? | Remove Punctuation? | Lemmatize/Stem? | Tokenizer |
|------------|-----------|---------------------|-----------------|-----------|
| **TF-IDF** | ✅ Yes | ✅ Yes | ✅ Yes | Manual or NLTK |
| **Word2Vec** | Optional | Optional | Optional | Gensim tokenizer |
| **LSTM (manual)** | ✅ Yes | ⚠️ Maybe | ⚠️ Maybe | Custom |
| **BERT** | ❌ Use model tokenizer | ❌ Keep it | ❌ Not needed | `BertTokenizer` |
| **GPT** | ❌ Use model tokenizer | ❌ Keep it | ❌ Not needed | `GPT2Tokenizer` |
| **T5** | ❌ Use model tokenizer | ❌ Keep it | ❌ Not needed | `T5Tokenizer` |

### ✅ Correct BERT Preprocessing (2025)

```python
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize tokenizer (handles everything automatically)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Raw text - NO manual preprocessing needed!
texts = [
    "I love this movie! 😊",
    "This is terrible... worst ever.",
    "It's okay, nothing special"
]

# Tokenize (automatically handles special tokens, padding, etc)
encoded = tokenizer(
    texts,
    padding=True,           # Pad to max length in batch
    truncation=True,        # Truncate if too long
    max_length=512,         # BERT max: 512 tokens
    return_tensors='pt'     # Return PyTorch tensors
)

# encoded contains:
# - input_ids: token IDs
# - attention_mask: which tokens are real vs padding
# - token_type_ids: segment IDs (for sentence pairs)
```

### ❌ Common Mistakes

**Mistake 1: Over-preprocessing**
```python
# WRONG for BERT
text = text.lower()                    # Tokenizer handles this
text = re.sub(r'[^\w\s]', '', text)   # Removes punctuation - BAD!
text = lemmatize(text)                 # Not needed!

# This removes information BERT needs
```

**Mistake 2: Not using model-specific tokenizer**
```python
# WRONG
tokens = text.split()  # Manual tokenization
model.predict(tokens)  # Won't work!

# CORRECT
encoded = tokenizer(text)
model(**encoded)
```

**Mistake 3: Forgetting max length**
```python
# WRONG - BERT max is 512 tokens
long_text = "..." * 10000  # Very long
encoded = tokenizer(long_text)  # Will truncate silently or error

# CORRECT
encoded = tokenizer(
    long_text,
    truncation=True,    # Explicitly truncate
    max_length=512      # BERT limit
)
```

### When You DO Need Preprocessing

```python
# These are still useful:

# 1. Remove HTML tags
import re
text = re.sub(r'<[^>]+>', '', text)

# 2. Handle excessive whitespace
text = ' '.join(text.split())

# 3. Remove URLs (optional, depends on task)
text = re.sub(r'http\S+', '', text)

# 4. Domain-specific cleaning
text = text.replace('[REDACTED]', '')  # Remove placeholders
```

---

## 10.11 Pretrained vs Fine-tuning vs Training from Scratch

### Decision Flow

```
Start: Do you have a NLP task?

├─ Have <1K samples?
│  └─ Use: Pretrained model + few-shot prompting (GPT-4)
│     OR: Pretrained embeddings → simple classifier
│
├─ Have 1K-10K samples?
│  └─ Use: Fine-tune pretrained model (BERT, T5)
│     - Start with checkpoint (e.g., bert-base-uncased)
│     - Train on your data (few epochs)
│
├─ Have 10K-100K samples?
│  └─ Use: Fine-tune pretrained model
│     - Best results with domain-specific pretraining
│     - Example: BioBERT for medical, LegalBERT for law
│
├─ Have 1M+ samples + unique domain?
│  └─ Consider: Continue pretraining → fine-tune
│     - Example: Train on your domain corpus first
│     - Then fine-tune on task
│
└─ Have 100M+ samples + billions in budget?
   └─ Consider: Train from scratch (rare!)
      - Only for new languages or truly novel domains
```

### Training Options Explained

#### 1. Use Pretrained (Zero-shot/Few-shot)

**What it is:** Use model as-is, no training

**When to use:**
- <100 samples
- Using GPT-4 or large LLMs
- Quick prototype

**Example:**
```python
from transformers import pipeline

# Zero-shot classification (no training!)
classifier = pipeline("zero-shot-classification",
                     model="facebook/bart-large-mnli")

text = "I love this restaurant!"
labels = ["positive", "negative", "neutral"]

result = classifier(text, labels)
# Works without training!
```

**Pros:**
- No training needed
- Works immediately
- Good for low-resource tasks

**Cons:**
- Lower accuracy than fine-tuning
- Expensive if using GPT-4 API
- Less control

---

#### 2. Fine-tuning Pretrained Model (RECOMMENDED)

**What it is:** Take pretrained model (e.g., BERT) and train on your data

**When to use:**
- 1K+ labeled samples
- Standard NLP tasks
- Need good accuracy

**Example:**
```python
from transformers import AutoModelForSequenceClassification, Trainer

# Load pretrained model
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3  # Your task: 3 classes
)

# Fine-tune on your data
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()  # Usually 2-5 epochs enough!
```

**How many epochs?**
- Small dataset (1K-10K): 3-5 epochs
- Medium dataset (10K-100K): 2-4 epochs
- Large dataset (100K+): 1-3 epochs

**Pros:**
- Best accuracy for most tasks
- Relatively fast (hours, not days)
- Works with small data (transfer learning)

**Cons:**
- Needs labeled data
- Requires GPU

---

#### 3. Continue Pretraining (Domain Adaptation)

**What it is:** First train on domain-specific unlabeled text, THEN fine-tune

**When to use:**
- Domain very different from general text
- Have lots of unlabeled domain data
- Examples: Medical, legal, scientific, code

**Example:**
```python
# Step 1: Continue pretraining on domain data (unlabeled)
from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Train on medical papers (unlabeled)
# Mask random words, predict them
# This creates "BioBERT"-like model

# Step 2: Fine-tune on your task (labeled)
model = AutoModelForSequenceClassification.from_pretrained(
    './domain-adapted-bert',
    num_labels=2
)
# Train on labeled medical classification data
```

**When needed:**
- Medical: Use BioBERT or domain-adapt BERT on PubMed
- Legal: Use LegalBERT or adapt on legal documents
- Scientific: Use SciBERT
- Code: Use CodeBERT or CodeT5

**Pros:**
- Better for specialized domains
- Handles domain vocabulary

**Cons:**
- Time-consuming (days)
- Needs large unlabeled corpus
- May not improve much for general domains

---

#### 4. Train from Scratch (RARELY NEEDED)

**What it is:** Train transformer from random initialization

**When to use:**
- New language not in pretrained models
- Extremely unique domain
- Privacy: cannot use external models
- Academic research

**Reality check:**
- Needs 1M-1B+ samples
- Costs $10K-$1M+ in compute
- Takes days/weeks

**Almost always better alternatives:**
- Multilingual BERT supports 100+ languages
- Continue pretraining cheaper than from scratch
- Fine-tuning works for 99% of use cases

**Pros:**
- Full control
- Potential for optimal performance

**Cons:**
- Extremely expensive
- Requires massive data
- Usually not worth it

---

### Sample Size Guidelines

| Samples | Strategy | Expected Accuracy | Time to Train |
|---------|----------|-------------------|---------------|
| <100 | Few-shot prompting (GPT-4) | 60-75% | Minutes |
| 100-1K | Pretrained + simple head | 65-80% | Minutes |
| 1K-10K | Fine-tune small model (DistilBERT) | 75-88% | 1-2 hours |
| 10K-50K | Fine-tune base model (BERT) | 82-92% | 2-6 hours |
| 50K-100K | Fine-tune large model (RoBERTa-large) | 85-94% | 4-12 hours |
| 100K-1M | Domain-adapt + fine-tune | 88-95% | 1-3 days |
| 1M+ | Consider from scratch (usually not worth it) | 90-96% | Weeks |

---

## 10.12 Hyperparameters for Transformer Fine-tuning

### Critical Parameters

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',

    # Learning rate (MOST IMPORTANT)
    learning_rate=2e-5,              # 2e-5 to 5e-5 for BERT
                                     # Lower for large models (1e-5)
                                     # Higher for small data (3e-5)

    # Epochs (careful - easy to overfit)
    num_train_epochs=3,              # 2-4 typical
                                     # More epochs for small datasets

    # Batch size (depends on GPU memory)
    per_device_train_batch_size=16,  # 8, 16, or 32
                                     # Smaller = more updates, more stable

    # Warmup (important!)
    warmup_steps=500,                # 5-10% of total steps
                                     # Or: warmup_ratio=0.1

    # Weight decay (regularization)
    weight_decay=0.01,               # 0.01 standard

    # Evaluation
    evaluation_strategy="epoch",     # Evaluate every epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",

    # Early stopping (prevent overfitting)
    # Need to add callback separately
)

# Add early stopping
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
```

### Learning Rate Guidelines

| Model Size | Typical LR | Range |
|------------|-----------|-------|
| BERT-base (110M) | 2e-5 | 1e-5 to 5e-5 |
| BERT-large (340M) | 1e-5 | 5e-6 to 3e-5 |
| RoBERTa-base | 2e-5 | 1e-5 to 5e-5 |
| DistilBERT | 3e-5 | 2e-5 to 5e-5 |
| T5-base | 1e-4 | 1e-5 to 1e-3 |
| GPT-2 | 5e-5 | 1e-5 to 1e-4 |

**Rule of thumb:** Start with 2e-5, lower if loss is unstable, raise if training is too slow.

### Batch Size Guidelines

| GPU Memory | Batch Size (BERT-base) | Batch Size (BERT-large) |
|------------|------------------------|-------------------------|
| 8GB (RTX 3070) | 8-16 | 2-4 |
| 16GB (V100, A100) | 16-32 | 8-16 |
| 24GB (RTX 3090) | 32-64 | 16-24 |
| 40GB (A100) | 64-128 | 24-32 |

**Tips:**
- Use gradient accumulation if batch size too small
- Smaller batch = slower but often better generalization

---

## 10.13 Common NLP Tasks → Model Selection

### Task-to-Model Quick Reference

| Task | Best Model Choice | Alternative | Difficulty |
|------|------------------|-------------|------------|
| **Sentiment Analysis** | RoBERTa-base fine-tuned | DistilBERT | Easy |
| **Topic Classification** | BERT-base fine-tuned | TF-IDF + LogReg (baseline) | Easy |
| **Named Entity Recognition** | BERT or RoBERTa | SpaCy (fast baseline) | Medium |
| **Question Answering (extractive)** | BERT for QA | RoBERTa | Medium |
| **Question Answering (generative)** | T5 or Flan-T5 | GPT-3.5 | Medium |
| **Text Summarization** | BART or T5 | Pegasus | Hard |
| **Translation** | MarianMT or T5 | GPT-4 API | Hard |
| **Text Generation** | GPT-3.5 or Llama | Mistral | Medium |
| **Semantic Similarity** | Sentence-BERT (SBERT) | USE (Universal Sentence Encoder) | Easy |
| **Zero-shot Classification** | BART-MNLI | GPT-4 | Easy (no training!) |

---

## 10.14 Common Pitfalls & Solutions

### Pitfall 1: Not Using Model-Specific Tokenizer

❌ **Wrong:**
```python
text = text.lower().split()
model.predict(text)  # Won't work!
```

✅ **Correct:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded = tokenizer(text, return_tensors='pt')
model(**encoded)
```

---

### Pitfall 2: Forgetting model.eval() During Inference

❌ **Wrong:**
```python
# Dropout is still active!
predictions = model(inputs)
```

✅ **Correct:**
```python
model.eval()  # Turn off dropout
with torch.no_grad():  # Disable gradient computation
    predictions = model(inputs)
```

---

### Pitfall 3: Overfitting on Small Datasets

**Symptoms:**
- Train accuracy: 99%
- Val accuracy: 70%

✅ **Solutions:**
```python
# 1. Reduce epochs
num_train_epochs=2  # Instead of 5

# 2. Add early stopping
callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]

# 3. Lower learning rate
learning_rate=1e-5  # Instead of 5e-5

# 4. Increase weight decay
weight_decay=0.1  # Instead of 0.01

# 5. Use smaller model
model = 'distilbert-base-uncased'  # Instead of bert-large

# 6. Data augmentation
from nlpaug.augmenter.word import SynonymAug
aug = SynonymAug()
augmented_text = aug.augment(text)
```

---

### Pitfall 4: Not Handling Class Imbalance

❌ **Problem:** 95% class A, 5% class B → Model predicts all A

✅ **Solutions:**
```python
# 1. Class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# In PyTorch
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

# 2. Oversample minority class
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# 3. Different metric
# Don't use accuracy! Use F1, precision, recall
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred, average='weighted')
```

---

### Pitfall 5: Truncating Long Documents Incorrectly

❌ **Wrong:**
```python
# Truncates to 512 tokens, loses end of document
encoded = tokenizer(long_text, truncation=True, max_length=512)
```

✅ **Better: Sliding Window**
```python
def split_long_text(text, tokenizer, max_length=512, stride=128):
    """Split long text into overlapping chunks"""
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_length - stride):
        chunk = tokens[i:i + max_length]
        chunks.append(chunk)

    return chunks

# Or use Longformer (up to 4096 tokens)
from transformers import LongformerForSequenceClassification
model = LongformerForSequenceClassification.from_pretrained(
    'allenai/longformer-base-4096'
)
```

---

### Pitfall 6: Not Considering Token Limits

| Model | Max Tokens | What Happens if Exceeded |
|-------|-----------|-------------------------|
| BERT | 512 | Truncates silently |
| GPT-2 | 1024 | Error or truncate |
| GPT-3.5 | 4096 | API error |
| GPT-4 | 8192 / 32K | API error (costs more for 32K) |
| Longformer | 4096 | Good for long docs |
| ModernBERT (2024) | 8192 | Best for long context |

**Solutions:**
- Summarize first, then process
- Use sliding window
- Use long-context models (Longformer, ModernBERT)

---

## 10.15 Quick Start Code Examples

### Example 1: Sentiment Analysis with BERT

```python
from transformers import pipeline

# Zero-shot (no training)
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Fine-tuning
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load data
dataset = load_dataset("imdb")

# Load pretrained model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Train
trainer.train()

# Evaluate
trainer.evaluate()
```

---

### Example 2: Named Entity Recognition

```python
from transformers import pipeline

# Pretrained NER
ner = pipeline("ner", model="dslim/bert-base-NER")

text = "Apple Inc. is located in Cupertino, California."
entities = ner(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity']} ({entity['score']:.2f})")

# Output:
# Apple: B-ORG (0.99)
# Inc: I-ORG (0.99)
# Cupertino: B-LOC (0.99)
# California: B-LOC (0.99)
```

---

### Example 3: Text Generation with GPT

```python
from transformers import pipeline

# Load generator
generator = pipeline("text-generation", model="gpt2")

# Generate
prompt = "Once upon a time, in a land far away,"
result = generator(
    prompt,
    max_length=100,
    num_return_sequences=3,
    temperature=0.7,  # Higher = more creative
    top_p=0.9,        # Nucleus sampling
)

for i, gen in enumerate(result):
    print(f"Generation {i+1}:")
    print(gen['generated_text'])
    print()
```

---

### Example 4: Question Answering with BERT

```python
from transformers import pipeline

# Load QA model
qa = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars
in Paris, France. It is named after the engineer Gustave Eiffel, whose
company designed and built the tower. Constructed from 1887 to 1889.
"""

question = "When was the Eiffel Tower built?"

answer = qa(question=question, context=context)
print(f"Answer: {answer['answer']}")
print(f"Confidence: {answer['score']:.2f}")

# Output:
# Answer: 1887 to 1889
# Confidence: 0.97
```

---

### Example 5: Summarization with T5

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="t5-base")

long_text = """
[Insert long article text here...]
"""

summary = summarizer(
    long_text,
    max_length=150,
    min_length=50,
    do_sample=False
)

print(summary[0]['summary_text'])
```

---

## 10.16 Summary Checklist

### Before Starting:
- [ ] Identify task: classification, generation, NER, Q&A, etc.
- [ ] Check dataset size (100? 1K? 10K? 100K?)
- [ ] Decide: baseline (TF-IDF) vs transformers
- [ ] Choose model: BERT (understanding), GPT (generation), T5 (versatile)

### Data Preparation:
- [ ] NO manual preprocessing for transformers (use tokenizer)
- [ ] Split data BEFORE any processing
- [ ] Check for class imbalance
- [ ] Handle long documents (truncate, sliding window, or Longformer)

### Model Selection:
- [ ] Small data (<1K) → Few-shot GPT-4 or pretrained
- [ ] Medium data (1K-100K) → Fine-tune BERT/RoBERTa
- [ ] Large data (100K+) → Fine-tune or domain-adapt
- [ ] Use DistilBERT for speed, RoBERTa for accuracy

### Training:
- [ ] Start with learning_rate=2e-5
- [ ] Use 2-4 epochs (more for small data)
- [ ] Add early stopping
- [ ] Monitor train vs val loss (check overfitting)
- [ ] Use class weights if imbalanced

### Evaluation:
- [ ] Set model.eval() during inference
- [ ] Use F1/precision/recall (not just accuracy)
- [ ] Test on holdout set
- [ ] Check predictions on edge cases

---

## 10.17 Resources & Further Reading

**HuggingFace (Main Resource):**
- https://huggingface.co/docs/transformers
- https://huggingface.co/models (pretrained models)
- https://huggingface.co/datasets (datasets)

**Tutorials:**
- HuggingFace Course: https://huggingface.co/course
- Fast.ai NLP: https://www.fast.ai/

**Papers:**
- BERT: https://arxiv.org/abs/1810.04805
- GPT-3: https://arxiv.org/abs/2005.14165
- T5: https://arxiv.org/abs/1910.10683
- Attention Is All You Need (Transformers): https://arxiv.org/abs/1706.03762

**Tools:**
- Weights & Biases (tracking): https://wandb.ai/
- Optuna (hyperparameter tuning): https://optuna.org/

---

**Last Updated:** 2025-10-12
**Next Section:** GANs and Generative Models (Phase 7)
