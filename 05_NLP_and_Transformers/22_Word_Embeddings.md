# 22. Word Embeddings

## Overview

Word embeddings are numeric representations of words in a lower-dimensional continuous vector space that capture semantic and syntactic relationships. They transform words into dense vectors where similar words have similar representations.

**Key Insight:** Words with similar meanings should have similar vector representations.

---

## 22.1 Why Word Embeddings?

### Limitations of One-Hot Encoding

```python
Vocabulary = ["cat", "dog", "king", "queen"]

cat   = [1, 0, 0, 0]
dog   = [0, 1, 0, 0]
king  = [0, 0, 1, 0]
queen = [0, 0, 0, 1]
```

**Problems:**
- High dimensionality (vocab size can be 50K-100K+)
- No semantic similarity: distance(cat, dog) = distance(cat, king)
- Sparse representations
- No generalization to unseen words

### Word Embeddings Solution

```python
# Dense, low-dimensional vectors (typically 50-300 dimensions)
cat   = [0.2, -0.4, 0.7, ..., 0.1]  # 300-dim
dog   = [0.3, -0.3, 0.8, ..., 0.2]  # 300-dim
king  = [0.5, 0.1, -0.2, ..., 0.4]  # 300-dim
queen = [0.4, 0.2, -0.1, ..., 0.3]  # 300-dim

# Now: distance(cat, dog) < distance(cat, king)
```

**Advantages:**
- Lower dimensionality (50-300 vs 50K-100K)
- Captures semantic relationships
- Generalizes to similar words
- Can perform analogies: king - man + woman ~= queen

---

## 22.2 Word2Vec

**Developed by:** Google (Mikolov et al., 2013)

**Core Idea:** Words that appear in similar contexts have similar meanings.

### Two Architectures

#### Continuous Bag-of-Words (CBOW)
Predicts target word from context words.

```
Context: [The, cat, sat, on, the]
Target: "mat"

Input: average embeddings of context words
Output: prediction of target word
```

**Advantages:**
- Faster training
- Better for frequent words
- Smoother representations

#### Skip-Gram
Predicts context words from target word.

```
Target: "mat"
Predictions: [The, cat, sat, on, the]

Input: target word embedding
Output: predictions of context words
```

**Advantages:**
- Better for rare words
- Captures more fine-grained semantics
- Generally preferred for smaller datasets

### Training Objective

**Negative Sampling:**
For each positive (word, context) pair, sample k negative contexts.

```
Positive: (cat, sat)
Negatives: (cat, building), (cat, economy), ..., (cat, quantum)
```

Maximize probability of positive pairs, minimize probability of negative pairs.

### Implementation

```python
from gensim.models import Word2Vec

sentences = [
    ["I", "love", "machine", "learning"],
    ["Deep", "learning", "is", "amazing"],
    ["Natural", "language", "processing"]
]

# Train Word2Vec
model = Word2Vec(
    sentences,
    vector_size=100,      # embedding dimension
    window=5,             # context window
    min_count=1,          # minimum word frequency
    workers=4,            # parallel threads
    sg=1                  # 1=skip-gram, 0=CBOW
)

# Get vector
vector = model.wv['learning']

# Find similar words
similar = model.wv.most_similar('learning', topn=5)
```

### Famous Analogies

```python
# king - man + woman ~= queen
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
# Output: queen

# Paris - France + Germany ~= Berlin
result = model.wv.most_similar(
    positive=['Paris', 'Germany'],
    negative=['France'],
    topn=1
)
# Output: Berlin
```

---

## 22.3 GloVe (Global Vectors)

**Developed by:** Stanford (Pennington et al., 2014)

**Core Idea:** Incorporate global word co-occurrence statistics.

### How GloVe Works

1. **Build co-occurrence matrix:** Count how often words appear together
2. **Factorize matrix:** Find low-rank approximation
3. **Learn vectors:** Minimize reconstruction error

**Objective Function:**
```
J = sum f(X_ij) (w_i^T w_j + b_i + b_j - log X_ij)^2
```

Where:
- X_ij = co-occurrence count of words i and j
- w_i, w_j = word vectors
- f(X_ij) = weighting function (caps influence of very frequent pairs)

### GloVe vs Word2Vec

**GloVe:**
- Global context (entire corpus statistics)
- Deterministic training
- Faster for large corpora
- Better on word similarity tasks

**Word2Vec:**
- Local context (sliding window)
- Stochastic training
- More efficient memory usage
- Better on analogy tasks

### Using Pretrained GloVe

```python
import numpy as np

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load GloVe
glove = load_glove_embeddings('glove.6B.100d.txt')

# Get vector
vec = glove['computer']

# Compute similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sim = cosine_similarity(glove['computer'], glove['technology'])
```

**Available Pretrained Models:**
- GloVe.6B: 6 billion tokens, Wikipedia + Gigaword
- GloVe.42B: 42 billion tokens, Common Crawl
- GloVe.840B: 840 billion tokens, Common Crawl
- Dimensions: 50, 100, 200, 300

---

## 22.4 FastText

**Developed by:** Facebook AI (Bojanowski et al., 2017)

**Key Innovation:** Represent words as bags of character n-grams.

### Character N-grams

Instead of treating "where" as atomic unit, represent as:

```
<wh, whe, her, ere, re>
+ special whole-word token
```

**Advantages:**
1. **Handle out-of-vocabulary (OOV) words**
   - "coronavirus" (new word) can be represented using character n-grams
   - Even if never seen in training

2. **Morphologically rich languages**
   - Turkish, Finnish, Arabic benefit greatly
   - Can understand word formation

3. **Rare words**
   - Better representations by sharing subword units
   - "unhappiness" shares n-grams with "happiness", "unhappy"

4. **Typos and misspellings**
   - "machien learning" still gets reasonable representation

### Example

```python
from gensim.models import FastText

sentences = [
    ["machine", "learning", "is", "awesome"],
    ["deep", "learning", "is", "powerful"]
]

# Train FastText
model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=3,              # minimum n-gram length
    max_n=6,              # maximum n-gram length
    workers=4
)

# Works for OOV words!
vector_oov = model.wv['machinelearning']  # Never seen but gets vector
```

### Use Cases

**Best for:**
- Morphologically rich languages
- Domains with many rare/technical terms
- Noisy text (social media, OCR)
- Small training corpora

**Comparison:**

| Model | OOV Handling | Training Speed | Memory |
|-------|-------------|----------------|---------|
| Word2Vec | None | Fast | Low |
| GloVe | None | Medium | Medium |
| FastText | Excellent | Slower | Higher |

---

## 22.5 Contextualized Embeddings

### Limitation of Static Embeddings

Word2Vec, GloVe, FastText produce same embedding regardless of context:

```
"I went to the bank to deposit money"  # financial institution
"I sat by the river bank"              # land alongside water

# "bank" gets SAME embedding in both sentences
```

### ELMo (Embeddings from Language Models)

**Developed by:** Allen Institute (Peters et al., 2018)

**Key Innovation:** Embeddings depend on entire sentence context.

```python
from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder()

sentences = [
    ["I", "went", "to", "the", "bank"],
    ["I", "sat", "by", "the", "bank"]
]

embeddings = elmo.embed_sentences(sentences)
# Different "bank" embeddings for each sentence!
```

**How it works:**
- Bidirectional LSTM language model
- Trained on large corpus
- Embeddings are function of entire input sentence
- Can use different layers for different tasks

### BERT Embeddings

**Transformer-based contextualized embeddings (2018+).**

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "I went to the bank"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# Last hidden state contains contextualized embeddings
embeddings = outputs.last_hidden_state
```

**Advantages over ELMo:**
- Bidirectional context (not just left-to-right + right-to-left)
- Attention mechanism captures long-range dependencies
- Better performance on downstream tasks
- Easier to fine-tune

---

## 22.6 Comparing Embedding Approaches

### Static vs Contextualized

**Static (Word2Vec, GloVe, FastText):**
- One vector per word type
- Fast inference
- Lower memory
- Good for traditional ML pipelines
- Still useful in 2025 for resource-constrained scenarios

**Contextualized (ELMo, BERT, RoBERTa):**
- Different vectors for same word in different contexts
- Captures polysemy and context
- Higher computational cost
- State-of-the-art performance
- Standard for most NLP tasks in 2025

---

## 22.7 Practical Applications

### Text Classification with Embeddings

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load embeddings
embeddings = load_glove_embeddings('glove.6B.100d.txt')

# Convert text to average embedding
def text_to_vector(text, embeddings):
    words = text.lower().split()
    vectors = [embeddings[w] for w in words if w in embeddings]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(100)

# Train classifier
X_train = [text_to_vector(text, embeddings) for text in train_texts]
clf = LogisticRegression()
clf.fit(X_train, y_train)
```

### Semantic Similarity

```python
def semantic_similarity(text1, text2, embeddings):
    vec1 = text_to_vector(text1, embeddings)
    vec2 = text_to_vector(text2, embeddings)
    return cosine_similarity(vec1, vec2)

sim = semantic_similarity(
    "machine learning is great",
    "deep learning is awesome",
    embeddings
)
```

### Document Clustering

```python
from sklearn.cluster import KMeans

# Convert documents to embeddings
doc_vectors = [text_to_vector(doc, embeddings) for doc in documents]

# Cluster
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(doc_vectors)
```

---

## 22.8 Evaluation Metrics

### Intrinsic Evaluation

**Word Similarity Tasks:**
- WordSim-353, SimLex-999 datasets
- Correlation with human similarity judgments
- Spearman's rank correlation

**Analogy Tasks:**
- Google analogy dataset
- Accuracy on semantic and syntactic analogies

### Extrinsic Evaluation

Downstream task performance:
- Text classification accuracy
- NER F1-score
- Sentiment analysis performance

**Note:** Extrinsic evaluation is generally more important.

---

## 22.9 Best Practices (2025)

### When to Use Static Embeddings

- Resource-constrained environments
- Fast inference required
- Traditional ML pipelines
- Transfer learning with limited data
- Interpretability important

**Recommended:**
- FastText for robustness to OOV
- GloVe for general-purpose English
- Domain-specific Word2Vec for specialized fields

### When to Use Contextualized Embeddings

- State-of-the-art performance needed
- Complex semantic understanding required
- Sufficient computational resources
- Fine-tuning capability desired

**Recommended:**
- BERT family for most tasks
- RoBERTa for better performance
- Domain-specific BERT models (BioBERT, SciBERT, FinBERT)

---

## 22.10 Modern Alternatives (2025)

### Sentence Embeddings

**Sentence-BERT (SBERT):**
Fine-tuned BERT for generating semantically meaningful sentence embeddings.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks"
]

embeddings = model.encode(sentences)
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
```

**Use Cases:**
- Semantic search
- Document similarity
- Clustering
- Information retrieval

---

## Resources

### Pretrained Embeddings

**Word2Vec:**
- Google News (300-dim, 100B words)

**GloVe:**
- http://nlp.stanford.edu/data/glove.6B.zip

**FastText:**
- https://fasttext.cc/docs/en/english-vectors.html
- 157 languages available

**Contextual:**
- Hugging Face Model Hub: https://huggingface.co/models

### Papers

- "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al., 2013) - Word2Vec
- "GloVe: Global Vectors for Word Representation" (Pennington et al., 2014)
- "Enriching Word Vectors with Subword Information" (Bojanowski et al., 2017) - FastText
- "Deep contextualized word representations" (Peters et al., 2018) - ELMo
