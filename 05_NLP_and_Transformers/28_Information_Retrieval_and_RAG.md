# Information Retrieval and RAG

## Table of Contents
1. [Introduction](#introduction)
2. [BM25 and TF-IDF](#bm25-and-tf-idf)
3. [Dense Retrieval](#dense-retrieval)
4. [Cross-Encoder Reranking](#cross-encoder-reranking)
5. [Hybrid Search](#hybrid-search)
6. [Vector Databases](#vector-databases)
7. [RAG Architecture](#rag-architecture)
8. [Chunking Strategies](#chunking-strategies)
9. [Advanced RAG](#advanced-rag)
10. [Retrieval Metrics](#retrieval-metrics)
11. [Embedding Fine-Tuning](#embedding-fine-tuning)
12. [Production RAG](#production-rag)
13. [RAG Failure Modes](#rag-failure-modes)
14. [See Also](#see-also)
15. [Resources](#resources)

---

## Introduction

**Information Retrieval (IR)** is the task of finding relevant documents from a corpus given a query. **Retrieval-Augmented Generation (RAG)** combines IR with language models to ground generation in retrieved context, reducing hallucinations and enabling dynamic knowledge.

**Key Concepts:**
- **Sparse Retrieval**: Term-based methods (TF-IDF, BM25) using inverted indices
- **Dense Retrieval**: Neural embeddings with approximate nearest neighbor search
- **Hybrid Search**: Combining sparse and dense methods for best of both worlds
- **Reranking**: Two-stage retrieval with expensive cross-encoders on top-k results
- **RAG Pipeline**: Query -> Retrieval -> Context -> Generation -> Response

**Why RAG:**
- Addresses LLM hallucination with grounded facts
- Dynamic knowledge without retraining
- Attribution and citability
- Domain-specific knowledge integration
- Cost-effective vs fine-tuning for knowledge updates

---

## BM25 and TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** scores terms by balancing frequency in document vs corpus. **BM25 (Best Matching 25)** is a probabilistic ranking function that improves on TF-IDF with saturation and length normalization.

### TF-IDF Formula

```
TF(t, d) = count(t in d) / total_terms(d)
IDF(t, D) = log(N / DF(t))
TF-IDF(t, d, D) = TF(t, d) * IDF(t, D)
```

Where:
- `t` = term, `d` = document, `D` = corpus
- `N` = total documents
- `DF(t)` = document frequency (docs containing t)

### BM25 Formula

```
BM25(d, q) = sum over terms t in q of:
  IDF(t) * (f(t,d) * (k1 + 1)) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))

IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

**Parameters:**
- **k1** (1.2-2.0): Term frequency saturation. Higher = more weight to term frequency
- **b** (0.75): Length normalization. 0 = no normalization, 1 = full normalization
- `f(t,d)` = term frequency in document
- `|d|` = document length, `avgdl` = average document length

**Key Properties:**
- Saturation: Diminishing returns for repeated terms
- Length normalization: Penalizes longer documents
- Probabilistic foundation: BM25 derived from probability ranking principle

### BM25 Implementation

```python
from rank_bm25 import BM25Okapi
import numpy as np

# Corpus
corpus = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are great pets",
    "Machine learning is a subset of artificial intelligence"
]

# Tokenize
tokenized_corpus = [doc.lower().split() for doc in corpus]

# Create BM25 index
bm25 = BM25Okapi(tokenized_corpus)

# Query
query = "cat and dog"
tokenized_query = query.lower().split()

# Get scores
scores = bm25.get_scores(tokenized_query)
print("Scores:", scores)

# Top-k retrieval
top_n = 2
top_indices = np.argsort(scores)[::-1][:top_n]
print("\nTop documents:")
for idx in top_indices:
    print(f"  [{idx}] {corpus[idx]} (score: {scores[idx]:.4f})")
```

### Custom BM25 with Tunable Parameters

```python
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        # Calculate document frequencies
        df = {}
        for document in corpus:
            self.doc_len.append(len(document))
            frequencies = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)

            for word in frequencies.keys():
                df[word] = df.get(word, 0) + 1

        # Calculate IDF
        for word, freq in df.items():
            self.idf[word] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query):
        scores = np.zeros(self.corpus_size)
        for word in query:
            if word not in self.idf:
                continue
            idf = self.idf[word]
            for idx, doc_freq in enumerate(self.doc_freqs):
                if word not in doc_freq:
                    continue
                freq = doc_freq[word]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                scores[idx] += idf * (numerator / denominator)
        return scores

# Usage
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25_custom = BM25(tokenized_corpus, k1=1.2, b=0.75)
scores = bm25_custom.get_scores(["cat", "dog"])
```

### When BM25 Beats Dense Retrieval

**BM25 advantages:**
- **Exact keyword matching**: Medical codes, product IDs, technical terms
- **Out-of-vocabulary terms**: New entities, acronyms, typos
- **Rare term importance**: Captures importance of unique terms
- **Interpretability**: Explainable term-based scores
- **No training required**: Works out-of-the-box
- **Low latency**: Inverted index is fast
- **Domain-specific**: When vocabulary is well-defined

**Best for:**
- Legal document search (case numbers, statutes)
- Code search (function names, variable names)
- E-commerce (SKUs, exact product names)
- Medical records (diagnosis codes, drug names)
- Question answering on structured data

---

## Dense Retrieval

**Dense retrieval** uses neural networks to encode queries and documents into dense vectors, then finds nearest neighbors in embedding space. Superior for semantic matching but requires training.

### Bi-Encoder Architecture

```
Query -> Encoder_q -> q_embedding
Document -> Encoder_d -> d_embedding
Similarity = cosine(q_embedding, d_embedding)
```

**Properties:**
- Separate encoding of query and documents
- Document embeddings can be precomputed and indexed
- Fast retrieval with approximate nearest neighbor (ANN)
- Trained with contrastive learning on (query, relevant_doc, irrelevant_doc) triplets

### SBERT Models

**Sentence-BERT (SBERT)** uses siamese networks to create semantically meaningful sentence embeddings.

```python
from sentence_transformers import SentenceTransformer, util
import torch

# Popular SBERT models
# all-MiniLM-L6-v2: Fast, 384 dims, good general purpose
# all-mpnet-base-v2: Better quality, 768 dims, slower
# multi-qa-mpnet-base-dot-v1: Optimized for QA/retrieval

model = SentenceTransformer('all-MiniLM-L6-v2')

# Documents
docs = [
    "Python is a high-level programming language",
    "Machine learning models learn from data",
    "Neural networks are inspired by the brain",
    "The cat sat on the mat"
]

# Encode documents (can be done offline and cached)
doc_embeddings = model.encode(docs, convert_to_tensor=True)

# Query
query = "What is a neural network?"
query_embedding = model.encode(query, convert_to_tensor=True)

# Calculate cosine similarity
cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

# Top-k retrieval
top_k = 2
top_results = torch.topk(cos_scores, k=min(top_k, len(docs)))

print(f"Query: {query}\n")
for score, idx in zip(top_results[0], top_results[1]):
    print(f"[{idx}] {docs[idx]} (score: {score:.4f})")
```

### Model Comparison

| Model | Dims | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General, latency-critical |
| all-mpnet-base-v2 | 768 | Medium | Better | General, balanced |
| multi-qa-mpnet-base-dot-v1 | 768 | Medium | Best (QA) | Question answering |
| msmarco-distilbert-base-v4 | 768 | Fast | Good (Search) | Web search |
| all-MiniLM-L12-v2 | 384 | Medium | Better | General |

### DPR (Dense Passage Retrieval)

**DPR** uses separate BERT encoders for questions and passages, trained on question-answer pairs.

```python
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch

# Load models
q_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

ctx_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Encode query
query = "Who wrote Romeo and Juliet?"
q_input = q_tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
q_embedding = q_encoder(**q_input).pooler_output

# Encode passages
passages = [
    "William Shakespeare wrote Romeo and Juliet in the 1590s",
    "Romeo and Juliet is a tragedy about two young lovers",
    "Python is a programming language"
]

ctx_embeddings = []
for passage in passages:
    ctx_input = ctx_tokenizer(passage, return_tensors='pt', truncation=True, max_length=512)
    ctx_embedding = ctx_encoder(**ctx_input).pooler_output
    ctx_embeddings.append(ctx_embedding)

ctx_embeddings = torch.cat(ctx_embeddings, dim=0)

# Compute similarity
scores = torch.matmul(q_embedding, ctx_embeddings.T)[0]
top_idx = torch.argmax(scores).item()
print(f"Top passage: {passages[top_idx]}")
print(f"Score: {scores[top_idx]:.4f}")
```

### ColBERT

**ColBERT (Contextualized Late Interaction over BERT)** computes token-level embeddings and uses late interaction (MaxSim) for retrieval.

```python
# ColBERT architecture:
# Query -> BERT -> [q1, q2, ..., qn] (token embeddings)
# Doc -> BERT -> [d1, d2, ..., dm] (token embeddings)
# Score = sum over qi of max over dj of sim(qi, dj)

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig

# Index documents (offline)
with Run().context(RunConfig(nranks=1)):
    indexer = Indexer(checkpoint='colbert-ir/colbertv2.0')
    indexer.index(name='my_index', collection='documents.tsv', overwrite=True)

# Search (online)
with Run().context(RunConfig(nranks=1)):
    searcher = Searcher(index='my_index', checkpoint='colbert-ir/colbertv2.0')
    results = searcher.search("What is machine learning?", k=10)

    for passage_id, passage_rank, passage_score in results:
        print(f"Rank {passage_rank}: {passage_score:.2f}")
```

**ColBERT advantages:**
- Fine-grained token matching
- Better than bi-encoders for precision
- Late interaction balances speed and quality

### E5 and BGE Models

**E5 (Embeddings from bidirectional Encoder rEpresentations)** and **BGE (BAAI General Embedding)** are state-of-the-art embedding models.

```python
from sentence_transformers import SentenceTransformer

# E5 models (add prefix for best results)
model_e5 = SentenceTransformer('intfloat/e5-base-v2')

query = "query: what is machine learning"  # prefix with "query: "
docs = [
    "passage: Machine learning is a subset of AI",  # prefix with "passage: "
    "passage: Python is a programming language"
]

query_emb = model_e5.encode(query, normalize_embeddings=True)
doc_embs = model_e5.encode(docs, normalize_embeddings=True)

scores = query_emb @ doc_embs.T
print(scores)

# BGE models (best open-source embeddings as of 2024)
model_bge = SentenceTransformer('BAAI/bge-base-en-v1.5')

# For retrieval tasks, add instruction
query = "Represent this sentence for searching relevant passages: machine learning"
docs = ["Machine learning is AI", "Python is a language"]

query_emb = model_bge.encode(query, normalize_embeddings=True)
doc_embs = model_bge.encode(docs, normalize_embeddings=True)

scores = query_emb @ doc_embs.T
print(scores)
```

**E5 variants:**
- `e5-small-v2`: 384 dims, fast
- `e5-base-v2`: 768 dims, balanced
- `e5-large-v2`: 1024 dims, best quality

**BGE variants:**
- `bge-small-en-v1.5`: 384 dims
- `bge-base-en-v1.5`: 768 dims
- `bge-large-en-v1.5`: 1024 dims

---

## Cross-Encoder Reranking

**Cross-encoders** process query and document together through a single encoder, allowing full attention between them. More accurate than bi-encoders but much slower (cannot precompute document embeddings).

### Bi-Encoder vs Cross-Encoder

| Aspect | Bi-Encoder | Cross-Encoder |
|--------|------------|---------------|
| Architecture | Separate encoders | Joint encoder |
| Input | Query or doc separately | [CLS] query [SEP] doc [SEP] |
| Embeddings | Can be cached | Must compute per query |
| Speed | Fast (ANN search) | Slow (N forward passes) |
| Quality | Good | Better |
| Use case | First-stage retrieval | Reranking top-k |

### Two-Stage Retrieval Pipeline

```
Stage 1 (Retrieval): Bi-encoder retrieves top-100 candidates (fast)
Stage 2 (Reranking): Cross-encoder reranks top-100 to top-10 (accurate)
```

### Cross-Encoder Implementation

```python
from sentence_transformers import CrossEncoder
import numpy as np

# Load cross-encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Simulated first-stage retrieval results
query = "What is machine learning?"
candidates = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a programming language used for ML",
    "The weather is nice today",
    "Neural networks learn from data using backpropagation",
    "Cats are popular pets"
]

# Create query-document pairs
pairs = [[query, doc] for doc in candidates]

# Score with cross-encoder
scores = cross_encoder.predict(pairs)

# Rank by score
ranked_indices = np.argsort(scores)[::-1]

print("Reranked results:")
for rank, idx in enumerate(ranked_indices, 1):
    print(f"{rank}. [{scores[idx]:.4f}] {candidates[idx]}")
```

### Full Two-Stage Pipeline

```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch

# Models
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Document corpus
corpus = [
    "Machine learning uses statistical techniques to give computers the ability to learn",
    "Deep learning is a subset of machine learning using neural networks",
    "Python is a popular programming language for data science",
    "Natural language processing enables computers to understand text",
    "Computer vision allows machines to interpret visual information",
    "The cat sat on the mat",
    "Reinforcement learning trains agents through rewards and penalties"
]

# Query
query = "What is deep learning?"

# Stage 1: Bi-encoder retrieval (top-k1)
print("Stage 1: Bi-encoder retrieval")
corpus_embeddings = bi_encoder.encode(corpus, convert_to_tensor=True)
query_embedding = bi_encoder.encode(query, convert_to_tensor=True)

cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
top_k1 = 5
top_results = torch.topk(cos_scores, k=top_k1)

print(f"\nTop-{top_k1} candidates:")
candidates = []
for score, idx in zip(top_results[0], top_results[1]):
    candidates.append(corpus[idx])
    print(f"  [{score:.4f}] {corpus[idx]}")

# Stage 2: Cross-encoder reranking (top-k2)
print(f"\nStage 2: Cross-encoder reranking")
pairs = [[query, doc] for doc in candidates]
cross_scores = cross_encoder.predict(pairs)

top_k2 = 3
reranked_indices = np.argsort(cross_scores)[::-1][:top_k2]

print(f"\nFinal top-{top_k2}:")
for rank, idx in enumerate(reranked_indices, 1):
    print(f"{rank}. [{cross_scores[idx]:.4f}] {candidates[idx]}")
```

### Popular Cross-Encoder Models

```python
# MS MARCO trained models (passage ranking)
models = [
    'cross-encoder/ms-marco-TinyBERT-L-2-v2',      # Fastest, 4 layers
    'cross-encoder/ms-marco-MiniLM-L-6-v2',        # Balanced, 6 layers
    'cross-encoder/ms-marco-MiniLM-L-12-v2',       # Better, 12 layers
]

# Natural Questions (NQ) trained
'cross-encoder/nq-distilbert-base-v1'

# MultiLingual
'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
```

---

## Hybrid Search

**Hybrid search** combines sparse (BM25) and dense (embedding) retrieval to leverage complementary strengths: exact matching + semantic understanding.

### Fusion Strategies

**1. Reciprocal Rank Fusion (RRF)**

```
RRF_score(d) = sum over all rankings r of: 1 / (k + rank_r(d))
```

Where `k` is a constant (typically 60) to avoid division by zero.

**2. Linear Combination**

```
score(d) = alpha * BM25(d) + (1 - alpha) * dense(d)
```

**3. Weighted Combination**

```
score(d) = w1 * normalize(BM25(d)) + w2 * normalize(dense(d))
```

### Reciprocal Rank Fusion Implementation

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import numpy as np

def reciprocal_rank_fusion(rankings_list, k=60):
    """
    Combine multiple rankings using RRF.

    Args:
        rankings_list: List of lists, each containing (doc_id, score) tuples
        k: Constant to prevent division by zero

    Returns:
        Combined ranking as list of (doc_id, rrf_score)
    """
    rrf_scores = {}

    for rankings in rankings_list:
        for rank, (doc_id, _) in enumerate(rankings, 1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (k + rank)

    # Sort by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

# Example usage
corpus = [
    "Machine learning is a field of artificial intelligence",
    "Python is great for data science and ML",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing helps computers understand text",
    "The quick brown fox jumps over the lazy dog"
]

query = "What is deep learning?"

# BM25 retrieval
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
bm25_scores = bm25.get_scores(query.lower().split())
bm25_rankings = [(i, score) for i, score in enumerate(bm25_scores)]
bm25_rankings.sort(key=lambda x: x[1], reverse=True)

# Dense retrieval
model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = model.encode(corpus)
query_embedding = model.encode(query)
dense_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].numpy()
dense_rankings = [(i, score) for i, score in enumerate(dense_scores)]
dense_rankings.sort(key=lambda x: x[1], reverse=True)

# Hybrid with RRF
hybrid_rankings = reciprocal_rank_fusion([bm25_rankings, dense_rankings], k=60)

print("BM25 Top-3:")
for doc_id, score in bm25_rankings[:3]:
    print(f"  [{score:.4f}] {corpus[doc_id]}")

print("\nDense Top-3:")
for doc_id, score in dense_rankings[:3]:
    print(f"  [{score:.4f}] {corpus[doc_id]}")

print("\nHybrid (RRF) Top-3:")
for doc_id, score in hybrid_rankings[:3]:
    print(f"  [{score:.4f}] {corpus[doc_id]}")
```

### Alpha Weighting

```python
def hybrid_search_alpha(query, corpus, alpha=0.5, top_k=5):
    """
    Hybrid search with alpha weighting.

    Args:
        alpha: Weight for BM25 (1-alpha for dense)
               alpha=1.0: pure BM25
               alpha=0.0: pure dense
               alpha=0.5: equal weight
    """
    # BM25
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())

    # Normalize BM25 scores to [0, 1]
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # Dense
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = model.encode(corpus)
    query_embedding = model.encode(query)
    dense_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].numpy()

    # Normalize dense scores to [0, 1]
    if dense_scores.max() > 0:
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())

    # Combine
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

    # Top-k
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'doc': corpus[idx],
            'hybrid_score': hybrid_scores[idx],
            'bm25_score': bm25_scores[idx],
            'dense_score': dense_scores[idx]
        })

    return results

# Test different alpha values
query = "neural network deep learning"
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    print(f"\nAlpha={alpha} (BM25 weight):")
    results = hybrid_search_alpha(query, corpus, alpha=alpha, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['hybrid_score']:.3f}] {r['doc'][:50]}...")
```

### When to Use Hybrid

**Use hybrid when:**
- Domain has both technical terms AND semantic concepts
- Need robustness to vocabulary mismatch
- Queries vary from keyword to natural language
- Want best of both worlds without choosing

**Tuning alpha:**
- High alpha (0.7-0.9): Technical domains, exact terms matter
- Balanced (0.5): General purpose
- Low alpha (0.1-0.3): Semantic similarity matters more

---

## Vector Databases

**Vector databases** are optimized for storing and searching high-dimensional embeddings at scale. Support approximate nearest neighbor (ANN) search.

### FAISS (Facebook AI Similarity Search)

**FAISS** is a library for efficient similarity search and clustering of dense vectors.

**Index Types:**

| Index | Build Time | Search Speed | Memory | Accuracy | Use Case |
|-------|------------|--------------|--------|----------|----------|
| Flat | Fast | Slow | High | 100% | Small datasets (<10k) |
| IVF | Medium | Fast | Medium | ~95% | Medium datasets |
| HNSW | Slow | Fastest | Highest | ~99% | Low latency needs |
| PQ | Fast | Medium | Lowest | ~90% | Large datasets, memory constrained |

### FAISS Implementation

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    "Machine learning is a subset of AI",
    "Python is a programming language",
    "Deep learning uses neural networks",
    "Natural language processing analyzes text"
] * 100  # Simulate larger dataset

embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype('float32')

d = embeddings.shape[1]  # Dimension
n = embeddings.shape[0]  # Number of vectors

print(f"Dataset: {n} vectors, {d} dimensions")

# 1. Flat Index (Exact search)
print("\n1. Flat Index (Exact Search)")
index_flat = faiss.IndexFlatL2(d)
index_flat.add(embeddings)

query = "What is neural networks?"
query_embedding = model.encode([query]).astype('float32')

k = 3
distances, indices = index_flat.search(query_embedding, k)
print(f"Top-{k} results:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"  {i+1}. [{dist:.4f}] {documents[idx]}")

# 2. IVF Index (Inverted File with clustering)
print("\n2. IVF Index")
nlist = 10  # Number of clusters
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train the index
index_ivf.train(embeddings)
index_ivf.add(embeddings)

# Search (nprobe = number of clusters to visit)
index_ivf.nprobe = 3
distances, indices = index_ivf.search(query_embedding, k)
print(f"Top-{k} results:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"  {i+1}. [{dist:.4f}] {documents[idx]}")

# 3. HNSW Index (Hierarchical Navigable Small World)
print("\n3. HNSW Index")
M = 32  # Number of connections per layer
index_hnsw = faiss.IndexHNSWFlat(d, M)
index_hnsw.add(embeddings)

distances, indices = index_hnsw.search(query_embedding, k)
print(f"Top-{k} results:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"  {i+1}. [{dist:.4f}] {documents[idx]}")

# 4. Product Quantization (Memory efficient)
print("\n4. PQ Index")
m = 8  # Number of sub-vectors
nbits = 8  # Bits per sub-vector
index_pq = faiss.IndexPQ(d, m, nbits)
index_pq.train(embeddings)
index_pq.add(embeddings)

distances, indices = index_pq.search(query_embedding, k)
print(f"Top-{k} results:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"  {i+1}. [{dist:.4f}] {documents[idx]}")
```

### FAISS with IVF + PQ (Production)

```python
# Combine IVF and PQ for best balance
nlist = 100  # Clusters
m = 8        # Sub-vectors for PQ
nbits = 8    # Bits per sub-vector

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# Train
index.train(embeddings)
index.add(embeddings)

# Search
index.nprobe = 10  # Search 10 clusters
distances, indices = index.search(query_embedding, k=5)
```

### Saving and Loading FAISS

```python
# Save
faiss.write_index(index, "vector.index")

# Load
index = faiss.read_index("vector.index")

# Save with IDs
import pickle
ids = list(range(len(documents)))
with open("doc_ids.pkl", "wb") as f:
    pickle.dump(ids, f)
```

### ChromaDB

**ChromaDB** is an open-source embedding database designed for LLM applications.

```python
import chromadb
from chromadb.config import Settings

# Initialize client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Create collection
collection = client.create_collection(
    name="my_documents",
    metadata={"description": "ML papers"}
)

# Add documents
documents = [
    "Machine learning is a subset of AI",
    "Python is great for data science",
    "Deep learning uses neural networks"
]
ids = ["doc1", "doc2", "doc3"]
metadatas = [
    {"source": "wiki", "topic": "ML"},
    {"source": "blog", "topic": "programming"},
    {"source": "paper", "topic": "DL"}
]

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

# Query
results = collection.query(
    query_texts=["What is deep learning?"],
    n_results=2,
    where={"topic": "DL"}  # Metadata filter
)

print("Results:", results['documents'])
print("Distances:", results['distances'])
print("Metadatas:", results['metadatas'])

# Update
collection.update(
    ids=["doc1"],
    documents=["Machine learning is a field of AI that learns from data"]
)

# Delete
collection.delete(ids=["doc3"])

# Persist
client.persist()
```

### Vector Database Comparison

| Database | Type | Features | Best For |
|----------|------|----------|----------|
| **FAISS** | Library | Fast, many index types, no metadata | High performance, offline |
| **ChromaDB** | Embedded | Easy API, metadata, persistence | Prototyping, small scale |
| **Pinecone** | Cloud | Managed, scalable, metadata filters | Production, serverless |
| **Weaviate** | Self-hosted | GraphQL, hybrid search, modules | Complex queries |
| **Qdrant** | Self-hosted/Cloud | Payload filters, fast, Rust | Production, filtering |
| **Milvus** | Self-hosted | Distributed, GPU support | Large scale |
| **pgvector** | Postgres extension | SQL, ACID, existing infra | Postgres users |

### Pinecone Example

```python
import pinecone

# Initialize
pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

# Create index
index_name = "ml-docs"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,
        metric="cosine"
    )

index = pinecone.Index(index_name)

# Upsert vectors
vectors = [
    ("doc1", embeddings[0].tolist(), {"text": documents[0], "category": "ML"}),
    ("doc2", embeddings[1].tolist(), {"text": documents[1], "category": "programming"})
]
index.upsert(vectors=vectors)

# Query
query_vector = query_embedding[0].tolist()
results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True,
    filter={"category": "ML"}
)

for match in results['matches']:
    print(f"[{match['score']:.4f}] {match['metadata']['text']}")
```

### Qdrant Example

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize
client = QdrantClient(path="./qdrant_db")  # or url="http://localhost:6333"

# Create collection
collection_name = "ml_docs"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Upload points
points = [
    PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={"text": documents[i], "category": "ML"}
    )
    for i in range(len(documents))
]
client.upsert(collection_name=collection_name, points=points)

# Search
results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding[0].tolist(),
    limit=3,
    query_filter={
        "must": [{"key": "category", "match": {"value": "ML"}}]
    }
)

for result in results:
    print(f"[{result.score:.4f}] {result.payload['text']}")
```

---

## 7. RAG Architecture

### 7.1 Retrieve-Then-Generate Pipeline

**RAG** (Retrieval-Augmented Generation) combines retrieval with generation to provide factual, grounded responses.

```python
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
from typing import List, Tuple

class BasicRAG:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.client = OpenAI()
        self.knowledge_base = []
        self.embeddings = None

    def index_documents(self, documents: List[str]):
        """Index documents for retrieval."""
        self.knowledge_base = documents
        self.embeddings = self.encoder.encode(
            documents,
            convert_to_tensor=False,
            show_progress_bar=True
        )

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top-k most relevant documents."""
        query_emb = self.encoder.encode([query])[0]

        # Compute cosine similarity
        scores = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [(self.knowledge_base[i], scores[i]) for i in top_k_idx]

    def generate(self, query: str, contexts: List[str]) -> str:
        """Generate answer using retrieved contexts."""
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""Answer the question using only the provided context. If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{context_str}

Question: {query}

Answer:"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content

    def query(self, question: str, k: int = 3) -> dict:
        """Complete RAG pipeline."""
        # Retrieve
        retrieved = self.retrieve(question, k)
        contexts = [doc for doc, _ in retrieved]
        scores = [score for _, score in retrieved]

        # Generate
        answer = self.generate(question, contexts)

        return {
            "answer": answer,
            "contexts": contexts,
            "scores": scores
        }

# Usage
rag = BasicRAG()
documents = [
    "The Eiffel Tower is located in Paris, France and was completed in 1889.",
    "Python is a high-level programming language created by Guido van Rossum.",
    "The Great Wall of China is over 13,000 miles long."
]
rag.index_documents(documents)

result = rag.query("Where is the Eiffel Tower?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['contexts']}")
```

### 7.2 Naive vs Advanced vs Modular RAG

```python
# Naive RAG: Simple retrieve + generate
class NaiveRAG:
    def query(self, question: str):
        contexts = self.retrieve(question, k=3)
        return self.generate(question, contexts)

# Advanced RAG: Query rewriting, hybrid search, reranking
class AdvancedRAG:
    def query(self, question: str):
        # Query expansion
        expanded_queries = self.expand_query(question)

        # Hybrid retrieval
        dense_results = self.dense_retrieve(expanded_queries)
        sparse_results = self.sparse_retrieve(expanded_queries)
        fused_results = self.fusion(dense_results, sparse_results)

        # Reranking
        reranked = self.rerank(question, fused_results)

        # Generate with citations
        return self.generate_with_citations(question, reranked)

# Modular RAG: Flexible routing and multi-step reasoning
class ModularRAG:
    def query(self, question: str):
        # Route to appropriate retrieval strategy
        strategy = self.route_query(question)

        if strategy == "multi_hop":
            return self.multi_hop_retrieve(question)
        elif strategy == "decompose":
            sub_questions = self.decompose(question)
            return self.answer_subquestions(sub_questions)
        else:
            return self.standard_retrieve(question)
```

### 7.3 LangChain RAG Implementation

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

# Load and split documents
loader = TextLoader("documents.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Create RAG chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Query
result = qa_chain({"query": "What is the main topic?"})
print(result["result"])
print(f"Sources: {[doc.page_content for doc in result['source_documents']]}")
```

### 7.4 LlamaIndex RAG Implementation

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure
Settings.llm = OpenAI(model="gpt-4", temperature=0)
Settings.embed_model = OpenAIEmbedding()

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Build index
index = VectorStoreIndex.from_documents(documents)

# Query with citations
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact"
)

response = query_engine.query("What is the main topic?")
print(f"Answer: {response}")
print(f"Sources: {[node.text for node in response.source_nodes]}")
```

### 7.5 Citation and Attribution

```python
def generate_with_citations(query: str, contexts: List[Tuple[str, str]]) -> str:
    """Generate answer with inline citations."""
    context_str = "\n\n".join([
        f"[{i+1}] (Source: {source})\n{text}"
        for i, (text, source) in enumerate(contexts)
    ])

    prompt = f"""Answer the question using the provided context. Include inline citations [1], [2], etc. when using information from specific sources.

Context:
{context_str}

Question: {query}

Answer (with citations):"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    return response.choices[0].message.content

# Usage
contexts = [
    ("Paris is the capital of France.", "doc1.txt"),
    ("The Eiffel Tower was built in 1889.", "doc2.txt")
]
answer = generate_with_citations("When was the Eiffel Tower built?", contexts)
# Output: "The Eiffel Tower was built in 1889 [2]."
```

---

## 8. Chunking Strategies

### 8.1 Fixed-Size Chunking

```python
def chunk_by_characters(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into fixed-size character chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

def chunk_by_tokens(text: str, chunk_size: int = 128, overlap: int = 20) -> List[str]:
    """Split text into fixed-size token chunks."""
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap

    return chunks

# Usage
text = "Long document text here..."
char_chunks = chunk_by_characters(text, chunk_size=500, overlap=50)
token_chunks = chunk_by_tokens(text, chunk_size=128, overlap=20)
```

### 8.2 Recursive Character Text Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Splits on multiple separators in order
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

text = """This is a long document.

It has multiple paragraphs.

Each paragraph should ideally stay together."""

chunks = splitter.split_text(text)
```

### 8.3 Semantic Chunking

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunking(text: str, threshold: float = 0.5) -> List[str]:
    """Chunk text based on semantic similarity between sentences."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Split into sentences
    sentences = text.split('. ')
    if not sentences:
        return [text]

    # Encode sentences
    embeddings = model.encode(sentences)

    # Compute similarity between consecutive sentences
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )

        if similarity > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    return chunks

# Usage
text = "Sentence 1 about topic A. Sentence 2 about topic A. Sentence 3 about topic B."
chunks = semantic_chunking(text, threshold=0.5)
```

### 8.4 Document-Aware Chunking

```python
def chunk_by_headers(markdown_text: str) -> List[dict]:
    """Split markdown by headers, preserving hierarchy."""
    import re

    lines = markdown_text.split('\n')
    chunks = []
    current_chunk = {"header": "", "content": []}

    for line in lines:
        if re.match(r'^#{1,6}\s', line):
            if current_chunk["content"]:
                chunks.append({
                    "header": current_chunk["header"],
                    "text": '\n'.join(current_chunk["content"])
                })
            current_chunk = {"header": line, "content": []}
        else:
            current_chunk["content"].append(line)

    if current_chunk["content"]:
        chunks.append({
            "header": current_chunk["header"],
            "text": '\n'.join(current_chunk["content"])
        })

    return chunks

# Usage
markdown = """# Main Title
Content under main title.

## Subsection
Content under subsection."""

chunks = chunk_by_headers(markdown)
```

### 8.5 Parent Document Retrieval

```python
class ParentDocumentRetriever:
    """Retrieve small chunks but return larger parent context."""

    def __init__(self, parent_chunk_size: int = 2000, child_chunk_size: int = 400):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_to_parent = {}
        self.child_embeddings = []
        self.child_chunks = []

    def index_document(self, text: str):
        """Create parent and child chunks."""
        # Create parent chunks
        parent_chunks = chunk_by_characters(text, self.parent_chunk_size, overlap=200)

        # Create child chunks from each parent
        for parent_id, parent_chunk in enumerate(parent_chunks):
            child_chunks = chunk_by_characters(parent_chunk, self.child_chunk_size, overlap=50)

            for child_chunk in child_chunks:
                child_id = len(self.child_chunks)
                self.child_chunks.append(child_chunk)
                self.child_to_parent[child_id] = parent_chunk

        # Encode child chunks
        self.child_embeddings = self.encoder.encode(self.child_chunks)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve parent chunks based on child chunk similarity."""
        query_emb = self.encoder.encode([query])[0]

        scores = np.dot(self.child_embeddings, query_emb) / (
            np.linalg.norm(self.child_embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        top_k_idx = np.argsort(scores)[-k:][::-1]

        # Return parent chunks (deduplicated)
        parent_chunks = []
        seen = set()
        for idx in top_k_idx:
            parent = self.child_to_parent[idx]
            if parent not in seen:
                parent_chunks.append(parent)
                seen.add(parent)

        return parent_chunks

# Usage
retriever = ParentDocumentRetriever(parent_chunk_size=2000, child_chunk_size=400)
retriever.index_document("Very long document text...")
parents = retriever.retrieve("query", k=3)
```

---

## 9. Advanced RAG Patterns

### 9.1 Multi-Hop RAG

```python
class MultiHopRAG:
    """Iteratively retrieve and reason over multiple steps."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def multi_hop_query(self, question: str, max_hops: int = 3) -> dict:
        """Perform multi-hop retrieval and reasoning."""
        all_contexts = []
        current_query = question

        for hop in range(max_hops):
            # Retrieve based on current query
            contexts = self.retriever.retrieve(current_query, k=3)
            all_contexts.extend(contexts)

            # Generate intermediate reasoning
            reasoning_prompt = f"""Based on the context, what additional information is needed to answer: {question}

Context: {' '.join(contexts)}

Next query (or "DONE" if sufficient):"""

            next_query = self.llm.generate(reasoning_prompt)

            if "DONE" in next_query.upper():
                break

            current_query = next_query

        # Final answer generation
        final_prompt = f"""Answer the question using all retrieved context.

Context: {' '.join(all_contexts)}

Question: {question}

Answer:"""

        answer = self.llm.generate(final_prompt)
        return {"answer": answer, "hops": hop + 1, "contexts": all_contexts}
```

### 9.2 Self-RAG (Self-Reflective RAG)

```python
class SelfRAG:
    """Self-reflective retrieval-augmented generation."""

    def query(self, question: str) -> str:
        # Step 1: Decide if retrieval is needed
        retrieval_decision = self.llm.generate(f"""Does this question require external knowledge? Answer YES or NO.

Question: {question}

Decision:""")

        if "NO" in retrieval_decision:
            return self.llm.generate(question)

        # Step 2: Retrieve
        contexts = self.retriever.retrieve(question, k=5)

        # Step 3: Generate candidate answers for each context
        candidates = []
        for ctx in contexts:
            answer = self.llm.generate(f"Context: {ctx}\n\nQuestion: {question}\n\nAnswer:")

            # Step 4: Self-evaluate relevance and support
            relevance = self.evaluate_relevance(ctx, question)
            support = self.evaluate_support(ctx, answer)

            candidates.append({
                "answer": answer,
                "context": ctx,
                "relevance": relevance,
                "support": support
            })

        # Step 5: Select best answer
        best = max(candidates, key=lambda x: x["relevance"] * x["support"])
        return best["answer"]

    def evaluate_relevance(self, context: str, question: str) -> float:
        """Score how relevant the context is to the question."""
        prompt = f"""Rate the relevance of the context to the question on a scale of 0-1.

Question: {question}
Context: {context}

Relevance score:"""
        score_str = self.llm.generate(prompt)
        return float(score_str.strip())

    def evaluate_support(self, context: str, answer: str) -> float:
        """Score how well the context supports the answer."""
        prompt = f"""Rate how well the context supports the answer on a scale of 0-1.

Context: {context}
Answer: {answer}

Support score:"""
        score_str = self.llm.generate(prompt)
        return float(score_str.strip())
```

### 9.3 Corrective RAG (CRAG)

```python
class CorrectiveRAG:
    """Verify and correct retrieval results."""

    def query(self, question: str) -> str:
        # Retrieve
        contexts = self.retriever.retrieve(question, k=5)

        # Evaluate retrieval quality
        relevant_contexts = []
        for ctx in contexts:
            relevance_score = self.evaluate_relevance(ctx, question)

            if relevance_score > 0.7:
                relevant_contexts.append(ctx)

        # Corrective action based on quality
        if len(relevant_contexts) == 0:
            # No relevant contexts: use web search or fallback
            return self.web_search_fallback(question)
        elif len(relevant_contexts) < 3:
            # Few relevant contexts: augment with web search
            web_results = self.web_search(question)
            relevant_contexts.extend(web_results)

        # Generate with verified contexts
        return self.generate(question, relevant_contexts)

    def evaluate_relevance(self, context: str, question: str) -> float:
        """Use LLM to evaluate context relevance."""
        prompt = f"""Score the relevance of this context to the question (0.0 to 1.0).

Question: {question}
Context: {context}

Score:"""
        return float(self.llm.generate(prompt).strip())
```

### 9.4 Hypothetical Document Embeddings (HyDE)

```python
class HyDE:
    """Generate hypothetical answer and use it for retrieval."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def query(self, question: str) -> str:
        # Generate hypothetical answer (without retrieval)
        hyde_prompt = f"""Write a detailed answer to this question (even if you're not certain):

Question: {question}

Hypothetical answer:"""

        hypothetical_doc = self.llm.generate(hyde_prompt)

        # Use hypothetical document for retrieval
        contexts = self.retriever.retrieve(hypothetical_doc, k=5)

        # Generate final answer with retrieved contexts
        final_prompt = f"""Answer the question using the provided context.

Context: {' '.join(contexts)}

Question: {question}

Answer:"""

        return self.llm.generate(final_prompt)
```

### 9.5 Query Decomposition

```python
def decompose_query(complex_question: str, llm) -> List[str]:
    """Break complex question into simpler sub-questions."""
    prompt = f"""Break down this complex question into 2-4 simpler sub-questions.

Question: {complex_question}

Sub-questions (one per line):"""

    response = llm.generate(prompt)
    sub_questions = [q.strip() for q in response.split('\n') if q.strip()]
    return sub_questions

def answer_with_decomposition(question: str, retriever, llm) -> str:
    """Answer by decomposing into sub-questions."""
    sub_questions = decompose_query(question, llm)

    sub_answers = []
    for sq in sub_questions:
        contexts = retriever.retrieve(sq, k=3)
        answer = llm.generate(f"Context: {' '.join(contexts)}\n\nQuestion: {sq}\n\nAnswer:")
        sub_answers.append(f"Q: {sq}\nA: {answer}")

    # Synthesize final answer
    synthesis_prompt = f"""Synthesize these sub-answers into a complete answer for the original question.

Original question: {question}

Sub-answers:
{chr(10).join(sub_answers)}

Final answer:"""

    return llm.generate(synthesis_prompt)
```

### 9.6 Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> List[str]:
    """Fuse multiple ranked lists using RRF."""
    scores = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += 1 / (k + rank + 1)

    # Sort by RRF score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in fused]

# Usage: Combine BM25 and dense retrieval
bm25_results = ["doc3", "doc1", "doc5"]
dense_results = ["doc1", "doc2", "doc3"]
fused = reciprocal_rank_fusion([bm25_results, dense_results])
```

---

## 10. Retrieval Quality Metrics

### 10.1 Mean Reciprocal Rank (MRR)

```python
def mean_reciprocal_rank(results: List[List[str]], relevant: List[str]) -> float:
    """Calculate MRR for a set of queries.

    Args:
        results: List of ranked results for each query
        relevant: List of relevant documents for each query
    """
    reciprocal_ranks = []

    for query_results, query_relevant in zip(results, relevant):
        for rank, doc in enumerate(query_results, 1):
            if doc == query_relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)

# Usage
results = [["doc1", "doc2", "doc3"], ["doc4", "doc1", "doc5"]]
relevant = ["doc2", "doc1"]
mrr = mean_reciprocal_rank(results, relevant)
print(f"MRR: {mrr:.3f}")  # 0.667
```

### 10.2 Normalized Discounted Cumulative Gain (NDCG)

```python
import numpy as np

def dcg_at_k(relevances: List[float], k: int) -> float:
    """Calculate DCG@k."""
    relevances = np.asarray(relevances)[:k]
    if relevances.size:
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0.0

def ndcg_at_k(relevances: List[float], k: int) -> float:
    """Calculate NDCG@k."""
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

# Usage
relevances = [3, 2, 3, 0, 1, 2]  # Relevance scores for top 6 results
ndcg_5 = ndcg_at_k(relevances, k=5)
print(f"NDCG@5: {ndcg_5:.3f}")
```

### 10.3 Recall@K and Precision@K

```python
def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Recall@K."""
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)

    if len(relevant_set) == 0:
        return 0.0

    return len(retrieved_at_k & relevant_set) / len(relevant_set)

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Precision@K."""
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)

    if k == 0:
        return 0.0

    return len(retrieved_at_k & relevant_set) / k

# Usage
retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant = ["doc2", "doc3", "doc7"]
recall_3 = recall_at_k(retrieved, relevant, k=3)
precision_3 = precision_at_k(retrieved, relevant, k=3)
print(f"Recall@3: {recall_3:.3f}, Precision@3: {precision_3:.3f}")
```

### 10.4 Mean Average Precision (MAP)

```python
def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """Calculate Average Precision for a single query."""
    relevant_set = set(relevant)
    precision_sum = 0.0
    num_relevant = 0

    for k, doc in enumerate(retrieved, 1):
        if doc in relevant_set:
            num_relevant += 1
            precision_sum += num_relevant / k

    if len(relevant_set) == 0:
        return 0.0

    return precision_sum / len(relevant_set)

def mean_average_precision(all_retrieved: List[List[str]], all_relevant: List[List[str]]) -> float:
    """Calculate MAP across multiple queries."""
    aps = [average_precision(ret, rel) for ret, rel in zip(all_retrieved, all_relevant)]
    return sum(aps) / len(aps)

# Usage
all_retrieved = [["doc1", "doc2", "doc3"], ["doc4", "doc5", "doc6"]]
all_relevant = [["doc2", "doc3"], ["doc4"]]
map_score = mean_average_precision(all_retrieved, all_relevant)
print(f"MAP: {map_score:.3f}")
```

### 10.5 RAGAS Evaluation Framework

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

# Prepare evaluation data
eval_data = {
    "question": ["What is the capital of France?", "When was Python created?"],
    "answer": ["Paris is the capital of France.", "Python was created in 1991."],
    "contexts": [
        ["Paris is the capital and largest city of France."],
        ["Python was created by Guido van Rossum and released in 1991."]
    ],
    "ground_truth": ["Paris", "1991"]
}

dataset = Dataset.from_dict(eval_data)

# Evaluate
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision]
)

print(result)
# {'faithfulness': 0.95, 'answer_relevancy': 0.92, 'context_recall': 1.0, 'context_precision': 0.88}
```

---

## 11. Embedding Fine-Tuning

### 11.1 When to Fine-Tune

Fine-tune embeddings when:
- **Domain-specific vocabulary** (medical, legal, technical)
- **Specialized retrieval tasks** (code search, product search)
- **Off-the-shelf models underperform** on your evaluation set
- **You have sufficient training data** (1000+ query-document pairs)

### 11.2 Training Data Generation

```python
import random
from typing import List, Tuple

def generate_hard_negatives(query: str, positive_doc: str, corpus: List[str], k: int = 5) -> List[str]:
    """Generate hard negatives using BM25 or dense retrieval."""
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(query.split())
    top_k_idx = np.argsort(scores)[-k*2:][::-1]

    # Filter out the positive document
    hard_negatives = [corpus[i] for i in top_k_idx if corpus[i] != positive_doc][:k]
    return hard_negatives

def create_training_triplets(queries: List[str], positives: List[str], corpus: List[str]) -> List[Tuple[str, str, str]]:
    """Create (query, positive, negative) triplets."""
    triplets = []

    for query, positive in zip(queries, positives):
        hard_negatives = generate_hard_negatives(query, positive, corpus, k=3)
        for negative in hard_negatives:
            triplets.append((query, positive, negative))

    return triplets
```

### 11.3 Fine-Tuning with sentence-transformers

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Prepare training data
train_examples = [
    InputExample(texts=["query 1", "positive doc 1", "negative doc 1"]),
    InputExample(texts=["query 2", "positive doc 2", "negative doc 2"]),
]

# Load base model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss (Multiple Negatives Ranking Loss)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='./fine-tuned-retriever'
)

# Save
model.save('./fine-tuned-retriever')
```

### 11.4 Contrastive Learning

```python
from sentence_transformers import losses

# Triplet loss: (anchor, positive, negative)
triplet_loss = losses.TripletLoss(model=model)

# Multiple Negatives Ranking Loss (more efficient)
mnr_loss = losses.MultipleNegativesRankingLoss(model)

# CoSENT loss (circle loss variant)
cosent_loss = losses.CoSENTLoss(model)

# Training with contrastive loss
model.fit(
    train_objectives=[(train_dataloader, mnr_loss)],
    epochs=3,
    evaluator=evaluator,
    evaluation_steps=500,
    warmup_steps=100
)
```

### 11.5 Matryoshka Representation Learning

```python
from sentence_transformers import losses

# Train embeddings that work at multiple dimensions
matryoshka_dims = [768, 512, 256, 128, 64]

matryoshka_loss = losses.MatryoshkaLoss(
    model=model,
    loss=losses.MultipleNegativesRankingLoss(model),
    matryoshka_dims=matryoshka_dims
)

model.fit(
    train_objectives=[(train_dataloader, matryoshka_loss)],
    epochs=3
)

# Use embeddings at different dimensions
full_emb = model.encode("query", convert_to_tensor=True)  # 768-dim
compact_emb = full_emb[:128]  # 128-dim (faster, slightly lower quality)
```

---

## 12. Production RAG Systems

### 12.1 Indexing Pipeline

```python
from typing import List, Iterator
import hashlib

class ProductionIndexer:
    """Production-ready document indexing pipeline."""

    def __init__(self, vectorstore, chunk_size: int = 512):
        self.vectorstore = vectorstore
        self.chunk_size = chunk_size
        self.processed_docs = set()

    def index_documents_batch(self, documents: Iterator[str], batch_size: int = 100):
        """Index documents in batches with deduplication."""
        batch = []

        for doc in documents:
            # Deduplication using hash
            doc_hash = hashlib.md5(doc.encode()).hexdigest()
            if doc_hash in self.processed_docs:
                continue

            self.processed_docs.add(doc_hash)

            # Chunk document
            chunks = self.chunk_document(doc)
            batch.extend(chunks)

            # Process batch
            if len(batch) >= batch_size:
                self.vectorstore.add_texts(batch)
                batch = []

        # Process remaining
        if batch:
            self.vectorstore.add_texts(batch)

    def chunk_document(self, doc: str) -> List[str]:
        """Chunk document with metadata preservation."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=50
        )
        return splitter.split_text(doc)
```

### 12.2 Caching Strategies

```python
from functools import lru_cache
import hashlib
import pickle
import redis

class RAGCache:
    """Multi-level caching for RAG systems."""

    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.embedding_cache = {}

    @lru_cache(maxsize=1000)
    def get_query_result(self, query: str) -> str:
        """In-memory cache for query results."""
        return self._execute_query(query)

    def get_embedding(self, text: str):
        """Cache embeddings in Redis."""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Check Redis cache
        if self.redis_client:
            cached = self.redis_client.get(f"emb:{text_hash}")
            if cached:
                return pickle.loads(cached)

        # Compute embedding
        embedding = self.encoder.encode(text)

        # Store in Redis
        if self.redis_client:
            self.redis_client.setex(
                f"emb:{text_hash}",
                86400,  # 24 hour TTL
                pickle.dumps(embedding)
            )

        return embedding
```

### 12.3 Query Routing

```python
class QueryRouter:
    """Route queries to appropriate retrieval strategies."""

    def route(self, query: str) -> str:
        """Determine query complexity and route accordingly."""
        query_lower = query.lower()

        # Simple factoid questions
        if any(word in query_lower for word in ["what", "when", "where", "who"]):
            return "simple_retrieval"

        # Complex reasoning questions
        elif any(word in query_lower for word in ["why", "how", "explain"]):
            return "multi_hop_retrieval"

        # Comparison questions
        elif "compare" in query_lower or "versus" in query_lower:
            return "comparison_retrieval"

        # Default
        return "standard_retrieval"

    def query(self, question: str):
        strategy = self.route(question)

        if strategy == "simple_retrieval":
            return self.simple_rag(question)
        elif strategy == "multi_hop_retrieval":
            return self.multi_hop_rag(question)
        else:
            return self.standard_rag(question)
```

### 12.4 Monitoring and Logging

```python
import logging
from datetime import datetime
import json

class RAGMonitor:
    """Monitor RAG system performance in production."""

    def __init__(self, log_file: str = "rag_metrics.log"):
        self.logger = logging.getLogger("RAG")
        handler = logging.FileHandler(log_file)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_query(self, query: str, contexts: List[str], answer: str, latency: float):
        """Log each query with metadata."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_contexts": len(contexts),
            "answer_length": len(answer),
            "latency_ms": latency * 1000,
            "avg_context_score": self._avg_score(contexts)
        }
        self.logger.info(json.dumps(log_data))

    def _avg_score(self, contexts: List[str]) -> float:
        """Compute average retrieval score."""
        # Implementation depends on your retrieval system
        return 0.85
```

### 12.5 Cost Optimization

```python
class CostOptimizedRAG:
    """Optimize costs for embedding and LLM calls."""

    def __init__(self):
        self.embedding_cache = {}
        self.query_cache = {}

    def embed_with_batching(self, texts: List[str], batch_size: int = 32):
        """Batch embeddings to reduce API calls."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.encoder.encode(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def generate_with_fallback(self, query: str, contexts: List[str]):
        """Use cheaper model for simple queries, expensive for complex."""
        if self.is_simple_query(query):
            # Use cheaper model (e.g., GPT-3.5)
            return self.cheap_llm.generate(query, contexts)
        else:
            # Use expensive model (e.g., GPT-4)
            return self.expensive_llm.generate(query, contexts)

    def is_simple_query(self, query: str) -> bool:
        """Heuristic to determine query complexity."""
        return len(query.split()) < 10 and "?" in query
```

---

## 13. Common RAG Failure Modes

### 13.1 Debugging Irrelevant Retrieval

```python
def debug_retrieval(query: str, retrieved_docs: List[str], k: int = 5):
    """Diagnose why retrieval failed."""
    print(f"Query: {query}\n")

    # Check if any relevant documents exist
    print("Top retrieved documents:")
    for i, doc in enumerate(retrieved_docs[:k], 1):
        print(f"{i}. {doc[:200]}...")

    # Analyze query-document similarity
    query_emb = encoder.encode([query])[0]
    doc_embs = encoder.encode(retrieved_docs[:k])

    similarities = [
        np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        for doc_emb in doc_embs
    ]

    print("\nSimilarity scores:")
    for i, sim in enumerate(similarities, 1):
        print(f"{i}. {sim:.3f}")

    # Suggestions
    if max(similarities) < 0.3:
        print("\nSuggestion: Low similarity scores. Consider query expansion or hybrid search.")
```

### 13.2 Detecting Hallucination

```python
def detect_hallucination(answer: str, contexts: List[str]) -> dict:
    """Check if answer is grounded in retrieved contexts."""
    prompt = f"""Evaluate if the answer is fully supported by the provided contexts. Answer with a score from 0 to 1, where 1 means fully supported and 0 means completely unsupported.

Contexts:
{chr(10).join(contexts)}

Answer:
{answer}

Support score (0-1):"""

    score_str = llm.generate(prompt)
    support_score = float(score_str.strip())

    result = {
        "support_score": support_score,
        "is_hallucination": support_score < 0.5,
        "recommendation": "Accept" if support_score >= 0.7 else "Review" if support_score >= 0.5 else "Reject"
    }

    return result
```

### 13.3 Handling Missing Information

```python
def handle_missing_info(query: str, contexts: List[str]) -> str:
    """Gracefully handle cases where information is not in knowledge base."""
    # Check context relevance
    relevance_scores = [evaluate_relevance(ctx, query) for ctx in contexts]

    if max(relevance_scores) < 0.3:
        return "I don't have enough information in my knowledge base to answer this question accurately. Please try rephrasing or asking a different question."

    # Generate answer with confidence
    prompt = f"""Answer the question using the context. If the context doesn't contain enough information, say so explicitly.

Context:
{chr(10).join(contexts)}

Question: {query}

Answer:"""

    return llm.generate(prompt)
```

### 13.4 Resolving Conflicting Information

```python
def resolve_conflicts(query: str, contexts: List[str]) -> str:
    """Handle conflicting information across documents."""
    prompt = f"""The following contexts contain potentially conflicting information. Analyze the conflicts and provide a balanced answer that acknowledges different perspectives or sources.

Contexts:
{chr(10).join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])}

Question: {query}

Answer (acknowledge conflicts):"""

    return llm.generate(prompt)
```

---

## 14. Resources and References

### 14.1 Key Libraries

```python
# Core RAG libraries
# pip install langchain llama-index haystack ragatouille

# Embedding models
# pip install sentence-transformers openai cohere

# Vector databases
# pip install faiss-cpu chromadb qdrant-client pinecone-client weaviate-client

# Evaluation
# pip install ragas
```

### 14.2 Vector Database Comparison

| Database | Open Source | Managed | Filtering | Hybrid Search | Best For |
|----------|-------------|---------|-----------|---------------|----------|
| FAISS | Yes | No | Limited | No | Research, prototyping |
| ChromaDB | Yes | Yes | Yes | No | Development, small-scale |
| Qdrant | Yes | Yes | Yes | Yes | Production, complex queries |
| Pinecone | No | Yes | Yes | Yes | Production, managed service |
| Weaviate | Yes | Yes | Yes | Yes | Production, GraphQL API |
| Milvus | Yes | Yes | Yes | Yes | Large-scale production |

### 14.3 Key Papers

**Retrieval Models:**
- **DPR** (Dense Passage Retrieval): Karpukhin et al., 2020
- **ColBERT**: Khattab & Zaharia, 2020
- **ANCE** (Approximate Nearest Neighbor Negative Contrastive Learning): Xiong et al., 2020
- **REALM** (Retrieval-Augmented Language Model Pre-Training): Guu et al., 2020

**RAG Advances:**
- **Self-RAG**: Asai et al., 2023
- **CRAG** (Corrective RAG): Yan et al., 2024
- **HyDE** (Hypothetical Document Embeddings): Gao et al., 2022
- **FLARE** (Forward-Looking Active Retrieval): Jiang et al., 2023

**Evaluation:**
- **RAGAS**: Explainable RAG Evaluation Framework
- **ARES**: Automated RAG Evaluation System

### 14.4 Benchmarks

**Retrieval Benchmarks:**
- **BEIR** (Benchmarking IR): 18 diverse datasets
- **MTEB** (Massive Text Embedding Benchmark): 56 embedding tasks
- **MS MARCO**: Large-scale passage ranking

**Code Examples:**
```python
# Evaluate on BEIR
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

dataset = "scifact"
data_path = util.download_and_unzip(dataset, "datasets")
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

# Your retrieval model here
results = your_model.search(corpus, queries, top_k=100)

# Evaluate
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values=[1,3,5,10,100])
print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
```

### 14.5 Additional Resources

**Documentation:**
- LangChain: https://python.langchain.com/docs/use_cases/question_answering/
- LlamaIndex: https://docs.llamaindex.ai/
- Haystack: https://docs.haystack.deepset.ai/
- sentence-transformers: https://www.sbert.net/

**Courses:**
- DeepLearning.AI: Building RAG Applications
- LangChain Academy: RAG from Scratch

**Community:**
- r/LocalLLaMA (Reddit)
- LangChain Discord
- Weights & Biases RAG Reports
