# 21. NLP Fundamentals

## Overview

Natural Language Processing (NLP) is a field at the intersection of computer science, artificial intelligence, and linguistics concerned with the interactions between computers and human language. In 2025, NLP has evolved from rule-based systems to sophisticated neural approaches powered by transformers and large language models.

---

## 21.1 Core NLP Tasks

### Text Classification
Assigning predefined categories to text documents.

**Applications:**
- Sentiment analysis (positive, negative, neutral)
- Spam detection
- Topic categorization
- Intent classification for chatbots

**Common Approaches:**
- Traditional: Bag-of-words + Naive Bayes, Logistic Regression
- Modern: Fine-tuned BERT, RoBERTa, or domain-specific transformers

### Named Entity Recognition (NER)
Identifying and classifying named entities (person, organization, location, date, etc.) in text.

**Example:**
```
Input: "Apple Inc. was founded by Steve Jobs in Cupertino."
Output: [Apple Inc.: ORGANIZATION], [Steve Jobs: PERSON], [Cupertino: LOCATION]
```

**Modern Approaches:**
- BiLSTM-CRF (traditional but effective)
- Transformer-based: spaCy with transformer models, Hugging Face NER pipelines
- Few-shot learning with GPT models

### Part-of-Speech (POS) Tagging
Labeling each word with its grammatical role (noun, verb, adjective, etc.).

**Use Cases:**
- Syntactic parsing
- Information extraction
- Grammar checking

### Machine Translation
Translating text from one language to another.

**Evolution:**
- Rule-based → Statistical MT (SMT) → Neural MT (NMT)
- 2025 State: Transformer-based models (Google Translate, DeepL)
- Zero-shot translation with multilingual models

### Question Answering
Automatically answering questions posed in natural language.

**Types:**
- Extractive QA: Extract answer from given context (SQuAD dataset)
- Open-domain QA: Search and retrieve from large corpora
- Generative QA: Generate answers (ChatGPT, Claude)

---

## 21.2 Text Preprocessing

### Tokenization
Breaking text into individual units (tokens).

**Levels:**
- Word tokenization: Split by whitespace and punctuation
- Subword tokenization: BPE, WordPiece, SentencePiece (used in modern transformers)
- Character tokenization: Useful for morphologically rich languages

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("Natural Language Processing is amazing!")
# Output: ['natural', 'language', 'processing', 'is', 'amazing', '!']
```

### Normalization
Standardizing text to reduce variability.

**Techniques:**
- Lowercasing: "Hello" → "hello"
- Removing punctuation and special characters
- Expanding contractions: "don't" → "do not"
- Removing accents: "café" → "cafe"

### Stemming and Lemmatization

**Stemming:** Crude heuristic to chop off word endings
```
running → run
better → better (not perfect)
```

**Lemmatization:** Using vocabulary and morphological analysis
```
running → run
better → good
am, are, is → be
```

### Stop Word Removal
Removing common words (the, is, at, which) that carry little semantic meaning.

**Note:** Modern transformer models typically don't require stop word removal as they learn contextual importance.

---

## 21.3 Traditional NLP Representations

### Bag-of-Words (BoW)
Represents text as a collection of word counts, ignoring grammar and word order.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love machine learning",
    "Machine learning is great",
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
```

**Limitations:**
- Ignores word order and context
- High dimensionality with large vocabularies
- No semantic understanding

### TF-IDF (Term Frequency-Inverse Document Frequency)
Weights words by their importance in a document relative to a corpus.

```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)

TF = (Count of word in document) / (Total words in document)
IDF = log(Total documents / Documents containing word)
```

**Advantages over BoW:**
- Down-weights common words
- Up-weights distinctive words
- Better for information retrieval

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
```

### N-grams
Contiguous sequences of n words to capture some word order.

**Examples:**
- Unigrams (1-gram): ["I", "love", "NLP"]
- Bigrams (2-gram): ["I love", "love NLP"]
- Trigrams (3-gram): ["I love NLP"]

```python
from sklearn.feature_extraction.text import CountVectorizer

# Bigrams
vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(corpus)
```

---

## 21.4 Language Models

### Statistical Language Models

**N-gram Language Models:**
Estimate probability of a word given previous n-1 words.

```
P(word_i | word_{i-1}, word_{i-2}, ..., word_{i-n+1})
```

**Limitations:**
- Data sparsity for higher-order n-grams
- No long-range dependencies
- Curse of dimensionality

### Neural Language Models
Use neural networks to learn word representations and predict next words.

**Evolution:**
1. **RNN-based LMs** (2010s): LSTM, GRU for sequential modeling
2. **Transformer-based LMs** (2017+): BERT, GPT, T5
3. **Large Language Models** (2020+): GPT-3/4, PaLM, Claude, LLaMA

---

## 21.5 Modern NLP Pipeline (2025)

### Typical Workflow

```python
# 1. Load pretrained model
from transformers import pipeline

# 2. Choose task
classifier = pipeline("sentiment-analysis")

# 3. Inference
result = classifier("I love this product!")
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Key Libraries (2025)

**Hugging Face Transformers:**
- De facto standard for transformer models
- 200k+ pretrained models
- Easy fine-tuning and inference

**spaCy:**
- Industrial-strength NLP
- Fast and efficient
- Now includes transformer models

**NLTK:**
- Classic toolkit for teaching and research
- Comprehensive linguistic resources

**Gensim:**
- Topic modeling and document similarity
- Word2Vec, Doc2Vec implementations

---

## 21.6 Evaluation Metrics

### For Classification Tasks

**Accuracy:** (Correct predictions) / (Total predictions)

**Precision, Recall, F1-Score:**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### For Sequence Tasks (NER, POS tagging)

**Token-level accuracy:** Percentage of correctly labeled tokens

**Span-level F1:** Evaluates exact entity boundary matches

### For Generation Tasks (Translation, Summarization)

**BLEU (Bilingual Evaluation Understudy):**
- Measures n-gram overlap with reference translations
- Range: 0-1 (or 0-100)

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
- Measures recall of n-grams
- Common for summarization

**METEOR:** Accounts for synonyms and stemming

**BERTScore:** Uses contextual embeddings for semantic similarity

---

## 21.7 Current Trends (2025)

### Prompt Engineering
Crafting inputs to elicit desired outputs from LLMs without fine-tuning.

**Techniques:**
- Zero-shot prompting
- Few-shot prompting (in-context learning)
- Chain-of-thought prompting
- Instruction tuning

### Retrieval-Augmented Generation (RAG)
Combining retrieval systems with LLMs to ground responses in external knowledge.

**Architecture:**
```
Query → Retrieval (vector DB) → Context + Query → LLM → Response
```

### Fine-tuning Strategies

**Full Fine-tuning:** Update all parameters (expensive)

**Parameter-Efficient Fine-Tuning (PEFT):**
- LoRA (Low-Rank Adaptation)
- Prefix tuning
- Adapters
- Only train small subset of parameters

---

## 21.8 Challenges and Limitations

### Current Challenges

**Hallucinations:** LLMs generating plausible but false information

**Context Length:** Limited to a fixed number of tokens (though increasing: 128K+ in 2025)

**Multilingual Performance:** English-centric bias in many models

**Domain Adaptation:** General models may underperform on specialized domains

**Computational Cost:** Large models require significant resources

### Bias and Fairness
Models can perpetuate biases present in training data.

**Mitigation:**
- Diverse training data
- Bias detection and measurement
- Fairness constraints during training
- Human oversight and feedback

---

## 21.9 Practical Tips

### When to Use What

**Simple tasks (spam detection, basic classification):**
- TF-IDF + Logistic Regression (fast, interpretable)
- Small BERT models

**Complex understanding tasks:**
- Large pretrained transformers (BERT, RoBERTa)
- Fine-tuned on domain data

**Generation tasks:**
- GPT-based models
- T5, BART for seq2seq
- Modern LLMs with API access

**Low-resource scenarios:**
- Few-shot learning with LLMs
- Data augmentation
- Cross-lingual transfer

---

## Resources

### Essential Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Language Models are Few-Shot Learners" (Brown et al., 2020)

### Datasets
- GLUE benchmark (general language understanding)
- SQuAD (question answering)
- CoNLL (NER, POS tagging)
- WMT (machine translation)

### Tools
- Hugging Face Hub: https://huggingface.co/
- spaCy: https://spacy.io/
- NLTK: https://www.nltk.org/
