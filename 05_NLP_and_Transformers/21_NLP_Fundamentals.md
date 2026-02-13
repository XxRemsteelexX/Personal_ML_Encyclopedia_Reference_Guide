# 21. NLP Fundamentals

## Table of Contents
- [1. Introduction to NLP](#1-introduction-to-nlp)
- [2. Text Preprocessing](#2-text-preprocessing)
- [3. Text Representation](#3-text-representation)
- [4. Regular Expressions for NLP](#4-regular-expressions-for-nlp)
- [5. Named Entity Recognition](#5-named-entity-recognition)
- [6. Part-of-Speech Tagging](#6-part-of-speech-tagging)
- [7. Sentiment Analysis](#7-sentiment-analysis)
- [8. Text Classification](#8-text-classification)
- [9. Topic Modeling](#9-topic-modeling)
- [10. Text Similarity](#10-text-similarity)
- [11. Practical NLP Pipeline](#11-practical-nlp-pipeline)
- [12. Resources and References](#12-resources-and-references)

---

## 1. Introduction to NLP

### 1.1 The NLP Landscape

**Natural Language Processing (NLP)** is the field concerned with enabling computers to understand, interpret, and generate human language. As of 2025, NLP has evolved from rule-based systems to sophisticated neural architectures dominated by transformer-based models and large language models (LLMs).

**Evolution Timeline:**
- **1950s-1990s**: Rule-based systems, symbolic NLP, regular expressions
- **2000s**: Statistical NLP, n-gram models, Naive Bayes, SVM
- **2010s**: Neural NLP, word embeddings (Word2Vec, GloVe), RNNs, LSTMs
- **2017+**: Transformer revolution (BERT, GPT), attention mechanisms
- **2020+**: Large Language Models, few-shot learning, prompt engineering
- **2025**: Multimodal models, retrieval-augmented generation, efficient fine-tuning

### 1.2 Classical vs Deep Learning NLP

**Classical NLP Approaches:**
- Feature engineering required (TF-IDF, BoW, n-grams)
- Fast training, interpretable results
- Works well on small datasets
- Limited ability to capture semantics
- Domain knowledge critical

**Deep Learning NLP:**
- Automatic feature learning from raw text
- Captures semantic relationships and context
- Requires large datasets and computational resources
- Transfer learning enables few-shot capabilities
- State-of-the-art performance on most tasks

**When to Use Classical:**
- Small datasets (< 10,000 examples)
- Real-time constraints (microsecond latency)
- Interpretability requirements
- Limited computational resources
- Simple pattern matching tasks

**When to Use Deep Learning:**
- Large datasets available
- Complex semantic understanding needed
- Transfer learning applicable
- Latency tolerance (milliseconds acceptable)
- State-of-the-art performance required

### 1.3 Text as Data

**Key Challenges:**
- **Ambiguity**: Words have multiple meanings (bank, bat)
- **Context-dependence**: Meaning changes with context
- **Variability**: Many ways to express same idea
- **Sparsity**: Long-tail distribution of words
- **Structure**: Syntax, grammar, discourse structure
- **Noise**: Typos, slang, code-switching

**Text Data Characteristics:**
- High dimensionality (vocabulary size)
- Sequential nature (word order matters)
- Hierarchical structure (words --> sentences --> documents)
- Discrete representations (symbols, not continuous)

```python
# Example: Text complexity
import numpy as np
from collections import Counter

def analyze_text_properties(text):
    """Analyze key properties of text data."""
    words = text.lower().split()

    # Vocabulary size vs total words (sparsity)
    vocab_size = len(set(words))
    total_words = len(words)
    vocab_ratio = vocab_size / total_words

    # Word frequency distribution (Zipf's law)
    word_freq = Counter(words)
    most_common = word_freq.most_common(10)

    # Average word length
    avg_word_len = np.mean([len(w) for w in words])

    return {
        'vocab_size': vocab_size,
        'total_words': total_words,
        'vocab_ratio': vocab_ratio,
        'most_common': most_common,
        'avg_word_length': avg_word_len
    }

# Example usage
text = """Natural language processing enables computers to understand
human language. NLP is a critical technology for modern AI systems."""

stats = analyze_text_properties(text)
print(f"Vocabulary: {stats['vocab_size']} unique words")
print(f"Vocab/Total ratio: {stats['vocab_ratio']:.2f}")
print(f"Most common: {stats['most_common'][:3]}")
```

---

## 2. Text Preprocessing

### 2.1 Tokenization

**Tokenization** splits text into individual units (tokens). The choice of tokenization strategy profoundly impacts model performance and vocabulary size.

#### Word Tokenization

**Simple Whitespace Splitting:**
```python
def simple_tokenize(text):
    """Basic whitespace tokenization."""
    return text.split()

# Limitation: doesn't handle punctuation
text = "Hello, world! How are you?"
tokens = simple_tokenize(text)
# ['Hello,', 'world!', 'How', 'are', 'you?']
```

**NLTK Word Tokenizer:**
```python
import nltk
from nltk.tokenize import word_tokenize

# Download required data (run once)
# nltk.download('punkt')

text = "Dr. Smith earned $150,000 in 2024. That's impressive!"
tokens = word_tokenize(text)
# ['Dr.', 'Smith', 'earned', '$', '150,000', 'in', '2024', '.',
#  'That', "'s", 'impressive', '!']

# Preserves punctuation, handles contractions
```

**SpaCy Tokenization:**
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple's iPhone 15 costs $999.")

tokens = [token.text for token in doc]
# ['Apple', "'s", 'iPhone', '15', 'costs', '$', '999', '.']

# Additional token attributes
for token in doc:
    print(f"{token.text}: lemma={token.lemma_}, pos={token.pos_}")
```

#### Subword Tokenization

**Critical for modern transformers:** Handles out-of-vocabulary words, reduces vocabulary size, captures morphology.

**Byte Pair Encoding (BPE):**
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Train custom BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=1000,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

# Train on corpus
corpus = [
    "the quick brown fox",
    "the lazy dog",
    # ... more text
]
tokenizer.train_from_iterator(corpus, trainer)

# Tokenize
output = tokenizer.encode("the quick fox")
print(output.tokens)
# Example: ['the', 'quick', 'fox']

# Save for reuse
tokenizer.save("bpe-tokenizer.json")
```

**WordPiece (BERT-style):**
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "unbelievable achievements"
tokens = tokenizer.tokenize(text)
# ['un', '##bel', '##ie', '##va', '##ble', 'achievements']

# Full encoding with special tokens
encoding = tokenizer(
    text,
    add_special_tokens=True,  # [CLS] and [SEP]
    return_attention_mask=True,
    return_tensors='pt'
)

print(f"Input IDs: {encoding['input_ids']}")
print(f"Attention mask: {encoding['attention_mask']}")
```

**SentencePiece (language-agnostic):**
```python
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='sp',
    vocab_size=8000,
    character_coverage=0.9995,
    model_type='unigram',  # or 'bpe'
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load('sp.model')

text = "This is a test sentence."
tokens = sp.encode_as_pieces(text)
ids = sp.encode_as_ids(text)

print(f"Tokens: {tokens}")
print(f"IDs: {ids}")

# Decode
decoded = sp.decode_ids(ids)
```

**Comparison of Tokenization Methods:**

| Method | Vocab Size | OOV Handling | Speed | Use Case |
|--------|-----------|--------------|-------|----------|
| Word | 50k-100k+ | Poor | Fast | Traditional ML |
| BPE | 10k-50k | Good | Medium | GPT models |
| WordPiece | 30k | Good | Medium | BERT models |
| SentencePiece | 8k-32k | Excellent | Fast | Multilingual |

### 2.2 Text Normalization

#### Lowercasing

```python
def normalize_case(text, method='lower'):
    """Normalize text case.

    Args:
        text: Input text
        method: 'lower', 'upper', or 'title'
    """
    if method == 'lower':
        return text.lower()
    elif method == 'upper':
        return text.upper()
    elif method == 'title':
        return text.title()
    else:
        return text

# When to lowercase:
# - Sentiment analysis (case rarely matters)
# - Topic modeling
# - Search/retrieval

# When NOT to lowercase:
# - Named entity recognition (John vs john)
# - POS tagging
# - Question answering (preserves proper nouns)
```

#### Removing Punctuation

```python
import string
import re

def remove_punctuation(text, keep_apostrophes=True):
    """Remove punctuation with options.

    Args:
        text: Input text
        keep_apostrophes: Preserve contractions like "don't"
    """
    if keep_apostrophes:
        # Remove all except apostrophes
        pattern = f"[{re.escape(string.punctuation.replace(\"'\", ''))}]"
        return re.sub(pattern, '', text)
    else:
        # Remove all punctuation
        return text.translate(str.maketrans('', '', string.punctuation))

# Example
text = "Hello, world! It's a great day... isn't it?"
cleaned = remove_punctuation(text)
# "Hello world! It's a great day isn't it"
```

#### Expanding Contractions

```python
# Contraction mapping
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}

def expand_contractions(text, contraction_map=CONTRACTIONS):
    """Expand contractions in text."""
    pattern = re.compile(r'\b(' + '|'.join(contraction_map.keys()) + r')\b')

    def replace(match):
        return contraction_map[match.group(0)]

    return pattern.sub(replace, text.lower())

# Example
text = "I'm sure you'll love NLP. It's amazing!"
expanded = expand_contractions(text)
# "i am sure you will love nlp. it is amazing!"
```

### 2.3 Stopword Removal

**Stopwords** are common words (the, is, at, which) that carry minimal semantic meaning.

```python
from nltk.corpus import stopwords
import nltk

# Download stopwords (run once)
# nltk.download('stopwords')

# Get English stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens, custom_stopwords=None):
    """Remove stopwords from token list.

    Args:
        tokens: List of tokens
        custom_stopwords: Additional stopwords to remove
    """
    stop_set = stop_words.copy()
    if custom_stopwords:
        stop_set.update(custom_stopwords)

    return [token for token in tokens if token.lower() not in stop_set]

# Example
tokens = ['this', 'is', 'a', 'sample', 'sentence', 'with', 'stopwords']
filtered = remove_stopwords(tokens)
# ['sample', 'sentence', 'stopwords']

# Custom stopwords for domain-specific tasks
custom = ['said', 'like', 'just']
filtered_custom = remove_stopwords(tokens, custom_stopwords=custom)
```

**When to remove stopwords:**
- Document classification with BoW/TF-IDF
- Topic modeling
- Search/information retrieval

**When NOT to remove:**
- Sentiment analysis ("not good" vs "good")
- Named entity recognition
- Modern transformer models (they learn importance)

### 2.4 Stemming vs Lemmatization

#### Stemming

**Crude rule-based suffix removal.** Fast but imprecise.

```python
from nltk.stem import PorterStemmer, SnowballStemmer

porter = PorterStemmer()
snowball = SnowballStemmer('english')

words = ['running', 'runs', 'ran', 'runner', 'easily', 'fairly']

print("Porter Stemmer:")
for word in words:
    print(f"{word} --> {porter.stem(word)}")
# running --> run
# runs --> run
# ran --> ran
# runner --> runner
# easily --> easili
# fairly --> fairli

print("\nSnowball Stemmer:")
for word in words:
    print(f"{word} --> {snowball.stem(word)}")
# Similar but slightly different rules
```

#### Lemmatization

**Dictionary-based reduction to base form (lemma).** Slower but accurate.

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

# Download required data (run once)
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to WordNet POS tag."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_with_pos(word):
    """Lemmatize with POS tagging for accuracy."""
    pos = get_wordnet_pos(word)
    return lemmatizer.lemmatize(word.lower(), pos)

# Examples
words = ['running', 'runs', 'ran', 'better', 'geese', 'was', 'are']

print("Without POS:")
for word in words:
    print(f"{word} --> {lemmatizer.lemmatize(word)}")

print("\nWith POS:")
for word in words:
    print(f"{word} --> {lemmatize_with_pos(word)}")
# running --> run
# runs --> run
# ran --> run
# better --> good
# geese --> goose
# was --> be
# are --> be
```

**Stemming vs Lemmatization Comparison:**

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| Speed | Fast | Slower |
| Accuracy | Lower | Higher |
| Output | May not be real word | Always real word |
| POS needed | No | Yes (for best results) |
| Use case | Search, clustering | NER, QA, generation |

### 2.5 Complete Preprocessing Pipeline

```python
import re
import string
import spacy
from typing import List, Dict

class TextPreprocessor:
    """Complete text preprocessing pipeline."""

    def __init__(self,
                 lowercase=True,
                 remove_punct=True,
                 remove_stopwords=False,
                 lemmatize=True,
                 remove_numbers=False,
                 min_token_length=2):
        """Initialize preprocessor with configuration.

        Args:
            lowercase: Convert to lowercase
            remove_punct: Remove punctuation
            remove_stopwords: Remove stopwords
            lemmatize: Lemmatize tokens
            remove_numbers: Remove numeric tokens
            min_token_length: Minimum token length to keep
        """
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_numbers = remove_numbers
        self.min_token_length = min_token_length

        # Load spaCy model
        self.nlp = spacy.load('en_core_web_sm')

        # Stopwords
        if remove_stopwords:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess(self, text: str) -> List[str]:
        """Full preprocessing pipeline.

        Args:
            text: Input text

        Returns:
            List of processed tokens
        """
        # Clean
        text = self.clean_text(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Process with spaCy
        doc = self.nlp(text)

        tokens = []
        for token in doc:
            # Skip punctuation
            if self.remove_punct and token.is_punct:
                continue

            # Skip stopwords
            if self.remove_stopwords and token.text in self.stop_words:
                continue

            # Skip numbers
            if self.remove_numbers and token.like_num:
                continue

            # Get token text or lemma
            if self.lemmatize:
                token_text = token.lemma_
            else:
                token_text = token.text

            # Filter by length
            if len(token_text) >= self.min_token_length:
                tokens.append(token_text)

        return tokens

    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """Preprocess multiple texts efficiently."""
        # Use spaCy's pipe for batch processing
        cleaned = [self.clean_text(text) for text in texts]

        if self.lowercase:
            cleaned = [text.lower() for text in cleaned]

        results = []
        for doc in self.nlp.pipe(cleaned, batch_size=50):
            tokens = []
            for token in doc:
                if self.remove_punct and token.is_punct:
                    continue
                if self.remove_stopwords and token.text in self.stop_words:
                    continue
                if self.remove_numbers and token.like_num:
                    continue

                token_text = token.lemma_ if self.lemmatize else token.text

                if len(token_text) >= self.min_token_length:
                    tokens.append(token_text)

            results.append(tokens)

        return results

# Usage examples
preprocessor = TextPreprocessor(
    lowercase=True,
    remove_punct=True,
    remove_stopwords=True,
    lemmatize=True
)

text = "The quick brown foxes are jumping over the lazy dogs! Visit http://example.com"
tokens = preprocessor.preprocess(text)
print(f"Tokens: {tokens}")
# ['quick', 'brown', 'fox', 'jump', 'lazy', 'dog']

# Batch processing
texts = [
    "I love machine learning!",
    "NLP is fascinating and powerful.",
    "Python makes NLP easy."
]
batch_results = preprocessor.preprocess_batch(texts)
for i, tokens in enumerate(batch_results):
    print(f"Text {i+1}: {tokens}")
```

---

## 3. Text Representation

### 3.1 Bag of Words (BoW)

**Bag of Words** represents text as an unordered collection of words, counting their frequencies while ignoring grammar and word order.

```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

# Create corpus
corpus = [
    "I love machine learning",
    "Machine learning is great",
    "I love deep learning",
    "Deep learning requires GPUs"
]

# Basic BoW
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Convert to dense array for inspection
X_dense = X.toarray()

# Show vocabulary
vocab = vectorizer.get_feature_names_out()
print(f"Vocabulary: {vocab}")
# ['deep' 'gpus' 'great' 'is' 'learning' 'love' 'machine' 'requires']

# Create DataFrame for visualization
df_bow = pd.DataFrame(X_dense, columns=vocab)
print("\nBag of Words Matrix:")
print(df_bow)
#    deep  gpus  great  is  learning  love  machine  requires
# 0     0     0      0   0         1     1        1         0
# 1     0     0      1   1         1     0        1         0
# 2     1     0      0   0         1     1        0         0
# 3     1     1      0   0         1     0        0         1

# Advanced configuration
vectorizer_advanced = CountVectorizer(
    max_features=1000,        # Limit vocabulary size
    min_df=2,                 # Ignore words appearing in < 2 docs
    max_df=0.8,              # Ignore words appearing in > 80% docs
    ngram_range=(1, 2),      # Unigrams and bigrams
    token_pattern=r'\b\w+\b', # Custom tokenization pattern
    lowercase=True,
    stop_words='english'
)

# Binary occurrence (instead of counts)
vectorizer_binary = CountVectorizer(binary=True)
X_binary = vectorizer_binary.fit_transform(corpus)
```

**Limitations:**
- Loses word order ("not good" = "good not")
- No semantic understanding
- High dimensionality (vocabulary size)
- Sparse representation
- No context awareness

### 3.2 TF-IDF (Term Frequency-Inverse Document Frequency)

**TF-IDF** weights words by importance, down-weighting common words and up-weighting distinctive terms.

**Formula:**
```
TF-IDF(t, d) = TF(t, d) * IDF(t)

TF(t, d) = (Count of term t in document d) / (Total terms in d)
IDF(t) = log((Total documents) / (Documents containing t))
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are enemies",
    "the mat is on the floor"
]

# Basic TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)

# Get feature names and scores
feature_names = tfidf.get_feature_names_out()
doc_0_tfidf = X_tfidf[0].toarray()[0]

# Show top terms for document 0
top_indices = doc_0_tfidf.argsort()[-5:][::-1]
print("Top 5 terms in document 0:")
for idx in top_indices:
    print(f"{feature_names[idx]}: {doc_0_tfidf[idx]:.3f}")

# Advanced TF-IDF configuration
tfidf_advanced = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 3),      # Unigrams, bigrams, trigrams
    sublinear_tf=True,       # Use 1 + log(tf) instead of tf
    use_idf=True,
    smooth_idf=True,         # Add 1 to document frequencies
    norm='l2',               # L2 normalization
    lowercase=True,
    stop_words='english'
)

X_tfidf_adv = tfidf_advanced.fit_transform(corpus)

# Custom analyzer for more control
def custom_analyzer(text):
    """Custom text analyzer."""
    # Custom preprocessing
    tokens = text.lower().split()
    # Remove short tokens
    tokens = [t for t in tokens if len(t) > 2]
    return tokens

tfidf_custom = TfidfVectorizer(analyzer=custom_analyzer)
X_custom = tfidf_custom.fit_transform(corpus)

# Save and load vectorizer
import pickle

# Save
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Load
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_loaded = pickle.load(f)

# Transform new documents
new_docs = ["the cat is on the floor"]
X_new = tfidf_loaded.transform(new_docs)
```

**TF-IDF Use Cases:**
- Document similarity/search
- Information retrieval
- Feature extraction for classification
- Keyword extraction

**Hyperparameter Tuning:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Parameter grid
param_grid = {
    'tfidf__max_features': [500, 1000, 2000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__min_df': [1, 2, 5],
    'tfidf__max_df': [0.7, 0.8, 0.9],
    'tfidf__sublinear_tf': [True, False],
    'clf__alpha': [0.1, 0.5, 1.0]
}

# Grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

# Fit (assuming X_train, y_train available)
# grid_search.fit(X_train, y_train)
# print(f"Best parameters: {grid_search.best_params_}")
```

### 3.3 N-grams

**N-grams** capture sequences of n consecutive words, preserving some local word order.

```python
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

corpus = [
    "the quick brown fox jumps",
    "the lazy brown dog sleeps"
]

# Unigrams (1-gram)
vectorizer_1gram = CountVectorizer(ngram_range=(1, 1))
X_1gram = vectorizer_1gram.fit_transform(corpus)
print(f"Unigrams: {vectorizer_1gram.get_feature_names_out()}")
# ['brown' 'dog' 'fox' 'jumps' 'lazy' 'quick' 'sleeps' 'the']

# Bigrams (2-gram)
vectorizer_2gram = CountVectorizer(ngram_range=(2, 2))
X_2gram = vectorizer_2gram.fit_transform(corpus)
print(f"\nBigrams: {vectorizer_2gram.get_feature_names_out()}")
# ['brown dog' 'brown fox' 'dog sleeps' 'fox jumps' 'lazy brown'
#  'quick brown' 'the lazy' 'the quick']

# Trigrams (3-gram)
vectorizer_3gram = CountVectorizer(ngram_range=(3, 3))
X_3gram = vectorizer_3gram.fit_transform(corpus)
print(f"\nTrigrams: {vectorizer_3gram.get_feature_names_out()}")

# Combined: unigrams + bigrams
vectorizer_combined = CountVectorizer(ngram_range=(1, 2))
X_combined = vectorizer_combined.fit_transform(corpus)
print(f"\nUnigrams + Bigrams: {vectorizer_combined.get_feature_names_out()}")

# Character n-grams (useful for morphology, typos)
vectorizer_char = CountVectorizer(
    analyzer='char',
    ngram_range=(2, 4)
)
X_char = vectorizer_char.fit_transform(corpus)

# Manual n-gram extraction
def extract_ngrams(text, n):
    """Extract n-grams from text.

    Args:
        text: Input text
        n: N-gram size

    Returns:
        List of n-gram tuples
    """
    tokens = text.split()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

text = "the quick brown fox jumps"
bigrams = extract_ngrams(text, 2)
print(f"\nManual bigrams: {bigrams}")
# ['the quick', 'quick brown', 'brown fox', 'fox jumps']

trigrams = extract_ngrams(text, 3)
print(f"Manual trigrams: {trigrams}")
# ['the quick brown', 'quick brown fox', 'brown fox jumps']

# N-gram frequency analysis
from nltk import ngrams
from collections import Counter

def ngram_frequency(text, n, top_k=10):
    """Compute most frequent n-grams.

    Args:
        text: Input text
        n: N-gram size
        top_k: Number of top n-grams to return
    """
    tokens = text.lower().split()
    n_grams = list(ngrams(tokens, n))
    freq = Counter(n_grams)
    return freq.most_common(top_k)

large_text = " ".join(corpus * 10)  # Simulate larger corpus
top_bigrams = ngram_frequency(large_text, 2, top_k=5)
print(f"\nTop bigrams: {top_bigrams}")
```

**N-gram Best Practices:**
- **Unigrams (n=1)**: Baseline, loses all context
- **Bigrams (n=2)**: Sweet spot for most tasks
- **Trigrams (n=3)**: Better context, but sparse
- **Higher n**: Very sparse, risk of overfitting
- **Character n-grams**: Robust to typos, good for short text

### 3.4 Hashing Vectorizer

**HashingVectorizer** uses hashing trick for memory-efficient text representation, useful for very large vocabularies or streaming data.

```python
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np

corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends"
]

# Basic hashing vectorizer
hash_vec = HashingVectorizer(
    n_features=2**10,  # 1024 features (power of 2)
    norm='l2',
    alternate_sign=True  # Reduces hash collisions impact
)

X_hash = hash_vec.fit_transform(corpus)
print(f"Shape: {X_hash.shape}")  # (3, 1024)

# Advantages over CountVectorizer:
# 1. Fixed memory footprint (n_features)
# 2. No vocabulary storage needed
# 3. Fast for large corpora
# 4. Supports streaming/online learning

# Disadvantage: Can't retrieve feature names (one-way hash)

# Advanced configuration
hash_vec_advanced = HashingVectorizer(
    n_features=2**18,      # 262144 features for large vocab
    ngram_range=(1, 2),    # Unigrams + bigrams
    norm='l2',
    alternate_sign=True,
    lowercase=True,
    stop_words='english',
    dtype=np.float32       # Memory efficiency
)

# Streaming usage
def document_generator():
    """Simulate streaming documents."""
    for i in range(1000):
        yield f"document {i} with some text content"

# Process streaming data
batch_size = 100
for i, batch_start in enumerate(range(0, 1000, batch_size)):
    batch = list(document_generator())[batch_start:batch_start+batch_size]
    X_batch = hash_vec.transform(batch)
    # Process batch (e.g., partial_fit for online learning)
    print(f"Batch {i+1}: {X_batch.shape}")
```

---

## 4. Regular Expressions for NLP

### 4.1 Common Regex Patterns

**Regular expressions** are essential for pattern matching, extraction, and text cleaning in NLP.

```python
import re

# Basic patterns
text = "Contact me at john@example.com or call 555-123-4567"

# Email extraction
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
emails = re.findall(email_pattern, text)
print(f"Emails: {emails}")  # ['john@example.com']

# Phone number extraction (US format)
phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
phones = re.findall(phone_pattern, text)
print(f"Phones: {phones}")  # ['555-123-4567']

# URL extraction
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
text_with_urls = "Visit https://www.example.com or http://test.org"
urls = re.findall(url_pattern, text_with_urls)
print(f"URLs: {urls}")

# Hashtag extraction
tweet = "Loving #NLP and #MachineLearning! #AI is amazing"
hashtag_pattern = r'#\w+'
hashtags = re.findall(hashtag_pattern, tweet)
print(f"Hashtags: {hashtags}")  # ['#NLP', '#MachineLearning', '#AI']

# Mention extraction (@username)
mention_pattern = r'@\w+'
tweet_with_mentions = "Thanks @johndoe and @janedoe for the help!"
mentions = re.findall(mention_pattern, tweet_with_mentions)
print(f"Mentions: {mentions}")  # ['@johndoe', '@janedoe']

# Number extraction
number_pattern = r'\b\d+(?:\.\d+)?\b'
text_numbers = "The price is $19.99 and quantity is 5"
numbers = re.findall(number_pattern, text_numbers)
print(f"Numbers: {numbers}")  # ['19.99', '5']

# Date extraction (MM/DD/YYYY or MM-DD-YYYY)
date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
text_dates = "Events on 12/25/2024 and 01-15-2025"
dates = re.findall(date_pattern, text_dates)
print(f"Dates: {dates}")  # ['12/25/2024', '01-15-2025']
```

### 4.2 Text Cleaning Patterns

```python
import re

class RegexCleaner:
    """Collection of regex-based text cleaning methods."""

    @staticmethod
    def remove_urls(text):
        """Remove URLs from text."""
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(pattern, '', text)

    @staticmethod
    def remove_emails(text):
        """Remove email addresses."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(pattern, '', text)

    @staticmethod
    def remove_phone_numbers(text):
        """Remove phone numbers."""
        pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        return re.sub(pattern, '', text)

    @staticmethod
    def remove_html_tags(text):
        """Remove HTML tags."""
        pattern = r'<[^>]+>'
        return re.sub(pattern, '', text)

    @staticmethod
    def remove_special_chars(text, keep_spaces=True):
        """Remove special characters, keep only alphanumeric."""
        if keep_spaces:
            pattern = r'[^a-zA-Z0-9\s]'
        else:
            pattern = r'[^a-zA-Z0-9]'
        return re.sub(pattern, '', text)

    @staticmethod
    def remove_extra_whitespace(text):
        """Remove extra whitespace, tabs, newlines."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing
        return text.strip()

    @staticmethod
    def remove_repeated_chars(text, max_repeat=2):
        """Remove repeated characters (e.g., 'sooooo' -> 'soo')."""
        pattern = r'(.)\1{' + str(max_repeat) + r',}'
        return re.sub(pattern, r'\1' * max_repeat, text)

    @staticmethod
    def normalize_whitespace(text):
        """Normalize different types of whitespace."""
        # Replace tabs, newlines, etc. with single space
        text = re.sub(r'[\t\n\r\f\v]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def extract_sentences(text):
        """Split text into sentences."""
        # Simple sentence boundary detection
        pattern = r'[.!?]+\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def clean_twitter_text(text):
        """Clean Twitter-specific elements."""
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags (keep text)
        text = re.sub(r'#(\w+)', r'\1', text)
        # Remove RT
        text = re.sub(r'\bRT\b', '', text)
        # Remove URLs
        text = RegexCleaner.remove_urls(text)
        # Clean whitespace
        text = RegexCleaner.remove_extra_whitespace(text)
        return text

# Usage examples
cleaner = RegexCleaner()

dirty_text = """
Visit https://example.com or email me@example.com
Call 555-123-4567 for more info!!!
<p>This is HTML</p>
Sooooo excited!!!
"""

clean_text = dirty_text
clean_text = cleaner.remove_urls(clean_text)
clean_text = cleaner.remove_emails(clean_text)
clean_text = cleaner.remove_phone_numbers(clean_text)
clean_text = cleaner.remove_html_tags(clean_text)
clean_text = cleaner.remove_repeated_chars(clean_text)
clean_text = cleaner.remove_extra_whitespace(clean_text)

print(f"Cleaned: {clean_text}")
# Cleaned: Visit or email me Call for more info!! Soo excited!!

# Twitter cleaning
tweet = "RT @johndoe: Loving #NLP and #AI! Check out https://example.com"
clean_tweet = cleaner.clean_twitter_text(tweet)
print(f"Clean tweet: {clean_tweet}")
# Clean tweet: Loving NLP and AI! Check out
```

### 4.3 Advanced Regex for NLP

```python
import re

# Context-aware extraction
def extract_with_context(text, pattern, context_words=5):
    """Extract pattern matches with surrounding context.

    Args:
        text: Input text
        pattern: Regex pattern
        context_words: Words of context on each side

    Returns:
        List of (match, context) tuples
    """
    results = []
    for match in re.finditer(pattern, text):
        start = match.start()
        end = match.end()

        # Get context
        words_before = text[:start].split()[-context_words:]
        words_after = text[end:].split()[:context_words]

        context = ' '.join(words_before + [match.group()] + words_after)
        results.append((match.group(), context))

    return results

# Example
text = "Apple Inc. announced record profits. Microsoft also performed well."
pattern = r'\b[A-Z][a-z]+ Inc\.'
results = extract_with_context(text, pattern, context_words=3)
for match, context in results:
    print(f"Match: {match}")
    print(f"Context: {context}\n")

# Named groups for structured extraction
def extract_structured_dates(text):
    """Extract dates with named groups."""
    pattern = r'(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})'

    dates = []
    for match in re.finditer(pattern, text):
        date_dict = match.groupdict()
        dates.append(date_dict)

    return dates

text = "Events on 12/25/2024 and 03/15/2025"
dates = extract_structured_dates(text)
print(f"Structured dates: {dates}")
# [{'month': '12', 'day': '25', 'year': '2024'},
#  {'month': '03', 'day': '15', 'year': '2025'}]

# Lookahead and lookbehind
def extract_prices(text):
    """Extract prices with currency symbol."""
    # Positive lookbehind for $
    pattern = r'(?<=\$)\d+(?:\.\d{2})?'
    return re.findall(pattern, text)

text = "The item costs $19.99 and shipping is $5.00"
prices = extract_prices(text)
print(f"Prices: {prices}")  # ['19.99', '5.00']

# Non-capturing groups
def extract_words_after_pattern(text, prefix):
    """Extract words following a pattern."""
    # (?:...) is non-capturing group
    pattern = rf'\b(?:{prefix})\s+(\w+)'
    return re.findall(pattern, text, re.IGNORECASE)

text = "I love Python and I love NLP"
words = extract_words_after_pattern(text, "love")
print(f"Words after 'love': {words}")  # ['Python', 'NLP']
```

---

## 5. Named Entity Recognition

### 5.1 SpaCy NER

**Named Entity Recognition (NER)** identifies and classifies named entities (person, organization, location, date, etc.) in text.

```python
import spacy
from spacy import displacy

# Load model
nlp = spacy.load("en_core_web_sm")

# Process text
text = """
Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.
The company released the iPhone in January 2007 for $499.
"""

doc = nlp(text)

# Extract entities
print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_:15s} {ent.start_char}-{ent.end_char}")

# Entity types in spaCy:
# PERSON      - People, including fictional
# NORP        - Nationalities, religious/political groups
# FAC         - Buildings, airports, highways, bridges
# ORG         - Companies, agencies, institutions
# GPE         - Countries, cities, states
# LOC         - Non-GPE locations, mountain ranges, water bodies
# PRODUCT     - Objects, vehicles, foods, etc.
# EVENT       - Named hurricanes, battles, wars, sports events
# WORK_OF_ART - Titles of books, songs, etc.
# LAW         - Named documents made into laws
# LANGUAGE    - Any named language
# DATE        - Absolute or relative dates/periods
# TIME        - Times smaller than a day
# PERCENT     - Percentage
# MONEY       - Monetary values
# QUANTITY    - Measurements
# ORDINAL     - "first", "second", etc.
# CARDINAL    - Numerals that don't fall under other types

# Visualize entities (in Jupyter)
# displacy.render(doc, style="ent", jupyter=True)

# Save visualization to HTML
html = displacy.render(doc, style="ent", page=True)
with open("entities.html", "w") as f:
    f.write(html)

# Filter by entity type
def get_entities_by_type(doc, entity_type):
    """Extract entities of specific type."""
    return [ent.text for ent in doc.ents if ent.label_ == entity_type]

organizations = get_entities_by_type(doc, "ORG")
persons = get_entities_by_type(doc, "PERSON")
locations = get_entities_by_type(doc, "GPE")

print(f"\nOrganizations: {organizations}")
print(f"Persons: {persons}")
print(f"Locations: {locations}")
```

### 5.2 Custom Entity Training

```python
import spacy
from spacy.training import Example
from spacy.util import minibatch
import random

# Training data format: (text, {"entities": [(start, end, label)]})
TRAIN_DATA = [
    ("Google was founded in 1998", {
        "entities": [(0, 6, "TECH_COMPANY")]
    }),
    ("Apple released the iPhone", {
        "entities": [(0, 5, "TECH_COMPANY"), (20, 26, "PRODUCT")]
    }),
    ("Microsoft develops Windows OS", {
        "entities": [(0, 9, "TECH_COMPANY"), (19, 29, "PRODUCT")]
    }),
    ("Tesla is an electric vehicle manufacturer", {
        "entities": [(0, 5, "TECH_COMPANY")]
    }),
    ("Amazon Web Services is a cloud platform", {
        "entities": [(0, 19, "TECH_COMPANY"), (25, 39, "PRODUCT")]
    })
]

def train_custom_ner(train_data, n_iter=30, model=None):
    """Train custom NER model.

    Args:
        train_data: List of (text, annotations) tuples
        n_iter: Number of training iterations
        model: Existing model to update (None to create new)

    Returns:
        Trained spaCy model
    """
    # Create blank model or load existing
    if model is not None:
        nlp = spacy.load(model)
    else:
        nlp = spacy.blank("en")

    # Add NER pipe if not present
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable other pipes during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):
        # Initialize optimizer
        optimizer = nlp.begin_training()

        # Training loop
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}

            # Batch the examples
            batches = minibatch(train_data, size=8)

            for batch in batches:
                examples = []
                for text, annots in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annots)
                    examples.append(example)

                # Update model
                nlp.update(
                    examples,
                    drop=0.5,
                    losses=losses,
                    sgd=optimizer
                )

            if (itn + 1) % 10 == 0:
                print(f"Iteration {itn+1}, Losses: {losses}")

    return nlp

# Train model
custom_nlp = train_custom_ner(TRAIN_DATA, n_iter=30)

# Save model
custom_nlp.to_disk("custom_ner_model")

# Load and test
nlp_loaded = spacy.load("custom_ner_model")
test_text = "Facebook and Instagram are social media platforms"
doc = nlp_loaded(test_text)

print("\nCustom NER Results:")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
```

### 5.3 Rule-Based Matching

```python
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")

# Pattern-based matching
matcher = Matcher(nlp.vocab)

# Pattern: {POS: "PROPN"} finds proper nouns
# {LOWER: "text"} finds lowercase "text"
# {IS_DIGIT: True} finds numbers

# Example: Match company names followed by Inc/Corp/Ltd
pattern1 = [
    {"POS": "PROPN", "OP": "+"},  # One or more proper nouns
    {"LOWER": {"IN": ["inc", "corp", "ltd", "llc"]}}
]
matcher.add("COMPANY", [pattern1])

# Example: Match email patterns
pattern2 = [
    {"LIKE_EMAIL": True}
]
matcher.add("EMAIL", [pattern2])

# Example: Match money patterns
pattern3 = [
    {"TEXT": "$"},
    {"IS_DIGIT": True}
]
matcher.add("MONEY", [pattern3])

text = "Apple Inc. and Microsoft Corp. are tech giants. Contact: info@example.com. Cost: $99"
doc = nlp(text)

matches = matcher(doc)
print("Pattern Matches:")
for match_id, start, end in matches:
    span = doc[start:end]
    label = nlp.vocab.strings[match_id]
    print(f"{label}: {span.text}")

# Phrase matching (for known entities)
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# List of known tech companies
tech_companies = ["apple", "google", "microsoft", "amazon", "facebook", "tesla"]
patterns = [nlp.make_doc(company) for company in tech_companies]
phrase_matcher.add("TECH_COMPANY", patterns)

text2 = "Google and Amazon are competing with Microsoft"
doc2 = nlp(text2)

matches2 = phrase_matcher(doc2)
print("\nPhrase Matches:")
for match_id, start, end in matches2:
    span = doc2[start:end]
    print(f"TECH_COMPANY: {span.text}")

# Add custom entities to doc
def add_custom_entities(doc, matches, label):
    """Add matched spans as custom entities."""
    new_ents = list(doc.ents)

    for match_id, start, end in matches:
        span = Span(doc, start, end, label=label)
        new_ents.append(span)

    # Remove duplicates and overlaps
    new_ents = spacy.util.filter_spans(new_ents)
    doc.ents = new_ents

    return doc

doc2 = add_custom_entities(doc2, matches2, "TECH_COMPANY")
print("\nEntities after adding custom:")
for ent in doc2.ents:
    print(f"{ent.text}: {ent.label_}")
```

---

## 6. Part-of-Speech Tagging

### 6.1 POS Tagging with SpaCy

**Part-of-Speech (POS) tagging** assigns grammatical categories (noun, verb, adjective, etc.) to each word.

```python
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog"
doc = nlp(text)

# Basic POS tagging
print("POS Tags:")
for token in doc:
    print(f"{token.text:10s} {token.pos_:10s} {token.tag_:10s} {token.dep_:10s}")

# POS Tag Meanings:
# ADJ      - adjective
# ADP      - adposition
# ADV      - adverb
# AUX      - auxiliary
# CCONJ    - coordinating conjunction
# DET      - determiner
# INTJ     - interjection
# NOUN     - noun
# NUM      - numeral
# PART     - particle
# PRON     - pronoun
# PROPN    - proper noun
# PUNCT    - punctuation
# SCONJ    - subordinating conjunction
# SYM      - symbol
# VERB     - verb
# X        - other

# Fine-grained tags (Penn Treebank)
# NN       - noun, singular
# NNS      - noun, plural
# VB       - verb, base form
# VBD      - verb, past tense
# VBG      - verb, gerund/present participle
# VBN      - verb, past participle
# JJ       - adjective
# JJR      - adjective, comparative
# JJS      - adjective, superlative
# RB       - adverb
# ... (and many more)

# Extract specific POS
def get_tokens_by_pos(doc, pos_tag):
    """Extract tokens with specific POS tag."""
    return [token.text for token in doc if token.pos_ == pos_tag]

# Extract all nouns
nouns = get_tokens_by_pos(doc, "NOUN")
print(f"\nNouns: {nouns}")

# Extract all verbs
verbs = get_tokens_by_pos(doc, "VERB")
print(f"Verbs: {verbs}")

# Extract all adjectives
adjectives = get_tokens_by_pos(doc, "ADJ")
print(f"Adjectives: {adjectives}")

# POS tag frequency
def pos_frequency(doc):
    """Count POS tag frequencies."""
    pos_tags = [token.pos_ for token in doc if not token.is_punct]
    return Counter(pos_tags)

text_longer = """
Natural language processing is a fascinating field of artificial intelligence.
It enables computers to understand and generate human language.
Modern NLP systems use deep learning and transformer architectures.
"""
doc_longer = nlp(text_longer)
pos_freq = pos_frequency(doc_longer)
print(f"\nPOS Frequency:")
for pos, count in pos_freq.most_common():
    print(f"{pos}: {count}")
```

### 6.2 Dependency Parsing

**Dependency parsing** identifies grammatical relationships between words.

```python
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

text = "The CEO announced a new product launch"
doc = nlp(text)

# Dependency relations
print("Dependency Parse:")
for token in doc:
    print(f"{token.text:15s} {token.dep_:15s} {token.head.text:15s}")

# Common dependency relations:
# nsubj    - nominal subject
# obj      - object
# iobj     - indirect object
# amod     - adjectival modifier
# advmod   - adverbial modifier
# det      - determiner
# prep     - prepositional modifier
# pobj     - object of preposition
# aux      - auxiliary
# conj     - conjunct
# cc       - coordinating conjunction

# Visualize dependency tree
# displacy.render(doc, style="dep", jupyter=True)

# Save visualization
svg = displacy.render(doc, style="dep")
with open("dependency.svg", "w") as f:
    f.write(svg)

# Extract subject-verb-object triples
def extract_svo_triples(doc):
    """Extract subject-verb-object triples."""
    triples = []

    for token in doc:
        if token.pos_ == "VERB":
            # Find subject
            subject = None
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subject = child
                    break

            # Find object
            obj = None
            for child in token.children:
                if child.dep_ in ["dobj", "obj"]:
                    obj = child
                    break

            if subject and obj:
                triples.append((subject.text, token.text, obj.text))

    return triples

text2 = "The company released a new product. Engineers developed innovative features."
doc2 = nlp(text2)
triples = extract_svo_triples(doc2)
print(f"\nSVO Triples:")
for subj, verb, obj in triples:
    print(f"{subj} --{verb}--> {obj}")

# Extract noun chunks
print(f"\nNoun Chunks:")
for chunk in doc2.noun_chunks:
    print(f"{chunk.text:20s} root: {chunk.root.text:10s} dep: {chunk.root.dep_}")

# Find root of sentence
def get_sentence_root(doc):
    """Find root verb of sentence."""
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None

root = get_sentence_root(doc)
print(f"\nSentence root: {root.text} (POS: {root.pos_})")
```

### 6.3 Constituency Parsing

**Constituency parsing** represents sentence structure as a tree of nested phrases.

```python
import spacy
from benepar.spacy_plugin import BeneparComponent

# Install: pip install benepar
# Download model: python -m spacy download en_core_web_md
# python -c "import benepar; benepar.download('benepar_en3')"

nlp = spacy.load("en_core_web_md")
if "benepar" not in nlp.pipe_names:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

text = "The quick brown fox jumps over the lazy dog"
doc = nlp(text)

# Get constituency parse
sent = list(doc.sents)[0]
print("Constituency Parse:")
print(sent._.parse_string)

# Example output:
# (S
#   (NP (DT The) (JJ quick) (JJ brown) (NN fox))
#   (VP (VBZ jumps)
#     (PP (IN over)
#       (NP (DT the) (JJ lazy) (NN dog)))))

# Extract specific phrases
def extract_noun_phrases(doc):
    """Extract noun phrases from constituency parse."""
    noun_phrases = []

    for sent in doc.sents:
        for constituent in sent._.constituents:
            if constituent._.labels[0] == "NP":
                noun_phrases.append(constituent.text)

    return noun_phrases

noun_phrases = extract_noun_phrases(doc)
print(f"\nNoun Phrases: {noun_phrases}")

# Extract verb phrases
def extract_verb_phrases(doc):
    """Extract verb phrases from constituency parse."""
    verb_phrases = []

    for sent in doc.sents:
        for constituent in sent._.constituents:
            if constituent._.labels[0] == "VP":
                verb_phrases.append(constituent.text)

    return verb_phrases

verb_phrases = extract_verb_phrases(doc)
print(f"Verb Phrases: {verb_phrases}")
```

---

## 7. Sentiment Analysis

### 7.1 Rule-Based Sentiment Analysis

**Rule-based approaches** use lexicons (dictionaries of words with sentiment scores) to determine text polarity.

#### VADER (Valence Aware Dictionary and sEntiment Reasoner)

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Install: pip install vaderSentiment

analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment
texts = [
    "I love this product! It's amazing!",
    "This is the worst experience ever.",
    "The product is okay, nothing special.",
    "Not bad, but could be better.",
    "AMAZING!!! Best purchase ever!!!"
]

print("VADER Sentiment Analysis:")
for text in texts:
    scores = analyzer.polarity_scores(text)
    print(f"\nText: {text}")
    print(f"Negative: {scores['neg']:.3f}")
    print(f"Neutral:  {scores['neu']:.3f}")
    print(f"Positive: {scores['pos']:.3f}")
    print(f"Compound: {scores['compound']:.3f}")

# Compound score interpretation:
# >= 0.05:  positive
# <= -0.05: negative
# else:     neutral

def classify_sentiment_vader(text, threshold=0.05):
    """Classify sentiment using VADER.

    Args:
        text: Input text
        threshold: Threshold for positive/negative classification

    Returns:
        Sentiment label (positive/negative/neutral)
    """
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    if compound >= threshold:
        return 'positive'
    elif compound <= -threshold:
        return 'negative'
    else:
        return 'neutral'

# VADER handles:
# - Capitalization (GREAT vs great)
# - Punctuation (good!!! vs good)
# - Negation (not good)
# - Degree modifiers (very good, extremely bad)
# - Emoticons and emojis

examples_vader = [
    "The food is AMAZING!!!",
    "The food is good",
    "The food is not good",
    "The food is not very good",
    "The food is okay :)"
]

for text in examples_vader:
    sentiment = classify_sentiment_vader(text)
    compound = analyzer.polarity_scores(text)['compound']
    print(f"{text:40s} --> {sentiment:10s} ({compound:.3f})")
```

#### TextBlob

```python
from textblob import TextBlob

# Install: pip install textblob
# python -m textblob.download_corpora

def analyze_with_textblob(text):
    """Analyze sentiment using TextBlob.

    Returns:
        Dictionary with polarity and subjectivity
    """
    blob = TextBlob(text)

    return {
        'polarity': blob.sentiment.polarity,      # -1 to 1
        'subjectivity': blob.sentiment.subjectivity  # 0 to 1
    }

texts = [
    "I absolutely love this product!",
    "This is terrible and disappointing.",
    "The weather is sunny today.",
    "In my opinion, this is the best movie ever!"
]

print("\nTextBlob Sentiment Analysis:")
for text in texts:
    result = analyze_with_textblob(text)
    print(f"\nText: {text}")
    print(f"Polarity:     {result['polarity']:.3f}")
    print(f"Subjectivity: {result['subjectivity']:.3f}")

# Polarity: -1 (negative) to 1 (positive)
# Subjectivity: 0 (objective) to 1 (subjective)
```

### 7.2 Machine Learning-Based Sentiment Analysis

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Sample dataset (in practice, use IMDB, Twitter, etc.)
texts = [
    "I love this product, it's amazing",
    "Terrible experience, very disappointed",
    "Great quality and fast shipping",
    "Waste of money, doesn't work",
    "Excellent service, highly recommend",
    "Poor quality, broke after one day",
    "Best purchase ever, so happy",
    "Awful customer service",
    "Fantastic product, exceeded expectations",
    "Not worth the price, very bad"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

# Feature extraction
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1,
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train multiple classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC(max_iter=1000)
}

print("Sentiment Classification Results:")
for name, clf in classifiers.items():
    # Train
    clf.fit(X_train_tfidf, y_train)

    # Cross-validation
    cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=3)

    # Test
    y_pred = clf.predict(X_test_tfidf)

    print(f"\n{name}:")
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Feature importance (for logistic regression)
    if name == 'Logistic Regression':
        feature_names = vectorizer.get_feature_names_out()
        coefficients = clf.coef_[0]

        # Top positive features
        top_positive_idx = np.argsort(coefficients)[-10:]
        print("\nTop positive features:")
        for idx in top_positive_idx:
            print(f"  {feature_names[idx]}: {coefficients[idx]:.3f}")

        # Top negative features
        top_negative_idx = np.argsort(coefficients)[:10]
        print("\nTop negative features:")
        for idx in top_negative_idx:
            print(f"  {feature_names[idx]}: {coefficients[idx]:.3f}")

# Complete sentiment analysis pipeline
class SentimentAnalyzer:
    """Complete sentiment analysis pipeline."""

    def __init__(self, method='vader'):
        """Initialize analyzer.

        Args:
            method: 'vader', 'textblob', or 'ml'
        """
        self.method = method

        if method == 'vader':
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.analyzer = SentimentIntensityAnalyzer()
        elif method == 'textblob':
            pass  # TextBlob doesn't need initialization
        elif method == 'ml':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            self.classifier = LogisticRegression(max_iter=1000)
            self.fitted = False

    def train(self, texts, labels):
        """Train ML-based classifier.

        Args:
            texts: List of training texts
            labels: List of labels (0=negative, 1=positive)
        """
        if self.method != 'ml':
            raise ValueError("Training only available for ML method")

        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.fitted = True

    def predict(self, text):
        """Predict sentiment.

        Args:
            text: Input text

        Returns:
            Dictionary with sentiment and score
        """
        if self.method == 'vader':
            scores = self.analyzer.polarity_scores(text)
            compound = scores['compound']

            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            return {
                'sentiment': sentiment,
                'score': compound,
                'details': scores
            }

        elif self.method == 'textblob':
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            return {
                'sentiment': sentiment,
                'score': polarity,
                'subjectivity': blob.sentiment.subjectivity
            }

        elif self.method == 'ml':
            if not self.fitted:
                raise ValueError("Model not trained. Call train() first.")

            X = self.vectorizer.transform([text])
            prediction = self.classifier.predict(X)[0]
            probability = self.classifier.predict_proba(X)[0]

            sentiment = 'positive' if prediction == 1 else 'negative'

            return {
                'sentiment': sentiment,
                'score': probability[1],  # Probability of positive
                'probabilities': {
                    'negative': probability[0],
                    'positive': probability[1]
                }
            }

    def predict_batch(self, texts):
        """Predict sentiment for multiple texts."""
        return [self.predict(text) for text in texts]

# Usage
analyzer_vader = SentimentAnalyzer(method='vader')
result = analyzer_vader.predict("I absolutely love this product!")
print(f"\nVADER Result: {result}")
```

### 7.3 Aspect-Based Sentiment Analysis

```python
import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

def extract_aspect_sentiments(text, aspects):
    """Extract sentiment for specific aspects.

    Args:
        text: Review text
        aspects: List of aspects to analyze (e.g., ['food', 'service'])

    Returns:
        Dictionary of aspect sentiments
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    doc = nlp(text)

    aspect_sentiments = defaultdict(list)

    # Find sentences containing aspects
    for sent in doc.sents:
        sent_text = sent.text.lower()

        for aspect in aspects:
            if aspect in sent_text:
                # Analyze sentiment of this sentence
                scores = analyzer.polarity_scores(sent.text)
                aspect_sentiments[aspect].append({
                    'sentence': sent.text,
                    'score': scores['compound']
                })

    # Aggregate scores
    results = {}
    for aspect, sentiments in aspect_sentiments.items():
        if sentiments:
            avg_score = np.mean([s['score'] for s in sentiments])
            results[aspect] = {
                'score': avg_score,
                'sentiment': 'positive' if avg_score >= 0.05 else 'negative' if avg_score <= -0.05 else 'neutral',
                'mentions': len(sentiments),
                'examples': sentiments[:3]  # First 3 mentions
            }

    return results

# Example
review = """
The food at this restaurant was absolutely delicious. I loved the pasta!
However, the service was quite slow and the staff seemed disorganized.
The ambiance was nice, but the food really stood out.
Overall, great food but poor service.
"""

aspects = ['food', 'service', 'ambiance']
results = extract_aspect_sentiments(review, aspects)

print("\nAspect-Based Sentiment Analysis:")
for aspect, data in results.items():
    print(f"\n{aspect.upper()}:")
    print(f"  Sentiment: {data['sentiment']} (score: {data['score']:.3f})")
    print(f"  Mentions: {data['mentions']}")
    print(f"  Examples:")
    for example in data['examples']:
        print(f"    - {example['sentence'][:60]}... ({example['score']:.3f})")
```

---

## 8. Text Classification

### 8.1 Naive Bayes for Text Classification

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np

# Sample dataset (20 newsgroups style)
texts = [
    "machine learning and neural networks",
    "deep learning with tensorflow",
    "basketball game results",
    "football match highlights",
    "stock market predictions",
    "cryptocurrency trading strategies",
    "python programming tutorial",
    "java development guide",
    "latest smartphone reviews",
    "laptop buying guide",
]

categories = [
    'ai', 'ai',
    'sports', 'sports',
    'finance', 'finance',
    'programming', 'programming',
    'tech', 'tech'
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, categories, test_size=0.3, random_state=42, stratify=categories
)

# Basic Naive Bayes
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

nb_classifier = MultinomialNB(alpha=1.0)
nb_classifier.fit(X_train_vec, y_train)

y_pred = nb_classifier.predict(X_test_vec)

print("Naive Bayes Classification:")
print(classification_report(y_test, y_pred))

# ComplementNB (often better for imbalanced datasets)
cnb_classifier = ComplementNB(alpha=1.0)
cnb_classifier.fit(X_train_vec, y_train)

y_pred_cnb = cnb_classifier.predict(X_test_vec)
print("\nComplement Naive Bayes:")
print(classification_report(y_test, y_pred_cnb))

# Pipeline with hyperparameter tuning
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

parameters = {
    'tfidf__max_features': [500, 1000, 2000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__use_idf': [True, False],
    'clf__alpha': [0.1, 0.5, 1.0, 2.0]
}

grid_search = GridSearchCV(
    pipeline,
    parameters,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

# grid_search.fit(X_train, y_train)
# print(f"\nBest parameters: {grid_search.best_params_}")
# print(f"Best score: {grid_search.best_score_:.3f}")
```

### 8.2 SVM for Text Classification

```python
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV

# Linear SVM (fast, good for high-dimensional text data)
svm_linear = LinearSVC(
    C=1.0,
    max_iter=1000,
    class_weight='balanced'  # Handle imbalanced classes
)

svm_linear.fit(X_train_vec, y_train)
y_pred_svm = svm_linear.predict(X_test_vec)

print("\nLinear SVM:")
print(classification_report(y_test, y_pred_svm))

# Get probability estimates (LinearSVC doesn't have predict_proba)
svm_calibrated = CalibratedClassifierCV(svm_linear, cv=3)
svm_calibrated.fit(X_train_vec, y_train)

probabilities = svm_calibrated.predict_proba(X_test_vec)
print("\nProbability estimates for first test sample:")
for i, category in enumerate(svm_calibrated.classes_):
    print(f"{category}: {probabilities[0][i]:.3f}")

# Kernel SVM (slower, can capture non-linear patterns)
svm_rbf = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True
)

# Note: RBF kernel can be slow on large datasets
# svm_rbf.fit(X_train_vec, y_train)

# Hyperparameter tuning for SVM
param_grid_svm = {
    'C': [0.1, 1.0, 10.0],
    'class_weight': ['balanced', None],
    'max_iter': [1000, 2000]
}

grid_svm = GridSearchCV(
    LinearSVC(),
    param_grid_svm,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1
)

# grid_svm.fit(X_train_vec, y_train)
```

### 8.3 Logistic Regression for Text Classification

```python
from sklearn.linear_model import LogisticRegression

# Multi-class logistic regression
lr_classifier = LogisticRegression(
    max_iter=1000,
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    multi_class='multinomial',
    class_weight='balanced'
)

lr_classifier.fit(X_train_vec, y_train)
y_pred_lr = lr_classifier.predict(X_test_vec)

print("\nLogistic Regression:")
print(classification_report(y_test, y_pred_lr))

# Feature importance analysis
feature_names = vectorizer.get_feature_names_out()

for i, category in enumerate(lr_classifier.classes_):
    top_10_idx = np.argsort(lr_classifier.coef_[i])[-10:]
    print(f"\nTop 10 features for '{category}':")
    for idx in top_10_idx:
        print(f"  {feature_names[idx]}: {lr_classifier.coef_[i][idx]:.3f}")

# Regularization comparison
regularizations = {
    'L1 (Lasso)': 'l1',
    'L2 (Ridge)': 'l2',
    'ElasticNet': 'elasticnet'
}

for name, penalty in regularizations.items():
    if penalty == 'elasticnet':
        clf = LogisticRegression(
            penalty=penalty,
            solver='saga',
            l1_ratio=0.5,
            max_iter=1000
        )
    elif penalty == 'l1':
        clf = LogisticRegression(
            penalty=penalty,
            solver='saga',
            max_iter=1000
        )
    else:
        clf = LogisticRegression(
            penalty=penalty,
            solver='lbfgs',
            max_iter=1000
        )

    clf.fit(X_train_vec, y_train)
    score = clf.score(X_test_vec, y_test)
    print(f"\n{name}: {score:.3f}")
```

### 8.4 Complete Text Classification Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pickle

class TextClassifier:
    """Complete text classification pipeline."""

    def __init__(self,
                 vectorizer_params=None,
                 classifier_params=None,
                 classifier_type='logistic'):
        """Initialize classifier.

        Args:
            vectorizer_params: Parameters for TfidfVectorizer
            classifier_params: Parameters for classifier
            classifier_type: 'logistic', 'naive_bayes', or 'svm'
        """
        # Default vectorizer parameters
        if vectorizer_params is None:
            vectorizer_params = {
                'max_features': 5000,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.8,
                'stop_words': 'english',
                'sublinear_tf': True
            }

        # Choose classifier
        if classifier_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            if classifier_params is None:
                classifier_params = {
                    'max_iter': 1000,
                    'C': 1.0,
                    'class_weight': 'balanced'
                }
            classifier = LogisticRegression(**classifier_params)

        elif classifier_type == 'naive_bayes':
            from sklearn.naive_bayes import MultinomialNB
            if classifier_params is None:
                classifier_params = {'alpha': 1.0}
            classifier = MultinomialNB(**classifier_params)

        elif classifier_type == 'svm':
            from sklearn.svm import LinearSVC
            if classifier_params is None:
                classifier_params = {
                    'C': 1.0,
                    'max_iter': 1000,
                    'class_weight': 'balanced'
                }
            classifier = LinearSVC(**classifier_params)

        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(**vectorizer_params)),
            ('classifier', classifier)
        ])

        self.fitted = False

    def fit(self, X, y):
        """Train the classifier."""
        self.pipeline.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X):
        """Predict class labels."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Handle classifiers without predict_proba
        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            return self.pipeline.named_steps['classifier'].predict_proba(
                self.pipeline.named_steps['vectorizer'].transform(X)
            )
        else:
            # Use calibration for LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            calibrated = CalibratedClassifierCV(
                self.pipeline.named_steps['classifier'],
                cv=3
            )
            X_vec = self.pipeline.named_steps['vectorizer'].transform(X)
            return calibrated.predict_proba(X_vec)

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        scores = cross_val_score(
            self.pipeline, X, y,
            cv=cv,
            scoring='f1_weighted'
        )
        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }

    def get_feature_importance(self, top_n=10):
        """Get top features for each class."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        vectorizer = self.pipeline.named_steps['vectorizer']
        classifier = self.pipeline.named_steps['classifier']

        feature_names = vectorizer.get_feature_names_out()

        if hasattr(classifier, 'coef_'):
            importance = {}
            for i, class_name in enumerate(classifier.classes_):
                top_idx = np.argsort(classifier.coef_[i])[-top_n:][::-1]
                importance[class_name] = [
                    (feature_names[idx], classifier.coef_[i][idx])
                    for idx in top_idx
                ]
            return importance
        else:
            return None

    def save(self, filepath):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load(self, filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.fitted = True

# Usage example
clf = TextClassifier(classifier_type='logistic')
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(f"Predictions: {predictions}")

# Cross-validation
cv_results = clf.cross_validate(X_train, y_train, cv=5)
print(f"\nCV Results: {cv_results['mean']:.3f} (+/- {cv_results['std']:.3f})")

# Feature importance
importance = clf.get_feature_importance(top_n=5)
if importance:
    print("\nTop features per class:")
    for class_name, features in importance.items():
        print(f"\n{class_name}:")
        for feature, score in features:
            print(f"  {feature}: {score:.3f}")

# Save model
clf.save('text_classifier.pkl')
```

---

## 9. Topic Modeling

### 9.1 Latent Dirichlet Allocation (LDA)

```python
from gensim import corpora
from gensim.models import LdaModel, LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
from pprint import pprint

# Sample corpus
documents = [
    "machine learning algorithms for data analysis",
    "deep learning neural networks tensorflow",
    "basketball game score highlights",
    "football match results commentary",
    "stock market trading strategies",
    "cryptocurrency bitcoin investment",
    "python programming tutorial guide",
    "java software development patterns",
    "smartphone camera reviews comparison",
    "laptop hardware specifications benchmark"
]

# Preprocessing
def preprocess_for_lda(documents):
    """Preprocess documents for LDA."""
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words('english'))
    processed = []

    for doc in documents:
        # Tokenize and lowercase
        tokens = word_tokenize(doc.lower())

        # Remove stopwords and short words
        tokens = [
            token for token in tokens
            if token not in stop_words and len(token) > 3
        ]

        processed.append(tokens)

    return processed

processed_docs = preprocess_for_lda(documents)

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)

# Filter extremes
dictionary.filter_extremes(
    no_below=1,      # Minimum document frequency
    no_above=0.8,    # Maximum document frequency
    keep_n=1000      # Keep top N most frequent
)

# Create bag-of-words corpus
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Train LDA model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,           # Number of topics
    random_state=42,
    update_every=1,
    chunksize=100,
    passes=10,              # Number of passes through corpus
    alpha='auto',           # Document-topic density
    eta='auto',             # Topic-word density
    per_word_topics=True
)

# Print topics
print("LDA Topics:")
topics = lda_model.print_topics(num_words=5)
for topic_id, topic in topics:
    print(f"\nTopic {topic_id}:")
    print(topic)

# Get topic distribution for documents
print("\n\nDocument-Topic Distribution:")
for i, doc in enumerate(corpus):
    topic_dist = lda_model.get_document_topics(doc)
    print(f"\nDocument {i}: {documents[i][:50]}...")
    for topic_id, prob in sorted(topic_dist, key=lambda x: x[1], reverse=True):
        print(f"  Topic {topic_id}: {prob:.3f}")

# Coherence score
coherence_model = CoherenceModel(
    model=lda_model,
    texts=processed_docs,
    dictionary=dictionary,
    coherence='c_v'
)

coherence_score = coherence_model.get_coherence()
print(f"\n\nCoherence Score (c_v): {coherence_score:.4f}")
```

### 9.2 Finding Optimal Number of Topics

```python
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    """Compute coherence scores for different numbers of topics.

    Args:
        dictionary: Gensim dictionary
        corpus: Gensim corpus
        texts: Tokenized documents
        limit: Max number of topics to test
        start: Starting number of topics
        step: Step size

    Returns:
        List of coherence scores
    """
    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        print(f"Training LDA with {num_topics} topics...")

        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            eta='auto'
        )

        model_list.append(model)

        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )

        coherence_values.append(coherence_model.get_coherence())

    return model_list, coherence_values

# Find optimal number of topics
model_list, coherence_values = compute_coherence_values(
    dictionary=dictionary,
    corpus=corpus,
    texts=processed_docs,
    start=2,
    limit=6,
    step=1
)

# Plot coherence scores
import matplotlib.pyplot as plt

x = range(2, 6, 1)
plt.figure(figsize=(10, 6))
plt.plot(x, coherence_values, marker='o')
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.title("Coherence Score vs Number of Topics")
plt.xticks(x)
plt.grid(True)
plt.savefig('lda_coherence.png', dpi=300, bbox_inches='tight')

# Best number of topics
optimal_idx = np.argmax(coherence_values)
optimal_topics = list(x)[optimal_idx]
print(f"\nOptimal number of topics: {optimal_topics}")
print(f"Best coherence score: {coherence_values[optimal_idx]:.4f}")
```

### 9.3 Non-Negative Matrix Factorization (NMF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np

# Prepare data
vectorizer = TfidfVectorizer(
    max_features=1000,
    max_df=0.8,
    min_df=2,
    stop_words='english'
)

tfidf_matrix = vectorizer.fit_transform(documents)

# Train NMF
n_topics = 3

nmf_model = NMF(
    n_components=n_topics,
    random_state=42,
    init='nndsvda',         # Initialization method
    max_iter=500,
    alpha=0.1,              # Regularization
    l1_ratio=0.5            # L1/L2 ratio for regularization
)

W = nmf_model.fit_transform(tfidf_matrix)  # Document-topic matrix
H = nmf_model.components_                   # Topic-word matrix

# Display topics
feature_names = vectorizer.get_feature_names_out()

print("NMF Topics:")
for topic_idx, topic in enumerate(H):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    top_scores = [topic[i] for i in top_words_idx]

    print(f"\nTopic {topic_idx}:")
    for word, score in zip(top_words, top_scores):
        print(f"  {word}: {score:.3f}")

# Document-topic distribution
print("\n\nDocument-Topic Distribution (NMF):")
for i, doc_topics in enumerate(W):
    print(f"\nDocument {i}: {documents[i][:50]}...")
    for topic_idx, score in enumerate(doc_topics):
        print(f"  Topic {topic_idx}: {score:.3f}")

# Compare LDA vs NMF
print("\n\nLDA vs NMF Comparison:")
print("LDA:")
print("  - Probabilistic model")
print("  - Topics are distributions over words")
print("  - Documents are mixtures of topics")
print("  - Better for interpretability")
print("  - Slower training")
print("\nNMF:")
print("  - Matrix factorization")
print("  - Deterministic given initialization")
print("  - Faster training")
print("  - Can handle sparse data well")
print("  - Good for large datasets")
```

### 9.4 Complete Topic Modeling Pipeline

```python
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TopicModeler:
    """Complete topic modeling pipeline."""

    def __init__(self, n_topics=5, method='lda'):
        """Initialize topic modeler.

        Args:
            n_topics: Number of topics
            method: 'lda' or 'nmf'
        """
        self.n_topics = n_topics
        self.method = method
        self.model = None
        self.dictionary = None
        self.corpus = None
        self.vectorizer = None
        self.fitted = False

    def preprocess(self, documents):
        """Preprocess documents."""
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        stop_words = set(stopwords.words('english'))
        processed = []

        for doc in documents:
            tokens = word_tokenize(doc.lower())
            tokens = [
                token for token in tokens
                if token.isalpha() and
                token not in stop_words and
                len(token) > 3
            ]
            processed.append(tokens)

        return processed

    def fit(self, documents):
        """Fit topic model."""
        if self.method == 'lda':
            # Preprocess
            processed_docs = self.preprocess(documents)

            # Create dictionary and corpus
            self.dictionary = corpora.Dictionary(processed_docs)
            self.dictionary.filter_extremes(no_below=2, no_above=0.8)
            self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]

            # Train LDA
            self.model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.n_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                eta='auto'
            )

            # Compute coherence
            coherence_model = CoherenceModel(
                model=self.model,
                texts=processed_docs,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            self.coherence_score = coherence_model.get_coherence()

        elif self.method == 'nmf':
            # Vectorize
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                max_df=0.8,
                min_df=2,
                stop_words='english'
            )
            tfidf_matrix = self.vectorizer.fit_transform(documents)

            # Train NMF
            self.model = NMF(
                n_components=self.n_topics,
                random_state=42,
                init='nndsvda',
                max_iter=500
            )
            self.model.fit(tfidf_matrix)

        self.fitted = True
        return self

    def get_topics(self, n_words=10):
        """Get top words for each topic."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        topics = []

        if self.method == 'lda':
            for topic_id in range(self.n_topics):
                topic_words = self.model.show_topic(topic_id, topn=n_words)
                topics.append([(word, float(score)) for word, score in topic_words])

        elif self.method == 'nmf':
            feature_names = self.vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(self.model.components_):
                top_idx = topic.argsort()[-n_words:][::-1]
                topic_words = [(feature_names[i], topic[i]) for i in top_idx]
                topics.append(topic_words)

        return topics

    def transform(self, documents):
        """Get topic distribution for documents."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.method == 'lda':
            processed_docs = self.preprocess(documents)
            corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]

            doc_topics = []
            for doc in corpus:
                topics = self.model.get_document_topics(doc)
                # Create dense array
                topic_dist = np.zeros(self.n_topics)
                for topic_id, prob in topics:
                    topic_dist[topic_id] = prob
                doc_topics.append(topic_dist)

            return np.array(doc_topics)

        elif self.method == 'nmf':
            tfidf_matrix = self.vectorizer.transform(documents)
            return self.model.transform(tfidf_matrix)

    def predict_topic(self, document):
        """Predict dominant topic for a document."""
        topic_dist = self.transform([document])[0]
        dominant_topic = np.argmax(topic_dist)
        return dominant_topic, topic_dist[dominant_topic]

# Usage
modeler = TopicModeler(n_topics=3, method='lda')
modeler.fit(documents)

# Get topics
topics = modeler.get_topics(n_words=5)
print("Topics:")
for i, topic_words in enumerate(topics):
    print(f"\nTopic {i}:")
    for word, score in topic_words:
        print(f"  {word}: {score:.3f}")

# Transform new documents
new_doc = "deep learning neural network architecture"
topic_id, score = modeler.predict_topic(new_doc)
print(f"\n\nDominant topic for '{new_doc}': Topic {topic_id} ({score:.3f})")
```

---

## 10. Text Similarity

### 10.1 Cosine Similarity

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
    "machine learning is a subset of artificial intelligence",
    "deep learning is a type of machine learning",
    "natural language processing deals with text data",
    "computer vision processes images and videos"
]

# Compute TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Cosine Similarity Matrix:")
print(similarity_matrix)
print()

# Find most similar documents
def find_similar_documents(query_idx, similarity_matrix, top_n=2):
    """Find most similar documents to query.

    Args:
        query_idx: Index of query document
        similarity_matrix: Pairwise similarity matrix
        top_n: Number of similar documents to return

    Returns:
        List of (doc_idx, similarity_score) tuples
    """
    # Get similarities for query document
    similarities = similarity_matrix[query_idx]

    # Sort by similarity (exclude self)
    similar_idx = np.argsort(similarities)[::-1][1:top_n+1]

    return [(idx, similarities[idx]) for idx in similar_idx]

# Example
query_idx = 0
similar_docs = find_similar_documents(query_idx, similarity_matrix, top_n=2)

print(f"Query document: {documents[query_idx]}")
print("\nMost similar documents:")
for doc_idx, score in similar_docs:
    print(f"  {documents[doc_idx]}")
    print(f"  Similarity: {score:.4f}\n")

# Custom cosine similarity function
def cosine_similarity_manual(vec1, vec2):
    """Compute cosine similarity manually.

    Formula: cos(theta) = (A * B) / (||A|| * ||B||)
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

# Verify
vec1 = tfidf_matrix[0].toarray()[0]
vec2 = tfidf_matrix[1].toarray()[0]
manual_sim = cosine_similarity_manual(vec1, vec2)
sklearn_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

print(f"Manual cosine similarity: {manual_sim:.4f}")
print(f"Sklearn cosine similarity: {sklearn_sim:.4f}")
```

### 10.2 Jaccard Similarity

```python
def jaccard_similarity(text1, text2):
    """Compute Jaccard similarity between two texts.

    Formula: |A intersection B| / |A union B|

    Args:
        text1, text2: Input texts

    Returns:
        Jaccard similarity (0 to 1)
    """
    # Tokenize
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    # Intersection and union
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)

# Example
text1 = "machine learning is amazing"
text2 = "deep learning is fascinating"
text3 = "machine learning and deep learning"

print(f"Jaccard similarity:")
print(f"  Text 1 vs Text 2: {jaccard_similarity(text1, text2):.4f}")
print(f"  Text 1 vs Text 3: {jaccard_similarity(text1, text3):.4f}")
print(f"  Text 2 vs Text 3: {jaccard_similarity(text2, text3):.4f}")

# Character-level Jaccard (good for typos)
def jaccard_similarity_char(text1, text2, n=2):
    """Character n-gram Jaccard similarity.

    Args:
        text1, text2: Input texts
        n: N-gram size

    Returns:
        Jaccard similarity
    """
    def get_char_ngrams(text, n):
        text = text.lower()
        return set([text[i:i+n] for i in range(len(text)-n+1)])

    ngrams1 = get_char_ngrams(text1, n)
    ngrams2 = get_char_ngrams(text2, n)

    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)

# Example
word1 = "python"
word2 = "pyton"  # Typo
word3 = "java"

print(f"\nCharacter Jaccard (bigrams):")
print(f"  {word1} vs {word2}: {jaccard_similarity_char(word1, word2, n=2):.4f}")
print(f"  {word1} vs {word3}: {jaccard_similarity_char(word1, word3, n=2):.4f}")
```

### 10.3 Edit Distance (Levenshtein Distance)

```python
def levenshtein_distance(str1, str2):
    """Compute Levenshtein (edit) distance.

    Minimum number of single-character edits (insertions, deletions,
    substitutions) required to change str1 into str2.

    Args:
        str1, str2: Input strings

    Returns:
        Edit distance
    """
    m, n = len(str1), len(str2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Deletion
                    dp[i][j-1],      # Insertion
                    dp[i-1][j-1]     # Substitution
                )

    return dp[m][n]

def normalized_levenshtein(str1, str2):
    """Normalized edit distance (0 to 1).

    Returns:
        Normalized distance (0 = identical, 1 = completely different)
    """
    distance = levenshtein_distance(str1, str2)
    max_len = max(len(str1), len(str2))

    if max_len == 0:
        return 0.0

    return distance / max_len

# Examples
print("Levenshtein Distance:")
pairs = [
    ("kitten", "sitting"),
    ("saturday", "sunday"),
    ("python", "pyton"),
    ("machine", "learning")
]

for str1, str2 in pairs:
    distance = levenshtein_distance(str1, str2)
    normalized = normalized_levenshtein(str1, str2)
    print(f"  '{str1}' vs '{str2}':")
    print(f"    Distance: {distance}")
    print(f"    Normalized: {normalized:.4f}\n")

# Using python-Levenshtein library (faster)
try:
    import Levenshtein

    print("Using python-Levenshtein library:")
    for str1, str2 in pairs:
        distance = Levenshtein.distance(str1, str2)
        ratio = Levenshtein.ratio(str1, str2)  # Similarity ratio
        print(f"  '{str1}' vs '{str2}': {distance} (ratio: {ratio:.4f})")

except ImportError:
    print("Install python-Levenshtein for faster computation:")
    print("  pip install python-Levenshtein")
```

### 10.4 Semantic Similarity

```python
import spacy

nlp = spacy.load("en_core_web_md")  # Requires medium or large model

def semantic_similarity_spacy(text1, text2):
    """Compute semantic similarity using spaCy word vectors.

    Args:
        text1, text2: Input texts

    Returns:
        Similarity score (0 to 1)
    """
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    return doc1.similarity(doc2)

# Examples
text_pairs = [
    ("I love machine learning", "I enjoy artificial intelligence"),
    ("The cat sits on the mat", "A dog lies on the floor"),
    ("Python is a programming language", "Java is used for software development"),
    ("The weather is sunny", "Stock market is volatile")
]

print("Semantic Similarity (spaCy):")
for text1, text2 in text_pairs:
    similarity = semantic_similarity_spacy(text1, text2)
    print(f"\n  Text 1: {text1}")
    print(f"  Text 2: {text2}")
    print(f"  Similarity: {similarity:.4f}")

# Complete similarity comparison
class TextSimilarity:
    """Compare texts using multiple similarity metrics."""

    def __init__(self):
        """Initialize with spaCy model."""
        import spacy
        self.nlp = spacy.load("en_core_web_md")

        # Initialize TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()

    def fit_tfidf(self, corpus):
        """Fit TF-IDF on corpus."""
        self.vectorizer.fit(corpus)

    def compare(self, text1, text2):
        """Compare texts using all methods.

        Returns:
            Dictionary with similarity scores
        """
        results = {}

        # Jaccard
        results['jaccard'] = jaccard_similarity(text1, text2)

        # Edit distance
        results['levenshtein_normalized'] = 1 - normalized_levenshtein(text1, text2)

        # Cosine (TF-IDF)
        tfidf = self.vectorizer.transform([text1, text2])
        results['cosine_tfidf'] = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

        # Semantic (spaCy)
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        results['semantic_spacy'] = doc1.similarity(doc2)

        return results

# Usage
similarity = TextSimilarity()
similarity.fit_tfidf(documents)

text1 = "machine learning is powerful"
text2 = "deep learning is effective"

results = similarity.compare(text1, text2)
print(f"\n\nSimilarity Comparison:")
print(f"Text 1: {text1}")
print(f"Text 2: {text2}")
print("\nScores:")
for method, score in results.items():
    print(f"  {method:25s}: {score:.4f}")
```

---

## 11. Practical NLP Pipeline

### 11.1 End-to-End Text Processing Class

```python
import re
import spacy
import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

class NLPPipeline:
    """Complete end-to-end NLP processing pipeline."""

    def __init__(self,
                 spacy_model='en_core_web_sm',
                 lowercase=True,
                 remove_punct=True,
                 remove_stopwords=False,
                 lemmatize=True):
        """Initialize NLP pipeline.

        Args:
            spacy_model: SpaCy model to use
            lowercase: Convert to lowercase
            remove_punct: Remove punctuation
            remove_stopwords: Remove stopwords
            lemmatize: Lemmatize tokens
        """
        self.nlp = spacy.load(spacy_model)
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

        if remove_stopwords:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Clean text (URLs, emails, etc.)."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)

        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess(self, text: str) -> List[str]:
        """Preprocess text into tokens.

        Args:
            text: Input text

        Returns:
            List of processed tokens
        """
        # Clean
        text = self.clean_text(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Process with spaCy
        doc = self.nlp(text)

        tokens = []
        for token in doc:
            # Skip punctuation
            if self.remove_punct and token.is_punct:
                continue

            # Skip stopwords
            if self.remove_stopwords and token.text in self.stop_words:
                continue

            # Skip spaces
            if token.is_space:
                continue

            # Get token text
            if self.lemmatize:
                token_text = token.lemma_
            else:
                token_text = token.text

            tokens.append(token_text)

        return tokens

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities.

        Returns:
            List of entity dictionaries
        """
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

        return entities

    def extract_pos_tags(self, text: str) -> List[Dict]:
        """Extract POS tags.

        Returns:
            List of token dictionaries with POS info
        """
        doc = self.nlp(text)

        pos_tags = []
        for token in doc:
            if not token.is_space:
                pos_tags.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,
                    'tag': token.tag_,
                    'dep': token.dep_
                })

        return pos_tags

    def extract_keywords(self, text: str, top_n=10) -> List[tuple]:
        """Extract keywords using TF-IDF-like scoring.

        Returns:
            List of (word, score) tuples
        """
        doc = self.nlp(text)

        # Get word frequencies
        word_freq = Counter()
        for token in doc:
            if (not token.is_stop and
                not token.is_punct and
                token.is_alpha):
                word_freq[token.lemma_] += 1

        # Simple scoring (can be replaced with TF-IDF)
        total = sum(word_freq.values())
        word_scores = {
            word: freq / total
            for word, freq in word_freq.items()
        }

        # Sort by score
        sorted_words = sorted(
            word_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_words[:top_n]

    def get_statistics(self, text: str) -> Dict:
        """Compute text statistics.

        Returns:
            Dictionary with various statistics
        """
        doc = self.nlp(text)

        # Basic counts
        n_chars = len(text)
        n_words = len([token for token in doc if not token.is_punct])
        n_sentences = len(list(doc.sents))

        # POS distribution
        pos_counts = Counter([token.pos_ for token in doc if not token.is_punct])

        # Average word length
        word_lengths = [len(token.text) for token in doc if token.is_alpha]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0

        # Average sentence length
        sent_lengths = [len([t for t in sent if not t.is_punct]) for sent in doc.sents]
        avg_sent_length = np.mean(sent_lengths) if sent_lengths else 0

        return {
            'n_characters': n_chars,
            'n_words': n_words,
            'n_sentences': n_sentences,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sent_length,
            'pos_distribution': dict(pos_counts),
            'unique_words': len(set([t.text.lower() for t in doc if t.is_alpha]))
        }

    def process_document(self, text: str) -> Dict:
        """Complete document processing.

        Returns:
            Dictionary with all extracted information
        """
        return {
            'original': text,
            'cleaned': self.clean_text(text),
            'tokens': self.preprocess(text),
            'entities': self.extract_entities(text),
            'pos_tags': self.extract_pos_tags(text),
            'keywords': self.extract_keywords(text),
            'statistics': self.get_statistics(text)
        }

# Usage example
pipeline = NLPPipeline(
    lowercase=True,
    remove_punct=True,
    remove_stopwords=True,
    lemmatize=True
)

sample_text = """
Apple Inc. announced record profits of $100 billion in 2024.
CEO Tim Cook stated that innovation drives success.
The company's AI initiatives continue to impress investors.
Visit https://apple.com for more information.
"""

result = pipeline.process_document(sample_text)

print("Processed Document:")
print(f"\nCleaned Text: {result['cleaned']}")
print(f"\nTokens: {result['tokens']}")
print(f"\nEntities:")
for ent in result['entities']:
    print(f"  {ent['text']:20s} {ent['label']}")
print(f"\nKeywords:")
for word, score in result['keywords']:
    print(f"  {word:15s} {score:.4f}")
print(f"\nStatistics:")
for key, value in result['statistics'].items():
    print(f"  {key}: {value}")
```

### 11.2 Handling Real-World Messy Text

```python
import re
from typing import List

class TextCleaner:
    """Robust text cleaning for real-world messy data."""

    @staticmethod
    def remove_emojis(text: str) -> str:
        """Remove emojis from text."""
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata

        # Normalize to NFKD form
        text = unicodedata.normalize('NFKD', text)

        # Remove non-ASCII
        text = text.encode('ascii', 'ignore').decode('ascii')

        return text

    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expand common contractions."""
        contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it's": "it is",
            "let's": "let us",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what's": "what is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have"
        }

        pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')

        def replace(match):
            return contractions[match.group(0)]

        return pattern.sub(replace, text.lower())

    @staticmethod
    def remove_repeated_chars(text: str, max_repeat=2) -> str:
        """Remove excessive character repetition."""
        pattern = r'(.)\1{' + str(max_repeat) + r',}'
        return re.sub(pattern, r'\1' * max_repeat, text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize all whitespace to single spaces."""
        # Replace tabs, newlines, etc.
        text = re.sub(r'[\t\n\r\f\v]', ' ', text)

        # Replace multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    @staticmethod
    def clean_social_media_text(text: str) -> str:
        """Clean social media text (Twitter, etc.)."""
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)

        # Remove hashtags (keep text)
        text = re.sub(r'#(\w+)', r'\1', text)

        # Remove RT
        text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        return text

    @classmethod
    def clean_all(cls, text: str, social_media=False) -> str:
        """Apply all cleaning steps.

        Args:
            text: Input text
            social_media: Apply social media cleaning

        Returns:
            Cleaned text
        """
        # Remove emojis
        text = cls.remove_emojis(text)

        # Normalize unicode
        text = cls.normalize_unicode(text)

        # Social media cleaning
        if social_media:
            text = cls.clean_social_media_text(text)

        # Expand contractions
        text = cls.expand_contractions(text)

        # Remove repeated characters
        text = cls.remove_repeated_chars(text)

        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)

        # Normalize whitespace
        text = cls.normalize_whitespace(text)

        return text

# Examples
messy_texts = [
    "OMG!!! This is soooooo amazing!!! ",
    "RT @user: Check this out https://example.com #awesome",
    "I'm so happyyy!!! Best day everrrr",
    "Can't believe this... it's unbelievable!!!!"
]

cleaner = TextCleaner()

print("Text Cleaning Examples:")
for text in messy_texts:
    cleaned = cleaner.clean_all(text, social_media=True)
    print(f"\nOriginal: {text}")
    print(f"Cleaned:  {cleaned}")
```

---

## 12. Resources and References

### 12.1 Essential Libraries

**Core NLP Libraries:**
- **spaCy**: Industrial-strength NLP (https://spacy.io/)
- **NLTK**: Natural Language Toolkit (https://www.nltk.org/)
- **Gensim**: Topic modeling and document similarity (https://radimrehurek.com/gensim/)
- **TextBlob**: Simplified text processing (https://textblob.readthedocs.io/)

**Text Processing:**
- **tokenizers**: Fast tokenizers (HuggingFace) (https://github.com/huggingface/tokenizers)
- **sentencepiece**: Unsupervised tokenizer (https://github.com/google/sentencepiece)
- **ftfy**: Fixes broken Unicode (https://github.com/rspeer/python-ftfy)

**Machine Learning:**
- **scikit-learn**: ML algorithms and vectorizers
- **XGBoost/LightGBM**: Gradient boosting for text classification
- **Vowpal Wabbit**: Fast online learning

**Sentiment Analysis:**
- **vaderSentiment**: Rule-based sentiment (https://github.com/cjhutto/vaderSentiment)
- **TextBlob**: Sentiment and subjectivity
- **transformers**: SOTA sentiment models (HuggingFace)

### 12.2 Key Concepts Summary

**Text Preprocessing:**
- Tokenization: word, subword (BPE, WordPiece, SentencePiece)
- Normalization: lowercasing, unicode, contractions
- Stopword removal: context-dependent decision
- Stemming vs Lemmatization: speed vs accuracy tradeoff

**Text Representation:**
- Bag of Words: simple, loses order
- TF-IDF: weights by importance
- N-grams: captures local context
- HashingVectorizer: memory-efficient for large vocabs

**NLP Tasks:**
- NER: spaCy, custom training, rule-based matching
- POS Tagging: grammatical categorization
- Sentiment Analysis: VADER, TextBlob, ML classifiers
- Text Classification: Naive Bayes, SVM, Logistic Regression
- Topic Modeling: LDA, NMF, coherence scoring

**Text Similarity:**
- Cosine: TF-IDF or embedding-based
- Jaccard: set-based overlap
- Edit Distance: character-level changes
- Semantic: word embedding similarity

### 12.3 Best Practices

**Data Preprocessing:**
1. Always inspect data before processing
2. Choose preprocessing based on task (keep stopwords for sentiment)
3. Use lemmatization for semantic tasks, stemming for search
4. Handle unicode properly for multilingual text
5. Clean but don't over-clean (preserve important signals)

**Feature Engineering:**
1. Start with TF-IDF baseline
2. Experiment with n-gram ranges (1,2) or (1,3)
3. Tune min_df and max_df to remove noise
4. Consider character n-grams for robustness
5. Use HashingVectorizer for very large datasets

**Model Selection:**
1. **Small datasets**: Naive Bayes, Logistic Regression
2. **Large datasets**: Linear SVM, Neural Networks
3. **Interpretability**: Logistic Regression with feature importance
4. **Speed**: Naive Bayes, Hashing + SGD
5. **Accuracy**: Ensemble methods, fine-tuned transformers

**Hyperparameter Tuning:**
- TF-IDF: max_features, ngram_range, min_df, max_df, sublinear_tf
- Naive Bayes: alpha (smoothing)
- SVM: C (regularization), kernel type
- Logistic Regression: C, penalty (l1/l2/elasticnet)

**Production Considerations:**
1. Save fitted vectorizers and models (pickle/joblib)
2. Version control preprocessing pipelines
3. Monitor for data drift
4. Handle OOV words gracefully
5. Set up proper logging and error handling
6. Benchmark inference latency
7. Consider model size for deployment

### 12.4 Common Pitfalls

**Avoid These Mistakes:**
1. **Fitting vectorizer on test data** --> Data leakage
2. **Not handling OOV words** --> Runtime errors
3. **Over-preprocessing** --> Losing important signals
4. **Ignoring class imbalance** --> Poor minority class performance
5. **Not validating preprocessing** --> Silent bugs
6. **Using wrong similarity metric** --> Misleading results
7. **Forgetting to lowercase** --> Vocabulary explosion
8. **Not removing rare words** --> Overfitting, memory issues

### 12.5 Further Learning

**Books:**
- "Natural Language Processing with Python" (Bird, Klein, Loper)
- "Speech and Language Processing" (Jurafsky, Martin)
- "Applied Text Analysis with Python" (Bengfort, Bilbro, Ojeda)

**Courses:**
- Stanford CS224N: Natural Language Processing with Deep Learning
- fast.ai: NLP Course
- Coursera: Natural Language Processing Specialization

**Datasets:**
- **IMDB**: Sentiment analysis (50k reviews)
- **20 Newsgroups**: Text classification
- **Reuters**: Topic classification
- **Yelp**: Multi-aspect sentiment
- **Twitter Sentiment**: Short text sentiment
- **Amazon Reviews**: Large-scale sentiment

**Benchmarks:**
- **GLUE**: General Language Understanding Evaluation
- **SuperGLUE**: More challenging tasks
- **SentEval**: Sentence representation evaluation

### 12.6 Code Templates

**Quick Start Template:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump((vectorizer, clf), f)
```

**Production Pipeline Template:**
```python
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

class ProductionNLPPipeline:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = None
        self.model = None

    def preprocess(self, text):
        doc = self.nlp(text.lower())
        return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

    def train(self, texts, labels):
        # Preprocess
        processed = [self.preprocess(text) for text in texts]

        # Vectorize
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(processed)

        # Train
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, labels)

    def predict(self, text):
        processed = self.preprocess(text)
        X = self.vectorizer.transform([processed])
        return self.model.predict(X)[0]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.vectorizer, self.model), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.vectorizer, self.model = pickle.load(f)

# Usage
pipeline = ProductionNLPPipeline()
# pipeline.train(train_texts, train_labels)
# pipeline.save('nlp_pipeline.pkl')
```

---

**End of NLP Fundamentals Guide**

This comprehensive guide covers classical NLP techniques that remain essential in 2025, forming the foundation for understanding modern transformer-based approaches. Master these fundamentals before moving to deep learning NLP methods.

