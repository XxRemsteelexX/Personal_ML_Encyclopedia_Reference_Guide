# 26. Large Language Models (LLMs)

## Overview

Large Language Models (LLMs) are transformer-based models trained on massive text corpora, capable of understanding and generating human-like text. They represent the state-of-the-art in NLP as of 2025, powering applications from chatbots to code generation.

**Key Characteristics:**
- Billions to trillions of parameters
- Pre-trained on diverse internet text
- Few-shot and zero-shot learning capabilities
- Emergent abilities at scale

---

## 26.1 Evolution of LLMs

### Timeline

**2018: BERT Era**
- BERT (110M-340M params): Bidirectional pre-training
- Masked language modeling
- Fine-tuning paradigm

**2019: GPT-2**
- 1.5B parameters
- Generative pre-training
- Zero-shot capabilities emerging

**2020: GPT-3**
- 175B parameters
- Few-shot in-context learning
- No fine-tuning needed for many tasks

**2021-2022: Specialized Models**
- Codex (code generation)
- InstructGPT (instruction following)
- ChatGPT (conversational AI)

**2023-2024: Multimodal & Reasoning**
- GPT-4 (multimodal, improved reasoning)
- Claude, Gemini (competitors)
- Open-source: LLaMA, Mistral

**2025: Current State**
- GPT-4o, Claude 3.5 Sonnet, Gemini 2.0
- Multimodal (text, image, audio, video)
- Reasoning-first architectures (OpenAI o1)
- Agentic capabilities

---

## 26.2 Architecture Types

### Encoder-Only (BERT Family)

**Architecture:** Stack of transformer encoders
**Training:** Masked Language Modeling (MLM)

```python
# Masked LM objective
Input:  "The [MASK] sat on the mat"
Target: "cat"
```

**Models:**
- BERT (2018): 110M-340M params
- RoBERTa (2019): Optimized BERT training
- ELECTRA (2020): Replaced token detection
- DeBERTa (2020): Disentangled attention

**Best For:**
- Classification tasks
- Named Entity Recognition
- Question Answering (extractive)
- Sentence similarity

### Decoder-Only (GPT Family)

**Architecture:** Stack of transformer decoders (causal self-attention only)
**Training:** Autoregressive Language Modeling

```python
# Next token prediction
Input:  "The cat sat on"
Target: "the"
```

**Models:**
- GPT-2 (2019): 1.5B params
- GPT-3 (2020): 175B params
- GPT-3.5/4 (2022-2023): ChatGPT, improved instruction following
- LLaMA (2023): Open-source, 7B-65B
- Claude 2/3 (2023-2024): Anthropic's models
- Gemini (2023-2024): Google's multimodal LLM

**Best For:**
- Text generation
- Few-shot learning
- Conversational AI
- Code generation

### Encoder-Decoder (T5 Family)

**Architecture:** Full transformer (encoder + decoder)
**Training:** Text-to-text format (all tasks as seq2seq)

```python
# Text-to-text format
Input:  "translate English to German: Hello"
Output: "Hallo"

Input:  "summarize: [long text]"
Output: "[summary]"
```

**Models:**
- T5 (2020): 11B params
- BART (2020): Denoising autoencoder
- mT5 (2021): Multilingual T5

**Best For:**
- Translation
- Summarization
- Any task frameable as text-to-text

---

## 26.3 Pre-training Objectives

### Masked Language Modeling (MLM)

```python
# BERT-style
def masked_lm_loss(model, tokens, mask_prob=0.15):
    # Randomly mask tokens
    masked_tokens = tokens.clone()
    mask = torch.rand(tokens.shape) < mask_prob
    masked_tokens[mask] = MASK_TOKEN

    # Predict original tokens
    predictions = model(masked_tokens)
    loss = cross_entropy(predictions[mask], tokens[mask])
    return loss
```

**Advantages:**
- Bidirectional context
- Good representations for understanding tasks

**Disadvantages:**
- Pre-train/fine-tune discrepancy (MASK not in real text)
- Cannot generate text naturally

### Causal Language Modeling (CLM)

```python
# GPT-style
def causal_lm_loss(model, tokens):
    # Predict next token at each position
    logits = model(tokens[:, :-1])
    loss = cross_entropy(
        logits.reshape(-1, vocab_size),
        tokens[:, 1:].reshape(-1)
    )
    return loss
```

**Advantages:**
- Natural generation
- No pre-train/fine-tune discrepancy
- Scales well (GPT-3, GPT-4)

**Disadvantages:**
- Unidirectional (can't see future tokens)

### Span Corruption (T5)

```python
# T5-style
Input:  "Thank you <X> me to your party <Y> week"
Output: "<X> for inviting <Y> last <Z>"
```

Randomly corrupt spans of text, train to reconstruct.

---

## 26.4 Scaling Laws

### Chinchilla Scaling Laws (2022)

**Key Finding:** Most LLMs are over-parametrized and under-trained.

**Optimal Compute Budget:**
```
For compute budget C:
- Model params N proportional to C^0.5
- Training tokens D proportional to C^0.5

Optimal: N and D should scale equally
```

**Implication:**
- GPT-3 (175B params, 300B tokens): Under-trained
- Chinchilla (70B params, 1.4T tokens): Better performance, smaller model

**2025 Practice:**
- Train smaller models on more data
- LLaMA 2: 70B on 2T tokens
- Efficient scaling

### Emergent Abilities

**Abilities that appear only at scale:**

**Few-shot learning:** GPT-3 could do it, GPT-2 couldn't
**Chain-of-thought reasoning:** Only in models >100B params
**Instruction following:** Emerges with scale + RLHF

**Example:**
```
<100B params: Cannot solve multi-step math
>100B params: Can solve with chain-of-thought prompting
```

---

## 26.5 Fine-Tuning Approaches

### Full Fine-Tuning

Update all parameters on downstream task.

```python
# Load pre-trained model
model = GPTModel.from_pretrained('gpt-3-175b')

# Fine-tune on task
optimizer = AdamW(model.parameters(), lr=1e-5)

for batch in task_data:
    loss = model(batch['input'], labels=batch['output'])
    loss.backward()
    optimizer.step()
```

**Pros:** Best performance
**Cons:** Expensive, need separate model per task

### Parameter-Efficient Fine-Tuning (PEFT)

#### LoRA (Low-Rank Adaptation)

**Idea:** Add trainable low-rank matrices to frozen weights.

```python
# Original: W in R^(dxk)
# LoRA: W + BA where B in R^(dxr), A in R^(rxk), r << d

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.W = nn.Linear(in_features, out_features)  # Frozen
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Freeze original weights
        self.W.weight.requires_grad = False

    def forward(self, x):
        return self.W(x) + self.lora_B(self.lora_A(x))
```

**Benefits:**
- Train <1% of parameters
- Merge LoRA weights for inference (no overhead)
- Multiple LoRA adapters for different tasks

#### Prefix Tuning

**Idea:** Prepend trainable vectors to each layer.

```python
class PrefixTuning(nn.Module):
    def __init__(self, num_layers, prefix_length, d_model):
        super().__init__()
        # Trainable prefix for each layer
        self.prefix = nn.Parameter(
            torch.randn(num_layers, prefix_length, d_model)
        )

    def forward(self, x, layer_idx):
        # Prepend prefix to input
        prefix = self.prefix[layer_idx].unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat([prefix, x], dim=1)
```

#### Adapters

**Idea:** Insert small bottleneck layers between transformer blocks.

```python
class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, d_model)

    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))
```

**Comparison:**

| Method | Trainable % | Performance | Inference Cost |
|--------|------------|-------------|----------------|
| Full FT | 100% | Best | 1x |
| LoRA | <1% | ~95% of full | 1x (merged) |
| Prefix | <0.1% | ~90% of full | 1.05x |
| Adapters | 1-3% | ~93% of full | 1.1x |

---

## 26.6 Alignment and RLHF

### Instruction Fine-Tuning

**Goal:** Make model follow instructions.

**Data Format:**
```json
{
  "instruction": "Translate to French",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

**Training:** Supervised fine-tuning on instruction-output pairs.

### RLHF (Reinforcement Learning from Human Feedback)

**Three-Step Process:**

**1. Supervised Fine-Tuning (SFT)**
```python
# Train on high-quality human demonstrations
loss = cross_entropy(model(prompt), demonstration)
```

**2. Reward Model Training**
```python
# Train model to predict human preferences
# Given: prompt, response_A, response_B, human_preference

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, tokens):
        hidden = self.base(tokens)
        return self.value_head(hidden[:, -1, :])  # Reward score

# Training
reward_A = reward_model(response_A)
reward_B = reward_model(response_B)
loss = -log_sigmoid(reward_winner - reward_loser)
```

**3. RL Fine-Tuning (PPO)**
```python
# Optimize policy to maximize reward
for prompt in prompts:
    response = policy_model.generate(prompt)
    reward = reward_model(prompt + response)

    # PPO objective
    ratio = new_policy(response) / old_policy(response)
    clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
    loss = -min(ratio * advantage, clipped_ratio * advantage)
```

**Result:** Model aligned with human preferences (helpful, harmless, honest).

### DPO (Direct Preference Optimization)

**2023 Alternative to RLHF (simpler, no RL):**

```python
# Directly optimize on preference pairs
loss = -log_sigmoid(
    beta * (log p_theta(y_w|x) - log p_theta(y_l|x))
    - beta * (log p_ref(y_w|x) - log p_ref(y_l|x))
)
```

Where:
- y_w = winning response
- y_l = losing response
- p_ref = reference model (SFT)
- beta = temperature

**Advantages over RLHF:**
- No reward model needed
- No RL (simpler, more stable)
- Comparable results

---

## 26.7 Prompting Techniques

### Zero-Shot Prompting

```python
prompt = "Translate to French: Hello, how are you?"
response = model.generate(prompt)
# Output: "Bonjour, comment allez-vous?"
```

### Few-Shot Prompting (In-Context Learning)

```python
prompt = """
Translate to French:
English: Hello --> French: Bonjour
English: Goodbye --> French: Au revoir
English: Thank you --> French: Merci
English: How are you? --> French:"""

response = model.generate(prompt)
# Output: "Comment allez-vous?"
```

### Chain-of-Thought (CoT)

```python
prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

A: Let's think step by step.
Roger started with 5 balls.
2 cans x 3 balls per can = 6 balls.
5 + 6 = 11 balls.
The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many do they have?

A: Let's think step by step."""

# Model will generate reasoning steps before answer
```

**Boost:** 2-3x improvement on complex reasoning tasks.

### Self-Consistency

```python
# Generate multiple reasoning paths
responses = [model.generate(cot_prompt, temperature=0.7) for _ in range(5)]

# Extract answers
answers = [extract_answer(r) for r in responses]

# Majority vote
final_answer = most_common(answers)
```

### Tree of Thoughts (ToT)

```python
# Explore multiple reasoning branches
def tree_of_thoughts(prompt, depth=3):
    if depth == 0:
        return model.generate(prompt)

    # Generate multiple next thoughts
    thoughts = [model.generate(f"{prompt}\nThought: ") for _ in range(3)]

    # Evaluate each thought
    scores = [evaluate_thought(prompt, t) for t in thoughts]

    # Recursively explore best thoughts
    best_thought = thoughts[argmax(scores)]
    return tree_of_thoughts(f"{prompt}\n{best_thought}", depth-1)
```

---

## 26.8 Modern LLM Architectures (2025)

### GPT-4 / GPT-4o

**Architecture:** Decoder-only transformer (details proprietary)
**Capabilities:**
- Multimodal (text, image, audio input)
- 128K context window
- Improved reasoning
- Function calling

**Unique Features:**
- Vision understanding
- Real-time voice interaction (4o)
- Structured output generation

### Claude 3.5 Sonnet

**Architecture:** Decoder-only transformer
**Capabilities:**
- 200K context window (largest in 2025)
- Strong coding abilities
- Better at following complex instructions
- Reduced hallucinations

**Unique Features:**
- Extended context without performance degradation
- Constitutional AI alignment

### Gemini 2.0

**Architecture:** Multimodal from ground up
**Capabilities:**
- Native multimodal (not bolt-on vision)
- Efficient inference
- Strong reasoning

### LLaMA 2/3

**Architecture:** Decoder-only, open-weights
**Sizes:** 7B, 13B, 70B parameters
**Training:** 2T tokens

**Advantages:**
- Open-source
- Efficient (can run locally)
- Strong performance for size

### Mistral/Mixtral

**Architecture:** Mixture of Experts (MoE)
**Innovation:** Sparse activation (only some experts active)

```python
# MoE layer
class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts=8, top_k=2):
        self.experts = nn.ModuleList([FFN() for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # Router: which experts to use
        gate_logits = self.gate(x)
        top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1).indices

        # Combine top-k experts
        output = sum(self.experts[i](x) for i in top_k_indices)
        return output / self.top_k
```

**Benefits:**
- Large model capacity (8 experts)
- Low compute (only 2 active)
- Better than dense models at same compute

---

## 26.9 Context Length Extensions

### Problem

Standard transformers: O(n^2) attention complexity.

Long sequences --> memory explosion.

### Solutions

#### 1. ALiBi (Attention with Linear Biases)

```python
# Add position bias to attention scores
def alibi_bias(seq_len, num_heads):
    slopes = 2 ** (-8 / num_heads * torch.arange(1, num_heads + 1))
    positions = torch.arange(seq_len)
    bias = -slopes[:, None] * positions[None, :]
    return bias

attention_scores = Q @ K.T + alibi_bias(seq_len, num_heads)
```

**Benefit:** Extrapolates to longer sequences at inference.

#### 2. Sparse Attention

Only attend to subset of positions (local + global).

#### 3. Flash Attention

Memory-efficient attention implementation (I/O optimization).

**2025 Standard:** Most LLMs use Flash Attention 2+.

#### 4. Extended Context Models

- GPT-4: 128K tokens
- Claude 2: 100K --> Claude 3: 200K
- Gemini: 1M+ tokens (2024)

---

## 26.10 Inference Optimization

### KV-Cache

**Problem:** Recompute past keys/values at each generation step.

**Solution:** Cache them.

```python
class CachedAttention:
    def __init__(self):
        self.kv_cache = []

    def forward(self, q, k, v, use_cache=True):
        if use_cache and len(self.kv_cache) > 0:
            # Concatenate with cached k, v
            k = torch.cat([self.kv_cache[0], k], dim=1)
            v = torch.cat([self.kv_cache[1], v], dim=1)

        # Store for next iteration
        self.kv_cache = [k, v]

        # Compute attention
        scores = q @ k.transpose(-2, -1)
        return softmax(scores) @ v
```

**Speedup:** 10-100x faster generation.

### Quantization

**8-bit quantization (bitsandbytes):**
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)
```

**4-bit quantization (GPTQ, AWQ):**
- 4x memory reduction
- <5% performance drop
- Enables 70B models on consumer GPUs

### Speculative Decoding

**Idea:** Use small model to draft, large model to verify.

```python
def speculative_decoding(large_model, small_model, prompt, k=5):
    # Small model generates k tokens (fast)
    draft = small_model.generate(prompt, max_new_tokens=k)

    # Large model scores all at once (parallel)
    scores = large_model.score(prompt + draft)

    # Accept tokens with high agreement
    for i, (draft_token, score) in enumerate(zip(draft, scores)):
        if score[draft_token] > threshold:
            prompt += draft_token
        else:
            # Resample from large model and stop
            prompt += large_model.sample(scores[i])
            break

    return prompt
```

**Speedup:** 2-3x with no quality loss.

---

## 26.11 Practical Usage (2025)

### Via APIs

```python
from openai import OpenAI

client = OpenAI(api_key="...")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Local Deployment (Hugging Face)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically distribute across GPUs
    load_in_8bit=True   # Quantize for efficiency
)

prompt = "Explain the theory of relativity in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### vLLM (Production Serving)

```python
from vllm import LLM, SamplingParams

# Initialize
llm = LLM(model="meta-llama/Llama-2-70b", tensor_parallel_size=4)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=500
)

# Generate (batched, optimized)
prompts = ["Prompt 1", "Prompt 2", ...]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

**vLLM Features:**
- PagedAttention (efficient memory)
- Continuous batching
- 2-10x faster than Hugging Face
- Production standard in 2025

---

## 26.12 Limitations and Challenges

### Current Limitations (2025)

**1. Hallucinations**
- Generate plausible but false information
- Mitigation: RAG, citations, human oversight

**2. Context Length**
- Still limited (128K-200K tokens)
- Can't process very long documents
- "Lost in the middle" problem

**3. Reasoning**
- Struggle with multi-step logic
- Can't verify own answers
- Improving with o1-style models

**4. Computational Cost**
- Expensive to train ($100M+ for GPT-4 class)
- Expensive to run (inference costs)

**5. Data Contamination**
- Trained on public data (including benchmarks)
- May memorize rather than generalize

### Safety Concerns

- Misuse (disinformation, spam)
- Bias and fairness
- Privacy (training data memorization)
- Alignment challenges (goals != human values)

---

## 26.13 Future Directions

### 2025+ Trends

**1. Multimodal Native**
- Not text + vision bolted on
- True joint understanding

**2. Agentic LLMs**
- Tool use
- Multi-step planning
- Autonomous task completion

**3. Reasoning-First**
- Explicitly train for reasoning (OpenAI o1)
- Verifiable outputs

**4. Smaller, Specialized Models**
- Domain-specific LLMs
- Efficient edge deployment

**5. Improved Alignment**
- Constitutional AI
- Scalable oversight
- Debate and recursive reward modeling

---

## Resources

### Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (GPT-3, Brown et al., 2020)
- "Training language models to follow instructions with human feedback" (InstructGPT, Ouyang et al., 2022)
- "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)

### APIs
- OpenAI: https://platform.openai.com/
- Anthropic Claude: https://www.anthropic.com/
- Google Gemini: https://ai.google.dev/

### Open-Source
- Hugging Face Transformers: https://huggingface.co/
- vLLM: https://github.com/vllm-project/vllm
- LLaMA: https://ai.meta.com/llama/

### Tools
- LangChain: https://www.langchain.com/
- LlamaIndex: https://www.llamaindex.ai/
- Guidance: https://github.com/guidance-ai/guidance
