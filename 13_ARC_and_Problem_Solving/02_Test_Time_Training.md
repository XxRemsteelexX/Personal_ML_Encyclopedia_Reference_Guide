# Test-Time Training for ARC - The 2024 Breakthrough

## Overview

**Test-Time Training (TTT)** emerged as the single most important technique for ARC challenge in 2024, enabling performance jumps from ~30% to 40%+ on individual models.

**Key Insight:** Instead of using a single pre-trained model for all tasks, adapt the model to each specific task using its training examples.

**Popularized by:** Jack Cole & Mohamed Osman (2023), Akyurek et al. (2024)

---

## Why Test-Time Training Works for ARC

### The Core Problem

**Traditional Approach:**
```python
# Train once on all ARC tasks
model = train_on_all_tasks(arc_dataset)

# Use same model for every test task
for task in test_tasks:
    output = model.predict(task.test_input)
```

**Problem:** Each ARC task has a unique transformation rule. A single model tries to memorize 400+ different rules.

---

### The TTT Solution

**Adaptive Approach:**
```python
# Pre-train on all ARC tasks
base_model = pretrain_on_all_tasks(arc_dataset)

# For each test task, adapt specifically
for task in test_tasks:
    # Fine-tune on THIS task's training examples
    task_model = fine_tune(base_model,
                           task.train_inputs,
                           task.train_outputs,
                           steps=300)

    # Now predict with task-specific model
    output = task_model.predict(task.test_input)
```

**Advantage:** Model can focus on learning the specific transformation rule for THIS task only.

---

## TTT Implementation: The Omni-ARC Recipe

### Architecture

**Base Model:** Qwen2.5-0.5B-Instruct (small, efficient)
- 500M parameters (tiny by LLM standards!)
- Instruction-tuned for following prompts
- Fast enough for test-time adaptation

**Why small models?**
- Need to fine-tune 400+ times (once per test task)
- Smaller = faster iteration
- Less prone to overfitting with 3-5 examples

---

### Pre-Training Phase

**Data:**
- 1,420 unique ARC tasks from public datasets
- Extensive augmentation (see below)
- Multiple task formats

**Configuration:**
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# LoRA configuration
lora_config = LoraConfig(
    r=128,              # Rank (higher = more capacity)
    lora_alpha=256,     # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)

# Training configuration
training_args = TrainingArguments(
    output_dir="./omni_arc_pretrain",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=10,  # ~200,000 steps total
    warmup_steps=1000,
    logging_steps=100,
    save_steps=1000,
    fp16=True  # Mixed precision for speed
)
```

**Training Time:** ~48 hours on 2x A6000 GPUs (48GB VRAM each)

---

### Data Augmentation (Critical!)

```python
import numpy as np

def augment_arc_task(task):
    """Generate multiple variations of an ARC task"""
    augmented_tasks = []

    # 1. Rotations
    for angle in [90, 180, 270]:
        rotated = {
            'train': [rotate_grid(ex, angle) for ex in task['train']],
            'test': [rotate_grid(ex, angle) for ex in task['test']]
        }
        augmented_tasks.append(rotated)

    # 2. Reflections
    for flip_type in ['horizontal', 'vertical']:
        flipped = {
            'train': [flip_grid(ex, flip_type) for ex in task['train']],
            'test': [flip_grid(ex, flip_type) for ex in task['test']]
        }
        augmented_tasks.append(flipped)

    # 3. Color permutations (huge impact!)
    for _ in range(5):  # Generate 5 random color remappings
        color_map = np.random.permutation(10)  # Colors 0-9
        remapped = {
            'train': [remap_colors(ex, color_map) for ex in task['train']],
            'test': [remap_colors(ex, color_map) for ex in task['test']]
        }
        augmented_tasks.append(remapped)

    # 4. Train/test swap (learn bidirectional rules)
    swapped = {
        'train': task['test'],
        'test': task['train']
    }
    augmented_tasks.append(swapped)

    return augmented_tasks

def rotate_grid(example, angle):
    """Rotate input and output grids"""
    input_grid = np.array(example['input'])
    output_grid = np.array(example['output'])

    k = angle // 90
    return {
        'input': np.rot90(input_grid, k).tolist(),
        'output': np.rot90(output_grid, k).tolist()
    }

def flip_grid(example, flip_type):
    """Flip grids horizontally or vertically"""
    input_grid = np.array(example['input'])
    output_grid = np.array(example['output'])

    if flip_type == 'horizontal':
        return {
            'input': np.fliplr(input_grid).tolist(),
            'output': np.fliplr(output_grid).tolist()
        }
    else:  # vertical
        return {
            'input': np.flipud(input_grid).tolist(),
            'output': np.flipud(output_grid).tolist()
        }

def remap_colors(example, color_map):
    """Change color values while preserving structure"""
    def apply_map(grid):
        return [[color_map[cell] for cell in row] for row in grid]

    return {
        'input': apply_map(example['input']),
        'output': apply_map(example['output'])
    }
```

**Impact:** Without augmentation ~25%, with augmentation ~40%

---

### Multi-Task Training

**Insight:** Don't just train on "input --> output". Train on multiple ARC-related tasks!

**Task 1: Output Generation (Primary)**
```python
def format_output_generation(task):
    """Standard ARC task: predict output from input"""
    prompt = "Given the following examples:\n\n"

    for i, example in enumerate(task['train']):
        prompt += f"Example {i+1}:\n"
        prompt += f"Input:\n{grid_to_string(example['input'])}\n"
        prompt += f"Output:\n{grid_to_string(example['output'])}\n\n"

    prompt += "Now predict the output for:\n"
    prompt += f"Input:\n{grid_to_string(task['test'][0]['input'])}\n"
    prompt += "Output:\n"

    target = grid_to_string(task['test'][0]['output'])

    return {"prompt": prompt, "target": target}
```

**Task 2: Input Distribution Learning**
```python
def format_input_generation(task):
    """Generate plausible input grids (helps model understand patterns)"""
    prompt = "Generate a new input grid that fits the pattern:\n\n"

    for i, example in enumerate(task['train']):
        prompt += f"Example {i+1} Input:\n{grid_to_string(example['input'])}\n\n"

    prompt += "New input:\n"

    # Target: Use one of the training inputs as the target
    target = grid_to_string(task['train'][-1]['input'])

    return {"prompt": prompt, "target": target}
```

**Task 3: Verification (Experimental)**
```python
def format_verification(task, candidate_output):
    """Verify if a proposed output is correct"""
    prompt = "Given these examples, is the proposed output correct?\n\n"

    for i, example in enumerate(task['train']):
        prompt += f"Example {i+1}:\n"
        prompt += f"Input: {grid_to_string(example['input'])}\n"
        prompt += f"Output: {grid_to_string(example['output'])}\n\n"

    prompt += f"Test Input: {grid_to_string(task['test'][0]['input'])}\n"
    prompt += f"Proposed Output: {grid_to_string(candidate_output)}\n"
    prompt += "Is this correct? (Yes/No):\n"

    target = "Yes" if candidate_output == task['test'][0]['output'] else "No"

    return {"prompt": prompt, "target": target}
```

**Why multi-task helps:**
- Model learns deeper understanding of ARC structure
- Input generation forces learning of "what makes a valid ARC input"
- Verification teaches error detection
- Shared representations improve all tasks

---

## Test-Time Fine-Tuning Phase

### The Critical Step: Task-Specific Adaptation

**For each test task:**

```python
def test_time_finetune(base_model, task, n_steps=300):
    """
    Fine-tune model on task's training examples

    Args:
        base_model: Pre-trained Omni-ARC model
        task: Single ARC task with train/test examples
        n_steps: Number of fine-tuning steps

    Returns:
        task_model: Model adapted to this specific task
    """
    # Clone base model
    task_model = clone_model(base_model)

    # Use n-1 training examples for fine-tuning
    # (hold out 1 for validation if needed)
    train_examples = task['train'][:-1]
    val_example = task['train'][-1]

    # Create fine-tuning dataset
    finetune_data = []
    for example in train_examples:
        # Repeat each example multiple times
        for _ in range(100):  # ~300 steps / 3 examples = 100 repeats
            formatted = format_output_generation_single(example)
            finetune_data.append(formatted)

    # Fine-tuning configuration
    finetune_args = TrainingArguments(
        output_dir=f"./ttt_{task['id']}",
        learning_rate=8e-5,  # Higher LR for quick adaptation
        per_device_train_batch_size=1,  # Small batch (only 3-5 examples!)
        num_train_epochs=1,
        max_steps=n_steps,
        logging_steps=50,
        save_strategy="no",  # Don't save intermediate checkpoints
        warmup_steps=0,  # No warmup for TTT
        fp16=True
    )

    # Fine-tune
    trainer = Trainer(
        model=task_model,
        args=finetune_args,
        train_dataset=finetune_data
    )

    trainer.train()

    return task_model

def format_output_generation_single(example):
    """Format a single example for fine-tuning"""
    prompt = "Input:\n" + grid_to_string(example['input']) + "\nOutput:\n"
    target = grid_to_string(example['output'])
    return {"prompt": prompt, "target": target}
```

**Key Parameters:**
- **Learning rate: 8e-5** (higher than pre-training's 5e-5)
  - Need fast adaptation with few examples
  - Risk of overfitting is acceptable (task-specific model)

- **Batch size: 1**
  - Only have 3-5 training examples
  - Small batch prevents averaging out task-specific patterns

- **Steps: ~300**
  - Empirically optimal (too few = underfit, too many = overfit)
  - ~100 iterations per training example

- **No warmup**
  - Start adapting immediately
  - Already at good initialization from pre-training

**Computational Cost:**
- ~30 seconds per task on A6000
- 400 test tasks x 30 sec = ~3.3 hours total inference time

---

### Preventing Overfitting During TTT

**Problem:** Only 3-5 examples! Easy to overfit.

**Solutions:**

**1. Stop Early (Validation-Based)**
```python
def test_time_finetune_with_validation(base_model, task, max_steps=500):
    """Stop when validation performance degrades"""
    train_examples = task['train'][:-1]
    val_example = task['train'][-1]

    best_model = None
    best_val_loss = float('inf')
    patience = 50
    no_improve_count = 0

    for step in range(max_steps):
        # Train one step
        task_model = train_step(task_model, train_examples)

        # Validate every 10 steps
        if step % 10 == 0:
            val_loss = evaluate(task_model, val_example)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = clone_model(task_model)
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                break  # Early stopping

    return best_model
```

**2. Augmentation During TTT**
```python
def test_time_finetune_with_augmentation(base_model, task, n_steps=300):
    """Augment training examples during fine-tuning"""
    train_examples = task['train']

    # Generate augmented versions
    augmented = []
    for example in train_examples:
        # Original
        augmented.append(example)

        # Rotations
        for angle in [90, 180, 270]:
            augmented.append(rotate_grid(example, angle))

        # Flips
        augmented.append(flip_grid(example, 'horizontal'))
        augmented.append(flip_grid(example, 'vertical'))

    # Fine-tune on augmented data
    # (only if augmentations preserve the task rule!)
    ...
```

**[WARNING] Warning:** Only augment if transformations preserve the rule!
- Rotation works if rule is rotation-invariant
- Color swap works if rule is color-independent
- Must check task properties first

**3. Regularization**
```python
# L2 regularization to stay close to base model
finetune_args = TrainingArguments(
    ...,
    weight_decay=0.01,  # L2 penalty
)
```

**4. Small LoRA Rank**
- Use rank 8-16 for TTT (vs 128 for pre-training)
- Limits model capacity = harder to overfit

---

## Inference: Generating Predictions

### Single Prediction

```python
def generate_output(model, test_input, max_length=512):
    """Generate output grid from test input"""
    # Format prompt
    prompt = "Input:\n" + grid_to_string(test_input) + "\nOutput:\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.0,  # Greedy decoding (deterministic)
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse grid from text
    output_grid = string_to_grid(generated_text)

    return output_grid
```

---

### Multiple Predictions with Voting (Critical!)

**Insight:** Model has uncertainty. Generate multiple outputs and vote!

```python
def generate_with_voting(model, test_input, n_predictions=96):
    """
    Generate multiple predictions and select most common

    This is the KEY technique used by Omni-ARC for 40% performance
    """
    predictions = []

    for i in range(n_predictions):
        # Even with temperature=0, slight variations can occur
        # (due to random seed, attention dropout in some implementations)
        output = generate_output(model, test_input)
        predictions.append(output)

    # Vote: Select most common prediction
    from collections import Counter

    # Convert grids to hashable format for counting
    predictions_hashable = [grid_to_tuple(p) for p in predictions]

    # Count occurrences
    vote_counts = Counter(predictions_hashable)

    # Get top 3 predictions (ARC allows 3 attempts)
    top_3 = vote_counts.most_common(3)

    # Convert back to grid format
    top_3_grids = [tuple_to_grid(pred) for pred, count in top_3]

    return top_3_grids

def grid_to_tuple(grid):
    """Convert grid to hashable tuple"""
    return tuple(tuple(row) for row in grid)

def tuple_to_grid(tup):
    """Convert tuple back to grid"""
    return [list(row) for row in tup]
```

**Why 96 predictions?**
- Empirically optimal (diminishing returns after ~100)
- Balances coverage vs computational cost
- Increases chance of finding correct answer

**Why voting works:**
- Model is uncertain about exact output
- Correct answer tends to be more "stable"
- Incorrect answers are more random/diverse
- Mode of distribution ~= most likely correct answer

---

## Grid Representation Strategies

### Challenge: How to Represent Grids as Text?

**Option 1: ASCII Art**
```
Input:
###..
#.#..
###..
.....

Output:
..###
...#.
..###
.....
```

**Option 2: Structured Text**
```
Input Grid (5x4):
Row 0: [1, 1, 1, 0, 0]
Row 1: [1, 0, 1, 0, 0]
Row 2: [1, 1, 1, 0, 0]
Row 3: [0, 0, 0, 0, 0]

Output Grid (5x4):
Row 0: [0, 0, 1, 1, 1]
Row 1: [0, 0, 0, 1, 0]
Row 2: [0, 0, 1, 1, 1]
Row 3: [0, 0, 0, 0, 0]
```

**Option 3: JSON**
```json
{
  "input": [[1,1,1,0,0], [1,0,1,0,0], [1,1,1,0,0], [0,0,0,0,0]],
  "output": [[0,0,1,1,1], [0,0,0,1,0], [0,0,1,1,1], [0,0,0,0,0]]
}
```

**Best Practice (Omni-ARC):** Structured text with row labels
- Clear boundaries
- Easy to parse
- Model-friendly (training data likely has similar formats)

---

## Advanced TTT Techniques

### 1. Multi-Stage Fine-Tuning

```python
def multi_stage_ttt(base_model, task):
    """
    Stage 1: Coarse adaptation (learn general pattern)
    Stage 2: Fine adaptation (learn specific details)
    """
    # Stage 1: High LR, more examples
    stage1_model = fine_tune(
        base_model,
        task['train'],
        learning_rate=1e-4,
        steps=200
    )

    # Stage 2: Low LR, focus on hard examples
    stage2_model = fine_tune(
        stage1_model,
        task['train'],
        learning_rate=1e-5,
        steps=100
    )

    return stage2_model
```

### 2. Task Difficulty Estimation

```python
def estimate_difficulty(task):
    """Estimate task difficulty to adjust TTT parameters"""
    difficulty_score = 0

    # Large grids are harder
    max_size = max(
        max(len(ex['input']), len(ex['input'][0]))
        for ex in task['train'] + task['test']
    )
    difficulty_score += max_size / 30.0

    # More colors are harder
    all_colors = set()
    for ex in task['train']:
        all_colors.update(cell for row in ex['input'] for cell in row)
    difficulty_score += len(all_colors) / 10.0

    # Fewer training examples are harder
    difficulty_score += (5 - len(task['train'])) / 5.0

    return difficulty_score

def adaptive_ttt(base_model, task):
    """Adjust TTT based on estimated difficulty"""
    difficulty = estimate_difficulty(task)

    if difficulty < 0.3:  # Easy task
        steps = 200
        lr = 5e-5
    elif difficulty < 0.7:  # Medium task
        steps = 300
        lr = 8e-5
    else:  # Hard task
        steps = 500
        lr = 1e-4

    return fine_tune(base_model, task['train'],
                     learning_rate=lr,
                     steps=steps)
```

### 3. Example Selection (n-1 Training)

```python
def select_training_examples(task, n_train):
    """Choose which examples to fine-tune on"""
    if len(task['train']) <= n_train:
        return task['train']

    # Strategy 1: Random selection
    import random
    return random.sample(task['train'], n_train)

    # Strategy 2: Diverse selection (better!)
    # Select examples that are most different from each other
    selected = []
    remaining = task['train'].copy()

    # Start with a random example
    selected.append(remaining.pop(random.randint(0, len(remaining)-1)))

    while len(selected) < n_train and remaining:
        # Find example most different from already selected
        max_diff = -1
        max_idx = 0

        for i, candidate in enumerate(remaining):
            min_similarity = min(
                similarity(candidate, sel) for sel in selected
            )
            if min_similarity > max_diff:
                max_diff = min_similarity
                max_idx = i

        selected.append(remaining.pop(max_idx))

    return selected
```

---

## Performance Analysis

### Omni-ARC Results (Test-Time Training)

**Configuration:**
- Base model: Qwen2.5-0.5B (500M params)
- Pre-training: 200K steps, 2x A6000, 48 hours
- TTT: 300 steps per task, ~30 sec per task
- Inference: 96 predictions + voting

**Performance:**
- **Without TTT:** ~25% accuracy
- **With TTT:** ~40% accuracy
- **Improvement:** +60% relative gain!

**Breakdown by Grid Size:**
| Grid Size | Without TTT | With TTT | Improvement |
|-----------|-------------|----------|-------------|
| <= 8x8     | 35%         | 55%      | +57%        |
| 8x8 - 15x15 | 22%       | 38%      | +73%        |
| > 15x15   | 12%         | 22%      | +83%        |

**Key Insight:** TTT helps more on harder (larger) tasks!

---

## Why TTT is So Effective

### 1. Task-Specific Specialization
- Base model must remember 400+ transformation rules
- TTT model only needs to learn 1 rule
- Focused learning --> better performance

### 2. Overcomes Data Scarcity
- Pre-training sees thousands of examples
- TTT squeezes maximum information from 3-5 examples
- Augmentation amplifies limited data

### 3. Adaptation to Task Distribution
- Test input might differ slightly from training distribution
- TTT can adapt to these task-specific quirks

### 4. Implicit Bayesian Inference
- Base model provides prior knowledge
- TTT performs Bayesian update given task evidence
- Posterior model is optimally adapted

---

## Limitations and Challenges

### 1. Computational Cost
- **400 tasks x 300 steps x 30 sec = 3.3 hours** inference time
- Compare to: standard model = 400 tasks x 0.1 sec = 40 seconds
- **100x slower!**

**Mitigation:**
- Parallelize across GPUs
- Use smaller models (500M vs 7B)
- Optimize implementation (vLLM, TensorRT)

### 2. Overfitting Risk
- Only 3-5 examples --> very easy to overfit
- Model might memorize training examples exactly
- Fails to generalize to test input if it's different

**Mitigation:**
- Early stopping with validation
- Regularization (weight decay, dropout)
- Augmentation
- Multiple predictions + voting

### 3. Negative Transfer
- Sometimes TTT makes model worse!
- If pre-training was good, TTT might degrade
- Need to detect and skip TTT for some tasks

**Mitigation:**
```python
def selective_ttt(base_model, task):
    """Only apply TTT if it helps"""
    # Generate with base model
    base_prediction = generate_output(base_model, task['test'][0]['input'])

    # Generate with TTT model
    ttt_model = test_time_finetune(base_model, task)
    ttt_prediction = generate_output(ttt_model, task['test'][0]['input'])

    # If available, validate on held-out training example
    if len(task['train']) > 3:
        val_example = task['train'][-1]
        base_val_loss = evaluate(base_model, val_example)
        ttt_val_loss = evaluate(ttt_model, val_example)

        if ttt_val_loss < base_val_loss:
            return ttt_prediction
        else:
            return base_prediction

    # Otherwise, return both as candidate answers
    return [ttt_prediction, base_prediction]
```

---

## Production Implementation

### Complete TTT Pipeline

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import numpy as np
from collections import Counter

class ARCTestTimeTrainer:
    def __init__(self, base_model_path, device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.base_model = get_peft_model(self.base_model, lora_config)

    def finetune_on_task(self, task, n_steps=300, lr=8e-5):
        """Fine-tune on a single ARC task"""
        # Clone model for task-specific adaptation
        task_model = clone_model(self.base_model)

        # Prepare training data
        train_data = []
        for example in task['train']:
            prompt = self._format_example(example['input'])
            target = self._format_grid(example['output'])
            train_data.append({"prompt": prompt, "target": target})

        # Fine-tune
        args = TrainingArguments(
            output_dir=f"./ttt_temp",
            learning_rate=lr,
            per_device_train_batch_size=1,
            max_steps=n_steps,
            save_strategy="no",
            logging_steps=100,
            fp16=True
        )

        trainer = Trainer(
            model=task_model,
            args=args,
            train_dataset=train_data
        )

        trainer.train()

        return task_model

    def predict_with_voting(self, model, test_input, n_predictions=96):
        """Generate multiple predictions and vote"""
        predictions = []

        for _ in range(n_predictions):
            output = self._generate_output(model, test_input)
            predictions.append(output)

        # Vote
        predictions_hashable = [self._grid_to_tuple(p) for p in predictions]
        vote_counts = Counter(predictions_hashable)
        top_3 = vote_counts.most_common(3)

        return [self._tuple_to_grid(pred) for pred, count in top_3]

    def solve_task(self, task):
        """Complete TTT pipeline for a single task"""
        # Fine-tune
        task_model = self.finetune_on_task(task)

        # Generate predictions for all test inputs
        all_predictions = []
        for test_input in [ex['input'] for ex in task['test']]:
            predictions = self.predict_with_voting(task_model, test_input)
            all_predictions.append(predictions)

        return all_predictions

    def _format_example(self, grid):
        """Convert grid to text prompt"""
        text = "Input:\n"
        for i, row in enumerate(grid):
            text += f"Row {i}: {row}\n"
        text += "Output:\n"
        return text

    def _format_grid(self, grid):
        """Convert grid to target text"""
        text = ""
        for i, row in enumerate(grid):
            text += f"Row {i}: {row}\n"
        return text

    def _generate_output(self, model, test_input):
        """Generate single output"""
        prompt = self._format_example(test_input)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_grid(generated_text)

    def _parse_grid(self, text):
        """Parse grid from generated text"""
        # Implementation depends on format
        # ... parsing logic ...
        pass

    def _grid_to_tuple(self, grid):
        return tuple(tuple(row) for row in grid)

    def _tuple_to_grid(self, tup):
        return [list(row) for row in tup]

# Usage
trainer = ARCTestTimeTrainer("Qwen/Qwen2.5-0.5B-Instruct")

# Solve a task
predictions = trainer.solve_task(arc_task)
print(f"Top 3 predictions: {predictions}")
```

---

## Key Takeaways

1. **TTT is essential** for ARC: +60% relative improvement
2. **Small models work** better than large models (500M vs 7B)
3. **Augmentation is critical** during pre-training
4. **300 steps is optimal** for TTT fine-tuning
5. **Voting dramatically helps**: 96 predictions --> mode
6. **Multi-task training** improves base model quality
7. **Computational cost** is the main limitation (100x slower)

**Next:** `03_Program_Synthesis_for_ARC.md` - How to search for executable programs that solve ARC tasks
