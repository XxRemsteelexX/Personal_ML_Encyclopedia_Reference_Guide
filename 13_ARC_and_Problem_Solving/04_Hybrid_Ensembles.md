# Hybrid Ensembles for ARC - Achieving 55%+ Accuracy

## Overview

**The 2024 Breakthrough:** No single method solves ARC. The key is intelligent ensembling.

**Finding:** Transduction-only OR induction-only approaches both cap at ~40%. Combining them reaches 55.5%+.

**Top teams used:**
- Test-time training (transduction)
- LLM-based reasoning (induction)
- Program synthesis (induction)
- Classic hand-coded solvers (hybrid)
- Intelligent voting/selection

---

## Why Ensembles Work

### Complementary Strengths

| Method | Strength | Weakness |
|--------|----------|----------|
| **Test-Time Training** | Pattern recognition, fast | Limited abstraction, needs similar patterns |
| **LLM Reasoning** | Abstract rules, code generation | Execution errors, hallucination |
| **Program Synthesis** | Precise logic, generalizable | Expensive search, limited creativity |
| **Hand-Coded Rules** | Perfect on known patterns | Doesn't generalize |

**Ensemble Strategy:** Use each method's strengths, avoid weaknesses.

---

## Transduction vs Induction

### Transduction: Direct Mapping

**Approach:** Learn input --> output mapping directly

**Example Methods:**
- Test-time fine-tuned models
- Nearest neighbor matching
- Neural network prediction

**Characteristics:**
-  Fast inference
-  Good at pattern matching
-  Poor abstraction
-  Struggles with novel patterns

**Performance:** ~40% alone

---

### Induction: Rule Discovery

**Approach:** Infer underlying transformation rule

**Example Methods:**
- Program synthesis
- LLM code generation
- Symbolic reasoning

**Characteristics:**
-  Better generalization
-  Explicit reasoning
-  Slow search
-  Hard to represent all rules

**Performance:** ~40% alone

---

### The Hybrid Solution

**Insight:** Different tasks favor different approaches

```python
def hybrid_solve(task):
    """Combine transduction and induction"""

    # Transduction: Test-time training
    ttt_predictions = test_time_training(task)

    # Induction: LLM reasoning
    llm_predictions = llm_reasoning(task)

    # Induction: Program synthesis
    program_predictions = program_synthesis(task)

    # Combine
    all_predictions = ttt_predictions + llm_predictions + program_predictions

    # Vote or intelligently select
    final_prediction = select_best(all_predictions, task)

    return final_prediction
```

**Performance:** ~55%+ ensemble

---

## The ARChitects Ensemble Architecture

### Multi-Stage Pipeline

```
Input Task
    v
+-----------------------------------------+
|  Stage 1: Quick Heuristics          |
|  - Hand-coded rules for common      |
|    patterns (rotation, flip, etc.)  |
|  - Exact match to training data     |
+-----------------------------------------+
    v (if no solution)
+-----------------------------------------+
|  Stage 2: Test-Time Training        |
|  - Fine-tune on task                |
|  - Generate 96 predictions          |
|  - Vote for top 3                   |
+-----------------------------------------+
    v (in parallel)
+-----------------------------------------+
|  Stage 3: LLM Reasoning              |
|  - Multi-perspective prompting      |
|  - Code generation + verification   |
|  - Self-correction loop             |
+-----------------------------------------+
    v (in parallel)
+-----------------------------------------+
|  Stage 4: Program Synthesis          |
|  - Search in DSL space              |
|  - Verify on training examples      |
|  - Return verified programs         |
+-----------------------------------------+
    v
+-----------------------------------------+
|  Stage 5: Ensemble Selection         |
|  - Collect all candidates           |
|  - Score by confidence/verification |
|  - Return top 3                     |
+-----------------------------------------+
    v
Final Predictions (3 attempts allowed)
```

---

## Implementation: Hybrid Solver

### Architecture

```python
import numpy as np
from collections import Counter
from typing import List, Dict, Any

class HybridARCSolver:
    """
    State-of-the-art hybrid ensemble for ARC
    Combines multiple solving strategies
    """

    def __init__(self):
        # Initialize all solvers
        self.ttt_solver = TestTimeTrainingSolver()
        self.llm_solver = LLMARCSolver()
        self.program_solver = ProgramSynthesisSolver()
        self.heuristic_solver = HeuristicSolver()

    def solve(self, task: Dict[str, Any]) -> List[Any]:
        """
        Solve ARC task using hybrid ensemble

        Args:
            task: ARC task with 'train' and 'test' examples

        Returns:
            List of top 3 predictions (ARC allows 3 attempts)
        """
        all_candidates = []

        # Stage 1: Quick heuristics (fast, high precision on simple tasks)
        print("Stage 1: Trying heuristics...")
        heuristic_candidates = self._try_heuristics(task)
        if heuristic_candidates:
            all_candidates.extend([
                {'prediction': c, 'source': 'heuristic', 'confidence': 1.0}
                for c in heuristic_candidates
            ])

        # Stage 2: Test-time training (slow, good for pattern matching)
        print("Stage 2: Test-time training...")
        ttt_candidates = self._try_ttt(task)
        all_candidates.extend([
            {'prediction': c, 'source': 'ttt', 'confidence': 0.8}
            for c in ttt_candidates
        ])

        # Stage 3: LLM reasoning (parallel with stage 2)
        print("Stage 3: LLM reasoning...")
        llm_candidates = self._try_llm(task)
        all_candidates.extend([
            {'prediction': c, 'source': 'llm', 'confidence': 0.7}
            for c in llm_candidates
        ])

        # Stage 4: Program synthesis (parallel, slow)
        print("Stage 4: Program synthesis...")
        program_candidates = self._try_program_synthesis(task)
        all_candidates.extend([
            {'prediction': c, 'source': 'program', 'confidence': 0.9}
            for c in program_candidates
        ])

        # Stage 5: Intelligent selection
        print("Stage 5: Selecting best predictions...")
        top_3 = self._select_top_predictions(all_candidates, task)

        return top_3

    def _try_heuristics(self, task):
        """Try hand-coded heuristics for common patterns"""
        candidates = []

        test_input = task['test'][0]['input']

        # Heuristic 1: Identity (output = input)
        if self._verify_on_training(lambda x: x, task['train']):
            candidates.append(test_input)

        # Heuristic 2: Rotations
        for k in [1, 2, 3]:  # 90, 180, 270 degrees
            transform = lambda x, k=k: np.rot90(np.array(x), k=k).tolist()
            if self._verify_on_training(transform, task['train']):
                candidates.append(transform(test_input))

        # Heuristic 3: Flips
        for flip_func in [np.fliplr, np.flipud]:
            transform = lambda x, f=flip_func: f(np.array(x)).tolist()
            if self._verify_on_training(transform, task['train']):
                candidates.append(transform(test_input))

        # Heuristic 4: Color replacements
        # (only if simple pattern detected)
        color_map = self._detect_color_mapping(task['train'])
        if color_map:
            transform = lambda x: self._apply_color_map(x, color_map)
            if self._verify_on_training(transform, task['train']):
                candidates.append(transform(test_input))

        return candidates[:3]  # Return top 3

    def _try_ttt(self, task):
        """Test-time training approach"""
        return self.ttt_solver.solve(task, n_predictions=96)[:3]

    def _try_llm(self, task):
        """LLM-based reasoning"""
        return self.llm_solver.solve_task(task, strategy='verification', max_attempts=5)

    def _try_program_synthesis(self, task):
        """Program synthesis approach"""
        return self.program_solver.search(task, max_programs=100, timeout=60)[:3]

    def _verify_on_training(self, transform_func, train_examples):
        """Verify transformation works on all training examples"""
        try:
            for example in train_examples:
                predicted = transform_func(example['input'])
                expected = example['output']
                if not np.array_equal(predicted, expected):
                    return False
            return True
        except:
            return False

    def _detect_color_mapping(self, train_examples):
        """Detect if task is simple color replacement"""
        # Check if all examples have same input/output shape
        if not all(
            len(ex['input']) == len(ex['output']) and
            len(ex['input'][0]) == len(ex['output'][0])
            for ex in train_examples
        ):
            return None

        # Try to find consistent color mapping
        color_map = {}
        for example in train_examples:
            for i in range(len(example['input'])):
                for j in range(len(example['input'][0])):
                    in_color = example['input'][i][j]
                    out_color = example['output'][i][j]

                    if in_color in color_map:
                        if color_map[in_color] != out_color:
                            return None  # Inconsistent mapping
                    else:
                        color_map[in_color] = out_color

        return color_map if color_map else None

    def _apply_color_map(self, grid, color_map):
        """Apply color mapping to grid"""
        return [[color_map.get(cell, cell) for cell in row] for row in grid]

    def _select_top_predictions(self, candidates, task):
        """
        Intelligently select top 3 predictions from all candidates

        Scoring criteria:
        1. Source confidence (heuristic > program > ttt > llm)
        2. Verification on training data
        3. Frequency across methods (voting)
        4. Consistency checks
        """
        if not candidates:
            return []

        # Remove duplicates while tracking sources
        unique_predictions = {}
        for cand in candidates:
            pred_key = self._grid_to_key(cand['prediction'])

            if pred_key not in unique_predictions:
                unique_predictions[pred_key] = {
                    'prediction': cand['prediction'],
                    'sources': [cand['source']],
                    'confidences': [cand['confidence']],
                    'count': 1
                }
            else:
                unique_predictions[pred_key]['sources'].append(cand['source'])
                unique_predictions[pred_key]['confidences'].append(cand['confidence'])
                unique_predictions[pred_key]['count'] += 1

        # Score each unique prediction
        scored_predictions = []
        for pred_key, pred_info in unique_predictions.items():
            score = self._score_prediction(pred_info, task)
            scored_predictions.append({
                'prediction': pred_info['prediction'],
                'score': score,
                'sources': pred_info['sources'],
                'count': pred_info['count']
            })

        # Sort by score
        scored_predictions.sort(key=lambda x: x['score'], reverse=True)

        # Return top 3 predictions
        return [p['prediction'] for p in scored_predictions[:3]]

    def _score_prediction(self, pred_info, task):
        """
        Score a prediction based on multiple factors

        Scoring:
        - Base confidence (from source)
        - Frequency bonus (appears in multiple methods)
        - Diversity bonus (from different method types)
        - Verification bonus (if verifiable on training)
        """
        # Base score: average confidence
        base_score = np.mean(pred_info['confidences'])

        # Frequency bonus: appears multiple times
        frequency_bonus = min(pred_info['count'] * 0.1, 0.5)

        # Diversity bonus: from different sources
        unique_sources = len(set(pred_info['sources']))
        diversity_bonus = unique_sources * 0.15

        # Source priority bonus
        source_priority = {
            'heuristic': 0.3,  # Highest (if heuristic works, it's usually right)
            'program': 0.2,    # High (verified programs are reliable)
            'ttt': 0.1,        # Medium
            'llm': 0.05        # Lower (LLMs can hallucinate)
        }
        priority_bonus = max(source_priority.get(s, 0) for s in pred_info['sources'])

        # Total score
        total_score = base_score + frequency_bonus + diversity_bonus + priority_bonus

        return total_score

    def _grid_to_key(self, grid):
        """Convert grid to hashable key for deduplication"""
        return str(grid)  # Simple string representation


# Utility functions used by solvers

class TestTimeTrainingSolver:
    """Wrapper for TTT solver (from 02_Test_Time_Training.md)"""
    def solve(self, task, n_predictions=96):
        # Implementation from previous file
        # Returns list of predictions
        pass

class LLMARCSolver:
    """Wrapper for LLM solver (from 03_LLM_Based_Reasoning.md)"""
    def solve_task(self, task, strategy='verification', max_attempts=5):
        # Implementation from previous file
        pass

class ProgramSynthesisSolver:
    """Program synthesis solver (simplified)"""
    def search(self, task, max_programs=100, timeout=60):
        # Search for programs that solve the task
        # Returns list of outputs from verified programs
        pass

class HeuristicSolver:
    """Hand-coded heuristics for common patterns"""
    pass
```

---

## Advanced Ensemble Techniques

### 1. Weighted Voting

```python
def weighted_vote(predictions, weights):
    """
    Vote with weights based on method performance

    Args:
        predictions: List of (prediction, method) tuples
        weights: Dict of {method: weight}

    Returns:
        Top prediction by weighted vote
    """
    vote_counts = {}

    for pred, method in predictions:
        pred_key = grid_to_key(pred)
        weight = weights.get(method, 1.0)

        if pred_key in vote_counts:
            vote_counts[pred_key]['votes'] += weight
            vote_counts[pred_key]['prediction'] = pred
        else:
            vote_counts[pred_key] = {
                'votes': weight,
                'prediction': pred
            }

    # Select prediction with most weighted votes
    best = max(vote_counts.values(), key=lambda x: x['votes'])
    return best['prediction']

# Example weights (based on empirical performance)
method_weights = {
    'heuristic': 2.0,   # High weight if verified
    'program': 1.5,     # High reliability
    'ttt': 1.0,         # Baseline
    'llm': 0.7          # Lower (more errors)
}
```

---

### 2. Stacking (Meta-Learning)

```python
def train_meta_model(validation_tasks):
    """
    Train a meta-model to select best prediction

    Inputs: Features from all methods
    Output: Probability each prediction is correct
    """
    # Collect meta-features
    meta_features = []
    meta_labels = []

    for task in validation_tasks:
        # Run all base solvers
        ttt_pred = ttt_solver.solve(task)
        llm_pred = llm_solver.solve(task)
        prog_pred = program_solver.solve(task)

        # Extract features
        features = {
            'ttt_confidence': ttt_solver.get_confidence(),
            'llm_verified': llm_solver.verified_on_train(task),
            'program_verified': program_solver.verified_on_train(task),
            'agreement_count': count_agreements([ttt_pred, llm_pred, prog_pred]),
            'task_difficulty': estimate_difficulty(task),
            # ... more features
        }

        # Label: which method was correct?
        ground_truth = task['test'][0]['output']
        labels = {
            'ttt': (ttt_pred == ground_truth),
            'llm': (llm_pred == ground_truth),
            'program': (prog_pred == ground_truth)
        }

        meta_features.append(features)
        meta_labels.append(labels)

    # Train meta-classifier (e.g., Random Forest)
    from sklearn.ensemble import RandomForestClassifier

    # One classifier per method
    meta_models = {}
    for method in ['ttt', 'llm', 'program']:
        X = np.array([list(f.values()) for f in meta_features])
        y = np.array([l[method] for l in meta_labels])

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X, y)

        meta_models[method] = clf

    return meta_models

def predict_with_stacking(task, meta_models):
    """Use meta-model to select best prediction"""
    # Run all base solvers
    predictions = {
        'ttt': ttt_solver.solve(task),
        'llm': llm_solver.solve(task),
        'program': program_solver.solve(task)
    }

    # Extract features for this task
    features = {
        'ttt_confidence': ttt_solver.get_confidence(),
        'llm_verified': llm_solver.verified_on_train(task),
        'program_verified': program_solver.verified_on_train(task),
        'agreement_count': count_agreements(list(predictions.values())),
        'task_difficulty': estimate_difficulty(task),
    }

    # Predict probability each method is correct
    X = np.array([list(features.values())])
    probs = {}
    for method, model in meta_models.items():
        probs[method] = model.predict_proba(X)[0][1]  # P(correct)

    # Select method with highest probability
    best_method = max(probs, key=probs.get)
    return predictions[best_method]
```

---

### 3. Task Routing

```python
def route_task(task):
    """
    Route task to most appropriate solver based on task characteristics

    Insight: Different solvers excel at different task types
    """
    # Analyze task properties
    properties = analyze_task(task)

    # Decision rules (learned from validation data)
    if properties['grid_size'] <= 8 and properties['simple_transformation']:
        # Small, simple tasks --> Heuristics work well
        return 'heuristic'

    elif properties['has_objects'] and properties['object_manipulation']:
        # Object-based tasks --> Program synthesis excels
        return 'program'

    elif properties['requires_counting'] or properties['arithmetic']:
        # Arithmetic tasks --> LLM reasoning helps
        return 'llm'

    elif properties['pattern_continuation']:
        # Pattern-based --> TTT works well
        return 'ttt'

    else:
        # Default: use ensemble
        return 'ensemble'

def analyze_task(task):
    """Extract task properties for routing"""
    properties = {}

    # Grid size
    max_h = max(len(ex['input']) for ex in task['train'])
    max_w = max(len(ex['input'][0]) for ex in task['train'])
    properties['grid_size'] = max(max_h, max_w)

    # Simple transformation check
    # (e.g., all examples have same input/output shape)
    properties['simple_transformation'] = all(
        len(ex['input']) == len(ex['output']) and
        len(ex['input'][0]) == len(ex['output'][0])
        for ex in task['train']
    )

    # Object detection
    properties['has_objects'] = detect_objects_in_task(task)

    # More properties...
    # properties['requires_counting'] = ...
    # properties['arithmetic'] = ...
    # properties['pattern_continuation'] = ...

    return properties
```

---

## Production-Ready Ensemble

### Complete Pipeline with All Optimizations

```python
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class ProductionARCEnsemble:
    """
    Production-ready ARC solver with:
    - Parallel execution
    - Timeout handling
    - Intelligent routing
    - Weighted ensemble
    - Logging and monitoring
    """

    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.solvers = self._initialize_solvers()
        self.meta_models = None  # Load if available

    def solve(self, task, time_limit=300):
        """
        Solve ARC task with time limit

        Args:
            task: ARC task dict
            time_limit: Maximum time in seconds

        Returns:
            Top 3 predictions
        """
        start_time = time.time()

        # Step 1: Quick routing decision
        route = self._route_task(task)
        print(f"Task routed to: {route}")

        # Step 2: If routed to specific solver, try it first
        if route != 'ensemble':
            quick_solution = self._try_quick_solve(task, route, time_limit=30)
            if quick_solution:
                print(f"Quick solution found by {route}")
                return quick_solution

        # Step 3: Run all solvers in parallel (with timeouts)
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed

        all_predictions = self._run_all_solvers_parallel(task, remaining_time)

        # Step 4: Ensemble selection
        top_3 = self._ensemble_select(all_predictions, task)

        total_time = time.time() - start_time
        print(f"Solved in {total_time:.1f}s")

        return top_3

    def _run_all_solvers_parallel(self, task, time_limit):
        """Run all solvers in parallel with timeout"""
        predictions = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all solvers
            futures = {}

            # Heuristics (fast, run first)
            futures['heuristic'] = executor.submit(
                self._run_with_timeout,
                self.solvers['heuristic'].solve,
                task,
                timeout=10
            )

            # TTT (medium speed)
            futures['ttt'] = executor.submit(
                self._run_with_timeout,
                self.solvers['ttt'].solve,
                task,
                timeout=min(120, time_limit * 0.4)
            )

            # LLM (medium speed)
            futures['llm'] = executor.submit(
                self._run_with_timeout,
                self.solvers['llm'].solve,
                task,
                timeout=min(60, time_limit * 0.2)
            )

            # Program synthesis (slow)
            futures['program'] = executor.submit(
                self._run_with_timeout,
                self.solvers['program'].solve,
                task,
                timeout=min(180, time_limit * 0.6)
            )

            # Collect results
            for method, future in futures.items():
                try:
                    result = future.result(timeout=time_limit)
                    if result:
                        for pred in result:
                            predictions.append({
                                'prediction': pred,
                                'source': method,
                                'confidence': self.config['method_weights'][method]
                            })
                except TimeoutError:
                    print(f"{method} solver timed out")
                except Exception as e:
                    print(f"{method} solver failed: {e}")

        return predictions

    def _run_with_timeout(self, func, *args, timeout=60):
        """Run function with timeout"""
        # Use the function directly (already in thread pool)
        return func(*args)

    def _ensemble_select(self, all_predictions, task):
        """Select top 3 using weighted voting + meta-model"""
        if not all_predictions:
            return []

        # If meta-model available, use it
        if self.meta_models:
            return self._select_with_meta_model(all_predictions, task)

        # Otherwise, use weighted voting
        return self._select_with_weighted_vote(all_predictions)

    def _select_with_weighted_vote(self, predictions):
        """Weighted voting selection"""
        vote_scores = {}

        for pred_info in predictions:
            pred_key = self._grid_to_key(pred_info['prediction'])
            weight = pred_info['confidence']

            if pred_key in vote_scores:
                vote_scores[pred_key]['score'] += weight
                vote_scores[pred_key]['count'] += 1
            else:
                vote_scores[pred_key] = {
                    'prediction': pred_info['prediction'],
                    'score': weight,
                    'count': 1,
                    'sources': [pred_info['source']]
                }

        # Boost score for predictions from multiple sources
        for key, info in vote_scores.items():
            if info['count'] > 1:
                info['score'] *= (1 + 0.2 * info['count'])  # 20% boost per additional source

        # Sort by score
        sorted_preds = sorted(vote_scores.values(), key=lambda x: x['score'], reverse=True)

        return [p['prediction'] for p in sorted_preds[:3]]

    def _try_quick_solve(self, task, method, time_limit=30):
        """Try to quickly solve with specific method"""
        try:
            result = self.solvers[method].solve(task)
            return result[:3] if result else None
        except:
            return None

    def _route_task(self, task):
        """Simple task routing"""
        # Analyze task
        max_size = max(
            max(len(ex['input']), len(ex['input'][0]))
            for ex in task['train'] + task['test']
        )

        # Simple heuristic routing
        if max_size <= 5:
            return 'heuristic'  # Small tasks often have simple rules
        elif max_size <= 12:
            return 'ttt'  # Medium tasks good for TTT
        else:
            return 'ensemble'  # Large tasks need full ensemble

    def _grid_to_key(self, grid):
        """Convert grid to hashable key"""
        return str(grid)

    def _initialize_solvers(self):
        """Initialize all solver instances"""
        return {
            'heuristic': HeuristicSolver(),
            'ttt': TestTimeTrainingSolver(),
            'llm': LLMARCSolver(),
            'program': ProgramSynthesisSolver()
        }

    def _default_config(self):
        """Default configuration"""
        return {
            'method_weights': {
                'heuristic': 2.0,
                'program': 1.5,
                'ttt': 1.0,
                'llm': 0.7
            },
            'time_limits': {
                'heuristic': 10,
                'ttt': 120,
                'llm': 60,
                'program': 180
            }
        }


# Usage
ensemble = ProductionARCEnsemble()

# Solve with 5 minute time limit
predictions = ensemble.solve(arc_task, time_limit=300)

print(f"Top 3 predictions: {predictions}")
```

---

## Performance Analysis

### Ensemble Component Contributions

| Component | Solo Accuracy | Ensemble Contribution |
|-----------|---------------|----------------------|
| Test-Time Training | 40% | +30% to ensemble |
| LLM Reasoning | 30% | +15% to ensemble |
| Program Synthesis | 35% | +20% to ensemble |
| Heuristics | 15% | +10% on simple tasks |
| **Full Ensemble** | **55.5%** | - |

**Key Insights:**
- TTT contributes most (solves many unique tasks)
- Program synthesis adds precision (high confidence when it works)
- LLM fills gaps (creative reasoning for novel patterns)
- Heuristics provide quick wins (10% of tasks are trivial)

---

## Best Practices

### 1. Always Run Multiple Methods
```python
# Don't rely on single solver
predictions = []
predictions.extend(method1.solve(task))
predictions.extend(method2.solve(task))
predictions.extend(method3.solve(task))
```

### 2. Verify Before Trusting
```python
# Verify predictions on training data when possible
verified_predictions = [
    p for p in predictions
    if verify_on_training(p, task['train'])
]
```

### 3. Use Timeouts
```python
# Don't let slow methods block entire pipeline
with timeout(120):
    expensive_predictions = program_synthesis.solve(task)
```

### 4. Weight by Historical Performance
```python
# Track which methods work best on which task types
weights = calculate_weights_from_history(task_type)
final = weighted_vote(predictions, weights)
```

### 5. Consider Computational Budget
```python
# If time is limited, skip expensive methods
if time_remaining < 60:
    skip_program_synthesis()
```

---

## Key Takeaways

1. **No single method solves ARC** - Ensembles are essential
2. **Transduction + Induction** both needed (~40% each --> 55%+ together)
3. **Weighted voting** better than simple majority
4. **Task routing** can save time (run fast methods first)
5. **Parallel execution** critical for production (run all methods simultaneously)
6. **Verification** on training data boosts confidence
7. **Meta-learning** can improve selection (stacking, learned routing)

**Current SOTA: 55.5%** (hybrid ensemble)
**Target: 85%** for $500K prize

**Next:** `05_Problem_Solving_Strategies.md` - General strategies beyond ARC
