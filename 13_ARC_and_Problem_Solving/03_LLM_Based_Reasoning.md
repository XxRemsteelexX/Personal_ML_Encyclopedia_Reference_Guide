# LLM-Based Reasoning for ARC - The ARChitects Approach

## Overview

**LLM-Based Reasoning** uses large language models (GPT-4, Claude, etc.) to understand ARC tasks, generate transformation rules, and produce code to solve them.

**Key Paper:** "The LLM ARChitect: Solving ARC-AGI Is A Matter of Perspective" (Franzen et al., 2024)

**Core Insight:** LLMs excel at abstract reasoning and code generation, but struggle with precise grid manipulation. Solution: Use LLMs for high-level reasoning, execute code for low-level operations.

---

## Why LLMs for ARC?

### Strengths

1. **Abstract Reasoning**
   - Can understand complex transformation rules
   - Reason about object properties, spatial relationships
   - Generate hypotheses from few examples

2. **Code Generation**
   - Write executable programs to implement transformations
   - Use programming abstractions (loops, conditions, functions)
   - Decompose complex tasks into steps

3. **Few-Shot Learning**
   - Designed to work with minimal examples
   - In-context learning from 3-5 examples
   - No fine-tuning required (though it helps)

### Weaknesses

1. **Precise Computation**
   - Can make arithmetic errors
   - Inconsistent with exact grid coordinates
   - Hallucinate non-existent patterns

2. **Visual Understanding**
   - Grids are represented as text (lossy conversion)
   - Hard to "see" spatial patterns
   - Miss subtle visual cues

3. **Execution**
   - Generate code but can't verify correctness
   - No feedback loop without running code
   - Bugs in generated code are common

---

## The ARChitects Approach

### Multi-Perspective Strategy

**Insight:** Different representations help LLM reason better.

**Perspectives:**
1. **Grid View** - Raw 2D array representation
2. **Object View** - Identify and track discrete objects
3. **Relational View** - Spatial relationships between objects
4. **Transformation View** - Step-by-step changes
5. **Code View** - Executable program

**Process:**
```
Input Task
    ↓
Generate Multiple Perspectives
    ↓
LLM Reasons Across Perspectives
    ↓
Generate Candidate Programs
    ↓
Execute and Verify
    ↓
Select Best Output
```

---

## Prompt Engineering for ARC

### Basic Prompt Structure

```python
def create_arc_prompt(task):
    """
    Basic prompt for GPT-4 to solve ARC task
    """
    prompt = """You are solving an abstract reasoning task. You will be given several input-output example pairs, and you need to determine the transformation rule and apply it to a new test input.

Here are the training examples:

"""
    # Add training examples
    for i, example in enumerate(task['train']):
        prompt += f"Example {i+1}:\n"
        prompt += "Input:\n" + grid_to_string(example['input']) + "\n"
        prompt += "Output:\n" + grid_to_string(example['output']) + "\n\n"

    prompt += """Now analyze these examples and:
1. Describe the transformation rule in plain English
2. Write Python code to implement this transformation
3. Apply your code to the test input below

Test Input:
""" + grid_to_string(task['test'][0]['input']) + "\n\n"

    prompt += """Please provide:
1. Your understanding of the rule
2. Python code (as a function `def transform(input_grid): ...`)
3. The output grid

Format your response as:
**Rule:** [your description]
**Code:** ```python [your code] ```
**Output:** [the output grid]
"""

    return prompt
```

**Response Example:**
```
**Rule:** The transformation rotates the input grid 90 degrees clockwise.

**Code:**
```python
def transform(input_grid):
    import numpy as np
    arr = np.array(input_grid)
    rotated = np.rot90(arr, k=-1)  # -1 for clockwise
    return rotated.tolist()
```

**Output:**
[[0, 1, 0],
 [1, 0, 1],
 [0, 1, 0]]
```

---

### Advanced: Chain-of-Thought Prompting

```python
def create_cot_prompt(task):
    """
    Chain-of-thought prompt for deeper reasoning
    """
    prompt = """You are an expert at abstract reasoning. Let's solve this step-by-step.

Training Examples:
"""
    for i, example in enumerate(task['train']):
        prompt += f"\nExample {i+1}:\n"
        prompt += "Input:\n" + grid_to_string(example['input']) + "\n"
        prompt += "Output:\n" + grid_to_string(example['output']) + "\n"

    prompt += """
Step 1: Analyze each example carefully. For each example, answer:
- What objects or patterns do you see in the input?
- How are they positioned?
- What colors are used?
- What changed in the output?
- What stayed the same?

Step 2: Identify commonalities across all examples:
- What transformation is applied consistently?
- Are there any rules or conditions?
- Are there special cases?

Step 3: Formulate a precise transformation rule.

Step 4: Write Python code to implement this rule.

Step 5: Test your code mentally on the training examples to verify it works.

Step 6: Apply your code to the test input:
"""
    prompt += grid_to_string(task['test'][0]['input'])

    return prompt
```

**Benefit:** Forces LLM to reason explicitly, reduces errors.

---

### Multi-Perspective Prompting (ARChitects)

```python
def create_multiperspective_prompt(task):
    """
    Generate multiple views of the same task
    """
    prompts = {}

    # Perspective 1: Grid View (Standard)
    prompts['grid'] = create_arc_prompt(task)

    # Perspective 2: Object-Centric View
    prompts['object'] = create_object_centric_prompt(task)

    # Perspective 3: Transformation Steps
    prompts['steps'] = create_stepwise_prompt(task)

    # Perspective 4: Natural Language Description
    prompts['description'] = create_descriptive_prompt(task)

    return prompts

def create_object_centric_prompt(task):
    """
    Describe task in terms of objects and their properties
    """
    prompt = """I'll describe these grids in terms of objects. An object is a connected region of cells with the same non-zero color.

Training Examples (Object View):
"""
    for i, example in enumerate(task['train']):
        prompt += f"\nExample {i+1}:\n"

        # Identify objects in input
        input_objects = find_objects(example['input'])
        prompt += f"Input Objects:\n"
        for j, obj in enumerate(input_objects):
            prompt += f"  Object {j}: Color {obj['color']}, Size {obj['size']}, Position {obj['position']}\n"

        # Identify objects in output
        output_objects = find_objects(example['output'])
        prompt += f"Output Objects:\n"
        for j, obj in enumerate(output_objects):
            prompt += f"  Object {j}: Color {obj['color']}, Size {obj['size']}, Position {obj['position']}\n"

    prompt += "\nBased on how objects change from input to output, determine the transformation rule and apply it to the test input.\n"

    return prompt

def find_objects(grid):
    """Find connected components (objects) in grid"""
    import numpy as np
    from scipy.ndimage import label

    arr = np.array(grid)
    objects = []

    # Find objects for each color
    for color in range(1, 10):  # 0 is background
        mask = (arr == color)
        if mask.sum() == 0:
            continue

        labeled, num_features = label(mask)

        for obj_id in range(1, num_features + 1):
            obj_mask = (labeled == obj_id)
            positions = np.argwhere(obj_mask)

            objects.append({
                'color': color,
                'size': len(positions),
                'position': positions[0].tolist(),  # top-left
                'bounding_box': [positions.min(axis=0).tolist(),
                                positions.max(axis=0).tolist()]
            })

    return objects
```

**Why multiple perspectives help:**
- Grid view: Good for spatial transformations (rotation, flip)
- Object view: Good for object manipulation (move, recolor, delete)
- Combining perspectives → better understanding

---

## Code Generation and Execution

### DSL (Domain-Specific Language) for ARC

Instead of asking LLM to generate arbitrary Python, provide a constrained DSL:

```python
# Define ARC-specific operations
class ARCDSL:
    """Domain-specific language for ARC transformations"""

    @staticmethod
    def rotate_90_cw(grid):
        """Rotate grid 90 degrees clockwise"""
        import numpy as np
        return np.rot90(np.array(grid), k=-1).tolist()

    @staticmethod
    def rotate_90_ccw(grid):
        """Rotate grid 90 degrees counter-clockwise"""
        import numpy as np
        return np.rot90(np.array(grid), k=1).tolist()

    @staticmethod
    def flip_horizontal(grid):
        """Flip grid horizontally"""
        return [row[::-1] for row in grid]

    @staticmethod
    def flip_vertical(grid):
        """Flip grid vertically"""
        return grid[::-1]

    @staticmethod
    def replace_color(grid, old_color, new_color):
        """Replace all instances of one color with another"""
        return [[new_color if cell == old_color else cell for cell in row]
                for row in grid]

    @staticmethod
    def extract_objects(grid, color):
        """Find all objects of a specific color"""
        import numpy as np
        from scipy.ndimage import label

        arr = np.array(grid)
        mask = (arr == color)
        labeled, num = label(mask)

        objects = []
        for obj_id in range(1, num + 1):
            positions = np.argwhere(labeled == obj_id)
            objects.append(positions.tolist())

        return objects

    @staticmethod
    def move_object(grid, object_positions, offset):
        """Move an object by a given offset"""
        import numpy as np
        arr = np.array(grid)

        # Get color of object
        color = arr[object_positions[0][0], object_positions[0][1]]

        # Create output (copy of input)
        output = arr.copy()

        # Erase object from original position
        for pos in object_positions:
            output[pos[0], pos[1]] = 0

        # Place object at new position
        for pos in object_positions:
            new_pos = (pos[0] + offset[0], pos[1] + offset[1])
            if (0 <= new_pos[0] < arr.shape[0] and
                0 <= new_pos[1] < arr.shape[1]):
                output[new_pos[0], new_pos[1]] = color

        return output.tolist()

    @staticmethod
    def resize_grid(grid, new_shape):
        """Resize grid to new shape (with padding/cropping)"""
        import numpy as np
        arr = np.array(grid)
        output = np.zeros(new_shape, dtype=int)

        # Copy what fits
        min_h = min(arr.shape[0], new_shape[0])
        min_w = min(arr.shape[1], new_shape[1])
        output[:min_h, :min_w] = arr[:min_h, :min_w]

        return output.tolist()

    @staticmethod
    def count_color(grid, color):
        """Count cells of a specific color"""
        import numpy as np
        return int((np.array(grid) == color).sum())

    @staticmethod
    def get_grid_shape(grid):
        """Get shape of grid as (height, width)"""
        return (len(grid), len(grid[0]) if grid else 0)

    @staticmethod
    def create_empty_grid(height, width, fill_color=0):
        """Create a new grid filled with a specific color"""
        return [[fill_color] * width for _ in range(height)]

    @staticmethod
    def overlay_grids(grid1, grid2, transparent_color=0):
        """Overlay grid2 on top of grid1 (grid2's non-transparent cells win)"""
        import numpy as np
        arr1 = np.array(grid1)
        arr2 = np.array(grid2)

        # Ensure same shape
        assert arr1.shape == arr2.shape

        output = arr1.copy()
        mask = (arr2 != transparent_color)
        output[mask] = arr2[mask]

        return output.tolist()

    # ... add more operations as needed
```

**Constrained Prompting:**
```python
def create_dsl_prompt(task):
    """Prompt LLM to use our DSL"""
    prompt = f"""You have access to these ARC operations:

{get_dsl_documentation()}

Your task is to compose these operations to solve the ARC challenge.

Training Examples:
"""
    for i, example in enumerate(task['train']):
        prompt += f"\nExample {i+1}:\n"
        prompt += "Input:\n" + grid_to_string(example['input']) + "\n"
        prompt += "Output:\n" + grid_to_string(example['output']) + "\n"

    prompt += """
Write a function using ONLY the operations above:

```python
def transform(grid):
    # Your code using ARCDSL operations
    result = ARCDSL.some_operation(grid, ...)
    return result
```

Test Input:
""" + grid_to_string(task['test'][0]['input'])

    return prompt

def get_dsl_documentation():
    """Generate documentation for DSL"""
    docs = ""
    for name, method in ARCDSL.__dict__.items():
        if not name.startswith('_') and callable(method):
            docs += f"- ARCDSL.{name}{inspect.signature(method)}\n"
            docs += f"  {method.__doc__}\n"
    return docs
```

**Benefits:**
- Reduces hallucination (LLM can only use predefined operations)
- Guarantees correct execution (operations are pre-tested)
- Easier to verify generated code
- Faster execution (optimized implementations)

---

## Self-Consistency and Verification

### Generate-and-Test Loop

```python
def solve_with_verification(task, llm, max_attempts=5):
    """
    Generate multiple candidate solutions and verify
    """
    candidates = []

    for attempt in range(max_attempts):
        # Generate solution
        prompt = create_arc_prompt(task)
        response = llm.generate(prompt, temperature=0.8)  # Higher temp for diversity

        # Parse code from response
        code = extract_code_from_response(response)

        # Execute code to get program
        try:
            namespace = {'ARCDSL': ARCDSL}
            exec(code, namespace)
            transform_func = namespace['transform']
        except Exception as e:
            print(f"Attempt {attempt}: Code execution failed - {e}")
            continue

        # Verify on training examples
        is_correct = True
        for example in task['train']:
            try:
                predicted_output = transform_func(example['input'])
                if predicted_output != example['output']:
                    is_correct = False
                    break
            except Exception as e:
                print(f"Attempt {attempt}: Transform failed on example - {e}")
                is_correct = False
                break

        if is_correct:
            # Apply to test input
            test_output = transform_func(task['test'][0]['input'])
            candidates.append({
                'code': code,
                'output': test_output,
                'attempt': attempt
            })

    return candidates
```

**Key Idea:** Generate multiple solutions, keep only those that pass training examples.

---

### Self-Correction with Feedback

```python
def solve_with_self_correction(task, llm, max_iterations=3):
    """
    Iteratively refine solution based on errors
    """
    prompt = create_arc_prompt(task)
    code = None

    for iteration in range(max_iterations):
        # Generate/refine code
        response = llm.generate(prompt)
        code = extract_code_from_response(response)

        # Test on training examples
        errors = []
        try:
            namespace = {'ARCDSL': ARCDSL}
            exec(code, namespace)
            transform_func = namespace['transform']

            for i, example in enumerate(task['train']):
                predicted = transform_func(example['input'])
                expected = example['output']

                if predicted != expected:
                    errors.append({
                        'example': i,
                        'predicted': predicted,
                        'expected': expected
                    })

        except Exception as e:
            errors.append({'exception': str(e)})

        # If no errors, we're done!
        if not errors:
            test_output = transform_func(task['test'][0]['input'])
            return test_output

        # Otherwise, provide feedback for next iteration
        feedback = f"\nYour code produced errors:\n"
        for error in errors:
            if 'exception' in error:
                feedback += f"- Exception: {error['exception']}\n"
            else:
                feedback += f"- Example {error['example']}: predicted {error['predicted']}, expected {error['expected']}\n"

        feedback += "\nPlease fix your code and try again.\n"

        prompt = prompt + "\n" + response + "\n" + feedback

    # Failed after max iterations
    return None
```

**Benefit:** LLM learns from its mistakes, iteratively improves.

---

## Practical Implementation

### Complete LLM-Based Solver

```python
import openai
import re

class LLMARCSolver:
    def __init__(self, model="gpt-4-turbo", api_key=None):
        self.model = model
        if api_key:
            openai.api_key = api_key

    def solve_task(self, task, strategy='verification', max_attempts=5):
        """
        Solve ARC task using LLM

        Args:
            task: ARC task dict
            strategy: 'basic', 'cot', 'verification', 'self_correction'
            max_attempts: Number of attempts for verification strategy

        Returns:
            Top 3 candidate outputs
        """
        if strategy == 'basic':
            return self._solve_basic(task)
        elif strategy == 'cot':
            return self._solve_cot(task)
        elif strategy == 'verification':
            return self._solve_with_verification(task, max_attempts)
        elif strategy == 'self_correction':
            return self._solve_with_self_correction(task)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _solve_basic(self, task):
        """Basic prompting"""
        prompt = create_arc_prompt(task)
        response = self._call_llm(prompt)
        code = self._extract_code(response)
        output = self._execute_code(code, task['test'][0]['input'])
        return [output]

    def _solve_cot(self, task):
        """Chain-of-thought prompting"""
        prompt = create_cot_prompt(task)
        response = self._call_llm(prompt)
        code = self._extract_code(response)
        output = self._execute_code(code, task['test'][0]['input'])
        return [output]

    def _solve_with_verification(self, task, max_attempts):
        """Generate multiple candidates, verify on training data"""
        candidates = []

        for attempt in range(max_attempts):
            prompt = create_dsl_prompt(task)
            response = self._call_llm(prompt, temperature=0.8)
            code = self._extract_code(response)

            # Verify on training examples
            if self._verify_code(code, task['train']):
                output = self._execute_code(code, task['test'][0]['input'])
                if output is not None:
                    candidates.append(output)

        # Return top 3 unique candidates
        unique_candidates = []
        for cand in candidates:
            if cand not in unique_candidates:
                unique_candidates.append(cand)
            if len(unique_candidates) >= 3:
                break

        return unique_candidates

    def _solve_with_self_correction(self, task):
        """Iterative refinement"""
        return solve_with_self_correction(task, self, max_iterations=3)

    def _call_llm(self, prompt, temperature=0.0):
        """Call OpenAI API"""
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at solving abstract reasoning tasks."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=2000
        )
        return response.choices[0].message.content

    def _extract_code(self, response):
        """Extract Python code from LLM response"""
        # Find code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]

        # Fallback: look for def transform
        match = re.search(r'(def transform\(.*?\):.*?)(\n\n|\nTest|\n```|\Z)', response, re.DOTALL)
        if match:
            return match.group(1)

        return None

    def _execute_code(self, code, test_input):
        """Execute generated code on test input"""
        if code is None:
            return None

        try:
            namespace = {'ARCDSL': ARCDSL, 'np': __import__('numpy')}
            exec(code, namespace)
            transform_func = namespace.get('transform')

            if transform_func:
                output = transform_func(test_input)
                return output
        except Exception as e:
            print(f"Execution error: {e}")
            return None

    def _verify_code(self, code, train_examples):
        """Verify code works on all training examples"""
        if code is None:
            return False

        try:
            namespace = {'ARCDSL': ARCDSL, 'np': __import__('numpy')}
            exec(code, namespace)
            transform_func = namespace.get('transform')

            if not transform_func:
                return False

            for example in train_examples:
                predicted = transform_func(example['input'])
                if predicted != example['output']:
                    return False

            return True
        except Exception:
            return False


# Usage
solver = LLMARCSolver(model="gpt-4-turbo", api_key="your-api-key")

# Solve with verification (best results)
predictions = solver.solve_task(arc_task, strategy='verification', max_attempts=10)

print(f"Generated {len(predictions)} candidate solutions")
```

---

## Performance and Limitations

### Performance

**GPT-4 (Basic Prompting):** ~5-10% accuracy
**GPT-4 (CoT + Verification):** ~20-30% accuracy
**GPT-4 (Multi-perspective + DSL):** ~30-40% accuracy

**Breakdown:**
- **Easy tasks (small grids, simple rules):** 60-70%
- **Medium tasks:** 20-40%
- **Hard tasks (large grids, complex rules):** 5-15%

### Why Not Higher?

1. **Visual Reasoning Gap**
   - Text representation loses spatial information
   - Hard to "see" patterns humans easily recognize

2. **Precise Execution**
   - LLMs make off-by-one errors
   - Inconsistent with exact coordinates

3. **Novel Pattern Recognition**
   - Training data doesn't cover all ARC patterns
   - Struggles with truly novel transformations

4. **Code Verification**
   - Generated code often has subtle bugs
   - Hard to verify correctness without execution

---

## Best Practices

### 1. Always Verify on Training Examples
```python
# Never trust LLM output without verification!
if not verify_code(code, task['train']):
    print("Code failed verification, generating new attempt")
```

### 2. Use Multiple Attempts
```python
# Generate 5-10 candidates, select best
for attempt in range(10):
    candidate = solve_attempt(task)
    if verify(candidate):
        candidates.append(candidate)
```

### 3. Provide Rich Context
```python
# Include multiple perspectives
prompt += grid_view(task)
prompt += object_view(task)
prompt += relationship_view(task)
```

### 4. Constrain Output Space
```python
# Use DSL to limit what LLM can generate
prompt += "Use ONLY these operations: rotate_90, flip_horizontal, replace_color"
```

### 5. Combine with Other Methods
```python
# Ensemble with test-time training
llm_output = llm_solver.solve(task)
ttt_output = ttt_solver.solve(task)
program_output = program_synthesis.solve(task)

# Vote or merge
final_output = ensemble([llm_output, ttt_output, program_output])
```

---

## Future Directions

1. **Vision-Language Models**
   - Use models that can "see" grids (GPT-4V, Gemini, Claude 3)
   - Reduce information loss from text representation

2. **Iterative Refinement**
   - Multi-turn conversations with LLM
   - Provide richer feedback (visual diffs, error explanations)

3. **Learned DSLs**
   - Automatically discover useful operations from data
   - Expand DSL based on task requirements

4. **Neuro-Symbolic Integration**
   - Combine LLM reasoning with symbolic search
   - Use LLM to guide program synthesis

---

## Key Takeaways

1. **LLMs are powerful** for abstract reasoning but need verification
2. **Chain-of-thought** improves performance significantly
3. **Multi-perspective prompting** helps LLM understand tasks better
4. **DSL constraints** reduce hallucination and errors
5. **Verification on training examples** is essential
6. **Ensemble with other methods** for best results
7. **Current limit:** ~30-40% accuracy alone, but critical component of 55%+ ensembles

**Next:** `04_Hybrid_Ensembles.md` - Combining TTT + LLM + Program Synthesis for SOTA
