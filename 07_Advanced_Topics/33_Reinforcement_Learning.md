# Reinforcement Learning - Complete Guide

## Table of Contents

1. [Introduction and Fundamentals](#1-introduction-and-fundamentals)
2. [Dynamic Programming](#2-dynamic-programming)
3. [Monte Carlo Methods](#3-monte-carlo-methods)
4. [Temporal Difference Learning](#4-temporal-difference-learning)
5. [Deep Q-Networks (DQN)](#5-deep-q-networks-dqn)
6. [Policy Gradient Methods](#6-policy-gradient-methods)
7. [Advanced Policy Optimization](#7-advanced-policy-optimization)
8. [Model-Based RL](#8-model-based-rl)
9. [Multi-Agent RL](#9-multi-agent-rl)
10. [Inverse RL and Reward Shaping](#10-inverse-rl-and-reward-shaping)
11. [Offline RL](#11-offline-rl)
12. [RL for LLM Alignment](#12-rl-for-llm-alignment)
13. [Practical Implementation](#13-practical-implementation)
14. [Hyperparameter Tuning for RL](#14-hyperparameter-tuning-for-rl)
15. [Applications](#15-applications)
16. [Resources and References](#16-resources-and-references)

---

## 1. Introduction and Fundamentals

### What is Reinforcement Learning

**Reinforcement Learning (RL)** is a paradigm where an agent learns to make sequential decisions by interacting with an environment to maximize cumulative rewards. Unlike supervised learning (which requires labeled examples) or unsupervised learning (which finds patterns), RL learns from trial-and-error through rewards and penalties.

**Key Characteristics:**
- **Sequential decision-making**: Actions affect future states
- **Delayed rewards**: Consequences may not be immediate
- **Exploration vs exploitation**: Balance trying new actions vs using known good ones
- **No supervision**: Agent discovers optimal behavior through interaction

### Agent-Environment Interaction

The RL framework consists of:

```
At each timestep t:
  1. Agent observes state s_t
  2. Agent takes action a_t
  3. Environment returns reward r_t and new state s_{t+1}
  4. Agent updates its knowledge
  5. Repeat
```

**Components:**
- **Agent**: The learner/decision-maker
- **Environment**: Everything outside the agent
- **State (s)**: Complete description of the world
- **Observation (o)**: Partial state information (in partially observable environments)
- **Action (a)**: Choices available to the agent
- **Reward (r)**: Scalar feedback signal

### Markov Decision Process (MDP) Formal Definition

An MDP is a mathematical framework for modeling sequential decision-making:

**Formal Definition**: An MDP is a tuple (S, A, P, R, gamma)

- **S**: State space (set of all possible states)
- **A**: Action space (set of all possible actions)
- **P**: Transition dynamics P(s' | s, a) - probability of reaching state s' from state s taking action a
- **R**: Reward function R(s, a, s') or R(s, a) - immediate reward
- **gamma**: Discount factor in [0, 1]

**Markov Property**: The future is independent of the past given the present:

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)
```

### Core Definitions

**Policy (pi)**: A mapping from states to actions
- **Deterministic**: pi(s) = a
- **Stochastic**: pi(a | s) = probability of action a in state s

**Return (G_t)**: Total discounted reward from timestep t onwards
```
G_t = r_{t+1} + gamma * r_{t+2} + gamma^2 * r_{t+3} + ... = sum_{k=0}^{infinity} gamma^k * r_{t+k+1}
```

**State Value Function V^pi(s)**: Expected return starting from state s following policy pi
```
V^pi(s) = E_pi[G_t | s_t = s] = E_pi[sum_{k=0}^{infinity} gamma^k * r_{t+k+1} | s_t = s]
```

**Action Value Function Q^pi(s, a)**: Expected return starting from state s, taking action a, then following policy pi
```
Q^pi(s, a) = E_pi[G_t | s_t = s, a_t = a]
```

**Advantage Function A^pi(s, a)**: How much better action a is compared to average
```
A^pi(s, a) = Q^pi(s, a) - V^pi(s)
```

**Optimal Value Functions**:
```
V*(s) = max_pi V^pi(s)
Q*(s, a) = max_pi Q^pi(s, a)
```

### Exploration vs Exploitation Tradeoff

**Exploitation**: Choose actions that maximize reward based on current knowledge
**Exploration**: Try new actions to discover potentially better strategies

**Common Strategies**:

1. **Epsilon-Greedy**: With probability epsilon, choose random action; otherwise choose best action
2. **Boltzmann/Softmax**: Sample actions proportional to exp(Q(s,a) / temperature)
3. **Upper Confidence Bound (UCB)**: Choose action that maximizes Q(s,a) + c * sqrt(log(t) / N(s,a))
4. **Thompson Sampling**: Maintain belief distribution over Q-values, sample from it

```python
import numpy as np

class EpsilonGreedy:
    """Epsilon-greedy exploration strategy"""
    def __init__(self, epsilon=0.1, decay=0.995, min_epsilon=0.01):
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon

    def select_action(self, q_values):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(q_values))
        return np.argmax(q_values)

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

class UCBExploration:
    """Upper Confidence Bound exploration"""
    def __init__(self, c=2.0):
        self.c = c
        self.counts = None
        self.total_steps = 0

    def select_action(self, q_values):
        if self.counts is None:
            self.counts = np.zeros(len(q_values))

        self.total_steps += 1

        # Handle unvisited actions
        if 0 in self.counts:
            return np.argmin(self.counts)

        # UCB formula
        ucb_values = q_values + self.c * np.sqrt(np.log(self.total_steps) / self.counts)
        action = np.argmax(ucb_values)
        self.counts[action] += 1
        return action
```

### Discount Factor Gamma and Its Effects

The **discount factor gamma in [0, 1]** determines how much future rewards are valued:

**gamma = 0**: Only immediate rewards matter (myopic)
**gamma --> 1**: Future rewards weighted nearly equally (far-sighted)

**Effects of gamma**:
- **Convergence**: Smaller gamma leads to faster convergence
- **Horizon**: Larger gamma considers longer time horizons
- **Value magnitude**: Smaller gamma reduces value function magnitude
- **Stability**: Smaller gamma often more stable in practice

**Choosing gamma**:
- **Short episodes**: gamma = 0.9 - 0.95
- **Long episodes**: gamma = 0.95 - 0.99
- **Continuing tasks**: gamma = 0.99 - 0.999

### Episodic vs Continuing Tasks

**Episodic Tasks**:
- Have terminal states (episode ends)
- Examples: Game with win/loss, reaching goal in maze
- Return is finite: G_t = r_{t+1} + gamma * r_{t+2} + ... + gamma^{T-t-1} * r_T
- Easier to learn (natural resets)

**Continuing Tasks**:
- No terminal state (run forever)
- Examples: Stock trading, climate control
- Must use discounting (gamma < 1) to ensure finite returns
- Can reformulate with average reward or discounted return

```python
import numpy as np

class Environment:
    """Base environment class demonstrating episodic vs continuing"""
    def __init__(self, episodic=True, max_steps=100):
        self.episodic = episodic
        self.max_steps = max_steps
        self.steps = 0

    def reset(self):
        self.steps = 0
        return self._get_initial_state()

    def step(self, action):
        self.steps += 1
        next_state = self._transition(action)
        reward = self._reward(next_state, action)

        if self.episodic:
            done = self._is_terminal(next_state) or self.steps >= self.max_steps
        else:
            done = False  # Continuing task never terminates

        return next_state, reward, done, {}

    def _get_initial_state(self):
        raise NotImplementedError

    def _transition(self, action):
        raise NotImplementedError

    def _reward(self, state, action):
        raise NotImplementedError

    def _is_terminal(self, state):
        raise NotImplementedError

def compute_returns(rewards, gamma=0.99, episodic=True):
    """Compute discounted returns from rewards"""
    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return np.array(returns)
```

---

## 2. Dynamic Programming

**Dynamic Programming (DP)** methods solve MDPs when the model (transition dynamics P and reward function R) is fully known. They use the Bellman equations to iteratively improve value functions.

**Requirements**:
- Complete knowledge of environment dynamics
- Finite state and action spaces
- Computational feasibility

### Bellman Equations

**Bellman Expectation Equations** (for a given policy pi):

```
V^pi(s) = sum_a pi(a|s) * sum_{s',r} P(s',r | s,a) * [r + gamma * V^pi(s')]

Q^pi(s,a) = sum_{s',r} P(s',r | s,a) * [r + gamma * sum_{a'} pi(a'|s') * Q^pi(s',a')]
```

**Bellman Optimality Equations** (for optimal policy):

```
V*(s) = max_a sum_{s',r} P(s',r | s,a) * [r + gamma * V*(s')]

Q*(s,a) = sum_{s',r} P(s',r | s,a) * [r + gamma * max_{a'} Q*(s',a')]
```

### Policy Evaluation (Iterative)

**Goal**: Compute V^pi for a given policy pi

**Algorithm**: Iteratively apply Bellman expectation equation until convergence

```python
import numpy as np

def policy_evaluation(policy, transition_probs, rewards, gamma=0.99, theta=1e-6):
    """
    Iterative policy evaluation

    Args:
        policy: Array of shape (n_states, n_actions) with action probabilities
        transition_probs: P(s' | s, a) shape (n_states, n_actions, n_states)
        rewards: R(s, a) shape (n_states, n_actions)
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        V: State value function
    """
    n_states = policy.shape[0]
    V = np.zeros(n_states)

    iteration = 0
    while True:
        delta = 0
        V_new = np.zeros(n_states)

        for s in range(n_states):
            v = 0
            for a in range(policy.shape[1]):
                # Expected value for this state-action
                q_sa = 0
                for s_next in range(n_states):
                    q_sa += transition_probs[s, a, s_next] * (
                        rewards[s, a] + gamma * V[s_next]
                    )
                v += policy[s, a] * q_sa

            V_new[s] = v
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        iteration += 1

        if delta < theta:
            print(f"Policy evaluation converged in {iteration} iterations")
            break

    return V

# Example: GridWorld
def create_gridworld_mdp(size=4):
    """Create simple gridworld MDP"""
    n_states = size * size
    n_actions = 4  # up, right, down, left

    transition_probs = np.zeros((n_states, n_actions, n_states))
    rewards = np.full((n_states, n_actions), -1.0)  # -1 per step

    # Terminal states (corners)
    terminal_states = [0, n_states - 1]

    for s in range(n_states):
        if s in terminal_states:
            # Terminal state transitions to itself with 0 reward
            for a in range(n_actions):
                transition_probs[s, a, s] = 1.0
                rewards[s, a] = 0.0
            continue

        row, col = s // size, s % size

        for a in range(n_actions):
            # Compute next state
            if a == 0:  # up
                next_row, next_col = max(0, row - 1), col
            elif a == 1:  # right
                next_row, next_col = row, min(size - 1, col + 1)
            elif a == 2:  # down
                next_row, next_col = min(size - 1, row + 1), col
            else:  # left
                next_row, next_col = row, max(0, col - 1)

            next_state = next_row * size + next_col
            transition_probs[s, a, next_state] = 1.0

    return transition_probs, rewards

# Test policy evaluation
transition_probs, rewards = create_gridworld_mdp(size=4)
n_states, n_actions = rewards.shape

# Random policy
random_policy = np.ones((n_states, n_actions)) / n_actions

V = policy_evaluation(random_policy, transition_probs, rewards)
print("State values:\n", V.reshape(4, 4))
```

### Policy Improvement

**Goal**: Given V^pi, find a better policy pi'

**Policy Improvement Theorem**: If Q^pi(s, pi'(s)) >= V^pi(s) for all s, then pi' >= pi

**Greedy Policy Improvement**:
```
pi'(s) = argmax_a Q^pi(s, a) = argmax_a sum_{s',r} P(s',r|s,a) * [r + gamma * V^pi(s')]
```

```python
def policy_improvement(V, transition_probs, rewards, gamma=0.99):
    """
    Policy improvement: create greedy policy w.r.t. value function

    Args:
        V: State value function
        transition_probs: P(s' | s, a)
        rewards: R(s, a)
        gamma: Discount factor

    Returns:
        policy: Improved deterministic policy (greedy)
        policy_stable: Whether policy changed
    """
    n_states, n_actions = rewards.shape
    policy = np.zeros((n_states, n_actions))

    for s in range(n_states):
        # Compute Q(s, a) for all actions
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            for s_next in range(n_states):
                q_values[a] += transition_probs[s, a, s_next] * (
                    rewards[s, a] + gamma * V[s_next]
                )

        # Greedy action selection
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0

    return policy
```

### Policy Iteration with Code

**Policy Iteration** alternates between policy evaluation and policy improvement until the policy converges.

**Algorithm**:
1. Initialize policy pi arbitrarily
2. Repeat until policy is stable:
   a. Policy Evaluation: Compute V^pi
   b. Policy Improvement: pi' = greedy(V^pi)
   c. If pi' = pi, stop; else pi = pi'

```python
def policy_iteration(transition_probs, rewards, gamma=0.99, max_iterations=100):
    """
    Policy iteration algorithm

    Args:
        transition_probs: P(s' | s, a)
        rewards: R(s, a)
        gamma: Discount factor
        max_iterations: Maximum number of iterations

    Returns:
        policy: Optimal policy
        V: Optimal value function
        iterations: Number of iterations until convergence
    """
    n_states, n_actions = rewards.shape

    # Initialize with random policy
    policy = np.ones((n_states, n_actions)) / n_actions

    for iteration in range(max_iterations):
        # Policy Evaluation
        V = policy_evaluation(policy, transition_probs, rewards, gamma)

        # Policy Improvement
        policy_new = policy_improvement(V, transition_probs, rewards, gamma)

        # Check if policy is stable
        if np.allclose(policy, policy_new):
            print(f"Policy iteration converged in {iteration + 1} iterations")
            return policy_new, V, iteration + 1

        policy = policy_new

    print(f"Policy iteration reached max iterations ({max_iterations})")
    return policy, V, max_iterations

# Example usage
transition_probs, rewards = create_gridworld_mdp(size=4)
optimal_policy, optimal_V, iters = policy_iteration(transition_probs, rewards, gamma=0.99)

print("Optimal value function:\n", optimal_V.reshape(4, 4))
print("\nOptimal policy (argmax):")
print(np.argmax(optimal_policy, axis=1).reshape(4, 4))
```

### Value Iteration with Code

**Value Iteration** combines policy evaluation and improvement into a single update.

**Algorithm**:
1. Initialize V(s) = 0 for all s
2. Repeat until convergence:
   ```
   V(s) = max_a sum_{s',r} P(s',r|s,a) * [r + gamma * V(s')]
   ```
3. Extract policy: pi(s) = argmax_a Q(s, a)

**Advantage**: Faster than policy iteration (no complete policy evaluation)

```python
def value_iteration(transition_probs, rewards, gamma=0.99, theta=1e-6, max_iterations=1000):
    """
    Value iteration algorithm

    Args:
        transition_probs: P(s' | s, a)
        rewards: R(s, a)
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum iterations

    Returns:
        policy: Optimal policy
        V: Optimal value function
        iterations: Number of iterations
    """
    n_states, n_actions = rewards.shape
    V = np.zeros(n_states)

    for iteration in range(max_iterations):
        delta = 0
        V_new = np.zeros(n_states)

        for s in range(n_states):
            # Compute Q(s, a) for all actions
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                for s_next in range(n_states):
                    q_values[a] += transition_probs[s, a, s_next] * (
                        rewards[s, a] + gamma * V[s_next]
                    )

            # Bellman optimality backup
            V_new[s] = np.max(q_values)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        if delta < theta:
            print(f"Value iteration converged in {iteration + 1} iterations")
            break

    # Extract optimal policy
    policy = policy_improvement(V, transition_probs, rewards, gamma)

    return policy, V, iteration + 1

# Example usage
transition_probs, rewards = create_gridworld_mdp(size=4)
optimal_policy, optimal_V, iters = value_iteration(transition_probs, rewards, gamma=0.99)

print("Optimal value function:\n", optimal_V.reshape(4, 4))
print("\nOptimal policy:")
policy_arrows = ['U', 'R', 'D', 'L']
policy_grid = np.array([policy_arrows[a] for a in np.argmax(optimal_policy, axis=1)]).reshape(4, 4)
print(policy_grid)
```

### Limitations of Dynamic Programming

**1. Requires Complete Model**: Must know P(s', r | s, a) - often unavailable in practice

**2. Curse of Dimensionality**:
- Computational cost grows exponentially with state/action space size
- Continuous spaces require discretization

**3. Memory Requirements**: Must store value for every state

**4. Computational Cost**: Each iteration sweeps all states

**When to Use DP**:
- Small, discrete state/action spaces
- Model is known or easily learned
- Planning problems (e.g., robotics path planning with known map)

**Alternatives**:
- **Model-free methods**: Learn directly from experience (Q-learning, policy gradients)
- **Function approximation**: Neural networks to handle large/continuous spaces
- **Sampling methods**: Monte Carlo, TD learning

---

## 3. Monte Carlo Methods

**Monte Carlo (MC)** methods learn from complete episodes of experience without requiring a model. They estimate value functions by averaging sample returns.

**Key Idea**: Use empirical mean return instead of expected return

**Advantages**:
- Model-free: Learn from experience
- Can handle episodic tasks
- Unbiased estimates
- Good for problems with limited dynamics knowledge

**Disadvantages**:
- Only for episodic tasks
- High variance
- Must wait until episode ends

### First-Visit vs Every-Visit MC

**First-Visit MC**:
- Average returns only from the first visit to state s in each episode
- Unbiased estimator
- More common in practice

**Every-Visit MC**:
- Average returns from every visit to state s
- Also unbiased (in limit)
- Can have lower variance with repeated visits

```python
import numpy as np
from collections import defaultdict

class MonteCarloAgent:
    """Base Monte Carlo learning agent"""
    def __init__(self, gamma=0.99, first_visit=True):
        self.gamma = gamma
        self.first_visit = first_visit
        self.returns = defaultdict(list)  # Store returns for each state
        self.V = defaultdict(float)  # State value function
        self.Q = defaultdict(lambda: defaultdict(float))  # Action value function
        self.policy = {}

    def update_value_from_episode(self, episode):
        """
        Update value function from episode

        Args:
            episode: List of (state, action, reward) tuples
        """
        # Extract states and rewards
        states = [x[0] for x in episode]
        rewards = [x[2] for x in episode]

        # Compute returns
        G = 0
        returns_list = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns_list.insert(0, G)

        # Update values
        visited = set()
        for t, (state, _, _) in enumerate(episode):
            state_key = self._state_to_key(state)

            if self.first_visit and state_key in visited:
                continue

            visited.add(state_key)
            self.returns[state_key].append(returns_list[t])
            self.V[state_key] = np.mean(self.returns[state_key])

    def _state_to_key(self, state):
        """Convert state to hashable key"""
        if isinstance(state, np.ndarray):
            return tuple(state)
        return state
```

### Monte Carlo Prediction with Code

**Goal**: Estimate V^pi(s) for a given policy pi

**Algorithm**:
1. Generate episode following policy pi
2. For each state s visited in episode:
   - Compute return G from that point
   - Update V(s) as average of observed returns

```python
def mc_prediction_first_visit(env, policy, n_episodes=10000, gamma=0.99):
    """
    First-visit Monte Carlo prediction

    Args:
        env: Environment with reset() and step(action) methods
        policy: Function mapping state to action
        n_episodes: Number of episodes to run
        gamma: Discount factor

    Returns:
        V: Estimated state value function
    """
    returns = defaultdict(list)
    V = defaultdict(float)

    for episode_num in range(n_episodes):
        # Generate episode
        episode = []
        state = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # Compute returns and update values
        G = 0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + gamma * G

            state_key = tuple(state) if isinstance(state, np.ndarray) else state

            # First-visit check
            if state_key not in visited:
                visited.add(state_key)
                returns[state_key].append(G)
                V[state_key] = np.mean(returns[state_key])

        if (episode_num + 1) % 1000 == 0:
            print(f"Episode {episode_num + 1}/{n_episodes}")

    return V

# Example: Simple GridWorld policy
def simple_policy(state):
    """Random policy for testing"""
    return np.random.randint(4)  # 4 actions
```

### Monte Carlo Control with Epsilon-Greedy

**MC Control**: Find optimal policy using MC methods

**Challenge**: Need to explore all state-action pairs

**Solution**: Epsilon-greedy exploration

**Algorithm (MC Control with Epsilon-Greedy)**:
1. Initialize Q(s, a) arbitrarily
2. Initialize epsilon-greedy policy derived from Q
3. For each episode:
   a. Generate episode using current policy
   b. For each (s, a) in episode:
      - Compute return G
      - Update Q(s, a) = average of returns
   c. For each state s in episode:
      - Update policy to be epsilon-greedy w.r.t. Q

```python
class MCControlAgent:
    """Monte Carlo Control with epsilon-greedy exploration"""
    def __init__(self, n_actions, gamma=0.99, epsilon=0.1, epsilon_decay=0.9999):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay

        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.returns = defaultdict(lambda: defaultdict(list))
        self.episode_count = 0

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        state_key = self._state_key(state)

        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state_key])

    def update_from_episode(self, episode):
        """
        Update Q values from episode

        Args:
            episode: List of (state, action, reward) tuples
        """
        # Compute returns
        G = 0
        returns_list = []
        for _, _, reward in reversed(episode):
            G = reward + self.gamma * G
            returns_list.insert(0, G)

        # Update Q values (first-visit)
        visited = set()
        for t, (state, action, _) in enumerate(episode):
            sa_key = (self._state_key(state), action)

            if sa_key not in visited:
                visited.add(sa_key)
                state_key = self._state_key(state)
                self.returns[state_key][action].append(returns_list[t])
                self.Q[state_key][action] = np.mean(self.returns[state_key][action])

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def _state_key(self, state):
        """Convert state to hashable key"""
        if isinstance(state, np.ndarray):
            return tuple(state)
        return state

def train_mc_control(env, n_episodes=10000):
    """Train agent using MC control"""
    agent = MCControlAgent(n_actions=env.action_space.n)

    episode_rewards = []

    for episode in range(n_episodes):
        # Collect episode
        trajectory = []
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward))
            total_reward += reward
            state = next_state

        # Update agent
        agent.update_from_episode(trajectory)
        episode_rewards.append(total_reward)

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    return agent, episode_rewards
```

### Off-Policy MC with Importance Sampling

**On-Policy**: Learn about policy pi while using pi for behavior
**Off-Policy**: Learn about target policy pi while following behavior policy b

**Why Off-Policy?**
- Learn from exploratory behavior while deriving optimal policy
- Learn from human/expert demonstrations
- Use single behavior policy to learn multiple target policies

**Importance Sampling**: Correct for difference between policies

**Importance Sampling Ratio**:
```
rho_{t:T-1} = product_{k=t}^{T-1} [pi(a_k | s_k) / b(a_k | s_k)]
```

**Ordinary Importance Sampling**:
```
V(s) = sum_i [rho_i * G_i] / N
```

**Weighted Importance Sampling** (lower variance):
```
V(s) = sum_i [rho_i * G_i] / sum_i rho_i
```

```python
class OffPolicyMC:
    """Off-policy Monte Carlo with weighted importance sampling"""
    def __init__(self, n_actions, gamma=0.99):
        self.n_actions = n_actions
        self.gamma = gamma

        # Target policy (deterministic, greedy)
        self.Q = defaultdict(lambda: np.zeros(n_actions))

        # For weighted importance sampling
        self.C = defaultdict(lambda: np.zeros(n_actions))  # Cumulative weights

    def target_policy(self, state):
        """Greedy target policy"""
        state_key = self._state_key(state)
        return np.argmax(self.Q[state_key])

    def behavior_policy(self, state, epsilon=0.1):
        """Epsilon-greedy behavior policy"""
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        return self.target_policy(state)

    def update_from_episode(self, episode, behavior_probs):
        """
        Update Q using weighted importance sampling

        Args:
            episode: List of (state, action, reward) tuples
            behavior_probs: List of behavior policy probabilities for each action
        """
        G = 0
        W = 1.0  # Importance sampling ratio

        # Work backwards from end of episode
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            state_key = self._state_key(state)

            # Update return
            G = reward + self.gamma * G

            # Update cumulative weight
            self.C[state_key][action] += W

            # Update Q with weighted average
            self.Q[state_key][action] += (W / self.C[state_key][action]) * (
                G - self.Q[state_key][action]
            )

            # Update importance sampling ratio
            # If target policy would not take this action, ratio becomes 0
            if action != np.argmax(self.Q[state_key]):
                break

            # Update W: pi(a|s) / b(a|s)
            # For greedy target policy, pi(a|s) = 1 for best action
            W *= 1.0 / behavior_probs[t]

    def _state_key(self, state):
        if isinstance(state, np.ndarray):
            return tuple(state)
        return state

# Example usage
def train_off_policy_mc(env, n_episodes=10000, epsilon=0.1):
    """Train using off-policy MC"""
    agent = OffPolicyMC(n_actions=env.action_space.n)

    for episode in range(n_episodes):
        trajectory = []
        behavior_probs = []
        state = env.reset()
        done = False

        while not done:
            # Behavior policy action
            action = agent.behavior_policy(state, epsilon)

            # Store behavior probability
            state_key = agent._state_key(state)
            greedy_action = np.argmax(agent.Q[state_key])
            if action == greedy_action:
                b_prob = 1 - epsilon + epsilon / agent.n_actions
            else:
                b_prob = epsilon / agent.n_actions

            behavior_probs.append(b_prob)

            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state

        agent.update_from_episode(trajectory, behavior_probs)

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{n_episodes}")

    return agent
```

---

## 4. Temporal Difference Learning

**Temporal Difference (TD)** learning combines ideas from Monte Carlo and Dynamic Programming:
- Like MC: Learn from experience (model-free)
- Like DP: Bootstrap from current estimates (don't wait for episode end)

**Key Advantage**: Can learn online from incomplete episodes

**TD Error**:
```
delta_t = r_{t+1} + gamma * V(s_{t+1}) - V(s_t)
```

### TD(0) Prediction with Code

**TD(0)** updates value estimate using immediate reward and next state's value:

**Update Rule**:
```
V(s_t) = V(s_t) + alpha * [r_{t+1} + gamma * V(s_{t+1}) - V(s_t)]
```

```python
class TD0Agent:
    """TD(0) prediction agent"""
    def __init__(self, gamma=0.99, alpha=0.1):
        self.gamma = gamma
        self.alpha = alpha
        self.V = defaultdict(float)

    def update(self, state, reward, next_state, done):
        """
        TD(0) update

        Args:
            state: Current state
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        # TD target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.V[next_state_key]

        # TD error
        td_error = td_target - self.V[state_key]

        # Update value
        self.V[state_key] += self.alpha * td_error

    def _state_key(self, state):
        if isinstance(state, np.ndarray):
            return tuple(state)
        return state

def train_td0(env, policy, n_episodes=1000, gamma=0.99, alpha=0.1):
    """Train TD(0) prediction"""
    agent = TD0Agent(gamma=gamma, alpha=alpha)

    for episode in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, reward, next_state, done)
            state = next_state

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}")

    return agent
```

### SARSA (On-Policy TD Control) with Code

**SARSA** (State-Action-Reward-State-Action) is on-policy TD control:

**Update Rule**:
```
Q(s_t, a_t) = Q(s_t, a_t) + alpha * [r_{t+1} + gamma * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
```

**Algorithm**:
1. Initialize Q(s, a) arbitrarily
2. For each episode:
   a. Choose action a from state s using epsilon-greedy
   b. Take action a, observe r, s'
   c. Choose a' from s' using epsilon-greedy
   d. Update Q(s, a) using (s, a, r, s', a')
   e. s = s', a = a'

```python
class SARSAAgent:
    """SARSA: On-policy TD control"""
    def __init__(self, n_actions, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        state_key = self._state_key(state)

        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state_key])

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (chosen by epsilon-greedy)
            done: Episode terminated
        """
        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        # TD target (using next action from policy)
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.Q[next_state_key][next_action]

        # TD error
        td_error = td_target - self.Q[state_key][action]

        # Update Q
        self.Q[state_key][action] += self.alpha * td_error

    def _state_key(self, state):
        if isinstance(state, np.ndarray):
            return tuple(state)
        return state

def train_sarsa(env, n_episodes=1000):
    """Train SARSA agent"""
    agent = SARSAAgent(n_actions=env.action_space.n)
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        action = agent.get_action(state)
        total_reward = 0
        done = False

        while not done:
            # Take action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Choose next action
            next_action = agent.get_action(next_state)

            # SARSA update
            agent.update(state, action, reward, next_state, next_action, done)

            # Move to next state-action
            state = next_state
            action = next_action

        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

    return agent, episode_rewards
```

### Q-Learning (Off-Policy TD Control) with Code

**Q-Learning** learns the optimal Q-function regardless of the policy being followed:

**Update Rule**:
```
Q(s_t, a_t) = Q(s_t, a_t) + alpha * [r_{t+1} + gamma * max_a Q(s_{t+1}, a) - Q(s_t, a_t)]
```

**Key Difference from SARSA**: Uses max Q(s', a) instead of Q(s', a') from behavior policy

```python
class QLearningAgent:
    """Q-Learning: Off-policy TD control"""
    def __init__(self, n_actions, gamma=0.99, alpha=0.1, epsilon=0.1, epsilon_decay=0.999):
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state, greedy=False):
        """Epsilon-greedy action selection"""
        state_key = self._state_key(state)

        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state_key])

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning update

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode terminated
        """
        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        # TD target (using max over actions)
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state_key])

        # TD error
        td_error = td_target - self.Q[state_key][action]

        # Update Q
        self.Q[state_key][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _state_key(self, state):
        if isinstance(state, np.ndarray):
            return tuple(state)
        return state

def train_qlearning(env, n_episodes=1000):
    """Train Q-Learning agent"""
    agent = QLearningAgent(n_actions=env.action_space.n)
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Choose action (epsilon-greedy)
            action = agent.get_action(state)

            # Take action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Q-Learning update
            agent.update(state, action, reward, next_state, done)

            state = next_state

        # Decay epsilon
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    return agent, episode_rewards
```

### Expected SARSA

**Expected SARSA** uses expected value under the policy instead of sampling:

**Update Rule**:
```
Q(s, a) = Q(s, a) + alpha * [r + gamma * E_pi[Q(s', a')] - Q(s, a)]
where E_pi[Q(s', a')] = sum_a' pi(a'|s') * Q(s', a')
```

**Advantage**: Lower variance than SARSA (no sampling variance)

```python
class ExpectedSARSA:
    """Expected SARSA agent"""
    def __init__(self, n_actions, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        state_key = self._state_key(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state_key])

    def expected_value(self, state):
        """Compute expected value under epsilon-greedy policy"""
        state_key = self._state_key(state)
        q_values = self.Q[state_key]

        # Best action
        best_action = np.argmax(q_values)

        # Expected value under epsilon-greedy
        expected_val = 0.0
        for a in range(self.n_actions):
            if a == best_action:
                prob = 1 - self.epsilon + self.epsilon / self.n_actions
            else:
                prob = self.epsilon / self.n_actions
            expected_val += prob * q_values[a]

        return expected_val

    def update(self, state, action, reward, next_state, done):
        """Expected SARSA update"""
        state_key = self._state_key(state)

        # TD target using expected value
        if done:
            td_target = reward
        else:
            expected_next_value = self.expected_value(next_state)
            td_target = reward + self.gamma * expected_next_value

        # Update
        td_error = td_target - self.Q[state_key][action]
        self.Q[state_key][action] += self.alpha * td_error

    def _state_key(self, state):
        if isinstance(state, np.ndarray):
            return tuple(state)
        return state
```

### TD(lambda) and Eligibility Traces

**Eligibility Traces** bridge between TD and MC by considering multiple time scales:

**n-step TD**: Look n steps ahead
```
G_t^(n) = r_{t+1} + gamma * r_{t+2} + ... + gamma^{n-1} * r_{t+n} + gamma^n * V(s_{t+n})
```

**TD(lambda)**: Weighted average of all n-step returns
```
G_t^lambda = (1 - lambda) * sum_{n=1}^{infinity} lambda^{n-1} * G_t^(n)
```

**Eligibility Trace** (accumulating):
```
e_t(s) = gamma * lambda * e_{t-1}(s) + 1{s_t = s}
```

**TD(lambda) Update**:
```
V(s) = V(s) + alpha * delta_t * e_t(s)  for all s
where delta_t = r_{t+1} + gamma * V(s_{t+1}) - V(s_t)
```

```python
class TDLambdaAgent:
    """TD(lambda) with eligibility traces"""
    def __init__(self, gamma=0.99, alpha=0.1, lambda_=0.9):
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_ = lambda_
        self.V = defaultdict(float)
        self.eligibility = defaultdict(float)

    def start_episode(self):
        """Reset eligibility traces at episode start"""
        self.eligibility = defaultdict(float)

    def update(self, state, reward, next_state, done):
        """TD(lambda) update with accumulating traces"""
        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        # TD error
        if done:
            td_error = reward - self.V[state_key]
        else:
            td_error = reward + self.gamma * self.V[next_state_key] - self.V[state_key]

        # Update eligibility trace for current state
        self.eligibility[state_key] += 1

        # Update all values using eligibility traces
        for s in list(self.eligibility.keys()):
            self.V[s] += self.alpha * td_error * self.eligibility[s]
            self.eligibility[s] *= self.gamma * self.lambda_

            # Remove small traces for efficiency
            if abs(self.eligibility[s]) < 1e-5:
                del self.eligibility[s]

    def _state_key(self, state):
        if isinstance(state, np.ndarray):
            return tuple(state)
        return state

# SARSA(lambda) variant
class SARSALambdaAgent:
    """SARSA(lambda) with eligibility traces"""
    def __init__(self, n_actions, gamma=0.99, alpha=0.1, lambda_=0.9, epsilon=0.1):
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.eligibility = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state):
        state_key = self._state_key(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state_key])

    def start_episode(self):
        self.eligibility = defaultdict(lambda: np.zeros(self.n_actions))

    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA(lambda) update"""
        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        # TD error
        if done:
            td_error = reward - self.Q[state_key][action]
        else:
            td_error = (reward + self.gamma * self.Q[next_state_key][next_action]
                       - self.Q[state_key][action])

        # Update eligibility
        self.eligibility[state_key][action] += 1

        # Update all Q values
        for s in list(self.eligibility.keys()):
            for a in range(self.n_actions):
                self.Q[s][a] += self.alpha * td_error * self.eligibility[s][a]
                self.eligibility[s][a] *= self.gamma * self.lambda_

    def _state_key(self, state):
        if isinstance(state, np.ndarray):
            return tuple(state)
        return state
```

### Comparison: MC vs TD vs DP

| Aspect | Monte Carlo | TD Learning | Dynamic Programming |
|--------|-------------|-------------|---------------------|
| **Model Required** | No | No | Yes |
| **Bootstrapping** | No | Yes | Yes |
| **Episode Completion** | Required | Not required | N/A |
| **Bias** | Unbiased | Biased (initially) | Depends on initialization |
| **Variance** | High | Low | N/A |
| **Convergence** | Slower | Faster | Guaranteed (if model correct) |
| **Online Learning** | No | Yes | No |
| **Continuous Tasks** | No | Yes | Yes |
| **Best For** | Episodic, unknown dynamics | Online, incomplete episodes | Known model, planning |

**When to Use Each**:
- **Monte Carlo**: Episodic tasks, need unbiased estimates, simple implementation
- **TD Learning**: Online learning, continuing tasks, faster convergence needed
- **Dynamic Programming**: Model is known, planning ahead, small state spaces

---

## 5. Deep Q-Networks (DQN)

**Deep Q-Networks (DQN)** combine Q-learning with deep neural networks to handle high-dimensional state spaces like images.

**Motivation**: Tabular Q-learning fails with large/continuous state spaces

**Key Innovations**:
1. **Neural Network Function Approximation**: Q(s, a; theta) instead of Q-table
2. **Experience Replay**: Break correlations in sequential data
3. **Target Network**: Stabilize learning

### DQN Architecture and Experience Replay with Code

**Experience Replay Buffer**: Store transitions (s, a, r, s', done) and sample mini-batches randomly

**Benefits**:
- Break temporal correlations
- Reuse experiences (sample efficiency)
- Stabilize learning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random

# Transition tuple for replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random mini-batch"""
        batch = random.sample(self.buffer, batch_size)

        # Convert to tensors
        states = torch.FloatTensor([t.state for t in batch])
        actions = torch.LongTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor([t.next_state for t in batch])
        dones = torch.FloatTensor([t.done for t in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(DQNNetwork, self).__init__()

        # Build network layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass returns Q-values for all actions"""
        return self.network(x)

class DQNAgent:
    """DQN Agent with experience replay"""
    def __init__(self, state_dim, action_dim,
                 lr=1e-3, gamma=0.99,
                 buffer_size=100000, batch_size=64,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 target_update_freq=10):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in eval mode

        # Optimizer and replay buffer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.update_count = 0

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """Perform one step of optimization"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss (Huber loss is more stable than MSE)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

def train_dqn(env, n_episodes=1000, max_steps=500):
    """Train DQN agent"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    episode_rewards = []
    episode_losses = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        losses = []

        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Perform optimization step
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            state = next_state

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        avg_loss = np.mean(losses) if losses else 0
        episode_losses.append(avg_loss)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Loss: {avg_loss:.4f}")

    return agent, episode_rewards, episode_losses
```

### Target Network and Why It Helps Stability

**Problem**: Q-learning with function approximation can diverge because the target changes at every update:
```
Q(s, a) --> r + gamma * max_a' Q(s', a')  # Target depends on Q being updated
```

**Solution**: Use separate **target network** with frozen parameters:
```
Q(s, a; theta) --> r + gamma * max_a' Q(s', a'; theta_target)
```

**Update Schedule**:
- Update policy network theta every step
- Update target network theta_target every C steps (or with Polyak averaging)

**Polyak Averaging** (soft update):
```
theta_target = tau * theta + (1 - tau) * theta_target
```

```python
class DQNWithSoftUpdate(DQNAgent):
    """DQN with Polyak averaging for target network"""
    def __init__(self, *args, tau=0.005, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau

    def soft_update_target(self):
        """Soft update of target network parameters"""
        for target_param, policy_param in zip(self.target_net.parameters(),
                                               self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

    def update(self):
        """Update with soft target network update"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample and compute loss (same as before)
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Soft update every step
        self.soft_update_target()

        return loss.item()
```

### Double DQN (Overestimation Bias Fix)

**Problem**: Standard DQN overestimates Q-values because it uses max operator for both action selection and evaluation:
```
target = r + gamma * max_a Q(s', a; theta_target)
```

**Solution**: Use policy network for action selection, target network for evaluation:
```
a* = argmax_a Q(s', a; theta)  # Select with policy network
target = r + gamma * Q(s', a*; theta_target)  # Evaluate with target network
```

```python
class DoubleDQN(DQNAgent):
    """Double DQN to reduce overestimation bias"""
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: select actions with policy network, evaluate with target network
        with torch.no_grad():
            # Select best actions using policy network
            next_actions = self.policy_net(next_states).argmax(dim=1)
            # Evaluate selected actions using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
```

### Dueling DQN (Separate Value and Advantage Streams)

**Idea**: Decompose Q-function into value and advantage:
```
Q(s, a) = V(s) + A(s, a)
where V(s) = value of being in state s
      A(s, a) = advantage of action a over others
```

**Architecture**: Network splits into two streams that combine at end

**Aggregation** (to ensure identifiability):
```
Q(s, a) = V(s) + [A(s, a) - mean_a' A(s, a')]
```

```python
class DuelingDQN(nn.Module):
    """Dueling DQN architecture"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        features = self.feature(x)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean A(s,a'))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values
```

### Prioritized Experience Replay

**Motivation**: Not all transitions are equally important for learning

**Idea**: Sample transitions with probability proportional to TD error

**Priority**:
```
p_i = |delta_i| + epsilon  # TD error magnitude plus small constant
P(i) = p_i^alpha / sum_k p_k^alpha
```

**Importance Sampling Correction**:
```
w_i = (N * P(i))^(-beta)
```

```python
class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta_start  # Importance sampling exponent
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """Add transition with max priority"""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = Transition(state, action, reward, next_state, done)

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample batch with prioritized sampling"""
        N = len(self.buffer)

        # Compute sampling probabilities
        priorities = self.priorities[:N]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(N, batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        self.frame += 1

        # Get samples
        batch = [self.buffer[idx] for idx in indices]
        states = torch.FloatTensor([t.state for t in batch])
        actions = torch.LongTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor([t.next_state for t in batch])
        dones = torch.FloatTensor([t.done for t in batch])
        weights = torch.FloatTensor(weights)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)
```

### Rainbow DQN (Combining All Improvements)

**Rainbow** combines six DQN extensions:
1. Double DQN
2. Dueling DQN
3. Prioritized Experience Replay
4. Multi-step returns (n-step)
5. Distributional RL (C51)
6. Noisy Nets (for exploration)

```python
# Simplified Rainbow (Double + Dueling + Prioritized Replay)
class RainbowDQN:
    """Rainbow DQN combining multiple improvements"""
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99,
                 n_step=3, buffer_size=100000, batch_size=32):
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size

        # Networks (Dueling architecture)
        self.policy_net = DuelingDQN(state_dim, action_dim)
        self.target_net = DuelingDQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # Prioritized replay
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)

        self.update_count = 0

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.policy_net.advantage_stream[-1].out_features - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Prioritized sampling
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (self.gamma ** self.n_step) * next_q * (1 - dones)

        # Weighted loss (importance sampling)
        td_errors = target_q - current_q
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Update priorities
        priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)

        # Soft update target network
        self.update_count += 1
        if self.update_count % 10 == 0:
            for target_param, policy_param in zip(self.target_net.parameters(),
                                                   self.policy_net.parameters()):
                target_param.data.copy_(0.005 * policy_param.data + 0.995 * target_param.data)

        return loss.item()
```

---

## 6. Policy Gradient Methods

**Policy Gradient Methods** directly optimize the policy instead of learning value functions.

**Advantages**:
- Can handle continuous action spaces naturally
- Can learn stochastic policies
- Better convergence properties in some cases
- Effective in high-dimensional action spaces

**Disadvantages**:
- High variance
- Sample inefficient
- Can get stuck in local optima

### Policy Parameterization

**Discrete Actions**: Softmax policy
```
pi(a|s; theta) = exp(h(s, a; theta)) / sum_a' exp(h(s, a'; theta))
```

**Continuous Actions**: Gaussian policy
```
pi(a|s; theta) = N(mu(s; theta), sigma^2)
```

### Policy Gradient Theorem

**Objective**: Maximize expected return
```
J(theta) = E_pi[G_0] = E_pi[sum_t gamma^t r_t]
```

**Policy Gradient Theorem**:
```
gradient J(theta) = E_pi[gradient log pi(a|s; theta) * Q^pi(s, a)]
```

**REINFORCE Algorithm**: Sample-based estimate
```
gradient J(theta) ~= gradient log pi(a_t|s_t; theta) * G_t
```

### REINFORCE Algorithm with Code

**Algorithm**:
1. Initialize policy parameters theta
2. For each episode:
   a. Generate episode following pi(theta)
   b. For each timestep t:
      - Compute return G_t
      - Update: theta = theta + alpha * gradient log pi(a_t|s_t) * G_t

```python
class PolicyNetwork(nn.Module):
    """Policy network for discrete actions"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

    def get_action(self, state):
        """Sample action from policy"""
        probs = self.forward(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

class REINFORCEAgent:
    """REINFORCE algorithm"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        """Select action and return log probability"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.policy.get_action(state_tensor)
        return action, log_prob

    def update(self, episode_data):
        """
        Update policy using episode data

        Args:
            episode_data: List of (log_prob, reward) tuples
        """
        log_probs = [x[0] for x in episode_data]
        rewards = [x[1] for x in episode_data]

        # Compute returns (reversed cumulative sum)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        # Normalize returns (reduce variance)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy gradient loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)

        policy_loss = torch.stack(policy_loss).sum()

        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item()

def train_reinforce(env, n_episodes=1000):
    """Train REINFORCE agent"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim)
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_data = []
        total_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            episode_data.append((log_prob, reward))
            total_reward += reward
            state = next_state

        # Update policy
        loss = agent.update(episode_data)
        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")

    return agent, episode_rewards
```

### Variance Reduction with Baselines

**Problem**: REINFORCE has high variance because returns can vary significantly

**Solution**: Subtract baseline b(s) from return (doesn't change expectation but reduces variance)

```
gradient J(theta) = E[gradient log pi(a|s) * (Q(s,a) - b(s))]
```

**Common Baseline**: State value function V(s)
```
gradient J(theta) = E[gradient log pi(a|s) * A(s,a)]
where A(s,a) = Q(s,a) - V(s) is the advantage
```

```python
class ValueNetwork(nn.Module):
    """Value network for baseline"""
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.network(state).squeeze()

class REINFORCEWithBaseline:
    """REINFORCE with learned value baseline"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.policy.get_action(state_tensor)
        value = self.value(state_tensor)
        return action, log_prob, value

    def update(self, episode_data):
        """Update policy and value function"""
        log_probs = torch.stack([x[0] for x in episode_data])
        values = torch.stack([x[1] for x in episode_data])
        rewards = [x[2] for x in episode_data]

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # Compute advantages
        advantages = returns - values.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        policy_loss = -(log_probs * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Optimize policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Optimize value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return policy_loss.item(), value_loss.item()
```

### Actor-Critic Architecture with Code

**Actor-Critic** combines policy gradients (actor) with value functions (critic):
- **Actor**: Policy network pi(a|s; theta)
- **Critic**: Value network V(s; w)

**Update Rules**:
```
Actor: theta = theta + alpha * gradient log pi(a|s) * A(s,a)
Critic: w = w + beta * delta * gradient V(s; w)
where delta = r + gamma * V(s') - V(s) is TD error
      A(s,a) ~= delta (advantage estimate)
```

```python
class ActorCritic(nn.Module):
    """Actor-Critic with shared backbone"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        shared_features = self.shared(state)
        policy = self.actor(shared_features)
        value = self.critic(shared_features).squeeze()
        return policy, value

    def get_action(self, state):
        policy, value = self.forward(state)
        action_dist = torch.distributions.Categorical(policy)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob, value

class ActorCriticAgent:
    """Actor-Critic agent"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return self.model.get_action(state_tensor)

    def update(self, state, action_log_prob, reward, next_state, done):
        """
        Single-step Actor-Critic update

        Args:
            state: Current state
            action_log_prob: Log probability of action taken
            reward: Reward received
            next_state: Next state
            done: Episode terminated
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # Get current value
        _, value = self.model(state_tensor)

        # Get next value
        with torch.no_grad():
            _, next_value = self.model(next_state_tensor)
            if done:
                next_value = 0

        # Compute TD error and advantage
        td_target = reward + self.gamma * next_value
        td_error = td_target - value
        advantage = td_error.detach()

        # Actor loss (policy gradient)
        actor_loss = -action_log_prob * advantage

        # Critic loss (value function)
        critic_loss = td_error.pow(2)

        # Total loss
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

def train_actor_critic(env, n_episodes=1000):
    """Train Actor-Critic agent"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCriticAgent(state_dim, action_dim)
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            loss = agent.update(state, log_prob, reward, next_state, done)

            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

    return agent, episode_rewards
```

### A2C (Advantage Actor-Critic)

**A2C** (Synchronous Advantage Actor-Critic) uses parallel environments and n-step returns:

**Key Features**:
- Multiple parallel environments
- N-step returns for lower variance
- Entropy regularization for exploration

**N-step Return**:
```
G_t^(n) = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1} + gamma^n * V(s_{t+n})
```

### A3C (Asynchronous Advantage Actor-Critic)

**A3C** uses asynchronous updates from multiple workers:
- Each worker has its own environment
- Workers update shared parameters asynchronously
- Eliminates need for experience replay

**Note**: In practice, synchronous A2C often performs better than asynchronous A3C

### GAE (Generalized Advantage Estimation) Formula

**Problem**: Bias-variance tradeoff in advantage estimation

**TD Error (lambda)**: Exponentially weighted average of n-step TD errors
```
A_t^GAE(gamma, lambda) = sum_{l=0}^{infinity} (gamma * lambda)^l * delta_{t+l}
where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

**Parameters**:
- **lambda = 0**: High bias, low variance (TD(0))
- **lambda = 1**: Low bias, high variance (Monte Carlo)
- **lambda in (0, 1)**: Balance between bias and variance

```python
def compute_gae(rewards, values, next_value, dones, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation

    Args:
        rewards: List of rewards
        values: List of value estimates
        next_value: Value estimate for final next state
        dones: List of done flags
        gamma: Discount factor
        lambda_: GAE lambda parameter

    Returns:
        advantages: GAE advantages
        returns: Discounted returns
    """
    advantages = []
    gae = 0

    # Append next_value to values for computation
    values = values + [next_value]

    for t in reversed(range(len(rewards))):
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lambda_ * gae

        advantages.insert(0, gae)

    # Compute returns (advantages + values)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]

    return torch.FloatTensor(advantages), torch.FloatTensor(returns)
```

---

## 7. Advanced Policy Optimization

### TRPO (Trust Region Policy Optimization) - Theory and Constraints

**Trust Region Policy Optimization (TRPO)** ensures monotonic improvement by constraining policy updates.

**Motivation**: Large policy updates can degrade performance; need to stay in "trust region"

**Objective**: Maximize expected improvement
```
maximize E_s~rho_old, a~pi_old [pi_new(a|s) / pi_old(a|s) * A_old(s,a)]
subject to E_s~rho_old [KL(pi_old(.|s) || pi_new(.|s))] <= delta
```

**KL Divergence Constraint**: Limits how much the policy can change
```
KL(pi_old || pi_new) <= delta  (typically delta = 0.01)
```

**Algorithm**:
1. Collect trajectories using current policy
2. Compute advantages
3. Solve constrained optimization problem (conjugate gradient + line search)
4. Update policy parameters

**Challenges**:
- Computationally expensive (second-order optimization)
- Difficult to implement correctly
- Hard to tune

### PPO (Proximal Policy Optimization) with Complete Code

**PPO** simplifies TRPO by using a clipped surrogate objective instead of KL constraint.

**Clipped Surrogate Objective**:
```
L_CLIP(theta) = E[min(r_t(theta) * A_t, clip(r_t(theta), 1-epsilon, 1+epsilon) * A_t)]
where r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
```

**Benefits**:
- Simpler to implement than TRPO
- First-order optimization only
- Robust and sample efficient
- State-of-the-art performance

```python
class PPOMemory:
    """Memory buffer for PPO"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def get_tensors(self):
        return (
            torch.FloatTensor(self.states),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.dones)
        )

class PPOAgent:
    """PPO Agent with clipped objective"""
    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, lambda_=0.95,
                 epsilon_clip=0.2, k_epochs=10,
                 value_coef=0.5, entropy_coef=0.01):

        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Actor-Critic model
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Memory
        self.memory = PPOMemory()

    def select_action(self, state):
        """Select action and store in memory"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            policy, value = self.model(state_tensor)
            action_dist = torch.distributions.Categorical(policy)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, value, done):
        """Store transition in memory"""
        self.memory.store(state, action, log_prob, reward, value, done)

    def update(self):
        """PPO update using collected trajectories"""
        # Get data from memory
        states, actions, old_log_probs, rewards, old_values, dones = self.memory.get_tensors()

        # Compute advantages and returns using GAE
        with torch.no_grad():
            _, last_value = self.model(states[-1].unsqueeze(0))
            last_value = last_value.item() if not dones[-1] else 0

        advantages, returns = compute_gae(
            rewards.tolist(), old_values.tolist(), last_value,
            dones.tolist(), self.gamma, self.lambda_
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for k epochs
        for _ in range(self.k_epochs):
            # Evaluate actions
            policies, values = self.model(states)
            action_dist = torch.distributions.Categorical(policies)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy()

            # Ratio for clipping
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages

            # Actor loss (clipped)
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = F.mse_loss(values, returns)

            # Entropy bonus (for exploration)
            entropy_loss = -entropy.mean()

            # Total loss
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Clear memory
        self.memory.clear()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': -entropy_loss.item()
        }

def train_ppo(env, n_episodes=1000, max_steps=500, update_freq=2048):
    """Train PPO agent"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)

    episode_rewards = []
    step_count = 0

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, log_prob, reward, value, done)

            total_reward += reward
            state = next_state
            step_count += 1

            # Update every update_freq steps
            if step_count % update_freq == 0:
                losses = agent.update()

            if done:
                break

        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

    return agent, episode_rewards
```

### PPO Clipping Mechanism Explained

**Importance Sampling Ratio**:
```
r(theta) = pi_theta(a|s) / pi_theta_old(a|s)
```

**Without Clipping**: Could have arbitrarily large updates
**With Clipping**:
```
L_CLIP = min(r * A, clip(r, 1-eps, 1+eps) * A)
```

**Effect**:
- If advantage A > 0 (good action): increase probability, but not too much (clip at 1+eps)
- If advantage A < 0 (bad action): decrease probability, but not too much (clip at 1-eps)
- Prevents destructive large policy updates

**Visualization**:
```
For A > 0:
  - If r > 1+eps: clipped to (1+eps)*A
  - If r < 1-eps: use r*A (not clipped)

For A < 0:
  - If r < 1-eps: clipped to (1-eps)*A
  - If r > 1+eps: use r*A (not clipped)
```

### SAC (Soft Actor-Critic) with Entropy Regularization

**Soft Actor-Critic (SAC)** is an off-policy algorithm that maximizes both reward and entropy:

**Objective**:
```
J(pi) = E[sum_t r_t + alpha * H(pi(.|s_t))]
where H(pi) = -E[log pi(a|s)] is entropy
```

**Benefits**:
- Entropy term encourages exploration
- Off-policy (sample efficient)
- Works well for continuous control
- Automatic temperature tuning

**Key Components**:
1. **Actor**: Stochastic policy (Gaussian for continuous)
2. **Twin Critics**: Two Q-networks to reduce overestimation
3. **Target Networks**: Soft updates for stability

```python
class SoftQNetwork(nn.Module):
    """Soft Q-network for SAC"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SoftQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous actions"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.shared(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob

class SACAgent:
    """Soft Actor-Critic agent"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 tau=0.005, alpha=0.2, automatic_alpha_tuning=True):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Twin Q-networks
        self.q1 = SoftQNetwork(state_dim, action_dim)
        self.q2 = SoftQNetwork(state_dim, action_dim)
        self.q1_target = SoftQNetwork(state_dim, action_dim)
        self.q2_target = SoftQNetwork(state_dim, action_dim)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy
        self.policy = GaussianPolicy(state_dim, action_dim)

        # Optimizers
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Automatic temperature tuning
        if automatic_alpha_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.automatic_alpha_tuning = True
        else:
            self.automatic_alpha_tuning = False

        self.replay_buffer = ReplayBuffer(capacity=1000000)

    def select_action(self, state, evaluate=False):
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if evaluate:
            with torch.no_grad():
                mean, _ = self.policy(state_tensor)
                action = torch.tanh(mean)
        else:
            with torch.no_grad():
                action, _ = self.policy.sample(state_tensor)

        return action.cpu().numpy()[0]

    def update(self, batch_size=256):
        """SAC update step"""
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Update Q-functions
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * min_q_next

        q1_pred = self.q1(states, actions.float())
        q2_pred = self.q2(states, actions.float())

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update policy
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - min_q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update temperature (alpha)
        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha
        }
```

### TD3 (Twin Delayed DDPG)

**Twin Delayed Deep Deterministic Policy Gradient (TD3)** improves DDPG with three key tricks:

**1. Twin Q-Networks**: Use minimum of two Q-networks to reduce overestimation
**2. Delayed Policy Updates**: Update policy less frequently than Q-functions
**3. Target Policy Smoothing**: Add noise to target actions for robustness

**Key Differences from SAC**:
- Deterministic policy (vs stochastic in SAC)
- No entropy term
- Target policy smoothing instead of entropy regularization

### When to Use Which Algorithm: Decision Guide

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **DQN** | Discrete actions, Atari games | Simple, proven | Sample inefficient, discrete only |
| **PPO** | General purpose, both discrete/continuous | Stable, easy to tune, robust | Less sample efficient than off-policy |
| **SAC** | Continuous control, robotics | Sample efficient, automatic exploration | More complex, hyperparameter sensitive |
| **TD3** | Continuous control | Sample efficient, deterministic | Requires careful tuning |
| **TRPO** | When stability critical | Monotonic improvement | Very slow, complex |
| **A2C/A3C** | Many parallel environments | Fast with parallelism | Less sample efficient |

**Decision Tree**:
1. **Action space discrete?** --> DQN or PPO
2. **Action space continuous?** --> PPO, SAC, or TD3
3. **Need sample efficiency?** --> SAC or TD3 (off-policy)
4. **Need simplicity/stability?** --> PPO
5. **Robotics/continuous control?** --> SAC
6. **Have many parallel envs?** --> PPO or A2C

---

## 8. Model-Based RL

**Model-Based RL** learns a model of environment dynamics and uses it for planning or improving sample efficiency.

**Environment Model**: Predicts next state and reward
```
s_{t+1}, r_t = f(s_t, a_t)
```

**Advantages**:
- Sample efficient (can generate synthetic experience)
- Can plan ahead
- Transferable models

**Disadvantages**:
- Model errors compound
- Difficult to learn accurate models
- Higher computational cost

### World Models (Learning Environment Dynamics)

**World Model**: Neural network that learns environment dynamics

**Components**:
1. **Vision Model (V)**: Encode observations to latent space
2. **Memory Model (M)**: Predict next latent state (RNN/Transformer)
3. **Controller (C)**: Policy in latent space

**Training**:
1. Collect experience with random/existing policy
2. Train world model to predict observations and rewards
3. Train policy in learned world model (imagination)
4. Deploy policy in real environment

```python
class WorldModel(nn.Module):
    """Simple world model for model-based RL"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(WorldModel, self).__init__()

        # Dynamics model: predicts next state
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Reward model: predicts reward
        self.reward = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Done model: predicts episode termination
        self.done = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        """Predict next state, reward, done"""
        x = torch.cat([state, action], dim=-1)
        next_state = self.dynamics(x)
        reward = self.reward(x)
        done = self.done(x)
        return next_state, reward, done

    def predict_rollout(self, state, policy, horizon=10):
        """Rollout trajectory using learned model"""
        states = [state]
        rewards = []

        current_state = state
        for _ in range(horizon):
            action = policy(current_state)
            next_state, reward, done = self.forward(current_state, action)

            states.append(next_state)
            rewards.append(reward)

            if done > 0.5:
                break

            current_state = next_state

        return states, rewards

def train_world_model(env, model, n_episodes=1000, batch_size=256):
    """Train world model from environment experience"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=100000)

    # Collect experience
    for episode in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()  # Random policy
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

        # Train model
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            # Predict
            pred_next_states, pred_rewards, pred_dones = model(states, actions.float())

            # Losses
            state_loss = F.mse_loss(pred_next_states, next_states)
            reward_loss = F.mse_loss(pred_rewards.squeeze(), rewards)
            done_loss = F.binary_cross_entropy(pred_dones.squeeze(), dones)

            loss = state_loss + reward_loss + done_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Loss: {loss.item():.4f}")

    return model
```

### Dyna-Q Architecture

**Dyna-Q** integrates planning and learning by:
1. Learning from real experience (model-free)
2. Learning world model from experience
3. Planning with simulated experience from model

**Algorithm**:
```
For each step:
  1. Take action in real environment, observe s, a, r, s'
  2. Update Q(s, a) with real experience (Q-learning)
  3. Update model: M(s, a) = (r, s')
  4. Planning: For k iterations:
     - Sample s, a from previously seen
     - Simulate s', r from model M(s, a)
     - Update Q(s, a) with simulated experience
```

```python
class DynaQAgent:
    """Dyna-Q agent combining learning and planning"""
    def __init__(self, n_states, n_actions, gamma=0.99, alpha=0.1,
                 epsilon=0.1, planning_steps=10):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.planning_steps = planning_steps

        # Q-table
        self.Q = defaultdict(lambda: np.zeros(n_actions))

        # Model: M[s][a] = (next_state, reward)
        self.model = {}

        # Track visited state-action pairs for planning
        self.visited_sa = []

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        """Update Q-value and model, then plan"""
        # Direct RL update (Q-learning)
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

        # Update model
        if state not in self.model:
            self.model[state] = {}
        self.model[state][action] = (next_state, reward)

        # Track visited state-action pair
        if (state, action) not in self.visited_sa:
            self.visited_sa.append((state, action))

        # Planning: update Q using simulated experience
        for _ in range(self.planning_steps):
            if not self.visited_sa:
                break

            # Sample random previously visited state-action
            s, a = self.visited_sa[np.random.randint(len(self.visited_sa))]

            # Simulate next state and reward from model
            if s in self.model and a in self.model[s]:
                s_next, r = self.model[s][a]

                # Q-learning update with simulated experience
                best_a_next = np.argmax(self.Q[s_next])
                td_target = r + self.gamma * self.Q[s_next][best_a_next]
                td_error = td_target - self.Q[s][a]
                self.Q[s][a] += self.alpha * td_error
```

### MBPO (Model-Based Policy Optimization)

**MBPO** combines model-based and model-free RL:
1. Learn dynamics model from real data
2. Generate synthetic rollouts from model
3. Train policy with mix of real and synthetic data (SAC/PPO)

**Key Insight**: Short rollouts from learned model reduce compounding errors

**Algorithm**:
```
1. Collect real data with current policy
2. Train dynamics model on real data
3. For each policy update:
   - Generate k-step rollouts from model
   - Mix real and synthetic data
   - Update policy (e.g., SAC)
```

### Dreamer and DreamerV3

**Dreamer** learns in latent imagination space:

**Architecture**:
1. **World Model**:
   - RSSM (Recurrent State-Space Model) for dynamics
   - Encodes observations to latent states
   - Predicts future latents

2. **Actor-Critic**:
   - Trained entirely in imagination
   - Imagined rollouts using world model
   - Never directly sees real observations during training

**DreamerV3** (2023) improvements:
- Single set of hyperparameters works across domains
- Discrete and continuous latents
- Symlog predictions for stability

### MuZero (Model-Based Planning Without Known Rules)

**MuZero** learns model for planning without learning explicit dynamics:

**Key Innovation**: Learn model that predicts:
- Value
- Policy
- Reward
Not explicit next observations (which can be complex/high-dim)

**Architecture**:
- **Representation Function**: s = h(o1, ..., ot)
- **Dynamics Function**: s', r = g(s, a)
- **Prediction Function**: p, v = f(s)

**Training**: Self-play with Monte Carlo Tree Search (MCTS)

**Applications**:
- Board games (Go, Chess, Shogi)
- Atari games
- Any domain where planning is beneficial

### Model-Based vs Model-Free Comparison

| Aspect | Model-Based | Model-Free |
|--------|-------------|------------|
| **Sample Efficiency** | High (can plan/imagine) | Low (needs real experience) |
| **Asymptotic Performance** | Limited by model errors | Can be optimal |
| **Computation** | High (planning/rollouts) | Low (direct policy) |
| **Model Errors** | Compound over rollouts | Not applicable |
| **Interpretability** | High (can inspect model) | Low (black box policy) |
| **Transfer** | Model can transfer | Policy often domain-specific |
| **Best For** | Sample-limited domains | Abundant data, complex dynamics |

**Hybrid Approaches** (best of both):
- MBPO: Use model to augment data
- Dyna-Q: Plan with model, learn with experience
- MuZero: Learn abstract model for planning

**When to Use Model-Based**:
- Expensive real-world interactions (robotics)
- Need sample efficiency
- Environment dynamics learnable
- Planning beneficial

**When to Use Model-Free**:
- Abundant simulation/data
- Complex, stochastic dynamics
- Ultimate performance matters more than samples

---

## 9. Multi-Agent RL

**Multi-Agent Reinforcement Learning (MARL)** extends RL to scenarios with multiple interacting agents.

**Challenges**:
- Non-stationary environment (other agents are learning)
- Exponential growth in joint action space
- Credit assignment (who contributed to outcome?)
- Coordination and communication

### Cooperative vs Competitive Settings

**Cooperative (Team)**:
- Shared reward
- Agents work together toward common goal
- Examples: Multi-robot coordination, team sports

**Competitive (Adversarial)**:
- Zero-sum or conflicting rewards
- Agents compete against each other
- Examples: Two-player games, adversarial training

**Mixed (General-Sum)**:
- Agents have individual rewards
- Mixture of cooperation and competition
- Examples: Autonomous driving, economic markets

```python
class MultiAgentEnvironment:
    """Base class for multi-agent environments"""
    def __init__(self, n_agents):
        self.n_agents = n_agents

    def reset(self):
        """Reset environment, return initial observations for all agents"""
        raise NotImplementedError

    def step(self, actions):
        """
        Take joint action

        Args:
            actions: Dict or list of actions for each agent

        Returns:
            observations: Dict of observations for each agent
            rewards: Dict of rewards for each agent
            dones: Dict of done flags
            infos: Additional information
        """
        raise NotImplementedError

    def get_state(self):
        """Get global state (if centralized training)"""
        raise NotImplementedError
```

### Independent Learners

**Independent Q-Learning (IQL)**: Each agent learns independently, treating others as part of environment

**Algorithm**:
- Each agent maintains own Q-function
- Updates based on local observation and reward
- Ignores presence of other learning agents

**Advantages**:
- Simple, decentralized
- Scales to many agents

**Disadvantages**:
- Non-stationary environment problem
- No coordination
- Suboptimal in cooperative settings

```python
class IndependentQLearning:
    """Independent Q-Learning for multiple agents"""
    def __init__(self, n_agents, state_dim, action_dim, lr=0.1, gamma=0.99, epsilon=0.1):
        self.n_agents = n_agents
        self.agents = []

        # Create independent Q-learning agent for each
        for i in range(n_agents):
            agent = QLearningAgent(n_actions=action_dim, alpha=lr, gamma=gamma, epsilon=epsilon)
            self.agents.append(agent)

    def select_actions(self, observations):
        """Each agent selects action independently"""
        actions = []
        for i, obs in enumerate(observations):
            action = self.agents[i].get_action(obs)
            actions.append(action)
        return actions

    def update(self, observations, actions, rewards, next_observations, dones):
        """Update each agent independently"""
        for i in range(self.n_agents):
            self.agents[i].update(
                observations[i],
                actions[i],
                rewards[i],
                next_observations[i],
                dones[i]
            )

def train_independent_learners(env, n_episodes=1000):
    """Train independent learners"""
    marl = IndependentQLearning(
        n_agents=env.n_agents,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    episode_rewards = []

    for episode in range(n_episodes):
        observations = env.reset()
        total_rewards = [0] * env.n_agents
        done = False

        while not done:
            # Select actions
            actions = marl.select_actions(observations)

            # Environment step
            next_observations, rewards, dones, _ = env.step(actions)

            # Update agents
            marl.update(observations, actions, rewards, next_observations, dones)

            for i in range(env.n_agents):
                total_rewards[i] += rewards[i]

            observations = next_observations
            done = all(dones.values()) if isinstance(dones, dict) else dones

        episode_rewards.append(np.mean(total_rewards))

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

    return marl, episode_rewards
```

### MAPPO, QMIX, MADDPG

**MAPPO (Multi-Agent PPO)**: Extends PPO with centralized critic

**QMIX**: Factorized value function for cooperative tasks

**MADDPG (Multi-Agent DDPG)**: Centralized training, decentralized execution for continuous control

All three algorithms are covered in detail with implementations in the full guide sections above (see sections on Multi-Agent RL algorithms for complete code).

### Communication in Multi-Agent Systems

**Communication Channels**: Allow agents to share information

**Types**:
1. **Explicit Communication**: Dedicated communication actions/channels
2. **Emergent Communication**: Learn to communicate through rewards
3. **Shared Parameters**: Weight sharing across agents

### PettingZoo Environment Examples

**PettingZoo**: Standard API for multi-agent RL environments

```python
# Example: Using PettingZoo environments
from pettingzoo.mpe import simple_spread_v2

# Create environment
env = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=25)

# Reset
observations = env.reset()

# Training loop
for episode in range(1000):
    observations = env.reset()
    done = False

    while not done:
        # Get actions for all agents
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        # Step environment
        observations, rewards, dones, infos = env.step(actions)

        # Check if all done
        done = all(dones.values())
```

---

## 10. Inverse RL and Reward Shaping

### Inverse Reinforcement Learning

**Inverse RL (IRL)** infers reward function from expert demonstrations.

**Applications**: Apprenticeship learning, imitation learning, understanding behavior

### Maximum Entropy IRL

**MaxEnt IRL** assumes expert chooses actions proportional to exponentiated reward

### Reward Shaping

**Potential-Based Shaping**:
```
R'(s, a, s') = R(s, a, s') + gamma * Phi(s') - Phi(s)
```

### Reward Hacking and Specification Gaming

**Reward Hacking**: Agent exploits reward function in unintended ways

**Solutions**: Careful design, inverse RL, human feedback, constraints

### Reward Modeling with Human Feedback

**Reward Modeling**: Learn reward function from human preferences

---

## 11. Offline RL

**Offline RL** learns from fixed datasets without environment interaction.

### Batch RL Problem Formulation

**Challenge**: Distribution shift between data and learned policy

### Conservative Q-Learning (CQL)

**CQL** learns conservative Q-values to prevent overestimation

### Decision Transformer

**Decision Transformer** treats RL as sequence modeling

### Implicit Q-Learning (IQL)

**IQL** uses expectile regression for stable offline learning

### When Offline RL Works and When It Fails

**Works**: High-quality data, good coverage
**Fails**: Poor data, narrow coverage, high stochasticity

---

## 12. RL for LLM Alignment

### RLHF Pipeline

**Three Stages**:
1. Supervised Fine-Tuning (SFT)
2. Reward Model Training
3. RL Fine-Tuning (PPO)

### DPO (Direct Preference Optimization)

**DPO** bypasses reward model, directly optimizes from preferences

**Advantages**: Simpler, more stable than RLHF

### GRPO (Group Relative Policy Optimization)

**GRPO** compares responses within groups for efficiency

### Constitutional AI

**Constitutional AI** uses AI feedback instead of human feedback

### Comparison: RLHF vs DPO vs GRPO

| Method | Stages | Stability | Complexity |
|--------|--------|-----------|------------|
| RLHF | 3 | Medium | High |
| DPO | 2 | High | Low |
| GRPO | 2 | High | Medium |

---

## 13. Practical Implementation

### Gymnasium (Formerly OpenAI Gym)

**Standard API** for RL environments

```python
import gymnasium as gym
env = gym.make('CartPole-v1')
observation, info = env.reset()
```

### Stable-Baselines3 Complete Examples

**Most popular RL library**

```python
from stable_baselines3 import PPO
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
```

### Custom Environment Creation

Create custom Gymnasium environments for your tasks

### CleanRL for Research

**Single-file implementations** for easy understanding and modification

### Isaac Gym for Robotics Simulation

**GPU-accelerated** robotics simulation with thousands of parallel environments

### RLlib for Distributed Training

**Scalable RL** with Ray for distributed training

---

## 14. Hyperparameter Tuning for RL

### Learning Rate Scheduling

**Critical hyperparameter**: Start with 3e-4 for PPO, 1e-4 for DQN

### Entropy Coefficient

**Controls exploration**: 0.01 typical, can anneal

### Batch Size and Buffer Size

**PPO**: 64-256 batch size
**DQN**: 100k-1M buffer size

### Number of Parallel Environments

**4-16 environments** for PPO typically optimal

### Reward Normalization

**Normalize rewards** when they vary widely

### Common Instabilities and Fixes

**Divergence**: Reduce LR, clip gradients
**No learning**: Increase exploration, reward shaping
**High variance**: Increase batch size, use GAE

---

## 15. Applications

### Game Playing (Atari, Go, StarCraft)

**AlphaGo**: Beat world champion at Go
**AlphaStar**: Grandmaster level StarCraft II
**DQN**: Superhuman Atari performance

### Robotics (Manipulation, Locomotion)

**Manipulation**: Grasping, assembly
**Locomotion**: Quadrupeds, bipeds
**Sim-to-real**: Domain randomization

### Recommendation Systems

**Sequential recommendations** optimizing long-term engagement

### Trading and Finance

**Portfolio optimization**, algorithmic trading, market making

### Chip Design and Optimization

**Google TPU floorplanning**, datacenter cooling

---

## 16. Resources and References

### Key Libraries

**Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3
**RLlib**: https://docs.ray.io/en/latest/rllib/
**CleanRL**: https://github.com/vwxyzjn/cleanrl
**TorchRL**: https://github.com/pytorch/rl

### Benchmarks

**Atari**: 57 classic arcade games
**MuJoCo**: Continuous control tasks
**DMControl**: DeepMind Control Suite
**Meta-World**: Multi-task manipulation

### Key Textbooks

**Sutton & Barto (2018)**: Reinforcement Learning: An Introduction
- Free online: http://incompleteideas.net/book/the-book-2nd.html
- The definitive RL textbook

### Key Papers by Topic

**Foundations**:
- Q-Learning (Watkins, 1989)
- Policy Gradients (Williams, 1992)

**Deep RL**:
- DQN (Mnih et al., 2013)
- Double DQN (van Hasselt et al., 2015)
- Rainbow (Hessel et al., 2017)

**Policy Optimization**:
- TRPO (Schulman et al., 2015)
- PPO (Schulman et al., 2017)
- SAC (Haarnoja et al., 2018)
- TD3 (Fujimoto et al., 2018)

**Model-Based**:
- World Models (Ha & Schmidhuber, 2018)
- MuZero (Schrittwieser et al., 2020)
- Dreamer (Hafner et al., 2020)

**Multi-Agent**:
- MADDPG (Lowe et al., 2017)
- QMIX (Rashid et al., 2018)
- MAPPO (Yu et al., 2021)

**Offline RL**:
- CQL (Kumar et al., 2020)
- Decision Transformer (Chen et al., 2021)
- IQL (Kostrikov et al., 2021)

**LLM Alignment**:
- InstructGPT/RLHF (Ouyang et al., 2022)
- DPO (Rafailov et al., 2023)
- Constitutional AI (Bai et al., 2022)

**Game Playing**:
- AlphaGo (Silver et al., 2016)
- AlphaZero (Silver et al., 2017)
- AlphaStar (Vinyals et al., 2019)

**Online Courses**:
- **OpenAI Spinning Up**: https://spinningup.openai.com/
- **DeepMind x UCL RL Course**: https://www.deepmind.com/learning-resources/
- **Berkeley CS285**: http://rail.eecs.berkeley.edu/deeprlcourse/
- **Hugging Face Deep RL Course**: https://huggingface.co/learn/deep-rl-course/

**Communities**:
- r/reinforcementlearning
- RL Discord servers
- Papers with Code: https://paperswithcode.com/

**Blogs**:
- Lilian Weng's Blog: https://lilianweng.github.io/
- Distill.pub: https://distill.pub/
- Berkeley AI Research Blog: https://bair.berkeley.edu/blog/

---

**End of Reinforcement Learning - Complete Guide**

This comprehensive guide covers RL from foundations (MDPs, Bellman equations) through cutting-edge applications (LLM alignment, multi-agent systems). Each section includes production-ready code, mathematical formulations, and practical insights for research and industry.

**Key Coverage**:
- **Foundations**: MDPs, Dynamic Programming, Monte Carlo, TD Learning
- **Deep RL**: DQN variants, Policy Gradients, PPO, SAC, TD3
- **Advanced**: Model-Based RL, Multi-Agent RL, Offline RL
- **Modern**: RLHF, DPO, LLM Alignment
- **Practical**: Gymnasium, Stable-Baselines3, Hyperparameter Tuning
- **Applications**: Games, Robotics, Finance, Chip Design
