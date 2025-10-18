# 33. Reinforcement Learning (RL)

## Overview

Reinforcement Learning trains agents to make sequential decisions by learning from rewards. The agent learns a policy that maximizes cumulative reward through trial and error.

**Key Components:**
- **Agent:** Decision maker
- **Environment:** What agent interacts with
- **State (s):** Current situation
- **Action (a):** What agent can do
- **Reward (r):** Feedback signal
- **Policy (π):** Agent's behavior (s → a)

---

## 33.1 Core Concepts

### Markov Decision Process (MDP)

```
MDP = (S, A, P, R, γ)

S: State space
A: Action space
P: Transition probabilities P(s'|s,a)
R: Reward function R(s,a)
γ: Discount factor [0,1]
```

### Value Functions

**State Value:**
```
V^π(s) = E[Σ γ^t r_t | s_0=s, π]
```

**Action Value (Q-function):**
```
Q^π(s,a) = E[Σ γ^t r_t | s_0=s, a_0=a, π]
```

### Bellman Equations

```
V^π(s) = E_π[r + γ V^π(s')]
Q^π(s,a) = E[r + γ Q^π(s', a')]
```

---

## 33.2 Q-Learning

**Off-policy TD learning:**

```python
# Update rule
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

### Implementation

```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
    
    def select_action(self, state):
        # ε-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])  # Explore
        return np.argmax(self.Q[state])  # Exploit
    
    def update(self, state, action, reward, next_state):
        # Q-learning update
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

# Training
agent = QLearning(n_states=100, n_actions=4)

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
```

---

## 33.3 Deep Q-Networks (DQN)

**Pioneered deep RL (Atari games):**

### Key Innovations

1. **Experience Replay:** Store transitions, sample randomly
2. **Target Network:** Separate network for stable targets
3. **Frame Stacking:** Use last 4 frames as state

### Implementation

```python
import torch
import torch.nn as nn
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(state),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done))

# Training
policy_net = DQN(state_dim=4, action_dim=2)
target_net = DQN(state_dim=4, action_dim=2)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer()

for episode in range(1000):
    state = env.reset()
    
    for t in range(500):
        # ε-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = policy_net(torch.FloatTensor(state))
                action = q_values.argmax().item()
        
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        
        # Train
        if len(replay_buffer.buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            # Current Q values
            q_values = policy_net(states).gather(1, actions.unsqueeze(1))
            
            # Target Q values
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)
            
            # Loss
            loss = nn.MSELoss()(q_values.squeeze(), targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if done:
            break
    
    # Update target network
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
```

---

## 33.4 Policy Gradient Methods

**Directly optimize policy:**

### REINFORCE Algorithm

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

def reinforce(policy, optimizer, gamma=0.99):
    states, actions, rewards = [], [], []
    
    # Collect episode
    state = env.reset()
    done = False
    
    while not done:
        state_tensor = torch.FloatTensor(state)
        probs = policy(state_tensor)
        action = torch.multinomial(probs, 1).item()
        
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
    
    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Update policy
    loss = 0
    for state, action, G in zip(states, actions, returns):
        probs = policy(torch.FloatTensor(state))
        log_prob = torch.log(probs[action])
        loss += -log_prob * G
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 33.5 Actor-Critic Methods

**Combine value and policy:**

### A3C (Asynchronous Advantage Actor-Critic)

**Key Features:**
- Parallel agents
- Shared model
- Advantage function: A(s,a) = Q(s,a) - V(s)

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        shared = self.shared(x)
        policy = self.actor(shared)
        value = self.critic(shared)
        return policy, value

def a3c_update(model, optimizer, states, actions, rewards, next_states, dones, gamma=0.99):
    # Convert to tensors
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    # Get policy and values
    policies, values = model(states)
    _, next_values = model(next_states)
    
    # Compute advantages
    td_targets = rewards + gamma * next_values.squeeze() * (1 - dones)
    advantages = td_targets - values.squeeze()
    
    # Actor loss (policy gradient with advantage)
    log_probs = torch.log(policies.gather(1, actions.unsqueeze(1)))
    actor_loss = -(log_probs.squeeze() * advantages.detach()).mean()
    
    # Critic loss (MSE)
    critic_loss = advantages.pow(2).mean()
    
    # Total loss
    loss = actor_loss + 0.5 * critic_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 33.6 PPO (Proximal Policy Optimization)

**2025 most popular: balanced efficiency and stability**

### Clipped Surrogate Objective

```python
ratio = π_new(a|s) / π_old(a|s)
L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

### Implementation

```python
class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
    
    def update(self, states, actions, old_log_probs, returns, advantages, clip_epsilon=0.2):
        for _ in range(10):  # Multiple epochs
            # Actor update
            new_log_probs = self.actor.get_log_prob(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Critic update
            values = self.critic(states)
            critic_loss = (returns - values).pow(2).mean()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
```

**Why PPO is Popular:**
- Stable training (clipping prevents large updates)
- Sample efficient
- Easy to implement
- Works across domains

---

## 33.7 Applications

### Game Playing
- AlphaGo (Go)
- OpenAI Five (Dota 2)
- AlphaStar (StarCraft II)

### Robotics
- Manipulation (grasping, assembly)
- Locomotion (walking, running)
- Navigation

### Autonomous Driving
- Lane keeping
- Path planning
- Decision making

### Finance
- Portfolio optimization
- Trading strategies

### Resource Management
- Data center cooling
- Traffic light control

---

## 33.8 Challenges

**Sample Efficiency:** Needs many interactions
**Credit Assignment:** Which action caused reward?
**Exploration vs Exploitation:** Balance needed
**Sparse Rewards:** Hard to learn with delayed feedback

**Solutions:**
- Curriculum learning
- Reward shaping
- Hindsight Experience Replay (HER)
- Intrinsic motivation

---

## Resources

- "Playing Atari with Deep RL" (Mnih et al., 2013) - DQN
- "Asynchronous Methods for Deep RL" (Mnih et al., 2016) - A3C
- "Proximal Policy Optimization" (Schulman et al., 2017) - PPO
- OpenAI Spinning Up: https://spinningup.openai.com/
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
