# 4. Bayesian Statistics

## 4.1 Bayes' Theorem

### Formula

```
P(H|E) = P(E|H) × P(H) / P(E)
```

**Components:**
- **P(H|E)**: Posterior probability (updated belief after seeing evidence)
- **P(E|H)**: Likelihood (probability of evidence given hypothesis)
- **P(H)**: Prior probability (initial belief before evidence)
- **P(E)**: Marginal probability of evidence

### Expanded Form

```
P(H|E) = P(E|H) × P(H) / [P(E|H) × P(H) + P(E|¬H) × P(¬H)]
```

### Example: Medical Diagnosis

```python
import numpy as np

# Disease prevalence (prior)
P_disease = 0.001

# Test accuracy
P_pos_given_disease = 0.99  # Sensitivity
P_neg_given_no_disease = 0.95  # Specificity
P_pos_given_no_disease = 1 - P_neg_given_no_disease  # 0.05

# Patient tests positive
# What's probability they have disease?

# Bayes' theorem
numerator = P_pos_given_disease * P_disease
denominator = (P_pos_given_disease * P_disease +
               P_pos_given_no_disease * (1 - P_disease))

P_disease_given_pos = numerator / denominator

print(f"Prior P(Disease): {P_disease:.4f}")
print(f"Posterior P(Disease|Positive Test): {P_disease_given_pos:.4f}")
print(f"\nEven with 99% accurate test, only {P_disease_given_pos*100:.1f}% chance of disease!")
```

---

## 4.2 Bayesian Inference

### Geometric Interpretation

Think of a 1×1 square representing all possibilities:
- Hypothesis occupies left portion with width P(H)
- Evidence restricts the space horizontally
- Posterior is the proportion of hypothesis in the restricted space

### Key Principles

**Prior:** Represents initial knowledge before seeing data

**Evidence Updates Beliefs:** New data modifies probabilities

**Coherent Belief Updating:** Mathematically consistent framework

---

## 4.3 Conjugate Priors

Prior and posterior have same distributional form.

### Beta-Binomial (Most Common)

**Application:** Estimating success probability

**Prior:** Beta(α, β)
**Likelihood:** Binomial
**Posterior:** Beta(α + successes, β + failures)

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Prior belief: Beta(2, 2) - slightly favor p=0.5
prior_alpha, prior_beta = 2, 2
prior = stats.beta(prior_alpha, prior_beta)

# Observe data: 7 successes out of 10 trials
successes = 7
failures = 3

# Posterior: Beta(2+7, 2+3) = Beta(9, 5)
post_alpha = prior_alpha + successes
post_beta = prior_beta + failures
posterior = stats.beta(post_alpha, post_beta)

# Plot
p = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 6))
plt.plot(p, prior.pdf(p), label='Prior Beta(2,2)', linestyle='--')
plt.plot(p, posterior.pdf(p), label='Posterior Beta(9,5)', linewidth=2)
plt.axvline(successes/(successes+failures), color='red', linestyle=':',
            label=f'MLE = {successes/(successes+failures):.2f}')
plt.xlabel('Probability p')
plt.ylabel('Density')
plt.title('Bayesian Updating: Prior → Posterior')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Posterior mean
post_mean = post_alpha / (post_alpha + post_beta)
print(f"Posterior mean: {post_mean:.3f}")

# 95% Credible interval
credible_interval = posterior.ppf([0.025, 0.975])
print(f"95% Credible Interval: [{credible_interval[0]:.3f}, {credible_interval[1]:.3f}]")
```

---

### Normal-Normal

**Application:** Estimating mean with known variance

**Prior:** N(μ₀, σ₀²)
**Likelihood:** N(μ, σ²) with known σ²
**Posterior:** Normal

```python
import numpy as np
from scipy import stats

# Prior belief: μ ~ N(170, 5²)
prior_mean = 170
prior_std = 5

# Data: 20 observations, known σ=10
data = np.array([172, 168, 175, 171, 169, 173, 170, 174, 172, 171,
                 170, 169, 172, 173, 168, 171, 170, 174, 172, 171])
known_std = 10
n = len(data)
data_mean = np.mean(data)

# Posterior parameters
precision_prior = 1 / prior_std**2
precision_data = n / known_std**2

post_mean = (precision_prior * prior_mean + precision_data * data_mean) / (precision_prior + precision_data)
post_var = 1 / (precision_prior + precision_data)
post_std = np.sqrt(post_var)

print(f"Prior: N({prior_mean}, {prior_std:.2f}²)")
print(f"Data mean: {data_mean:.2f}")
print(f"Posterior: N({post_mean:.2f}, {post_std:.2f}²)")
print(f"\nPosterior is weighted average of prior and data")
```

---

## 4.4 Markov Chain Monte Carlo (MCMC)

When posterior is intractable, use sampling methods.

### Metropolis-Hastings Algorithm

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def metropolis_hastings(log_posterior, initial, n_samples, proposal_std=1.0):
    """
    Metropolis-Hastings MCMC sampler

    Args:
        log_posterior: Function computing log P(θ|data)
        initial: Starting value
        n_samples: Number of samples
        proposal_std: Std dev of proposal distribution
    """
    samples = [initial]
    current = initial
    accepted = 0

    for i in range(n_samples - 1):
        # Propose new value
        proposal = current + np.random.normal(0, proposal_std)

        # Acceptance ratio (in log space)
        log_ratio = log_posterior(proposal) - log_posterior(current)

        # Accept or reject
        if np.log(np.random.random()) < log_ratio:
            current = proposal
            accepted += 1

        samples.append(current)

    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate

# Example: Sample from posterior of mean
# Prior: N(0, 10²), Likelihood: N(μ, 1²), Data: [1, 2, 3, 2, 1]
data = np.array([1, 2, 3, 2, 1])
prior_mean, prior_std = 0, 10
likelihood_std = 1

def log_posterior(mu):
    # Log prior
    log_prior = stats.norm.logpdf(mu, prior_mean, prior_std)

    # Log likelihood
    log_likelihood = np.sum(stats.norm.logpdf(data, mu, likelihood_std))

    return log_prior + log_likelihood

# Run MCMC
np.random.seed(42)
samples, acceptance = metropolis_hastings(log_posterior, initial=0, n_samples=10000, proposal_std=0.5)

print(f"Acceptance rate: {acceptance:.2%}")
print(f"Posterior mean: {np.mean(samples[1000:]):.3f}")  # Discard burn-in
print(f"Posterior std: {np.std(samples[1000:]):.3f}")

# Plot trace
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(samples)
plt.xlabel('Iteration')
plt.ylabel('μ')
plt.title('MCMC Trace')

plt.subplot(1, 2, 2)
plt.hist(samples[1000:], bins=50, density=True, alpha=0.7)
plt.xlabel('μ')
plt.ylabel('Density')
plt.title('Posterior Distribution')
plt.tight_layout()
plt.show()
```

---

### PyMC for Modern Bayesian Inference (2025)

```python
# Modern approach using PyMC
import pymc as pm
import arviz as az
import numpy as np

# Data
data = np.array([1, 2, 3, 2, 1])

# Model
with pm.Model() as model:
    # Prior
    mu = pm.Normal('mu', mu=0, sigma=10)

    # Likelihood
    obs = pm.Normal('obs', mu=mu, sigma=1, observed=data)

    # Sample
    trace = pm.sample(2000, return_inferencedata=True, random_seed=42)

# Summary
print(az.summary(trace))

# Plot
az.plot_trace(trace)
az.plot_posterior(trace)
```

---

## 4.5 Bayesian Neural Networks (2025 Production Applications)

### Uncertainty Quantification

Critical for production systems.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty"""
    def __init__(self, in_features, out_features):
        super().__init__()

        # Weight parameters: mean and log variance
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1)

    def forward(self, x):
        # Sample weights from posterior
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)

        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)

        return F.linear(x, weight, bias)

class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Prediction with uncertainty
model = BayesianNN(input_dim=10, hidden_dim=50, output_dim=1)

# Monte Carlo sampling for prediction
def predict_with_uncertainty(model, x, n_samples=100):
    predictions = []
    model.eval()

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred)

    predictions = torch.stack(predictions)

    # Mean and std of predictions
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)

    return mean, std

# Example
x = torch.randn(5, 10)
mean, uncertainty = predict_with_uncertainty(model, x, n_samples=100)

print(f"Predictions: {mean.squeeze()}")
print(f"Uncertainty: {uncertainty.squeeze()}")
```

### MC Dropout (Practical Alternative)

```python
import torch
import torch.nn as nn

class MCDropoutNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

def predict_with_mc_dropout(model, x, n_samples=100):
    """Use dropout at test time for uncertainty"""
    model.train()  # Keep dropout active

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred)

    predictions = torch.stack(predictions)
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)

    return mean, std

# Example
model = MCDropoutNN(input_dim=10, hidden_dim=50, output_dim=1)
x = torch.randn(5, 10)

mean, uncertainty = predict_with_mc_dropout(model, x, n_samples=100)
print(f"Predictions: {mean.squeeze()}")
print(f"Uncertainty (epistemic): {uncertainty.squeeze()}")
```

### Production Applications (2025)

**Remaining Useful Life (RUL) Prediction:**
- NASA Turbofan Engine Dataset
- BNN + MC Dropout
- Quantifies aleatoric + epistemic uncertainty
- Critical for maintenance planning

**Predictive Quality in Manufacturing:**
- Quality characteristic prediction from process data
- Uncertainty per prediction
- Decision-making with confidence levels

**Materials Science:**
- Material property prediction
- Uncertainty quantification for informed decisions
- Physics-guided Bayesian NNs

```python
# Example: RUL prediction with uncertainty
class RULPredictor(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        rul = self.fc(dropped)
        return rul

# Predict with credible intervals
def predict_rul_with_uncertainty(model, sensor_data, n_samples=100):
    model.train()  # MC Dropout

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            rul = model(sensor_data)
            predictions.append(rul)

    predictions = torch.cat(predictions, dim=1)

    mean_rul = predictions.mean(dim=1)
    lower_ci = predictions.quantile(0.025, dim=1)
    upper_ci = predictions.quantile(0.975, dim=1)

    return mean_rul, lower_ci, upper_ci

# Usage
model = RULPredictor()
sensor_data = torch.randn(1, 30, 14)  # 1 engine, 30 timesteps, 14 sensors

mean, lower, upper = predict_rul_with_uncertainty(model, sensor_data)
print(f"Predicted RUL: {mean.item():.1f} cycles")
print(f"95% CI: [{lower.item():.1f}, {upper.item():.1f}]")
```

---

## 4.6 Applications in ML

### Naive Bayes Classifier

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predictions with probabilities
y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)

print(f"Accuracy: {nb.score(X_test, y_test):.3f}")
print(f"Class probabilities (first 5): {y_prob[:5]}")
```

---

### Bayesian Optimization for Hyperparameter Tuning

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np

def bayesian_optimization(f, bounds, n_iterations=20):
    """
    Simple Bayesian optimization

    Args:
        f: Function to minimize
        bounds: [(low, high), ...] for each parameter
        n_iterations: Number of iterations
    """
    # Initialize with random points
    X_samples = []
    y_samples = []

    for _ in range(5):
        x = np.array([np.random.uniform(low, high) for low, high in bounds])
        y = f(x)
        X_samples.append(x)
        y_samples.append(y)

    X_samples = np.array(X_samples)
    y_samples = np.array(y_samples).reshape(-1, 1)

    # Gaussian Process
    gp = GaussianProcessRegressor(kernel=RBF())

    for iteration in range(n_iterations):
        # Fit GP
        gp.fit(X_samples, y_samples)

        # Acquisition function: Expected Improvement
        # (Simplified: just sample and pick best)
        candidates = []
        for _ in range(1000):
            x = np.array([np.random.uniform(low, high) for low, high in bounds])
            mu, sigma = gp.predict([x], return_std=True)
            # Expected improvement
            improvement = np.max(y_samples) - mu
            ei = improvement * sigma
            candidates.append((x, ei))

        # Select best candidate
        next_x = max(candidates, key=lambda x: x[1])[0]
        next_y = f(next_x)

        # Update samples
        X_samples = np.vstack([X_samples, next_x])
        y_samples = np.vstack([y_samples, next_y])

        print(f"Iteration {iteration+1}: f(x) = {next_y:.4f}")

    # Return best
    best_idx = np.argmax(y_samples)
    return X_samples[best_idx], y_samples[best_idx]

# Example: Optimize hyperparameters
def objective(params):
    # Simulate model performance
    learning_rate, n_estimators = params
    # (In practice, train model and return validation score)
    return -(learning_rate - 0.01)**2 - (n_estimators - 100)**2 / 10000

bounds = [(0.001, 0.1), (10, 200)]  # learning_rate, n_estimators
best_params, best_score = bayesian_optimization(objective, bounds, n_iterations=15)

print(f"\nBest params: learning_rate={best_params[0]:.4f}, n_estimators={int(best_params[1])}")
print(f"Best score: {best_score[0]:.4f}")
```

---

## Resources

- **Classic:**
  - "Bayesian Data Analysis" by Gelman et al.
  - "The Theory That Would Not Die" by McGrayne (history)

- **Modern:**
  - "Statistical Rethinking" by McElreath
  - "Bayesian Methods for Hackers" (online)

- **2025 Applications:**
  - PyMC documentation: https://www.pymc.io/
  - ArviZ (visualization): https://arviz-devs.github.io/
  - Papers: Bayesian NNs for uncertainty quantification in production (2024-2025)
