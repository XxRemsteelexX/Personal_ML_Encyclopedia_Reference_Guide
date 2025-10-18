# 6. Causal Inference (2025 Cutting-Edge)

## Overview

Causal inference goes beyond correlation to understand cause-and-effect relationships. In 2025, the integration of causal inference with deep learning represents a major frontier in data science.

**Core Question:** Does X cause Y, or is it just correlation?

---

## 6.1 Frameworks for Causal Inference

### 1. Potential Outcomes Framework (Rubin Causal Model)

**Key Concept:** For each unit, there exist potential outcomes under different treatments.

**Notation:**
- Y₁(i): Outcome for unit i if treated
- Y₀(i): Outcome for unit i if not treated
- **Individual Treatment Effect (ITE):** τᵢ = Y₁(i) - Y₀(i)

**Fundamental Problem:** Can only observe one potential outcome per unit!

**Average Treatment Effect (ATE):**
```
τ = E[Y₁ - Y₀] = E[Y₁] - E[Y₀]
```

**Assumptions:**
- **SUTVA** (Stable Unit Treatment Value Assumption): No interference between units
- **Ignorability**: Treatment assignment independent of potential outcomes (given covariates)
- **Overlap**: All units have non-zero probability of treatment

```python
import numpy as np
from scipy import stats

# Simulate potential outcomes
np.random.seed(42)
n = 1000

# Each person has two potential outcomes (only one observed)
Y0 = np.random.normal(10, 2, n)  # Outcome without treatment
Y1 = Y0 + np.random.normal(2, 1, n)  # Outcome with treatment

# True ATE (we can see this in simulation, not in real data!)
true_ate = np.mean(Y1 - Y0)
print(f"True ATE: {true_ate:.3f}")

# Random treatment assignment
treatment = np.random.binomial(1, 0.5, n)

# Observed outcomes (fundamental problem: can't see both!)
Y_observed = np.where(treatment == 1, Y1, Y0)

# Naive estimate (if randomized)
ate_estimate = np.mean(Y_observed[treatment == 1]) - np.mean(Y_observed[treatment == 0])
print(f"Estimated ATE: {ate_estimate:.3f}")
```

---

### 2. Structural Causal Models (Pearl's Framework)

**Key Concept:** Use directed acyclic graphs (DAGs) to represent causal relationships.

**Components:**
- **Nodes:** Variables
- **Edges:** Direct causal effects
- **Structural equations:** How variables are generated

**Example DAG:**
```
Treatment (T) → Outcome (Y)
     ↑              ↑
     └─ Confounder (X) ─┘
```

**Do-Calculus:** Mathematical framework for causal inference from observational data.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create DAG
G = nx.DiGraph()
G.add_edges_from([
    ('X', 'T'),  # Confounder affects treatment
    ('X', 'Y'),  # Confounder affects outcome
    ('T', 'Y')   # Treatment affects outcome (causal effect)
])

# Visualize
pos = {'X': (0, 1), 'T': (0, 0), 'Y': (1, 0)}
plt.figure(figsize=(6, 4))
nx.draw(G, pos, with_labels=True, node_color='lightblue',
        node_size=2000, font_size=14, arrows=True, arrowsize=20)
plt.title('Causal DAG: X is a confounder')
plt.show()

# Backdoor paths: X ← T → Y (blocked by conditioning on X)
# After conditioning on X, association = causation
```

---

## 6.2 Methods for Causal Inference

### Randomized Controlled Trials (RCT)

**Gold Standard:** Random assignment breaks all confounding.

```
Random Assignment → E[Y₀|T=1] = E[Y₀|T=0]
```

Therefore: ATE = E[Y|T=1] - E[Y|T=0]

**Advantages:**
- Eliminates selection bias
- Simple analysis
- Clear causal interpretation

**Disadvantages:**
- Expensive
- Ethical concerns
- Not always feasible

---

### Propensity Score Matching

**Goal:** Mimic randomization using observational data.

**Propensity Score:** e(x) = P(T=1|X=x)

**Idea:** Match treated and control units with similar propensity scores.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Generate observational data with confounding
np.random.seed(42)
n = 1000

# Confounder
X = np.random.normal(0, 1, (n, 5))

# Treatment depends on X (selection bias!)
treatment_prob = 1 / (1 + np.exp(-X @ np.array([1, 0.5, -0.3, 0.2, 0.1])))
treatment = np.random.binomial(1, treatment_prob)

# Outcome depends on X and treatment
Y0 = X @ np.array([2, 1, -1, 0.5, 0.3]) + np.random.normal(0, 1, n)
Y1 = Y0 + 3  # Treatment effect = 3
Y = np.where(treatment == 1, Y1, Y0)

# Naive estimate (biased due to confounding)
naive_ate = np.mean(Y[treatment == 1]) - np.mean(Y[treatment == 0])
print(f"Naive ATE: {naive_ate:.3f} (True: 3.0)")

# Propensity score matching
# 1. Estimate propensity scores
ps_model = LogisticRegression()
ps_model.fit(X, treatment)
propensity_scores = ps_model.predict_proba(X)[:, 1]

# 2. Match treated to controls with similar propensity
treated_idx = np.where(treatment == 1)[0]
control_idx = np.where(treatment == 0)[0]

# For each treated unit, find nearest control
nn = NearestNeighbors(n_neighbors=1)
nn.fit(propensity_scores[control_idx].reshape(-1, 1))
distances, matches = nn.kneighbors(propensity_scores[treated_idx].reshape(-1, 1))

# Matched control outcomes
matched_control_outcomes = Y[control_idx[matches.flatten()]]

# Matched ATE
matched_ate = np.mean(Y[treated_idx]) - np.mean(matched_control_outcomes)
print(f"Matched ATE: {matched_ate:.3f}")
```

---

### Inverse Propensity Weighting (IPW)

**Idea:** Weight observations by inverse of propensity score.

**ATE Estimator:**
```
τ̂ = (1/n) Σ [T·Y/e(X) - (1-T)·Y/(1-e(X))]
```

```python
# IPW estimator
weights_treated = 1 / propensity_scores[treatment == 1]
weights_control = 1 / (1 - propensity_scores[treatment == 0])

weighted_treated_mean = np.average(Y[treatment == 1], weights=weights_treated)
weighted_control_mean = np.average(Y[treatment == 0], weights=weights_control)

ipw_ate = weighted_treated_mean - weighted_control_mean
print(f"IPW ATE: {ipw_ate:.3f}")
```

---

### Difference-in-Differences (DiD)

**Setup:** Panel data with treatment introduced at specific time.

**Idea:** Compare change over time in treated vs. control groups.

**Estimator:**
```
τ = (Ȳ_treated,after - Ȳ_treated,before) - (Ȳ_control,after - Ȳ_control,before)
```

**Parallel Trends Assumption:** Without treatment, both groups would have same trend.

```python
import pandas as pd
import numpy as np

# Generate DiD data
np.random.seed(42)
n_states = 20
n_years = 10

data = []
for state in range(n_states):
    treated = state < 10  # Half treated

    for year in range(n_years):
        post = year >= 5  # Treatment starts at year 5

        # Outcome = state effect + year effect + treatment effect + noise
        state_effect = state * 0.5
        year_effect = year * 0.3
        treatment_effect = 2.0 if (treated and post) else 0.0

        outcome = state_effect + year_effect + treatment_effect + np.random.normal(0, 0.5)

        data.append({
            'state': state,
            'year': year,
            'treated': treated,
            'post': post,
            'outcome': outcome
        })

df = pd.DataFrame(data)

# DiD estimation
did_estimate = (
    df[(df['treated'] == True) & (df['post'] == True)]['outcome'].mean() -
    df[(df['treated'] == True) & (df['post'] == False)]['outcome'].mean()
) - (
    df[(df['treated'] == False) & (df['post'] == True)]['outcome'].mean() -
    df[(df['treated'] == False) & (df['post'] == False)]['outcome'].mean()
)

print(f"DiD Estimate: {did_estimate:.3f} (True: 2.0)")

# Regression DiD
import statsmodels.formula.api as smf

model = smf.ols('outcome ~ treated + post + treated:post', data=df).fit()
print("\nRegression DiD:")
print(model.summary().tables[1])
print(f"\nTreatment effect: {model.params['treated:post']:.3f}")
```

---

### Instrumental Variables (IV)

**Problem:** Unobserved confounders.

**Solution:** Find an "instrument" Z that:
1. Affects treatment (relevance)
2. Only affects outcome through treatment (exclusion restriction)
3. Independent of confounders (exogeneity)

**Example:** Effect of education on earnings
- Instrument: Distance to college
- Affects education (closer → more likely to attend)
- Doesn't directly affect earnings (except through education)

**Two-Stage Least Squares (2SLS):**

```python
from statsmodels.sandbox.regression.gmm import IV2SLS
import numpy as np

# Generate IV data
np.random.seed(42)
n = 1000

# Instrument (e.g., distance to college)
Z = np.random.normal(0, 1, n)

# Unobserved confounder
U = np.random.normal(0, 1, n)

# Treatment (education) depends on instrument and confounder
T = 2*Z + U + np.random.normal(0, 1, n)

# Outcome (earnings) depends on treatment and confounder
Y = 3*T + 2*U + np.random.normal(0, 1, n)  # True effect = 3

# Naive OLS (biased)
from sklearn.linear_model import LinearRegression
naive_model = LinearRegression().fit(T.reshape(-1, 1), Y)
print(f"Naive OLS: {naive_model.coef_[0]:.3f} (biased due to U)")

# IV/2SLS
# Stage 1: Regress T on Z
stage1 = LinearRegression().fit(Z.reshape(-1, 1), T)
T_hat = stage1.predict(Z.reshape(-1, 1))

# Stage 2: Regress Y on predicted T
stage2 = LinearRegression().fit(T_hat.reshape(-1, 1), Y)
print(f"IV/2SLS: {stage2.coef_[0]:.3f} (True: 3.0)")
```

---

## 6.3 Deep Learning for Causal Inference (2025)

### Motivation

**Traditional ML:** Excels at prediction, captures spurious correlations

**Causal ML:** Ensures robustness, interpretability, counterfactual reasoning

### Deep Structural Causal Models (DSCM)

**Idea:** Use neural networks as components in structural causal models.

**Benefits:**
- Handle high-dimensional data
- Learn complex nonlinear relationships
- Enable counterfactual inference

```python
import torch
import torch.nn as nn

class DeepSCM(nn.Module):
    """
    Deep Structural Causal Model

    X → T → Y (with unobserved U)
    """
    def __init__(self, dim_x=10, dim_hidden=50):
        super().__init__()

        # Structural equation for T given X, U_T
        self.f_T = nn.Sequential(
            nn.Linear(dim_x + 1, dim_hidden),  # +1 for noise
            nn.ReLU(),
            nn.Linear(dim_hidden, 1)
        )

        # Structural equation for Y given X, T, U_Y
        self.f_Y = nn.Sequential(
            nn.Linear(dim_x + 1 + 1, dim_hidden),  # X, T, noise
            nn.ReLU(),
            nn.Linear(dim_hidden, 1)
        )

    def forward(self, X, U_T, U_Y):
        # Generate T from X and noise
        T = self.f_T(torch.cat([X, U_T], dim=1))

        # Generate Y from X, T, and noise
        Y = self.f_Y(torch.cat([X, T, U_Y], dim=1))

        return T, Y

    def do_intervention(self, X, t_value):
        """
        Intervene: set T = t_value (do-operator)

        This computes counterfactual: what would Y be if we set T=t?
        """
        T_intervention = torch.full((X.shape[0], 1), t_value)

        # Sample noise for Y
        U_Y = torch.randn(X.shape[0], 1)

        Y_counterfactual = self.f_Y(torch.cat([X, T_intervention, U_Y], dim=1))

        return Y_counterfactual

# Example usage
model = DeepSCM(dim_x=5, dim_hidden=32)

# Observational data
X = torch.randn(100, 5)
U_T = torch.randn(100, 1)
U_Y = torch.randn(100, 1)

T, Y = model(X, U_T, U_Y)

# Intervention: What if we set T=1 for everyone?
Y_intervention = model.do_intervention(X, t_value=1.0)

print(f"Observed Y mean: {Y.mean().item():.3f}")
print(f"Counterfactual Y mean (T=1): {Y_intervention.mean().item():.3f}")
```

---

### Causal Effect Estimation with Deep Learning

**CEVAE (Causal Effect VAE):**

Uses variational autoencoders to learn latent confounders.

```python
class CEVAE(nn.Module):
    """
    Causal Effect Variational Autoencoder

    Handles unobserved confounding by learning latent variable Z
    """
    def __init__(self, dim_x=10, dim_z=5, dim_hidden=32):
        super().__init__()

        # Inference network: q(z|x,t,y)
        self.encoder = nn.Sequential(
            nn.Linear(dim_x + 2, dim_hidden),  # x, t, y
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_z * 2)  # mean and logvar
        )

        # Generative network: p(x,t,y|z)
        self.decoder_x = nn.Sequential(
            nn.Linear(dim_z, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_x)
        )

        self.decoder_t = nn.Sequential(
            nn.Linear(dim_z + dim_x, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )

        self.decoder_y = nn.Sequential(
            nn.Linear(dim_z + dim_x + 1, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1)
        )

    def encode(self, x, t, y):
        # q(z|x,t,y)
        input = torch.cat([x, t, y], dim=1)
        h = self.encoder(input)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, t):
        # p(x|z)
        x = self.decoder_x(z)

        # p(t|x,z)
        t_prob = self.decoder_t(torch.cat([z, x], dim=1))

        # p(y|x,t,z)
        y = self.decoder_y(torch.cat([z, x, t], dim=1))

        return x, t_prob, y

    def forward(self, x, t, y):
        mu, logvar = self.encode(x, t, y)
        z = self.reparameterize(mu, logvar)
        x_recon, t_recon, y_recon = self.decode(z, t)
        return x_recon, t_recon, y_recon, mu, logvar

    def estimate_ate(self, x, n_samples=100):
        """Estimate Average Treatment Effect"""
        y0_samples = []
        y1_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Sample z ~ q(z|x,t,y) with t=0
                # (Approximate with prior)
                z = torch.randn(x.shape[0], 5)

                # Counterfactual: Y(0)
                _, _, y0 = self.decode(z, torch.zeros(x.shape[0], 1))
                y0_samples.append(y0)

                # Counterfactual: Y(1)
                _, _, y1 = self.decode(z, torch.ones(x.shape[0], 1))
                y1_samples.append(y1)

        y0_mean = torch.stack(y0_samples).mean(dim=0)
        y1_mean = torch.stack(y1_samples).mean(dim=0)

        ate = (y1_mean - y0_mean).mean()
        return ate.item()

# Example
model = CEVAE(dim_x=10, dim_z=5)

# After training on observational data...
X_test = torch.randn(100, 10)
ate_estimate = model.estimate_ate(X_test)
print(f"Estimated ATE: {ate_estimate:.3f}")
```

---

## 6.4 Industry Applications (2025)

### Healthcare and Precision Medicine

**Application:** Personalized treatment effects

**Challenge:** Patient heterogeneity, unobserved confounders

**Solution:** Deep learning + causal inference
- Learn latent health states
- Estimate individual treatment effects
- Optimal treatment assignment

---

### Manufacturing Fault Diagnosis

**Application:** Root cause analysis

**Method:** Causal graphs + VAEs
- Construct interpretable fault propagation graphs
- Integrate physics equations with neural networks
- Enable exploration of deep-rooted causes

**Example:** Continuous Catalytic Reforming heat exchangers
- Granger causality + Variational Autoencoders
- Explicit causal graphs for fault diagnosis

---

### E-commerce and Marketing

**Application:** Customer lifetime value estimation

**Challenge:** Selection bias (customers choose to engage)

**Solution:** Propensity score matching + deep learning
- Estimate treatment effect of marketing campaigns
- Account for self-selection
- Optimize spending

---

## 6.5 Software Tools (2025)

### DoWhy (Microsoft)

```python
import dowhy
from dowhy import CausalModel

# Specify causal graph
model = CausalModel(
    data=df,
    treatment='treatment',
    outcome='outcome',
    common_causes=['confounder1', 'confounder2']
)

# Identify causal effect
identified_estimand = model.identify_effect()

# Estimate
causal_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)

print(causal_estimate)

# Refute (sensitivity analysis)
refute_result = model.refute_estimate(
    identified_estimand,
    causal_estimate,
    method_name="random_common_cause"
)
print(refute_result)
```

---

### EconML (Microsoft)

Econometric + Machine Learning for heterogeneous treatment effects.

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

# Causal Forest
est = CausalForestDML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestRegressor()
)

# Fit
est.fit(Y, T, X=X, W=W)  # Y: outcome, T: treatment, X: features, W: confounders

# Estimate individual treatment effects
treatment_effects = est.effect(X)

# Confidence intervals
lb, ub = est.effect_interval(X, alpha=0.05)
```

---

## 6.6 Pearl's Ladder of Causation

**Level 1: Association** (Seeing)
- P(Y|X) - "What is?"
- Passive observation
- Traditional ML

**Level 2: Intervention** (Doing)
- P(Y|do(X)) - "What if?"
- Randomized experiments
- A/B tests

**Level 3: Counterfactuals** (Imagining)
- P(Y_x|X', Y') - "What if I had done differently?"
- Retrospective
- Most powerful

**2025 Frontier:** Deep learning enables all three levels through Deep SCMs.

---

## Resources

**Classic Books:**
- "Causality" by Judea Pearl (2009)
- "Causal Inference: The Mixtape" by Cunningham (2021)
- "The Book of Why" by Pearl & Mackenzie (2018)

**2025 Research:**
- "Causal Inference Meets Deep Learning: A Comprehensive Survey" (2024)
- "Deep Structural Causal Models for Tractable Counterfactual Inference" (2020)
- "A survey of deep causal models and their industrial applications" (2024)

**Software:**
- DoWhy: https://github.com/py-why/dowhy
- EconML: https://github.com/py-why/EconML
- CausalML: https://github.com/uber/causalml
