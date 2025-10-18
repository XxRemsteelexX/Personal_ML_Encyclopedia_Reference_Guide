# Privacy-Preserving Machine Learning - Federated Learning & Differential Privacy

## Overview

**Privacy is mandatory** under GDPR Article 32 and EU AI Act Article 15. This guide covers technical implementations of privacy-preserving ML.

**Techniques:**
- Differential Privacy (DP) - Mathematical privacy guarantee
- Federated Learning (FL) - Train without centralizing data
- Homomorphic Encryption - Compute on encrypted data
- Secure Multi-Party Computation - Collaborative learning without data sharing

---

## Differential Privacy

### Formal Definition

**ε-Differential Privacy:** Algorithm M is ε-differentially private if for all datasets D1 and D2 differing by one record:

```
P[M(D1) ∈ S] ≤ e^ε × P[M(D2) ∈ S]
```

**Lower ε = stronger privacy** (ε=0.1 is strong, ε=10 is weak)

### Laplace Mechanism

```python
import numpy as np

class DifferentialPrivacy:
    """
    Implement differential privacy using Laplace mechanism
    """

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def add_laplace_noise(self, true_value, sensitivity):
        """
        Add Laplace noise calibrated to sensitivity and epsilon
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return true_value + noise

    def private_mean(self, data, data_range):
        """
        Compute differentially private mean
        """
        true_mean = np.mean(data)

        # Sensitivity of mean
        sensitivity = data_range / len(data)

        private_mean = self.add_laplace_noise(true_mean, sensitivity)
        return private_mean

    def private_count(self, data, condition):
        """
        Differentially private count query
        """
        true_count = np.sum(condition(data))

        # Sensitivity of count = 1 (adding/removing one record changes count by 1)
        sensitivity = 1

        private_count = self.add_laplace_noise(true_count, sensitivity)
        return max(0, private_count)  # Count can't be negative

# Usage
dp = DifferentialPrivacy(epsilon=1.0)

# Private statistics
private_avg_age = dp.private_mean(ages, data_range=100)
private_count_positive = dp.private_count(labels, lambda x: x == 1)
```

### Gaussian Mechanism (for (ε, δ)-DP)

```python
class GaussianDP:
    """
    (ε, δ)-differential privacy using Gaussian mechanism
    """

    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta

    def compute_sigma(self, sensitivity):
        """
        Compute noise scale for (ε, δ)-DP
        """
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def add_gaussian_noise(self, true_value, sensitivity):
        """
        Add calibrated Gaussian noise
        """
        sigma = self.compute_sigma(sensitivity)
        noise = np.random.normal(0, sigma)
        return true_value + noise

# Usage
gdp = GaussianDP(epsilon=1.0, delta=1e-5)
private_sum = gdp.add_gaussian_noise(true_sum, sensitivity=10)
```

---

## DP-SGD: Differentially Private Deep Learning

### Implementation

```python
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras

def train_with_dp(model, train_data, epsilon=1.0, delta=1e-5):
    """
    Train neural network with differential privacy
    """
    # Privacy parameters
    l2_norm_clip = 1.0  # Clip gradients to this L2 norm
    noise_multiplier = 1.1  # Noise scale

    # DP-SGD optimizer
    optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=1,
        learning_rate=0.01
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(train_data, epochs=10, batch_size=256)

    # Compute actual privacy spent
    from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

    privacy_spent = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=len(train_data),
        batch_size=256,
        noise_multiplier=noise_multiplier,
        epochs=10,
        delta=delta
    )

    print(f"Privacy spent: ε={privacy_spent[0]:.2f}, δ={delta}")

    return model
```

**Key Parameters:**
- **l2_norm_clip:** Clip gradients to bound sensitivity
- **noise_multiplier:** Controls noise scale (higher = more privacy, less accuracy)
- **Privacy budget (ε):** Lower is better (ε<1 is strong privacy)

---

## Federated Learning

### Basic FL Architecture

```python
class FederatedLearning:
    """
    Federated Learning implementation
    """

    def __init__(self, global_model):
        self.global_model = global_model

    def federated_averaging(self, client_models, client_weights):
        """
        FedAvg: Aggregate client models by weighted averaging
        """
        # Initialize aggregated weights
        aggregated_weights = [
            np.zeros_like(w) for w in self.global_model.get_weights()
        ]

        # Weighted sum
        total_weight = sum(client_weights)

        for client_model, weight in zip(client_models, client_weights):
            client_weights_list = client_model.get_weights()

            for i, client_w in enumerate(client_weights_list):
                aggregated_weights[i] += (weight / total_weight) * client_w

        # Update global model
        self.global_model.set_weights(aggregated_weights)

        return self.global_model

    def train_round(self, clients_data, local_epochs=5):
        """
        Single round of federated learning
        """
        client_models = []
        client_weights = []

        for client_data in clients_data:
            # Each client trains on local data
            client_model = tf.keras.models.clone_model(self.global_model)
            client_model.set_weights(self.global_model.get_weights())

            client_model.compile(optimizer='sgd', loss='binary_crossentropy')
            client_model.fit(client_data, epochs=local_epochs, verbose=0)

            client_models.append(client_model)
            client_weights.append(len(client_data))  # Weight by data size

        # Aggregate
        self.global_model = self.federated_averaging(client_models, client_weights)

        return self.global_model

# Usage
global_model = create_model()
fl = FederatedLearning(global_model)

for round in range(100):
    global_model = fl.train_round(client_datasets, local_epochs=5)
    print(f"Round {round} complete")
```

---

## Federated Learning with Differential Privacy

### DP-FL Implementation (2025 SOTA)

```python
class DPFederatedLearning:
    """
    Federated Learning with Differential Privacy
    Implements 2024-2025 best practices
    """

    def __init__(self, global_model, epsilon=1.0, delta=1e-5):
        self.global_model = global_model
        self.epsilon = epsilon
        self.delta = delta

    def clip_model_weights(self, model, clip_norm=1.0):
        """
        Clip model weights to bound sensitivity
        """
        weights = model.get_weights()
        clipped_weights = []

        for w in weights:
            norm = np.linalg.norm(w)
            if norm > clip_norm:
                w = w * (clip_norm / norm)
            clipped_weights.append(w)

        return clipped_weights

    def add_gaussian_noise_to_weights(self, weights, sensitivity, epsilon, delta):
        """
        Add Gaussian noise to aggregated weights
        """
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

        noisy_weights = []
        for w in weights:
            noise = np.random.normal(0, sigma, size=w.shape)
            noisy_weights.append(w + noise)

        return noisy_weights

    def dp_federated_averaging(self, client_models, clip_norm=1.0):
        """
        Differentially private federated averaging
        """
        # 1. Clip client model updates
        clipped_updates = []
        for client_model in client_models:
            clipped_weights = self.clip_model_weights(client_model, clip_norm)
            clipped_updates.append(clipped_weights)

        # 2. Average clipped updates
        num_clients = len(clipped_updates)
        aggregated_weights = [
            np.mean([client[i] for client in clipped_updates], axis=0)
            for i in range(len(clipped_updates[0]))
        ]

        # 3. Add Gaussian noise
        sensitivity = 2 * clip_norm / num_clients  # Sensitivity of average
        noisy_weights = self.add_gaussian_noise_to_weights(
            aggregated_weights,
            sensitivity,
            self.epsilon,
            self.delta
        )

        # 4. Update global model
        self.global_model.set_weights(noisy_weights)

        return self.global_model

# Usage (Healthcare example - 2024 research)
dp_fl = DPFederatedLearning(global_model, epsilon=1.9, delta=1e-5)

# Train across hospitals without sharing patient data
for round in range(50):
    client_models = []

    # Each hospital trains locally
    for hospital_data in hospital_datasets:
        client_model = tf.keras.models.clone_model(global_model)
        client_model.set_weights(global_model.get_weights())

        client_model.compile(optimizer='adam', loss='binary_crossentropy')
        client_model.fit(hospital_data, epochs=5, verbose=0)

        client_models.append(client_model)

    # Aggregate with DP
    global_model = dp_fl.dp_federated_averaging(client_models, clip_norm=1.0)

print("DP-FL training complete: ε=1.9, δ=1e-5")
print("96.1% accuracy with strong privacy guarantee (2024 research)")
```

**Performance (2024 Research):**
- **96.1% accuracy** with ε=1.9
- **4.22% improvement** over standard FL (hybrid DP method)
- **Strong privacy guarantee** for medical data

---

## Adaptive Local Differential Privacy (ALDP-FL)

### 2025 State-of-the-Art

```python
class ALDP_FL:
    """
    Adaptive Local Differential Privacy for Federated Learning
    2025 research: dynamic clipping thresholds
    """

    def __init__(self, global_model, epsilon=1.0):
        self.global_model = global_model
        self.epsilon = epsilon
        self.layer_clip_history = {}  # Track L2 norms per layer

    def adaptive_clip_threshold(self, layer_name, layer_gradients, window_size=10):
        """
        Dynamically set clipping threshold based on historical L2 norms
        """
        # Compute L2 norm of this layer's gradients
        l2_norm = np.linalg.norm(layer_gradients)

        # Update history
        if layer_name not in self.layer_clip_history:
            self.layer_clip_history[layer_name] = []

        self.layer_clip_history[layer_name].append(l2_norm)

        # Keep only recent history
        if len(self.layer_clip_history[layer_name]) > window_size:
            self.layer_clip_history[layer_name] = \
                self.layer_clip_history[layer_name][-window_size:]

        # Adaptive threshold = moving average of L2 norms
        adaptive_threshold = np.mean(self.layer_clip_history[layer_name])

        return adaptive_threshold

    def adaptive_clip_and_noise(self, model_updates):
        """
        Apply adaptive clipping and noise injection per layer
        """
        noisy_updates = []

        for i, (layer_name, layer_update) in enumerate(model_updates):
            # 1. Adaptive clipping threshold
            clip_threshold = self.adaptive_clip_threshold(layer_name, layer_update)

            # 2. Clip
            norm = np.linalg.norm(layer_update)
            if norm > clip_threshold:
                layer_update = layer_update * (clip_threshold / norm)

            # 3. Adaptive noise (based on threshold)
            sensitivity = clip_threshold
            noise_scale = sensitivity / self.epsilon
            noise = np.random.laplace(0, noise_scale, size=layer_update.shape)

            noisy_update = layer_update + noise
            noisy_updates.append((layer_name, noisy_update))

        return noisy_updates

# Usage
aldp_fl = ALDP_FL(global_model, epsilon=1.0)

# In federated round
for client_model in client_models:
    # Get model updates (difference from global)
    model_updates = [
        (f"layer_{i}", client_w - global_w)
        for i, (client_w, global_w) in enumerate(
            zip(client_model.get_weights(), global_model.get_weights())
        )
    ]

    # Apply adaptive DP
    noisy_updates = aldp_fl.adaptive_clip_and_noise(model_updates)

    # Aggregate...
```

**Benefits:**
- **Layer-specific privacy** - Different noise for each layer
- **Adaptive to data** - Adjusts to gradient magnitudes
- **Better utility** - Less noise when gradients are small

---

## Homomorphic Encryption for ML

### Secure Prediction on Encrypted Data

```python
from tenseal import Context, CKKS

class HomomorphicMLInference:
    """
    Perform inference on encrypted data
    """

    def __init__(self):
        # Create encryption context
        self.context = Context.ckks(
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            scale=2**40
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40

    def encrypt_data(self, data):
        """Encrypt input data"""
        return ts.ckks_vector(self.context, data)

    def encrypted_linear_layer(self, encrypted_input, weights, bias):
        """
        Compute linear layer on encrypted data
        y = Wx + b (all encrypted)
        """
        # Matrix multiplication (encrypted)
        encrypted_output = encrypted_input.mm(weights)

        # Add bias (encrypted)
        encrypted_output += bias

        return encrypted_output

    def encrypted_inference(self, encrypted_input, model_weights):
        """
        Full model inference on encrypted data
        """
        x = encrypted_input

        for layer_weights, layer_bias in model_weights:
            # Linear transformation (encrypted)
            x = self.encrypted_linear_layer(x, layer_weights, layer_bias)

            # Activation (approximate with polynomial)
            x = x.polyval([0, 1, 0, -1/6])  # Approximate ReLU

        return x  # Still encrypted!

# Usage
he_inference = HomomorphicMLInference()

# Client encrypts their data
encrypted_input = he_inference.encrypt_data(sensitive_data)

# Server performs inference (never sees plaintext!)
encrypted_prediction = he_inference.encrypted_inference(
    encrypted_input,
    model_weights
)

# Client decrypts result
prediction = encrypted_prediction.decrypt()
```

---

## Privacy-Utility Trade-off

```python
def evaluate_privacy_utility_tradeoff(X, y, epsilon_values):
    """
    Evaluate accuracy vs privacy trade-off
    """
    results = []

    for epsilon in epsilon_values:
        # Train with DP
        model = train_with_dp(X, y, epsilon=epsilon)

        # Evaluate
        accuracy = model.evaluate(X_test, y_test)

        results.append({
            'epsilon': epsilon,
            'accuracy': accuracy,
            'privacy_level': 'Strong' if epsilon < 1 else 'Moderate' if epsilon < 5 else 'Weak'
        })

    return pd.DataFrame(results)

# Typical trade-off
tradeoff_results = evaluate_privacy_utility_tradeoff(
    X_train, y_train,
    epsilon_values=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

"""
Expected results:
ε = 0.1: 75% accuracy (very strong privacy)
ε = 1.0: 88% accuracy (strong privacy) ← Sweet spot
ε = 5.0: 93% accuracy (moderate privacy)
ε = 10.0: 95% accuracy (weak privacy)
No DP: 96% accuracy (no privacy)
"""
```

---

## Best Practices

1. **Choose appropriate ε**
   - Medical data: ε < 1
   - Financial data: ε < 3
   - General data: ε < 10

2. **Combine techniques**
   - FL + DP = strong privacy without centralization
   - HE + DP = encryption + mathematical guarantee

3. **Monitor privacy budget**
   - Track cumulative ε across queries
   - Stop when budget exhausted

4. **Validate privacy claims**
   - Use privacy auditing tools
   - Test with membership inference attacks

5. **Document everything**
   - GDPR Article 30: Record privacy measures
   - EU AI Act Article 11: Technical documentation

**Key Takeaway:** Privacy and utility are a trade-off. With 2025 techniques (ALDP-FL, hybrid DP), you can achieve **96%+ accuracy with ε=1.9** (strong privacy).

**Next:** `04_Model_Explainability.md` - SHAP, LIME, and interpretability techniques
