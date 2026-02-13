# Anomaly Detection with Deep Learning

## Table of Contents

- [Introduction](#introduction)
- [Autoencoder-Based Anomaly Detection](#autoencoder-based-anomaly-detection)
  - [Standard Autoencoder](#standard-autoencoder)
  - [Denoising Autoencoder](#denoising-autoencoder)
  - [Sparse Autoencoder](#sparse-autoencoder)
  - [Complete Autoencoder Anomaly Detector](#complete-autoencoder-anomaly-detector)
- [Variational Autoencoder Anomaly Detection](#variational-autoencoder-anomaly-detection)
  - [ELBO Loss and Anomaly Scoring](#elbo-loss-and-anomaly-scoring)
  - [VAE Anomaly Detector Implementation](#vae-anomaly-detector-implementation)
- [GAN-Based Anomaly Detection](#gan-based-anomaly-detection)
  - [AnoGAN](#anogan)
  - [f-AnoGAN](#f-anogan)
  - [GANomaly](#ganomaly)
  - [Skip-GANomaly](#skip-ganomaly)
  - [GANomaly Implementation](#ganomaly-implementation)
- [Contrastive Learning for OOD Detection](#contrastive-learning-for-ood-detection)
  - [SimCLR-Based Anomaly Detection](#simclr-based-anomaly-detection)
  - [Deep SVDD](#deep-svdd)
  - [Contrastive OOD Detection Implementation](#contrastive-ood-detection-implementation)
- [Transformer-Based Anomaly Detection](#transformer-based-anomaly-detection)
  - [Anomaly Transformer](#anomaly-transformer)
  - [Patch-Based Transformers for Image Anomaly Detection](#patch-based-transformers-for-image-anomaly-detection)
  - [Transformer Anomaly Detector Implementation](#transformer-anomaly-detector-implementation)
- [Graph Neural Network Anomaly Detection](#graph-neural-network-anomaly-detection)
  - [Node-Level Anomaly Detection](#node-level-anomaly-detection)
  - [Edge-Level Anomaly Detection](#edge-level-anomaly-detection)
  - [GNN Anomaly Detection Implementation](#gnn-anomaly-detection-implementation)
- [Image Anomaly Detection Industrial](#image-anomaly-detection-industrial)
  - [PatchCore](#patchcore)
  - [PaDiM](#padim)
  - [SPADE](#spade)
  - [Anomalib Library](#anomalib-library)
  - [PatchCore with Anomalib Implementation](#patchcore-with-anomalib-implementation)
- [Self-Supervised Approaches](#self-supervised-approaches)
  - [Rotation Prediction](#rotation-prediction)
  - [CutPaste](#cutpaste)
  - [DRAEM](#draem)
  - [Natural Synthetic Anomalies](#natural-synthetic-anomalies)
- [Video Anomaly Detection](#video-anomaly-detection)
  - [Frame Prediction Methods](#frame-prediction-methods)
  - [Memory-Augmented Networks](#memory-augmented-networks)
  - [Weakly Supervised Video Anomaly Detection](#weakly-supervised-video-anomaly-detection)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [Production Considerations](#production-considerations)
- [See Also](#see-also)
- [Resources](#resources)

---

## Introduction

**Anomaly detection** (also called outlier detection or novelty detection) identifies data points that deviate significantly from expected patterns. Deep learning approaches have transformed this field by learning complex, high-dimensional representations of normality without hand-crafted features.

**Core paradigm**: Train a model to understand "normal" data, then flag deviations as anomalies. This is fundamentally a one-class or semi-supervised learning problem -- labeled anomalies are scarce or absent during training.

**Advantages of deep learning over classical methods**:

- **Automatic feature extraction**: No need for domain-specific feature engineering; networks learn relevant representations from raw data (images, time series, graphs).
- **Scalability**: Handles high-dimensional data (megapixel images, long time series) where classical methods like Isolation Forest or One-Class SVM struggle.
- **Non-linear decision boundaries**: Captures complex manifold structures that linear or kernel methods cannot efficiently represent.
- **Multi-modal fusion**: Naturally combines heterogeneous data sources (sensor readings, images, logs) within unified architectures.
- **Localization**: Deep models can pinpoint where in an image or when in a time series the anomaly occurs, not just flag the entire sample.

**Taxonomy of deep anomaly detection**:

| Category | Method | Key Idea |
|---|---|---|
| Reconstruction-based | Autoencoder, VAE | High reconstruction error signals anomaly |
| Generative | GAN, Flow | Anomalies fall outside learned distribution |
| Self-supervised | Rotation, CutPaste | Pretext tasks define normality boundary |
| Contrastive | Deep SVDD, SimCLR | Collapse normal data to compact representation |
| Attention-based | Anomaly Transformer | Association discrepancy in attention patterns |
| Memory-based | PatchCore, MemAE | Compare against stored normal prototypes |
| Graph-based | GNN methods | Structural and attribute anomalies in graphs |

**When to use deep anomaly detection**: When data is high-dimensional, when you have abundant normal data but few or no labeled anomalies, when you need anomaly localization, or when classical methods plateau on your problem.

---

## Autoencoder-Based Anomaly Detection

### Standard Autoencoder

The **autoencoder (AE)** is the foundational architecture for reconstruction-based anomaly detection. The core idea: an autoencoder trained only on normal data will reconstruct normal samples well but fail on anomalies, producing high reconstruction error.

**Architecture**: An encoder maps input x to a lower-dimensional bottleneck representation z, and a decoder reconstructs x-hat from z. The bottleneck forces the network to learn a compressed representation of the normal data manifold.

```
Input x  -->  Encoder  -->  z (bottleneck)  -->  Decoder  -->  x_hat
               f(x)         dim << input dim       g(z)
```

**Loss function**: Mean Squared Error (MSE) between input and reconstruction:

```
L(x, x_hat) = (1/n) * sum((x_i - x_hat_i)^2)
```

**Anomaly scoring**: At inference time, compute the reconstruction error for each sample. Anomalies produce higher error because the autoencoder has not learned to reconstruct patterns outside the normal distribution.

**Threshold selection**: Fit a distribution to reconstruction errors on a held-out validation set of normal data, then set threshold as:

```
threshold = mean(errors_val) + k * std(errors_val)
```

Common values for k range from 2 to 4. Alternatively, use a percentile-based threshold (e.g., 99th percentile of validation errors).

**Architecture design considerations**:

- **Bottleneck size**: Too large and anomalies can be reconstructed; too small and normal data reconstruction degrades. Typical ratios: 5-20% of input dimension.
- **Depth**: Deeper encoders capture hierarchical features but risk overfitting. Start with 2-4 layers and increase as needed.
- **Activation**: ReLU for hidden layers, sigmoid or none for output (depending on input normalization).
- **Skip connections**: Avoid them -- they allow the network to bypass the bottleneck and reconstruct anomalies.

### Denoising Autoencoder

The **denoising autoencoder (DAE)** adds noise to the input during training, forcing the network to learn robust features rather than identity mapping. This improves anomaly detection because the model learns the underlying data manifold structure more effectively.

**Training procedure**: Corrupt input x with noise (Gaussian, masking, or salt-and-pepper) to produce x_noisy, then train the autoencoder to reconstruct the clean x from x_noisy.

```
L_DAE(x) = ||x - g(f(x + noise))||^2
```

**Noise types for anomaly detection**:
- **Gaussian noise**: x_noisy = x + epsilon, epsilon ~ N(0, sigma^2). Sigma typically 0.1-0.5.
- **Masking noise**: Randomly zero out a fraction of input features. Masking rate 0.1-0.3.
- **Dropout noise**: Apply dropout in the encoder during training.

### Sparse Autoencoder

The **sparse autoencoder** adds a sparsity constraint to the bottleneck activations, encouraging the network to use only a few active neurons for each input. This produces more distinctive representations of normal data.

**Sparsity penalty**: Add KL divergence between average activation and a target sparsity level rho (typically 0.05):

```
L_sparse = L_reconstruction + beta * sum(KL(rho || rho_hat_j))
```

where rho_hat_j is the average activation of hidden unit j over the training batch.

### Complete Autoencoder Anomaly Detector

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional


class AnomalyAutoencoder(nn.Module):
    """Autoencoder for anomaly detection with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        bottleneck_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = bottleneck_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class AEAnomalyDetector:
    """Complete anomaly detection pipeline using autoencoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        bottleneck_dim: int = 16,
        lr: float = 1e-3,
        threshold_sigma: float = 3.0,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = AnomalyAutoencoder(
            input_dim, hidden_dims, bottleneck_dim
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss(reduction="none")
        self.threshold = None
        self.threshold_sigma = threshold_sigma
        self.train_errors_mean = None
        self.train_errors_std = None

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        early_stopping_patience: int = 10,
    ) -> dict:
        """Train the autoencoder on normal data."""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                x_hat, _ = self.model(batch_x)
                loss = self.criterion(x_hat, batch_x).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation phase
            if X_val is not None:
                val_loss = self._compute_loss(X_val)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.model.load_state_dict(best_state)
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}"
                if X_val is not None:
                    msg += f" - Val Loss: {history['val_loss'][-1]:.6f}"
                print(msg)

        # Compute threshold from training data reconstruction errors
        val_data = X_val if X_val is not None else X_train
        errors = self.compute_anomaly_scores(val_data)
        self.train_errors_mean = np.mean(errors)
        self.train_errors_std = np.std(errors)
        self.threshold = (
            self.train_errors_mean + self.threshold_sigma * self.train_errors_std
        )
        print(f"Threshold set to {self.threshold:.6f} "
              f"(mean={self.train_errors_mean:.6f}, "
              f"std={self.train_errors_std:.6f})")
        return history

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction error."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            x_hat, _ = self.model(X_tensor)
            # Per-sample MSE
            errors = self.criterion(x_hat, X_tensor).mean(dim=1).cpu().numpy()
        return errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions: 1 = anomaly, 0 = normal."""
        scores = self.compute_anomaly_scores(X)
        return (scores > self.threshold).astype(int)

    def _compute_loss(self, X: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            x_hat, _ = self.model(X_tensor)
            return self.criterion(x_hat, X_tensor).mean().item()


# Usage example
if __name__ == "__main__":
    np.random.seed(42)
    X_normal = np.random.randn(5000, 50).astype(np.float32)
    X_anomaly = np.random.randn(100, 50).astype(np.float32) + 4.0

    split = int(0.8 * len(X_normal))
    X_train, X_val = X_normal[:split], X_normal[split:]

    detector = AEAnomalyDetector(
        input_dim=50,
        hidden_dims=[128, 64, 32],
        bottleneck_dim=8,
        threshold_sigma=3.0,
        device="cpu",
    )
    history = detector.fit(X_train, X_val, epochs=50, batch_size=128)

    preds_normal = detector.predict(X_val)
    preds_anomaly = detector.predict(X_anomaly)
    print(f"Normal flagged as anomaly: {preds_normal.mean():.2%}")
    print(f"Anomalies detected: {preds_anomaly.mean():.2%}")
```

---

## Variational Autoencoder Anomaly Detection

### ELBO Loss and Anomaly Scoring

The **Variational Autoencoder (VAE)** extends the standard autoencoder with a probabilistic framework. Instead of mapping to a deterministic bottleneck, the encoder produces parameters of a latent distribution q(z|x), typically Gaussian.

**ELBO (Evidence Lower Bound) loss**:

```
L_VAE = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
       = Reconstruction Term  - KL Divergence Term
```

- **Reconstruction term**: How well the decoder reconstructs x from sampled z. Uses MSE or binary cross-entropy.
- **KL divergence term**: Regularizes the latent space to stay close to the prior p(z) = N(0, I). For Gaussian encoder q(z|x) = N(mu, sigma^2):

```
KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
```

**Anomaly scoring with VAE** -- multiple options:

1. **Reconstruction probability**: Sample z multiple times, compute average reconstruction likelihood. Low probability indicates anomaly.
2. **ELBO score**: Use the full ELBO as anomaly score. Anomalies have lower ELBO (higher loss).
3. **KL divergence only**: Anomalies may produce latent representations far from the prior.
4. **Mahalanobis distance in latent space**: Compute distance of encoded mu from the origin (or cluster center) in latent space.

**Advantages of VAE over AE for anomaly detection**:
- Structured latent space enables density estimation.
- Sampling provides uncertainty quantification.
- Less prone to "holes" in latent space where anomalies might be reconstructed well.
- Multiple anomaly scoring options for different use cases.

### VAE Anomaly Detector Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple


class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            prev_dim = h_dim
        self.shared = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAEAnomalyDetector(nn.Module):
    """Variational Autoencoder for anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        latent_dim: int = 16,
        beta: float = 1.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.beta = beta
        self.encoder = VAEEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = VAEDecoder(latent_dim, list(reversed(hidden_dims)), input_dim)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ELBO loss with beta weighting on KL term."""
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def compute_anomaly_score(
        self, x: torch.Tensor, n_samples: int = 10
    ) -> torch.Tensor:
        """Compute anomaly score using Monte Carlo sampling of ELBO."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            scores = []
            for _ in range(n_samples):
                z = self.reparameterize(mu, logvar)
                x_hat = self.decoder(z)
                # Per-sample reconstruction error
                recon_error = F.mse_loss(
                    x_hat, x, reduction="none"
                ).mean(dim=1)
                # Per-sample KL divergence
                kl_div = -0.5 * (
                    1 + logvar - mu.pow(2) - logvar.exp()
                ).sum(dim=1)
                scores.append(recon_error + self.beta * kl_div)
            return torch.stack(scores).mean(dim=0)


def train_vae_detector(
    model: VAEAnomalyDetector,
    X_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """Training loop for VAE anomaly detector."""
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.FloatTensor(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    history = {"total": [], "recon": [], "kl": []}

    for epoch in range(epochs):
        model.train()
        epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            x_hat, mu, logvar = model(batch_x)
            total, recon, kl = model.loss_function(batch_x, x_hat, mu, logvar)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_total += total.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()

        n = len(loader)
        history["total"].append(epoch_total / n)
        history["recon"].append(epoch_recon / n)
        history["kl"].append(epoch_kl / n)

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Total: {history['total'][-1]:.4f}, "
                f"Recon: {history['recon'][-1]:.4f}, "
                f"KL: {history['kl'][-1]:.4f}"
            )
    return history
```

---

## GAN-Based Anomaly Detection

### AnoGAN

**AnoGAN** (Schlegl et al., 2017) was the first GAN-based anomaly detection method. The key insight: a GAN trained on normal data can only generate normal samples. If a test sample cannot be well-represented in the GAN's latent space, it is anomalous.

**Procedure**:
1. Train a standard GAN (generator G, discriminator D) on normal data only.
2. For each test sample x, find z* that minimizes: `L(z) = lambda_r * ||x - G(z)||_1 + lambda_d * ||f_D(x) - f_D(G(z))||_1` where f_D extracts intermediate discriminator features.
3. Anomaly score = optimized loss L(z*).

**Major limitation**: The iterative optimization over z at inference time is extremely slow (hundreds of gradient steps per sample).

### f-AnoGAN

**f-AnoGAN** (fast AnoGAN, Schlegl et al., 2019) addresses the speed limitation by training an encoder network to directly map inputs to the GAN latent space.

**Two-stage training**:
1. Train WGAN on normal data.
2. Train encoder E to map x to z such that G(E(x)) approximates x.

This replaces the expensive optimization loop with a single forward pass, achieving real-time inference.

### GANomaly

**GANomaly** (Akcay et al., 2018) uses an encoder-decoder-encoder architecture that captures the anomaly signal more effectively than reconstruction error alone.

**Architecture**:
- **Generator**: Encoder1 maps x to z1, Decoder maps z1 to x_hat.
- **Sub-encoder**: Encoder2 maps x_hat to z2.
- **Discriminator**: Classifies real vs. generated images.

**Loss function** (three components):
- **Adversarial loss**: Standard GAN loss for realistic generation.
- **Contextual loss**: ||x - x_hat||_1 (reconstruction quality).
- **Encoder loss**: ||z1 - z2||_2 (latent consistency).

**Anomaly score**: ||E1(x) - E2(G(E1(x)))||_2. Normal data produces consistent latent codes; anomalies cause discrepancy between the two encodings.

### Skip-GANomaly

**Skip-GANomaly** (Akay et al., 2019) extends GANomaly with U-Net-style skip connections in the generator, specifically designed for texture anomaly detection. The skip connections preserve spatial details while the anomaly scoring still relies on latent code consistency. This architecture is particularly effective for surface inspection tasks.

### GANomaly Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple


class GANomalyEncoder(nn.Module):
    """Encoder network for GANomaly (convolutional)."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 100, img_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        nf = 64  # base number of filters

        self.conv_layers = nn.Sequential(
            # img_size -> img_size/2
            nn.Conv2d(in_channels, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # img_size/2 -> img_size/4
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # img_size/4 -> img_size/8
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # img_size/8 -> img_size/16
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Compute flattened size: nf*8 * (img_size/16) * (img_size/16)
        feat_size = nf * 8 * (img_size // 16) * (img_size // 16)
        self.fc = nn.Linear(feat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        return self.fc(features.view(features.size(0), -1))


class GANomalyDecoder(nn.Module):
    """Decoder network for GANomaly (convolutional)."""

    def __init__(
        self, out_channels: int = 1, latent_dim: int = 100, img_size: int = 64
    ):
        super().__init__()
        nf = 64
        self.init_size = img_size // 16
        self.fc = nn.Linear(latent_dim, nf * 8 * self.init_size * self.init_size)
        self.nf = nf

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.nf * 8, self.init_size, self.init_size)
        return self.deconv_layers(h)


class GANomalyGenerator(nn.Module):
    """GANomaly generator: Encoder1 -> Decoder -> Encoder2."""

    def __init__(
        self, in_channels: int = 1, latent_dim: int = 100, img_size: int = 64
    ):
        super().__init__()
        self.encoder1 = GANomalyEncoder(in_channels, latent_dim, img_size)
        self.decoder = GANomalyDecoder(in_channels, latent_dim, img_size)
        self.encoder2 = GANomalyEncoder(in_channels, latent_dim, img_size)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.encoder1(x)
        x_hat = self.decoder(z1)
        z2 = self.encoder2(x_hat)
        return x_hat, z1, z2


class GANomalyDiscriminator(nn.Module):
    """PatchGAN discriminator for GANomaly."""

    def __init__(self, in_channels: int = 1, img_size: int = 64):
        super().__init__()
        nf = 64
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        feat_size = nf * 4 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Linear(feat_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        return self.classifier(features.view(features.size(0), -1))


class GANomalyTrainer:
    """Training and inference for GANomaly."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 100,
        img_size: int = 64,
        lr: float = 2e-4,
        w_adv: float = 1.0,
        w_con: float = 50.0,
        w_enc: float = 1.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.generator = GANomalyGenerator(
            in_channels, latent_dim, img_size
        ).to(self.device)
        self.discriminator = GANomalyDiscriminator(
            in_channels, img_size
        ).to(self.device)

        self.opt_g = optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        self.opt_d = optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
        )

        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.w_adv = w_adv
        self.w_con = w_con
        self.w_enc = w_enc

    def train_step(self, real: torch.Tensor) -> dict:
        real = real.to(self.device)
        batch_size = real.size(0)
        real_label = torch.ones(batch_size, 1, device=self.device)
        fake_label = torch.zeros(batch_size, 1, device=self.device)

        # --- Update Discriminator ---
        self.opt_d.zero_grad()
        x_hat, _, _ = self.generator(real)
        pred_real = self.discriminator(real)
        pred_fake = self.discriminator(x_hat.detach())
        loss_d = (
            self.bce(pred_real, real_label) + self.bce(pred_fake, fake_label)
        ) * 0.5
        loss_d.backward()
        self.opt_d.step()

        # --- Update Generator ---
        self.opt_g.zero_grad()
        x_hat, z1, z2 = self.generator(real)
        pred_fake = self.discriminator(x_hat)

        loss_adv = self.bce(pred_fake, real_label)
        loss_con = self.l1(x_hat, real)
        loss_enc = self.l2(z1, z2)

        loss_g = (
            self.w_adv * loss_adv
            + self.w_con * loss_con
            + self.w_enc * loss_enc
        )
        loss_g.backward()
        self.opt_g.step()

        return {
            "loss_d": loss_d.item(),
            "loss_g": loss_g.item(),
            "loss_adv": loss_adv.item(),
            "loss_con": loss_con.item(),
            "loss_enc": loss_enc.item(),
        }

    def compute_anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """Anomaly score = ||z1 - z2||_2 per sample."""
        self.generator.eval()
        with torch.no_grad():
            x = x.to(self.device)
            _, z1, z2 = self.generator(x)
            scores = torch.mean((z1 - z2) ** 2, dim=1).cpu().numpy()
        return scores
```

---

## Contrastive Learning for OOD Detection

### SimCLR-Based Anomaly Detection

**Contrastive learning** trains models to pull representations of similar data points together while pushing dissimilar points apart. For anomaly detection, the model learns a compact representation of normal data; test samples that fall far from this cluster are anomalous.

**SimCLR framework adapted for anomaly detection**:
1. Apply random augmentations (crop, flip, color jitter) to create two views of each normal sample.
2. Train with NT-Xent (Normalized Temperature-Scaled Cross-Entropy) loss to bring augmented views of the same sample together.
3. At inference, compute cosine similarity between test sample embedding and nearest normal embeddings. Low similarity indicates anomaly.

**NT-Xent loss** for a positive pair (i, j) in a batch of N pairs:

```
L(i,j) = -log( exp(sim(z_i, z_j) / tau) / sum_k[k!=i]( exp(sim(z_i, z_k) / tau) ) )
```

where sim(u, v) = (u . v) / (||u|| * ||v||) is cosine similarity and tau is a temperature parameter (typically 0.1-0.5).

### Deep SVDD

**Deep Support Vector Data Description** (Ruff et al., 2018) learns a neural network mapping that collapses normal data representations to a single point (hypersphere center) in the latent space. Anomalies will be mapped far from this center.

**Objective**: Minimize the volume of the hypersphere enclosing normal data representations:

```
L_SVDD = (1/N) * sum( ||phi(x_i; W) - c||^2 ) + (lambda / 2) * ||W||^2
```

where c is the center of the hypersphere (initialized as the mean of representations from a forward pass) and phi is the learned feature mapping.

**Key constraint**: Bias terms must be removed from the network to prevent the trivial solution of mapping everything to c. Use networks without bias and without bounded activations (no sigmoid/tanh in hidden layers; use ReLU).

**Anomaly score**: ||phi(x; W) - c||^2 -- the squared distance from the hypersphere center.

### Contrastive OOD Detection Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class DeepSVDD(nn.Module):
    """Deep Support Vector Data Description for anomaly detection."""

    def __init__(self, input_dim: int, hidden_dims: list = None, rep_dim: int = 32):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            # No bias terms to prevent trivial collapse
            layers.extend([
                nn.Linear(prev_dim, h_dim, bias=False),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, rep_dim, bias=False))
        self.net = nn.Sequential(*layers)
        self.rep_dim = rep_dim
        self.center = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def init_center(self, loader: DataLoader, device: torch.device) -> None:
        """Initialize hypersphere center as mean of initial representations."""
        self.eval()
        representations = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                rep = self.forward(batch_x)
                representations.append(rep)
        all_reps = torch.cat(representations, dim=0)
        self.center = all_reps.mean(dim=0).to(device)
        # Avoid center being too close to zero
        self.center[(torch.abs(self.center) < 1e-6)] = 1e-6


class ContrastiveOODDetector:
    """Anomaly detection combining Deep SVDD with contrastive pretraining."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        rep_dim: int = 32,
        lr: float = 1e-3,
        svdd_nu: float = 0.1,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = DeepSVDD(input_dim, hidden_dims, rep_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-6
        )
        self.nu = svdd_nu
        self.threshold = None

    def pretrain_contrastive(
        self,
        X_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        temperature: float = 0.1,
        noise_std: float = 0.1,
    ) -> None:
        """Contrastive pretraining with augmented views."""
        proj_head = ProjectionHead(
            self.model.rep_dim, 128, 64
        ).to(self.device)
        params = list(self.model.parameters()) + list(proj_head.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            proj_head.train()
            total_loss = 0.0

            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                bs = batch_x.size(0)

                # Create two augmented views (additive noise)
                view1 = batch_x + noise_std * torch.randn_like(batch_x)
                view2 = batch_x + noise_std * torch.randn_like(batch_x)

                z1 = proj_head(self.model(view1))
                z2 = proj_head(self.model(view2))

                # NT-Xent loss
                z = torch.cat([z1, z2], dim=0)
                sim_matrix = torch.mm(z, z.t()) / temperature
                # Mask out self-similarity
                mask = ~torch.eye(2 * bs, dtype=torch.bool, device=self.device)
                sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

                # Positive pairs: (i, i+bs) and (i+bs, i)
                pos_sim = torch.sum(z1 * z2, dim=1) / temperature
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

                loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
                loss = loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Contrastive Epoch {epoch+1}/{epochs} "
                    f"- Loss: {total_loss / len(loader):.4f}"
                )

    def fit_svdd(
        self,
        X_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
    ) -> dict:
        """Train Deep SVDD objective after optional contrastive pretraining."""
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(torch.FloatTensor(X_train))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize center
        self.model.init_center(loader, self.device)
        history = {"loss": []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                rep = self.model(batch_x)
                dist = torch.sum((rep - self.model.center) ** 2, dim=1)
                loss = torch.mean(dist)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            history["loss"].append(epoch_loss / len(loader))

            if (epoch + 1) % 20 == 0:
                print(
                    f"SVDD Epoch {epoch+1}/{epochs} "
                    f"- Loss: {history['loss'][-1]:.4f}"
                )

        # Set threshold
        scores = self.compute_anomaly_scores(X_train)
        self.threshold = np.percentile(scores, 100 * (1 - self.nu))
        print(f"Threshold set to {self.threshold:.4f}")
        return history

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score = squared distance to hypersphere center."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            rep = self.model(X_t)
            dist = torch.sum(
                (rep - self.model.center) ** 2, dim=1
            ).cpu().numpy()
        return dist

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.compute_anomaly_scores(X)
        return (scores > self.threshold).astype(int)
```

---

## Transformer-Based Anomaly Detection

### Anomaly Transformer

The **Anomaly Transformer** (Xu et al., ICLR 2022) introduced the concept of **association discrepancy** for unsupervised time series anomaly detection. The key insight: anomaly points have difficulty building associations with the entire series, while normal points form strong associations with adjacent points.

**Architecture components**:
- **Anomaly-Attention mechanism**: Computes two types of associations:
  - **Prior-association**: Learnable Gaussian kernel modeling expected local attention patterns.
  - **Series-association**: Standard self-attention computed from data.
- **Association discrepancy**: KL divergence between prior and series associations. Normal points show high discrepancy (strong local prior vs. broad series attention). Anomaly points show different patterns because they cannot associate normally.

**Minimax optimization**: The model jointly optimizes reconstruction quality and maximizes the association discrepancy for more distinguishable anomaly criteria.

### Patch-Based Transformers for Image Anomaly Detection

For image anomaly detection, **Vision Transformer (ViT)** approaches divide images into patches and detect anomalies at the patch level:

1. Extract patch embeddings using a pretrained ViT backbone.
2. Model the distribution of normal patch embeddings.
3. Detect anomalous patches by measuring deviation from normal distribution.

**Advantages**: Global context through self-attention captures long-range spatial dependencies that convolutional methods miss, useful for detecting anomalies that span multiple regions.

### Transformer Anomaly Detector Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time series."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AnomalyAttention(nn.Module):
    """Anomaly attention with prior and series association."""

    def __init__(self, d_model: int, n_heads: int, seq_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Learnable prior: Gaussian kernel scale per head
        self.sigma = nn.Parameter(torch.ones(n_heads, 1, 1))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape

        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Series association (standard attention)
        series_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        series_attn = F.softmax(series_scores, dim=-1)

        # Prior association (Gaussian kernel)
        distances = torch.arange(L, device=x.device).float()
        distances = (distances.unsqueeze(0) - distances.unsqueeze(1)) ** 2
        sigma = torch.clamp(self.sigma, min=1e-4)
        prior_attn = torch.exp(-distances / (2 * sigma ** 2))
        prior_attn = prior_attn / prior_attn.sum(dim=-1, keepdim=True)
        prior_attn = prior_attn.unsqueeze(0).expand(B, -1, -1, -1)

        # Attention output
        out = torch.matmul(series_attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.W_o(out)

        return out, series_attn, prior_attn


class AnomalyTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int, d_ff: int = 256):
        super().__init__()
        self.attention = AnomalyAttention(d_model, n_heads, seq_len)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out, series_attn, prior_attn = self.attention(self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, series_attn, prior_attn


class AnomalyTransformer(nn.Module):
    """Simplified Anomaly Transformer for time series anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        seq_len: int = 100,
        d_ff: int = 256,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)

        self.layers = nn.ModuleList([
            AnomalyTransformerLayer(d_model, n_heads, seq_len, d_ff)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, list, list]:
        h = self.pos_enc(self.embedding(x))

        all_series = []
        all_prior = []
        for layer in self.layers:
            h, series_attn, prior_attn = layer(h)
            all_series.append(series_attn)
            all_prior.append(prior_attn)

        output = self.output_proj(h)
        return output, all_series, all_prior

    def compute_association_discrepancy(
        self,
        series_attns: list,
        prior_attns: list,
    ) -> torch.Tensor:
        """Compute KL divergence between series and prior associations."""
        discrepancies = []
        for series, prior in zip(series_attns, prior_attns):
            # Symmetrized KL divergence
            kl_sp = F.kl_div(
                series.log(), prior, reduction="none"
            ).sum(dim=-1).mean(dim=1)
            kl_ps = F.kl_div(
                prior.log(), series, reduction="none"
            ).sum(dim=-1).mean(dim=1)
            discrepancies.append(kl_sp + kl_ps)
        # Average across layers, sum across time steps
        disc = torch.stack(discrepancies).mean(dim=0)
        return disc.mean(dim=-1)  # Shape: (batch,)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Combined reconstruction + association discrepancy score."""
        x_hat, series_attns, prior_attns = self.forward(x)
        recon_error = F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))
        assoc_disc = self.compute_association_discrepancy(
            series_attns, prior_attns
        )
        return recon_error * assoc_disc
```

---

## Graph Neural Network Anomaly Detection

### Node-Level Anomaly Detection

**Node-level anomaly detection** identifies individual nodes in a graph whose attributes or structural roles deviate from the majority. This is critical in fraud detection (anomalous users), network intrusion detection (compromised hosts), and social network analysis (bot accounts).

**Approaches**:
- **Attribute reconstruction**: Use a GNN autoencoder to reconstruct node features. Nodes with high reconstruction error are anomalous.
- **Structure reconstruction**: Predict adjacency matrix from learned node embeddings. Anomalous nodes produce poor structural reconstructions.
- **Dual reconstruction**: Jointly reconstruct attributes and structure for more robust detection.
- **Contrastive**: Learn node representations via graph contrastive learning (e.g., GCA, GRACE). Nodes far from the normal cluster in embedding space are anomalies.

### Edge-Level Anomaly Detection

**Edge anomalies** represent unusual relationships in a graph: fraudulent transactions between accounts, unexpected communication patterns, or novel chemical bonds. Methods typically compute edge scores from concatenated or element-wise products of connected node embeddings, then threshold to identify anomalous edges.

**Structural anomalies** in graphs go beyond individual nodes or edges. They involve unusual subgraph patterns: unexpected dense connections (communities), star patterns (hub anomalies), or bridge edges connecting otherwise disconnected components.

**Applications**:
- **Fraud detection**: Unusual transaction patterns in financial networks.
- **Network intrusion**: Anomalous traffic patterns in computer networks.
- **Social network abuse**: Coordinated inauthentic behavior detection.
- **Molecular anomalies**: Unusual chemical structures in drug discovery.

### GNN Anomaly Detection Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

try:
    from torch_geometric.nn import GCNConv, GAE
    from torch_geometric.data import Data
    from torch_geometric.utils import negative_sampling
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("torch_geometric not installed. Install with: "
          "pip install torch-geometric")


class GNNEncoder(nn.Module):
    """GCN-based encoder for graph anomaly detection."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class AttributeDecoder(nn.Module):
    """Decode node attributes from embeddings."""

    def __init__(self, embed_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GraphAnomalyDetector(nn.Module):
    """Dual-objective graph anomaly detector:
    reconstructs both node attributes and graph structure."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        embed_dim: int = 32,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels, embed_dim)
        self.attr_decoder = AttributeDecoder(embed_dim, hidden_channels, in_channels)
        self.alpha = alpha  # Weight between structure and attribute loss

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, edge_index)
        x_hat = self.attr_decoder(z)
        return z, x_hat

    def structure_loss(
        self, z: torch.Tensor, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Binary cross-entropy for edge prediction."""
        # Positive edges
        pos_score = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score)
        )

        # Negative edges
        neg_edge_index = negative_sampling(
            edge_index, num_nodes=num_nodes,
            num_neg_samples=edge_index.size(1)
        )
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score)
        )
        return (pos_loss + neg_loss) / 2

    def attribute_loss(
        self, x: torch.Tensor, x_hat: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=1)

    def compute_anomaly_scores(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> np.ndarray:
        """Per-node anomaly scores combining structure and attribute."""
        self.eval()
        with torch.no_grad():
            z, x_hat = self.forward(x, edge_index)

            # Attribute reconstruction error per node
            attr_error = F.mse_loss(
                x_hat, x, reduction="none"
            ).mean(dim=1)

            # Structure reconstruction error per node
            adj_pred = torch.sigmoid(torch.mm(z, z.t()))
            # Build dense adjacency for comparison
            num_nodes = x.size(0)
            adj_true = torch.zeros(num_nodes, num_nodes, device=x.device)
            adj_true[edge_index[0], edge_index[1]] = 1.0
            struct_error = F.binary_cross_entropy(
                adj_pred, adj_true, reduction="none"
            ).mean(dim=1)

            scores = (
                self.alpha * struct_error + (1 - self.alpha) * attr_error
            )
        return scores.cpu().numpy()


def train_graph_anomaly_detector(
    model: GraphAnomalyDetector,
    data: "Data",
    epochs: int = 200,
    lr: float = 1e-3,
) -> list:
    """Train graph anomaly detector."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z, x_hat = model(data.x, data.edge_index)
        s_loss = model.structure_loss(z, data.edge_index, data.num_nodes)
        a_loss = model.attribute_loss(data.x, x_hat).mean()
        loss = model.alpha * s_loss + (1 - model.alpha) * a_loss

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Loss: {loss.item():.4f} "
                f"(struct: {s_loss.item():.4f}, "
                f"attr: {a_loss.item():.4f})"
            )
    return losses
```

---

## Image Anomaly Detection Industrial

Industrial image anomaly detection focuses on defect detection in manufacturing: scratches, dents, stains, missing components, and structural deformations on products. The challenge is extreme class imbalance -- defective samples are rare and diverse.

### PatchCore

**PatchCore** (Roth et al., CVPR 2022) achieves state-of-the-art results by building a memory bank of patch-level features from normal training images, then comparing test patches against this bank.

**Algorithm**:
1. **Feature extraction**: Use a pretrained backbone (e.g., WideResNet50) to extract intermediate feature maps from normal training images.
2. **Patch-level features**: Extract features at each spatial location, yielding a patch feature per position.
3. **Coreset subsampling**: Reduce memory bank size with greedy coreset selection (typically 1-10% of all patches) while maintaining coverage.
4. **Inference**: For each test patch, find the nearest neighbor in the memory bank. The distance is the patch-level anomaly score.
5. **Image-level score**: Maximum patch score across all positions.
6. **Anomaly map**: Spatial map of patch-level scores provides pixel-level localization.

**Strengths**: No training required (feature extraction only), extremely fast adaptation to new products, state-of-the-art on MVTec AD.

### PaDiM

**PaDiM** (Defard et al., 2021) models the distribution of normal patch embeddings using multivariate Gaussian distributions. For each patch position, it estimates mean and covariance from normal training data. At test time, the Mahalanobis distance from the position-specific Gaussian serves as the anomaly score.

**Key design choices**:
- Use features from multiple layers of a pretrained CNN (multi-scale).
- Concatenate features from different layers for richer patch representations.
- Random dimensionality reduction to manage computational cost of covariance estimation.

### SPADE

**SPADE** (Cohen and Hoshen, 2020) (Sub-Image Anomaly Detection with Deep Pyramid Correspondences) uses multi-resolution feature alignment. It extracts features at multiple scales from a pretrained network and computes k-nearest-neighbor distances between test and normal feature sets at each scale.

### Anomalib Library

**Anomalib** (Intel, open-source) provides a unified framework for image anomaly detection, implementing PatchCore, PaDiM, SPADE, STFPM, GANomaly, DRAEM, and many more methods with consistent APIs, training pipelines, and evaluation.

**Key features**:
- Standardized data loading for MVTec and custom datasets.
- Built-in metrics: AUROC (image-level and pixel-level), F1, PRO.
- OpenVINO integration for optimized inference.
- Lightning-based training with logging.

### PatchCore with Anomalib Implementation

```python
# Using anomalib for industrial anomaly detection
# pip install anomalib

from pathlib import Path
import numpy as np


def run_patchcore_anomalib():
    """PatchCore anomaly detection using anomalib library."""
    from anomalib.data import MVTec
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    # Configure datamodule for MVTec AD
    datamodule = MVTec(
        root="./datasets/MVTec",
        category="bottle",  # Choose category
        image_size=(256, 256),
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )

    # Initialize PatchCore model
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers_to_extract_from=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    # Train and test using Engine
    engine = Engine(
        max_epochs=1,  # PatchCore needs only 1 epoch (feature extraction)
        accelerator="auto",
        devices=1,
    )

    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)
    print(f"Image AUROC: {results[0]['image_AUROC']:.4f}")
    print(f"Pixel AUROC: {results[0]['pixel_AUROC']:.4f}")

    return model, engine


def custom_patchcore_inference():
    """Manual PatchCore implementation for understanding."""
    import torch
    import torch.nn.functional as F
    from torchvision import models, transforms
    from scipy.spatial.distance import cdist

    # Feature extractor from pretrained backbone
    backbone = models.wide_resnet50_2(pretrained=True)
    backbone.eval()

    # Hook to extract intermediate features
    features = {}

    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output
        return hook

    backbone.layer2.register_forward_hook(hook_fn("layer2"))
    backbone.layer3.register_forward_hook(hook_fn("layer3"))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def extract_patch_features(images: torch.Tensor) -> np.ndarray:
        """Extract and concatenate multi-scale patch features."""
        with torch.no_grad():
            _ = backbone(images)

        # Get features from both layers
        f2 = features["layer2"]  # (B, C2, H2, W2)
        f3 = features["layer3"]  # (B, C3, H3, W3)

        # Upsample layer3 features to match layer2 spatial size
        f3_up = F.interpolate(
            f3, size=f2.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate along channel dimension
        combined = torch.cat([f2, f3_up], dim=1)

        # Reshape to (B * H * W, C) patch features
        B, C, H, W = combined.shape
        patch_feats = combined.permute(0, 2, 3, 1).reshape(-1, C)
        return patch_feats.cpu().numpy(), (B, H, W)

    def build_memory_bank(
        train_features: np.ndarray, sampling_ratio: float = 0.1
    ) -> np.ndarray:
        """Coreset subsampling of patch features."""
        n_samples = max(1, int(len(train_features) * sampling_ratio))

        # Greedy coreset selection
        selected_indices = [np.random.randint(len(train_features))]
        min_distances = cdist(
            train_features,
            train_features[selected_indices],
            metric="euclidean",
        ).min(axis=1)

        for _ in range(n_samples - 1):
            new_idx = np.argmax(min_distances)
            selected_indices.append(new_idx)
            new_dist = cdist(
                train_features,
                train_features[[new_idx]],
                metric="euclidean",
            ).squeeze()
            min_distances = np.minimum(min_distances, new_dist)

        return train_features[selected_indices]

    def compute_anomaly_map(
        test_features: np.ndarray,
        memory_bank: np.ndarray,
        spatial_shape: tuple,
        k: int = 9,
    ) -> np.ndarray:
        """Compute anomaly map from patch distances to memory bank."""
        B, H, W = spatial_shape
        distances = cdist(test_features, memory_bank, metric="euclidean")
        # k-nearest neighbor distance
        knn_dist = np.sort(distances, axis=1)[:, :k].mean(axis=1)
        anomaly_map = knn_dist.reshape(B, H, W)
        return anomaly_map

    return extract_patch_features, build_memory_bank, compute_anomaly_map
```

---

## Self-Supervised Approaches

Self-supervised anomaly detection creates pretext tasks that define what "normal" looks like without explicit anomaly labels. The model learns features from normal data through these proxy objectives, and anomalies violate the learned patterns.

### Rotation Prediction

**Rotation prediction** (Gidaris et al., 2018 adapted for AD by Hendrycks et al., 2019): Train a classifier to predict the rotation angle (0, 90, 180, 270 degrees) applied to normal images. At test time, anomalous images produce higher prediction uncertainty or lower confidence because the model has not learned their rotation-equivariant features.

**Anomaly score**: Entropy of the rotation prediction distribution or negative max probability:

```
score(x) = -max_r P(rotation = r | x)
```

### CutPaste

**CutPaste** (Li et al., CVPR 2021) creates synthetic anomalies by cutting a patch from a normal image and pasting it at a random location, then trains a binary classifier to distinguish original from augmented images.

**Variants**:
- **CutPaste**: Random rectangular patch cut-and-paste.
- **CutPaste-Scar**: Thin, elongated patches mimicking scratches.
- **3-way CutPaste**: Train a 3-class classifier (normal, CutPaste, CutPaste-Scar).

**At inference**: Use the trained representations (penultimate layer features) to fit a Gaussian density estimator. Anomaly score = Mahalanobis distance from the normal distribution.

### DRAEM

**DRAEM** (Zavrtanik et al., ICCV 2021) -- Discriminatively Trained Anomaly Detection by Reconstruction of Anomalous Regions. It generates synthetic anomalies using Perlin noise textures overlaid on normal images, then trains two networks jointly:

1. **Reconstructive sub-network**: Autoencoder trained to reconstruct clean images from synthetically corrupted inputs.
2. **Discriminative sub-network**: Segmentation network that identifies anomalous regions from the difference between input and reconstruction.

This dual approach achieves both detection and localization, with the discriminative network learning to identify subtle reconstruction artifacts.

### Natural Synthetic Anomalies

**NSA** (Schluter et al., 2022) (Natural Synthetic Anomalies) improves upon CutPaste by using Poisson image editing to seamlessly blend patches from external images into normal training images. The blending creates more realistic synthetic anomalies, leading to better detection boundaries.

**Procedure**:
1. Sample a random patch from an external image source (e.g., DTD texture dataset).
2. Create a random irregular mask.
3. Use Poisson image editing (seamless cloning) to blend the patch into a normal image.
4. Train a segmentation network to detect the blended regions.

---

## Video Anomaly Detection

Video anomaly detection identifies unusual events in surveillance footage, traffic monitoring, or manufacturing lines. The temporal dimension adds complexity compared to image anomaly detection.

### Frame Prediction Methods

**Future frame prediction**: Train a model (typically a convolutional LSTM or 3D U-Net) to predict the next frame given previous frames. At inference, high prediction error indicates that the observed event deviates from learned temporal patterns.

**Key architectures**:
- **ConvLSTM prediction**: Encode spatial-temporal context with convolutional LSTMs, predict next frame.
- **U-Net with temporal skip connections**: Combine multi-scale spatial features with temporal information.
- **Video prediction GANs**: Adversarial training for sharper predictions; discriminator loss serves as additional anomaly signal.

**Optical flow-based methods**: Instead of predicting raw frames, predict optical flow fields. Anomalous events produce unexpected motion patterns. Dual-stream approaches combine appearance (RGB) and motion (flow) predictions.

### Memory-Augmented Networks

**Memory-augmented autoencoders** (Gong et al., ICCV 2019) address the problem of autoencoders reconstructing anomalies too well. They add an external memory module that stores prototypical normal patterns:

1. Encoder produces a query from the input frame.
2. Memory module retrieves and combines the most relevant stored patterns.
3. Decoder reconstructs from the memory-augmented representation.

Since the memory stores only normal patterns, anomalous inputs must be reconstructed from normal prototypes, increasing reconstruction error.

### Weakly Supervised Video Anomaly Detection

**Weakly supervised** approaches (Sultani et al., CVPR 2018) use video-level labels (this video contains an anomaly) without frame-level annotation:

**Multiple Instance Learning (MIL) framework**:
- Treat each video as a bag of frame segments.
- Normal videos: all segments are normal.
- Anomalous videos: at least one segment is anomalous.
- Train with MIL ranking loss: max-scoring segment in anomalous video should score higher than max-scoring segment in normal video.

```
L = max(0, 1 - max_i(s_a_i) + max_j(s_n_j)) + lambda * smoothness_term
```

This approach significantly reduces annotation cost while achieving reasonable detection performance.

---

## Benchmarks and Datasets

| Dataset | Domain | Samples | Anomaly Types | Metric | Notes |
|---|---|---|---|---|---|
| MVTec AD | Industrial inspection | 5354 images, 15 categories | Scratches, dents, contamination | Image/Pixel AUROC | Gold standard for industrial AD |
| MVTec LOCO AD | Industrial logical anomalies | 3644 images, 5 categories | Structural and logical defects | sPRO | Tests logical constraint violations |
| CIFAR-10 | Natural images | 60K images, 10 classes | One-vs-rest setup | AUROC | Each class as normal, rest as anomaly |
| MNIST | Handwritten digits | 70K images, 10 classes | One-vs-rest setup | AUROC | Simpler benchmark for validation |
| KDD Cup 99 | Network intrusion | 4.9M connections | 22 attack types | F1, AUROC | Classic but dated; use NSL-KDD instead |
| NSL-KDD | Network intrusion | 150K connections | 4 attack categories | F1, AUROC | Cleaned version of KDD Cup 99 |
| SWaT | Industrial control | 11 days time series | 36 attack scenarios | F1, precision, recall | Secure Water Treatment testbed |
| WADI | Industrial control | 16 days time series | 15 attack scenarios | F1, precision, recall | Water Distribution testbed |
| Yahoo S5 | Time series | 367 time series | Point and pattern anomalies | F1 | Real and synthetic benchmarks |
| NAB | Time series | 58 time series | Various temporal anomalies | NAB score | Numenta Anomaly Benchmark |
| UCSD Ped1/2 | Video surveillance | 70/28 clips | Pedestrian anomalies | Frame-level AUC | Bikes, carts, skaters on walkway |
| ShanghaiTech | Video surveillance | 437 videos | 130 abnormal events | Frame-level AUC | 13 scenes, diverse anomalies |
| VisA | Industrial inspection | 10821 images, 12 categories | Surface and structural | Image/Pixel AUROC | Newer, more challenging than MVTec |

**Evaluation metrics**:
- **AUROC (Area Under ROC Curve)**: Standard metric for binary anomaly classification. Threshold-independent. Image-level for detection, pixel-level for localization.
- **AUPRC (Area Under Precision-Recall Curve)**: Better for highly imbalanced datasets where AUROC can be misleadingly high.
- **F1 Score**: Requires threshold selection. Report best F1 or F1 at a fixed false positive rate.
- **PRO (Per-Region Overlap)**: For localization quality; weights each anomaly region equally regardless of size.

**Common evaluation protocols**:
- **One-vs-rest**: One class is "normal," all others are "anomalous." Average AUROC across all classes.
- **Leave-one-out**: Train on N-1 classes, test on held-out class.
- **Defect detection**: Train on defect-free samples only, test on both normal and defective.

---

## Production Considerations

Deploying deep anomaly detection in production introduces challenges beyond model accuracy.

**Inference speed requirements**:
- Real-time industrial inspection: less than 50ms per image (20+ FPS).
- Network intrusion: less than 1ms per packet/flow for inline detection.
- PatchCore with coreset: approximately 20-100ms per image depending on memory bank size.
- Autoencoder: approximately 1-5ms per sample (single forward pass).
- GAN-based (AnoGAN): seconds per sample due to optimization loop; f-AnoGAN fixes this.

**Memory constraints**:
- PatchCore memory bank: 10-500MB per product category. Coreset sampling reduces this by 10-100x.
- Model size: Autoencoders can be tiny (less than 1MB). Pretrained backbones (ResNet, ViT) range from 25-300MB.
- Edge deployment: Use knowledge distillation, pruning, or quantization. Anomalib supports OpenVINO for optimized inference.

```python
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from collections import deque


@dataclass
class AnomalyDetectionConfig:
    """Production configuration for anomaly detection system."""
    model_path: str = "models/anomaly_detector.pt"
    threshold: float = 0.5
    threshold_percentile: float = 99.0
    max_inference_time_ms: float = 50.0
    batch_size: int = 32
    enable_drift_detection: bool = True
    drift_window_size: int = 1000
    drift_significance: float = 0.05
    alert_cooldown_seconds: float = 60.0
    min_samples_for_threshold: int = 100


class AdaptiveThreshold:
    """Dynamically adjust anomaly threshold based on recent score distribution."""

    def __init__(
        self,
        initial_threshold: float,
        window_size: int = 1000,
        adaptation_rate: float = 0.01,
        min_threshold: float = 0.1,
        max_threshold: float = 10.0,
    ):
        self.threshold = initial_threshold
        self.scores = deque(maxlen=window_size)
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.target_percentile = 99.0

    def update(self, score: float) -> None:
        self.scores.append(score)
        if len(self.scores) >= 100:
            target = np.percentile(list(self.scores), self.target_percentile)
            # Exponential moving average update
            self.threshold = (
                (1 - self.adaptation_rate) * self.threshold
                + self.adaptation_rate * target
            )
            self.threshold = np.clip(
                self.threshold, self.min_threshold, self.max_threshold
            )

    def is_anomaly(self, score: float) -> bool:
        return score > self.threshold


class ProductionAnomalyDetector:
    """Production-ready anomaly detection wrapper with monitoring."""

    def __init__(
        self,
        model: Callable,
        config: AnomalyDetectionConfig,
    ):
        self.model = model
        self.config = config
        self.adaptive_threshold = AdaptiveThreshold(
            initial_threshold=config.threshold,
            window_size=config.drift_window_size,
        )
        self.inference_times: deque = deque(maxlen=1000)
        self.anomaly_count = 0
        self.total_count = 0
        self.last_alert_time = 0.0

    def predict(self, x: np.ndarray) -> dict:
        """Run inference with monitoring and adaptive thresholding."""
        start_time = time.time()

        # Compute anomaly score
        score = self.model(x)

        inference_time_ms = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time_ms)

        # Adaptive threshold update
        self.adaptive_threshold.update(score)
        is_anomaly = self.adaptive_threshold.is_anomaly(score)

        self.total_count += 1
        if is_anomaly:
            self.anomaly_count += 1

        # Check for latency violations
        latency_ok = inference_time_ms <= self.config.max_inference_time_ms

        result = {
            "score": float(score),
            "is_anomaly": is_anomaly,
            "threshold": self.adaptive_threshold.threshold,
            "inference_time_ms": inference_time_ms,
            "latency_ok": latency_ok,
            "anomaly_rate": self.anomaly_count / max(1, self.total_count),
        }

        # Alert logic with cooldown
        if is_anomaly:
            current_time = time.time()
            if (current_time - self.last_alert_time
                    > self.config.alert_cooldown_seconds):
                self.last_alert_time = current_time
                result["alert"] = True

        return result

    def get_health_metrics(self) -> dict:
        """Return system health metrics for monitoring dashboards."""
        times = list(self.inference_times)
        return {
            "total_predictions": self.total_count,
            "total_anomalies": self.anomaly_count,
            "anomaly_rate": self.anomaly_count / max(1, self.total_count),
            "current_threshold": self.adaptive_threshold.threshold,
            "avg_inference_ms": np.mean(times) if times else 0.0,
            "p95_inference_ms": np.percentile(times, 95) if times else 0.0,
            "p99_inference_ms": np.percentile(times, 99) if times else 0.0,
            "latency_violations": sum(
                1 for t in times if t > self.config.max_inference_time_ms
            ),
        }
```

**Continuous learning and model updates**:
- **Concept drift**: Normal behavior evolves over time (seasonal patterns, product changes). Periodically retrain or fine-tune on recent normal data.
- **Feedback loop**: When operators confirm or deny flagged anomalies, incorporate this feedback to refine thresholds and potentially retrain.
- **A/B testing**: Deploy new model versions alongside existing ones, compare detection rates and false positive rates before full rollout.
- **Model versioning**: Track model versions, training data snapshots, and threshold configurations for reproducibility and rollback.

**Multi-modal anomaly detection**:
- Combine sensor time series with images for richer anomaly signals.
- Fuse text logs with numerical metrics for IT operations anomaly detection.
- Use attention-based fusion or late fusion (separate models per modality, combined scoring).

**Operational best practices**:
- Set alerts on anomaly rate spikes, not individual anomalies, to reduce alert fatigue.
- Maintain a "golden set" of known anomalies for regression testing after model updates.
- Log all anomaly scores (not just binary decisions) for post-hoc threshold analysis.
- Monitor model input distributions to detect data pipeline issues before they cause false anomalies.

---

## See Also

- **35_Time_Series_Deep_Learning.md** - Time series forecasting and representation learning, relevant for temporal anomaly detection baselines.
- **34_Graph_Neural_Networks.md** - Foundational GNN architectures used in graph anomaly detection.
- **31_Transfer_Learning.md** - Transfer learning and pretrained features are central to methods like PatchCore and PaDiM.
- **32_Meta_Learning.md** - Few-shot anomaly detection borrows ideas from meta-learning for rapid adaptation to new anomaly types.

---

## Resources

**Foundational Papers**:
- Schlegl et al. (2017). "Unsupervised Anomaly Detection with Generative Adversarial Networks" (AnoGAN).
- Ruff et al. (2018). "Deep One-Class Classification" (Deep SVDD).
- Akcay et al. (2018). "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training."
- Gong et al. (2019). "Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection."
- Roth et al. (2022). "Towards Total Recall in Industrial Anomaly Detection" (PatchCore), CVPR 2022.
- Xu et al. (2022). "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy," ICLR 2022.
- Li et al. (2021). "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization," CVPR 2021.
- Zavrtanik et al. (2021). "DRAEM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection," ICCV 2021.
- Defard et al. (2021). "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization."
- Sultani et al. (2018). "Real-World Anomaly Detection in Surveillance Videos," CVPR 2018.

**Libraries and Frameworks**:
- Anomalib (Intel): https://github.com/openvinotoolkit/anomalib -- Unified image anomaly detection library.
- PyOD: https://github.com/yzhao062/pyod -- Python Outlier Detection library (classical + deep methods).
- ADBench: https://github.com/Minqi824/ADBench -- Anomaly detection benchmark suite.

**Surveys**:
- Pang et al. (2021). "Deep Learning for Anomaly Detection: A Review." ACM Computing Surveys.
- Ruff et al. (2021). "A Unifying Review of Deep and Shallow Anomaly Detection." Proceedings of the IEEE.
- Liu et al. (2024). "Deep Industrial Image Anomaly Detection: A Survey." Machine Intelligence Research.

**Datasets**:
- MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- SWaT/WADI: iTrust, Singapore University of Technology and Design.
- Numenta Anomaly Benchmark (NAB): https://github.com/numenta/NAB
