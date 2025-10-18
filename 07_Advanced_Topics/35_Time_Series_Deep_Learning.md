# Time Series Deep Learning - N-BEATS, NeuralProphet, TFT & Transformers

## Overview

**Deep learning revolutionizes time series forecasting** with neural architectures that automatically learn temporal patterns without manual feature engineering.

**Key Architectures (2024-2025):**
- N-BEATS (interpretable, pure DL)
- NeuralProphet (hybrid classical + DL)
- Temporal Fusion Transformer (multi-horizon with attention)
- Informer/Autoformer (long-sequence transformers)
- iTransformer (2024 SOTA)

---

## N-BEATS: Neural Basis Expansion Analysis

### Architecture Overview

**AAAI 2021 breakthrough:** Pure deep learning without RNNs, no domain knowledge needed.

**Key Innovation:** Stack of fully-connected blocks with forward/backward connections.

```python
import numpy as np
import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    """Single N-BEATS block with basis expansion"""

    def __init__(self, input_size, theta_size, basis_function, layers=4, layer_size=256):
        super().__init__()

        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function

        # Fully connected stack
        self.fc_stack = nn.ModuleList([
            nn.Linear(input_size if i == 0 else layer_size, layer_size)
            for i in range(layers)
        ])

        # Basis expansion parameters
        self.theta_b = nn.Linear(layer_size, theta_size)  # Backcast
        self.theta_f = nn.Linear(layer_size, theta_size)  # Forecast

    def forward(self, x):
        # Pass through FC stack
        h = x
        for fc in self.fc_stack:
            h = torch.relu(fc(h))

        # Generate theta parameters
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)

        # Basis expansion
        backcast = self.basis_function(theta_b, self.input_size)
        forecast = self.basis_function(theta_f, self.input_size)

        return backcast, forecast


class GenericBasis(nn.Module):
    """Generic basis function (non-interpretable, high performance)"""

    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_fc = nn.Linear(backcast_size, backcast_size)
        self.forecast_fc = nn.Linear(forecast_size, forecast_size)

    def forward(self, theta, target_size):
        return self.backcast_fc(theta) if target_size == self.backcast_fc.in_features else self.forecast_fc(theta)


class TrendBasis(nn.Module):
    """Trend basis function (interpretable)"""

    def __init__(self, degree_of_polynomial, backcast_size, forecast_size):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta, target_size):
        # Polynomial basis: [1, t, t^2, ..., t^p]
        t = torch.arange(0, target_size, dtype=torch.float32) / target_size
        T = torch.stack([t ** i for i in range(self.polynomial_size)], dim=1)

        # theta @ T^T
        return torch.matmul(theta, T.T)


class SeasonalityBasis(nn.Module):
    """Seasonality basis function (interpretable)"""

    def __init__(self, harmonics, backcast_size, forecast_size):
        super().__init__()
        self.harmonics = harmonics
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta, target_size):
        # Fourier basis: [sin(2πt), cos(2πt), sin(4πt), ...]
        t = torch.arange(0, target_size, dtype=torch.float32) / target_size

        S = []
        for i in range(1, self.harmonics + 1):
            S.append(torch.sin(2 * np.pi * i * t))
            S.append(torch.cos(2 * np.pi * i * t))

        S = torch.stack(S, dim=1)
        return torch.matmul(theta, S.T)


class NBeatsNet(nn.Module):
    """Complete N-BEATS model"""

    def __init__(self, backcast_size, forecast_size, stacks):
        super().__init__()

        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.stacks = nn.ModuleList(stacks)

    def forward(self, x):
        # Initialize
        forecast = torch.zeros(x.size(0), self.forecast_size)

        for stack in self.stacks:
            # Each block in stack
            for block in stack:
                backcast, block_forecast = block(x)

                # Residual connection
                x = x - backcast

                # Accumulate forecast
                forecast = forecast + block_forecast

        return forecast


# Build interpretable N-BEATS
def build_interpretable_nbeats(backcast_size=24, forecast_size=12):
    """N-BEATS with interpretable decomposition (trend + seasonality)"""

    # Trend stack
    trend_stack = [
        NBeatsBlock(
            input_size=backcast_size,
            theta_size=3,  # Polynomial degree = 2
            basis_function=TrendBasis(degree_of_polynomial=2,
                                      backcast_size=backcast_size,
                                      forecast_size=forecast_size),
            layers=4,
            layer_size=256
        )
        for _ in range(3)  # 3 blocks
    ]

    # Seasonality stack
    seasonality_stack = [
        NBeatsBlock(
            input_size=backcast_size,
            theta_size=10,  # 5 harmonics = 10 coefficients
            basis_function=SeasonalityBasis(harmonics=5,
                                            backcast_size=backcast_size,
                                            forecast_size=forecast_size),
            layers=4,
            layer_size=2048
        )
        for _ in range(3)
    ]

    model = NBeatsNet(
        backcast_size=backcast_size,
        forecast_size=forecast_size,
        stacks=[trend_stack, seasonality_stack]
    )

    return model


# Training
def train_nbeats(model, train_loader, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward
            forecast = model(batch_x)
            loss = criterion(forecast, batch_y)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

    return model


# Usage
model = build_interpretable_nbeats(backcast_size=24, forecast_size=12)

# Train
trained_model = train_nbeats(model, train_loader, epochs=100)

# Inference
with torch.no_grad():
    forecast = trained_model(historical_window)
```

**Performance:** Outperforms statistical methods (ARIMA, ETS) and matches complex RNNs with simpler architecture.

---

## NeuralProphet: Explainable Forecasting at Scale

### Hybrid Classical + Deep Learning

**Successor to Facebook Prophet:** Combines decomposition with neural networks.

```python
from neuralprophet import NeuralProphet
import pandas as pd

# Data preparation (requires 'ds' and 'y' columns)
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=1000, freq='D'),
    'y': np.sin(np.arange(1000) * 2 * np.pi / 365) + np.random.randn(1000) * 0.1
})

# Initialize model
model = NeuralProphet(
    growth='linear',              # 'linear' or 'discontinuous'
    changepoints_range=0.9,       # Detect trend changes in first 90%
    n_changepoints=10,            # Number of changepoints

    # Seasonality
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive',  # 'additive' or 'multiplicative'

    # Auto-regression (key for accuracy!)
    n_lags=30,                    # Use 30 historical values
    ar_layers=[64, 32],           # Neural network for AR

    # Forecasting
    n_forecasts=7,                # Multi-step ahead

    # Training
    learning_rate=0.1,
    epochs=100,
    batch_size=64
)

# Fit
metrics = model.fit(df, freq='D')

# Forecast
future = model.make_future_dataframe(df, periods=30)
forecast = model.predict(future)

# Visualize components
fig_components = model.plot_components(forecast)

# Plot forecast
fig_forecast = model.plot(forecast)
```

### Adding External Regressors

```python
# Future regressors (known in advance)
df['promo'] = (df['ds'].dt.dayofweek >= 5).astype(int)  # Weekend promos

model = NeuralProphet(n_lags=14)
model.add_future_regressor('promo')

metrics = model.fit(df, freq='D')

# For forecasting, provide future values
future = model.make_future_dataframe(df, periods=30)
future['promo'] = (future['ds'].dt.dayofweek >= 5).astype(int)

forecast = model.predict(future)
```

### Lagged Regressors (Past Events)

```python
# Events that affect future (e.g., website visits → sales)
df['website_visits'] = np.random.poisson(100, size=len(df))

model = NeuralProphet(n_lags=7)
model.add_lagged_regressor(
    name='website_visits',
    n_lags=7,              # Use 7 days of history
    regularization=0.1
)

metrics = model.fit(df, freq='D')
forecast = model.predict(df)
```

**Best Use Cases:**
- Business forecasting (sales, demand)
- Higher-frequency data (hourly, daily)
- When interpretability is crucial
- Domain with known seasonality

---

## Temporal Fusion Transformer (TFT)

### Multi-Horizon Forecasting with Attention

**Google Research 2019, Nature MI 2023:** Interpretable multi-step forecasting.

```python
import torch
import torch.nn as nn

class VariableSelectionNetwork(nn.Module):
    """Select most relevant features per timestep"""

    def __init__(self, input_size, hidden_size, num_features):
        super().__init__()

        # Variable selection weights
        self.grn = GatedResidualNetwork(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch, time, features]

        # Compute selection weights
        flattened = x.view(x.size(0), -1)
        weights = self.softmax(self.grn(flattened))

        # Weight features
        weights = weights.unsqueeze(1)
        selected = x * weights

        return selected, weights


class GatedResidualNetwork(nn.Module):
    """Gated skip connection for gradient flow"""

    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1):
        super().__init__()

        if output_size is None:
            output_size = input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        # Gating mechanism
        self.gate = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        # Skip connection
        if input_size != output_size:
            self.skip = nn.Linear(input_size, output_size)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        # Main path
        h = self.elu(self.fc1(x))
        h = self.fc2(h)
        h = self.dropout(h)

        # Gating
        gate = self.sigmoid(self.gate(self.elu(self.fc1(x))))

        # Gated residual
        output = gate * h + (1 - gate) * self.skip(x)

        return output


class TemporalFusionTransformer(nn.Module):
    """TFT for multi-horizon forecasting"""

    def __init__(self,
                 static_size,           # Static features (e.g., product ID)
                 historical_size,       # Historical inputs
                 future_known_size,     # Future known (e.g., day of week)
                 hidden_size=128,
                 num_heads=4,
                 forecast_horizon=24):
        super().__init__()

        self.forecast_horizon = forecast_horizon

        # Variable selection
        self.static_selection = VariableSelectionNetwork(static_size, hidden_size, static_size)
        self.historical_selection = VariableSelectionNetwork(historical_size, hidden_size, historical_size)
        self.future_selection = VariableSelectionNetwork(future_known_size, hidden_size, future_known_size)

        # LSTM for temporal processing
        self.lstm_encoder = nn.LSTM(
            input_size=historical_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.lstm_decoder = nn.LSTM(
            input_size=future_known_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

        # Output
        self.output_layer = GatedResidualNetwork(hidden_size, hidden_size, forecast_horizon)

    def forward(self, static_features, historical_inputs, future_known_inputs):
        batch_size = historical_inputs.size(0)

        # Variable selection
        static_selected, static_weights = self.static_selection(static_features)
        historical_selected, hist_weights = self.historical_selection(historical_inputs)
        future_selected, future_weights = self.future_selection(future_known_inputs)

        # Encode historical
        lstm_out_enc, (h, c) = self.lstm_encoder(historical_selected)

        # Decode future
        lstm_out_dec, _ = self.lstm_decoder(future_selected, (h, c))

        # Attention over encoded states
        attn_out, attn_weights = self.attention(
            query=lstm_out_dec,
            key=lstm_out_enc,
            value=lstm_out_enc
        )

        # Output projection
        forecast = self.output_layer(attn_out)

        return forecast, {
            'static_weights': static_weights,
            'historical_weights': hist_weights,
            'future_weights': future_weights,
            'attention_weights': attn_weights
        }


# Usage with PyTorch Forecasting library (easier)
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# Create dataset
training = TimeSeriesDataSet(
    data=df,
    time_idx="time_idx",
    target="sales",
    group_ids=["product_id"],
    max_encoder_length=30,
    max_prediction_length=7,

    static_categoricals=["product_id", "category"],
    static_reals=["avg_price"],

    time_varying_known_categoricals=["day_of_week", "month"],
    time_varying_known_reals=["promotion"],

    time_varying_unknown_reals=["sales"],

    target_normalizer=GroupNormalizer(groups=["product_id"])
)

# Train
model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=128,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    loss=QuantileLoss()
)

trainer = pl.Trainer(max_epochs=30, gpus=1)
trainer.fit(model, train_dataloaders=train_dataloader)

# Interpret
interpretation = model.interpret_output(predictions, reduction="sum")
model.plot_interpretation(interpretation)
```

**Key Features:**
- Multi-horizon forecasting (predict next N steps)
- Variable importance (which features matter?)
- Temporal attention (which timesteps are important?)
- Quantile regression (prediction intervals)

**Recent Success (2024):**
- VN1 Forecasting Competition: 4th place overall
- Medical: Blood pressure prediction 7 min ahead
- Energy: Smart grid optimization

---

## Transformer-Based Models

### Informer (AAAI 2021 Best Paper)

**Problem:** Standard Transformers are O(L²) for sequence length L.

**Solution:** ProbSparse self-attention → O(L log L)

```python
class ProbSparseSelfAttention(nn.Module):
    """Efficient attention for long sequences"""

    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        B, L, D = queries.shape

        # Project
        Q = self.query_projection(queries)
        K = self.key_projection(keys)
        V = self.value_projection(values)

        # Sample top-k queries (ProbSparse trick)
        U = self.factor * int(np.log(L))  # Sample size

        # Compute sparsity measurement
        Q_K = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(D)
        M = Q_K.max(dim=-1)[0] - Q_K.mean(dim=-1)

        # Top-k selection
        _, top_indices = M.topk(U, dim=-1)

        # Compute attention only for top queries
        Q_reduced = Q.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, D))
        scores = torch.matmul(Q_reduced, K.transpose(-2, -1)) / np.sqrt(D)
        attn = torch.softmax(scores, dim=-1)

        output = torch.matmul(attn, V)

        return output
```

### Autoformer (NeurIPS 2021)

**Innovation:** Auto-correlation mechanism (better than attention for series).

```python
class AutoCorrelation(nn.Module):
    """Auto-correlation for periodic patterns"""

    def __init__(self, factor=1):
        super().__init__()
        self.factor = factor

    def time_delay_agg(self, values, corr):
        """Aggregate values based on correlation delays"""

        # Find top-k delays
        _, top_k = torch.topk(corr, self.factor, dim=-1)

        # Aggregate
        batch = values.shape[0]
        channel = values.shape[2]

        # Roll and aggregate
        aggregated = torch.zeros_like(values)
        for i in range(top_k.shape[1]):
            delay = top_k[:, i]
            pattern = torch.roll(values, shifts=delay.item(), dims=1)
            aggregated += pattern

        return aggregated / top_k.shape[1]

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape

        # Compute auto-correlation via FFT
        q_fft = torch.fft.rfft(queries, dim=1)
        k_fft = torch.fft.rfft(keys, dim=1)

        # Auto-correlation in frequency domain
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=1)

        # Time delay aggregation
        V = self.time_delay_agg(values, corr)

        return V
```

**Performance:** 38% improvement over previous SOTA on 6 benchmarks.

### iTransformer (2024 SOTA)

**Paradigm shift:** Attend over **variables** instead of time.

```python
class iTransformer(nn.Module):
    """Inverted Transformer - attend over variables"""

    def __init__(self, n_vars, seq_len, d_model=512, n_heads=8, n_layers=3):
        super().__init__()

        # Embed each variable's time series
        self.var_embeddings = nn.ModuleList([
            nn.Linear(seq_len, d_model) for _ in range(n_vars)
        ])

        # Transformer over variables (not time!)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Project back to forecast
        self.forecast_heads = nn.ModuleList([
            nn.Linear(d_model, seq_len) for _ in range(n_vars)
        ])

    def forward(self, x):
        # x: [batch, time, variables]
        B, T, V = x.shape

        # Transpose to [batch, variables, time]
        x = x.transpose(1, 2)

        # Embed each variable
        embeddings = []
        for i in range(V):
            emb = self.var_embeddings[i](x[:, i, :])  # [batch, d_model]
            embeddings.append(emb)

        embeddings = torch.stack(embeddings, dim=1)  # [batch, vars, d_model]

        # Transformer over variables
        transformed = self.transformer(embeddings)

        # Forecast each variable
        forecasts = []
        for i in range(V):
            forecast = self.forecast_heads[i](transformed[:, i, :])
            forecasts.append(forecast)

        forecasts = torch.stack(forecasts, dim=2)  # [batch, time, vars]

        return forecasts
```

**Why it works:** Captures inter-variable dependencies better than time dependencies.

---

## Best Practices

### 1. Model Selection Guide

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **N-BEATS** | Pure forecasting, no features | Interpretable, no domain knowledge | Univariate only |
| **NeuralProphet** | Business metrics, seasonality | Easy to use, interpretable | Limited complexity |
| **TFT** | Multi-horizon, many features | Handles covariates, quantiles | Complex setup |
| **Informer** | Long sequences (>1000) | Efficient, scales well | Less interpretable |
| **Autoformer** | Periodic data | Auto-correlation works well | Needs clear periodicity |
| **iTransformer** | Multivariate dependencies | Captures variable relationships | Newer, less tested |

### 2. Training Tips

```python
# Cross-validation for time series
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"Validation RMSE: {score}")
```

### 3. Ensembling

```python
class ForecastEnsemble:
    """Combine multiple models"""

    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)

    def predict(self, X):
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)

        return ensemble_pred

# Usage
ensemble = ForecastEnsemble([
    nbeats_model,
    neuralprophet_model,
    tft_model
], weights=[0.4, 0.3, 0.3])

forecast = ensemble.predict(test_data)
```

### 4. Evaluation Metrics

```python
def evaluate_forecast(y_true, y_pred):
    """Comprehensive evaluation"""

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Directional accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }
```

---

## Production Deployment

```python
class TimeSeriesPipeline:
    """Production-ready forecasting pipeline"""

    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler or StandardScaler()

    def preprocess(self, data):
        """Handle missing values, scaling"""

        # Fill missing
        data = data.fillna(method='ffill').fillna(method='bfill')

        # Scale
        scaled = self.scaler.fit_transform(data)

        return scaled

    def forecast(self, historical_data, horizon=24):
        """Generate forecast with confidence intervals"""

        # Preprocess
        X = self.preprocess(historical_data)

        # Forecast
        if hasattr(self.model, 'predict_quantiles'):
            # Quantile predictions (TFT)
            forecast = self.model.predict_quantiles(X, quantiles=[0.1, 0.5, 0.9])

            return {
                'forecast': self.scaler.inverse_transform(forecast['0.5']),
                'lower_bound': self.scaler.inverse_transform(forecast['0.1']),
                'upper_bound': self.scaler.inverse_transform(forecast['0.9'])
            }
        else:
            # Point predictions
            forecast = self.model.predict(X)

            return {
                'forecast': self.scaler.inverse_transform(forecast)
            }

    def monitor_drift(self, recent_data):
        """Detect distribution shift"""

        from scipy.stats import ks_2samp

        # Compare recent vs training distribution
        stat, pvalue = ks_2samp(self.training_data, recent_data)

        if pvalue < 0.05:
            print("⚠️ Distribution drift detected! Consider retraining.")

        return {'drift_detected': pvalue < 0.05, 'p_value': pvalue}
```

---

## Key Takeaways

1. **N-BEATS** - Pure DL, interpretable, no feature engineering
2. **NeuralProphet** - Hybrid approach, easy to use, good for business metrics
3. **TFT** - Best for multi-horizon with many covariates, interpretable attention
4. **Transformers (Informer/Autoformer)** - Efficient for long sequences (>1000 steps)
5. **iTransformer** - 2024 SOTA, attends over variables instead of time
6. **Always cross-validate** with time series splits (not random!)
7. **Ensemble models** for robustness
8. **Monitor drift** in production

**Next:** See `12_Cutting_Edge_2025/` for AutoML and Neural Architecture Search
