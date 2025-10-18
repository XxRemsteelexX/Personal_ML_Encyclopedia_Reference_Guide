# Model Deployment (2025 State-of-the-Art)

## Table of Contents
1. [Introduction](#introduction)
2. [Deployment Strategies](#deployment-strategies)
3. [Model Serving Frameworks](#model-serving-frameworks)
4. [Containerization](#containerization)
5. [CI/CD Pipelines for ML](#cicd-pipelines-for-ml)
6. [Advanced Deployment Patterns](#advanced-deployment-patterns)
7. [Model Versioning and Registry](#model-versioning-and-registry)
8. [API Design for ML Models](#api-design-for-ml-models)
9. [Production Examples](#production-examples)
10. [Best Practices](#best-practices)

---

## Introduction

Model deployment is the process of integrating a trained machine learning model into a production environment where it can make predictions on new data. According to 2025 industry research, comprehensive MLOps practices enable **60% faster deployment** compared to ad-hoc approaches.

### Why Deployment is Critical

- **Business Value:** Models provide no value until deployed
- **Iteration Speed:** Fast deployment enables rapid experimentation
- **Reliability:** Proper deployment reduces production incidents by 40%
- **Scalability:** Production systems must handle varying loads

### Key Challenges

1. **Model-Code Gap:** Research code rarely production-ready
2. **Dependencies:** Managing libraries, versions, hardware requirements
3. **Scalability:** Handling variable traffic patterns
4. **Monitoring:** Detecting degradation and drift
5. **Latency:** Meeting real-time requirements
6. **Cost:** Balancing performance with infrastructure costs

### 2025 Landscape

- **Edge AI:** On-device deployment for privacy and low latency
- **Serverless ML:** Auto-scaling, pay-per-use inference
- **Multi-Cloud:** Avoiding vendor lock-in
- **Green AI:** Carbon-aware deployment strategies
- **Regulatory Compliance:** EU AI Act, model cards, transparency

---

## Deployment Strategies

### 1. Batch Inference

**When to Use:**
- Predictions needed periodically (daily, hourly)
- Large volumes of data
- Latency not critical (minutes to hours acceptable)
- Cost-sensitive applications

**Advantages:**
- Efficient resource utilization
- Simple implementation
- Easy debugging
- Lower cost per prediction

**Disadvantages:**
- Not real-time
- Stale predictions between runs
- Scheduling complexity

**Production Example:**

```python
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

class BatchInferenceEngine:
    """Production batch inference engine with monitoring and error handling."""

    def __init__(self, model_path: str, output_dir: str, batch_size: int = 512):
        self.model = torch.jit.load(model_path)  # TorchScript for production
        self.model.eval()
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def preprocess(self, df: pd.DataFrame) -> torch.Tensor:
        """Preprocess data for model input."""
        # Example: normalize features
        features = df[['feature1', 'feature2', 'feature3']].values
        return torch.FloatTensor(features)

    def predict_batch(self, data_path: str) -> pd.DataFrame:
        """Run batch prediction on entire dataset."""
        start_time = datetime.now()
        self.logger.info(f"Starting batch inference from {data_path}")

        # Load data
        df = pd.read_parquet(data_path)
        self.logger.info(f"Loaded {len(df)} records")

        # Prepare tensors
        X = self.preprocess(df)

        # Batch prediction
        predictions = []
        total_batches = (len(X) + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i + self.batch_size].to(self.device)
                batch_pred = self.model(batch).cpu().numpy()
                predictions.extend(batch_pred)

                if (i // self.batch_size) % 10 == 0:
                    self.logger.info(
                        f"Processed {i // self.batch_size}/{total_batches} batches"
                    )

        # Create output dataframe
        df['prediction'] = predictions
        df['prediction_timestamp'] = datetime.now()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"predictions_{timestamp}.parquet"
        df.to_parquet(output_path)

        duration = (datetime.now() - start_time).total_seconds()
        throughput = len(df) / duration

        self.logger.info(
            f"Batch inference complete. "
            f"Records: {len(df)}, Duration: {duration:.2f}s, "
            f"Throughput: {throughput:.0f} records/sec"
        )

        return df

# Usage with scheduling (Apache Airflow)
if __name__ == "__main__":
    engine = BatchInferenceEngine(
        model_path="/models/production/model_v3.pt",
        output_dir="/data/predictions",
        batch_size=1024
    )

    results = engine.predict_batch("/data/input/daily_data.parquet")
```

### 2. Real-Time Inference (REST API)

**When to Use:**
- Immediate predictions required (< 100ms)
- User-facing applications
- Interactive systems
- Variable request patterns

**Production Example with FastAPI:**

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, validator
import torch
import numpy as np
from typing import List, Optional
import time
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import uvicorn

# Prometheus metrics
REQUEST_COUNT = Counter(
    'model_requests_total',
    'Total prediction requests',
    ['model_version', 'status']
)
REQUEST_LATENCY = Histogram(
    'model_request_latency_seconds',
    'Request latency in seconds',
    ['model_version']
)

app = FastAPI(title="ML Model API", version="2.0")

# Input/Output schemas
class PredictionRequest(BaseModel):
    features: List[float]
    model_version: Optional[str] = "latest"

    @validator('features')
    def validate_features(cls, v):
        if len(v) != 10:  # Expected feature count
            raise ValueError('Expected 10 features')
        if any(np.isnan(val) or np.isinf(val) for val in v):
            raise ValueError('Features contain NaN or Inf')
        return v

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    latency_ms: float

class ModelServer:
    """Production model server with caching and monitoring."""

    def __init__(self, model_path: str):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.version = "v3.2.1"

        # Warm up model (important for GPU)
        self._warmup()

    def _warmup(self):
        """Warm up model with dummy data."""
        dummy_input = torch.randn(1, 10).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)

    def predict(self, features: List[float]) -> tuple:
        """Make prediction with confidence estimate."""
        start_time = time.time()

        # Prepare input
        x = torch.FloatTensor([features]).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(x)
            prediction = output[0].item()
            # If model outputs logits, convert to probability
            confidence = torch.sigmoid(output[0]).item()

        latency = (time.time() - start_time) * 1000  # ms

        return prediction, confidence, latency

# Global model instance
model_server = ModelServer("/models/production/model_v3.pt")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prediction endpoint with monitoring."""
    try:
        prediction, confidence, latency = model_server.predict(request.features)

        # Update metrics
        REQUEST_COUNT.labels(
            model_version=model_server.version,
            status="success"
        ).inc()
        REQUEST_LATENCY.labels(
            model_version=model_server.version
        ).observe(latency / 1000)

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version=model_server.version,
            latency_ms=latency
        )

    except Exception as e:
        REQUEST_COUNT.labels(
            model_version=model_server.version,
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes."""
    return {"status": "healthy", "version": model_server.version}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4  # Multiple workers for concurrency
    )
```

### 3. Streaming Inference

**When to Use:**
- Continuous data streams (IoT, logs, events)
- Low-latency processing of sequential data
- Event-driven architectures

**Production Example with Kafka:**

```python
from kafka import KafkaConsumer, KafkaProducer
import json
import torch
from typing import Dict
import logging

class StreamingInferenceService:
    """Real-time streaming inference with Kafka."""

    def __init__(
        self,
        model_path: str,
        input_topic: str = "input_features",
        output_topic: str = "predictions",
        bootstrap_servers: str = "localhost:9092"
    ):
        # Load model
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Kafka consumer
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='ml_inference_group'
        )

        # Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        self.output_topic = output_topic
        self.logger = logging.getLogger(__name__)

        # Batch processing for efficiency
        self.batch_size = 32
        self.batch_timeout = 0.1  # seconds
        self.batch = []
        self.last_batch_time = time.time()

    def process_message(self, message: Dict) -> Dict:
        """Process single message."""
        features = torch.FloatTensor(message['features']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(features).cpu().item()

        return {
            'id': message['id'],
            'prediction': prediction,
            'timestamp': time.time()
        }

    def process_batch(self, messages: list) -> list:
        """Process batch of messages for efficiency."""
        if not messages:
            return []

        # Stack features
        features = torch.stack([
            torch.FloatTensor(msg['features']) for msg in messages
        ]).to(self.device)

        # Batch inference
        with torch.no_grad():
            predictions = self.model(features).cpu().numpy()

        # Prepare outputs
        results = []
        for msg, pred in zip(messages, predictions):
            results.append({
                'id': msg['id'],
                'prediction': float(pred),
                'timestamp': time.time()
            })

        return results

    def run(self):
        """Main processing loop."""
        self.logger.info("Starting streaming inference service")

        try:
            for message in self.consumer:
                data = message.value
                self.batch.append(data)

                # Process batch if full or timeout
                current_time = time.time()
                should_process = (
                    len(self.batch) >= self.batch_size or
                    (current_time - self.last_batch_time) >= self.batch_timeout
                )

                if should_process and self.batch:
                    # Process batch
                    results = self.process_batch(self.batch)

                    # Send predictions
                    for result in results:
                        self.producer.send(self.output_topic, value=result)

                    self.logger.info(f"Processed batch of {len(self.batch)} messages")

                    # Reset batch
                    self.batch = []
                    self.last_batch_time = current_time

        except KeyboardInterrupt:
            self.logger.info("Shutting down streaming service")
        finally:
            self.consumer.close()
            self.producer.close()

# Usage
if __name__ == "__main__":
    service = StreamingInferenceService(
        model_path="/models/production/model_v3.pt",
        bootstrap_servers="kafka:9092"
    )
    service.run()
```

### 4. Edge Deployment

**When to Use:**
- Privacy-sensitive applications
- Low-latency requirements (< 10ms)
- Offline operation needed
- Bandwidth constraints

**Model Optimization for Edge:**

```python
import torch
from torch.quantization import quantize_dynamic
import onnx
import onnxruntime as ort

class EdgeModelOptimizer:
    """Optimize models for edge deployment."""

    @staticmethod
    def quantize_model(model_path: str, output_path: str) -> None:
        """Dynamic quantization for CPU inference."""
        model = torch.jit.load(model_path)

        # Quantize to INT8
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )

        # Save quantized model
        torch.jit.save(quantized_model, output_path)

        # Check size reduction
        original_size = Path(model_path).stat().st_size / (1024 * 1024)
        quantized_size = Path(output_path).stat().st_size / (1024 * 1024)

        print(f"Original: {original_size:.2f} MB")
        print(f"Quantized: {quantized_size:.2f} MB")
        print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")

    @staticmethod
    def export_to_onnx(
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        output_path: str
    ) -> None:
        """Export PyTorch model to ONNX for cross-platform inference."""
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,  # Optimization
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model exported to {output_path}")

    @staticmethod
    def benchmark_onnx(model_path: str, dummy_input: np.ndarray) -> None:
        """Benchmark ONNX model inference speed."""
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )

        # Warmup
        for _ in range(10):
            _ = session.run(None, {'input': dummy_input})

        # Benchmark
        import time
        iterations = 1000
        start = time.time()
        for _ in range(iterations):
            _ = session.run(None, {'input': dummy_input})

        duration = time.time() - start
        latency = (duration / iterations) * 1000

        print(f"Average latency: {latency:.2f} ms")
        print(f"Throughput: {iterations/duration:.0f} inferences/sec")

# Usage
optimizer = EdgeModelOptimizer()

# Quantize for mobile
optimizer.quantize_model(
    "/models/model.pt",
    "/models/model_quantized.pt"
)

# Export to ONNX for cross-platform
model = torch.load("/models/model.pt")
dummy_input = torch.randn(1, 10)
optimizer.export_to_onnx(model, dummy_input, "/models/model.onnx")
```

---

## Model Serving Frameworks

### 1. TorchServe (PyTorch)

**Best For:** PyTorch models, high throughput, AWS integration

```python
# model_handler.py - Custom handler for TorchServe
from ts.torch_handler.base_handler import BaseHandler
import torch
import logging

class CustomModelHandler(BaseHandler):
    """Custom handler with preprocessing and postprocessing."""

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        """Initialize model and preprocessing."""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load model
        self.model = torch.jit.load(f"{model_dir}/model.pt")
        self.model.eval()

        # GPU if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.initialized = True
        logging.info("Model initialized successfully")

    def preprocess(self, data):
        """Preprocess input data."""
        # data is list of requests
        inputs = []
        for row in data:
            input_data = row.get("data") or row.get("body")
            # Convert to tensor
            tensor = torch.FloatTensor(input_data)
            inputs.append(tensor)

        return torch.stack(inputs).to(self.device)

    def inference(self, data):
        """Run inference."""
        with torch.no_grad():
            predictions = self.model(data)
        return predictions

    def postprocess(self, data):
        """Postprocess predictions."""
        return data.cpu().numpy().tolist()
```

**TorchServe Configuration:**

```yaml
# config.properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=32
job_queue_size=1000
model_store=/models
load_models=all
```

**Docker Deployment:**

```dockerfile
# Dockerfile.torchserve
FROM pytorch/torchserve:latest-gpu

# Copy model archive
COPY model-store /home/model-server/model-store

# Copy config
COPY config.properties /home/model-server/config.properties

# Expose ports
EXPOSE 8080 8081 8082

CMD ["torchserve", \
     "--start", \
     "--model-store=/home/model-server/model-store", \
     "--models=my_model=my_model.mar"]
```

### 2. TensorFlow Serving

**Best For:** TensorFlow models, gRPC performance

```python
import tensorflow as tf
import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import numpy as np

class TFServingClient:
    """Client for TensorFlow Serving."""

    def __init__(self, host: str = "localhost", port: int = 8500):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)

    def predict(self, model_name: str, data: np.ndarray, version: int = None):
        """Send prediction request."""
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name

        if version:
            request.model_spec.version.value = version

        # Convert data to tensor proto
        request.inputs['input'].CopyFrom(
            tf.make_tensor_proto(data, dtype=tf.float32)
        )

        # Call predict
        result = self.stub.Predict(request, timeout=10.0)

        # Extract output
        output = tf.make_ndarray(result.outputs['output'])
        return output

    def close(self):
        """Close gRPC channel."""
        self.channel.close()

# Usage
client = TFServingClient(host="tf-serving", port=8500)
data = np.random.randn(1, 10).astype(np.float32)
prediction = client.predict("my_model", data, version=3)
print(f"Prediction: {prediction}")
```

### 3. BentoML (Multi-Framework)

**Best For:** Python-first, multi-framework, microservices

```python
import bentoml
from bentoml.io import NumpyNdarray, JSON
import numpy as np

# Save model to BentoML
model = torch.load("model.pt")
bentoml.pytorch.save_model("fraud_detection", model)

# Create service
@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 10},
)
class FraudDetectionService:
    """BentoML service for fraud detection."""

    # Load model on service initialization
    model_ref = bentoml.pytorch.get("fraud_detection:latest")

    def __init__(self):
        self.model = self.model_ref.load_model()
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @bentoml.api
    def predict(self, input_data: NumpyNdarray) -> JSON:
        """Prediction API."""
        # Convert to tensor
        tensor = torch.FloatTensor(input_data).to(self.device)

        # Predict
        with torch.no_grad():
            prediction = self.model(tensor).cpu().numpy()

        return {
            "prediction": prediction.tolist(),
            "is_fraud": bool(prediction[0] > 0.5)
        }

    @bentoml.api
    def predict_batch(self, input_data: NumpyNdarray) -> JSON:
        """Batch prediction API."""
        tensor = torch.FloatTensor(input_data).to(self.device)

        with torch.no_grad():
            predictions = self.model(tensor).cpu().numpy()

        return {
            "predictions": predictions.tolist(),
            "count": len(predictions)
        }
```

**BentoML Deployment:**

```bash
# Build container
bentoml containerize fraud_detection:latest -t fraud-detection:v1

# Deploy to Kubernetes
bentoml deploy fraud_detection:latest \
    --platform kubernetes \
    --replicas 3 \
    --gpu 1
```

---

## Containerization

### Docker Best Practices

```dockerfile
# Multi-stage build for smaller images
FROM python:3.11-slim AS builder

# Install dependencies in builder stage
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 mluser

# Copy dependencies from builder
COPY --from=builder /root/.local /home/mluser/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=mluser:mluser . .

# Switch to non-root user
USER mluser

# Add local bin to PATH
ENV PATH=/home/mluser/.local/bin:$PATH

# Health check
HEALTH CHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  labels:
    app: ml-model
    version: v3
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
        version: v3
    spec:
      containers:
      - name: ml-model
        image: myregistry/ml-model:v3.2.1
        ports:
        - containerPort: 8000
          name: http

        # Resource limits
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1

        # Environment variables
        env:
        - name: MODEL_VERSION
          value: "v3.2.1"
        - name: LOG_LEVEL
          value: "INFO"

        # Liveness probe
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        # Readiness probe
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3

        # Volume mounts
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true

      # Volumes
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  type: LoadBalancer
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
    name: http

---
# hpa.yaml - Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## CI/CD Pipelines for ML

### GitHub Actions Example

```yaml
# .github/workflows/ml-deploy.yml
name: ML Model CI/CD

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'src/**'
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ml-model

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run unit tests
      run: |
        pytest tests/ --cov=src --cov-report=xml

    - name: Model validation tests
      run: |
        python -m pytest tests/test_model_performance.py

    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    permissions:
      contents: read
      packages: write

    steps:
    - uses: actions/checkout@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=sha,prefix=,format=short
          type=raw,value=latest

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging

    steps:
    - name: Deploy to staging
      run: |
        # Update Kubernetes deployment
        kubectl set image deployment/ml-model-staging \
          ml-model=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

        # Wait for rollout
        kubectl rollout status deployment/ml-model-staging

    - name: Run smoke tests
      run: |
        python tests/smoke_tests.py --env=staging

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Canary deployment
      run: |
        # Deploy to 10% of traffic first
        kubectl apply -f k8s/canary-deployment.yaml

        # Monitor for 10 minutes
        python scripts/monitor_canary.py --duration=600

    - name: Full deployment
      run: |
        # If canary succeeds, deploy to all
        kubectl set image deployment/ml-model-production \
          ml-model=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

        kubectl rollout status deployment/ml-model-production
```

---

## Advanced Deployment Patterns

### 1. A/B Testing

```python
import random
from fastapi import FastAPI, Header
from typing import Optional

app = FastAPI()

class ABTestingRouter:
    """Route requests to different model versions for A/B testing."""

    def __init__(self, model_a_path: str, model_b_path: str, traffic_split: float = 0.5):
        self.model_a = torch.jit.load(model_a_path)
        self.model_b = torch.jit.load(model_b_path)
        self.traffic_split = traffic_split  # Percentage to model B

        # Metrics tracking
        self.metrics = {
            'model_a': {'requests': 0, 'latency': []},
            'model_b': {'requests': 0, 'latency': []}
        }

    def route_request(self, user_id: Optional[str] = None) -> str:
        """Deterministic routing based on user_id, or random."""
        if user_id:
            # Consistent hashing for same user
            hash_val = hash(user_id) % 100
            return 'model_b' if hash_val < (self.traffic_split * 100) else 'model_a'
        else:
            # Random routing
            return 'model_b' if random.random() < self.traffic_split else 'model_a'

    def predict(self, features: torch.Tensor, model_version: str) -> dict:
        """Make prediction with selected model."""
        import time
        start = time.time()

        model = self.model_b if model_version == 'model_b' else self.model_a

        with torch.no_grad():
            prediction = model(features).item()

        latency = time.time() - start

        # Track metrics
        self.metrics[model_version]['requests'] += 1
        self.metrics[model_version]['latency'].append(latency)

        return {
            'prediction': prediction,
            'model_version': model_version,
            'latency': latency
        }

router = ABTestingRouter(
    model_a_path="/models/baseline_v3.pt",
    model_b_path="/models/candidate_v4.pt",
    traffic_split=0.1  # 10% to new model
)

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    user_id: Optional[str] = Header(None)
):
    """Prediction with A/B testing."""
    # Route request
    model_version = router.route_request(user_id)

    # Make prediction
    features = torch.FloatTensor([request.features])
    result = router.predict(features, model_version)

    return result

@app.get("/ab_metrics")
async def get_ab_metrics():
    """Get A/B test metrics."""
    import numpy as np

    metrics = {}
    for version, data in router.metrics.items():
        if data['requests'] > 0:
            metrics[version] = {
                'requests': data['requests'],
                'avg_latency': np.mean(data['latency']),
                'p95_latency': np.percentile(data['latency'], 95),
                'p99_latency': np.percentile(data['latency'], 99)
            }

    return metrics
```

### 2. Canary Deployment

```yaml
# canary-deployment.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model
spec:
  selector:
    app: ml-model
  ports:
  - port: 80
    targetPort: 8000

---
# Stable deployment (90% traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-stable
spec:
  replicas: 9
  selector:
    matchLabels:
      app: ml-model
      version: stable
  template:
    metadata:
      labels:
        app: ml-model
        version: stable
    spec:
      containers:
      - name: ml-model
        image: myregistry/ml-model:v3.2.0
        ports:
        - containerPort: 8000

---
# Canary deployment (10% traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-model
      version: canary
  template:
    metadata:
      labels:
        app: ml-model
        version: canary
    spec:
      containers:
      - name: ml-model
        image: myregistry/ml-model:v3.2.1
        ports:
        - containerPort: 8000
```

**Canary Monitoring Script:**

```python
import requests
import time
from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class CanaryMetrics:
    error_rate: float
    latency_p95: float
    latency_p99: float
    throughput: float

class CanaryMonitor:
    """Monitor canary deployment and decide on promotion/rollback."""

    def __init__(
        self,
        canary_endpoint: str,
        stable_endpoint: str,
        error_threshold: float = 0.05,  # 5% error rate
        latency_threshold: float = 1.5   # 1.5x latency increase
    ):
        self.canary_endpoint = canary_endpoint
        self.stable_endpoint = stable_endpoint
        self.error_threshold = error_threshold
        self.latency_threshold = latency_threshold

    def collect_metrics(self, endpoint: str, duration: int = 300) -> CanaryMetrics:
        """Collect metrics from endpoint for specified duration (seconds)."""
        errors = 0
        requests_count = 0
        latencies = []

        start_time = time.time()

        while (time.time() - start_time) < duration:
            try:
                start = time.time()
                response = requests.get(f"{endpoint}/metrics")
                latency = time.time() - start

                if response.status_code != 200:
                    errors += 1

                latencies.append(latency)
                requests_count += 1

                time.sleep(1)  # 1 request per second

            except Exception as e:
                errors += 1
                requests_count += 1

        error_rate = errors / requests_count if requests_count > 0 else 1.0

        return CanaryMetrics(
            error_rate=error_rate,
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            throughput=requests_count / duration
        )

    def evaluate_canary(self, duration: int = 300) -> bool:
        """Evaluate if canary should be promoted."""
        print(f"Monitoring canary for {duration} seconds...")

        # Collect metrics from both
        canary_metrics = self.collect_metrics(self.canary_endpoint, duration)
        stable_metrics = self.collect_metrics(self.stable_endpoint, duration)

        print(f"\nCanary Metrics:")
        print(f"  Error Rate: {canary_metrics.error_rate:.2%}")
        print(f"  P95 Latency: {canary_metrics.latency_p95:.3f}s")
        print(f"  P99 Latency: {canary_metrics.latency_p99:.3f}s")

        print(f"\nStable Metrics:")
        print(f"  Error Rate: {stable_metrics.error_rate:.2%}")
        print(f"  P95 Latency: {stable_metrics.latency_p95:.3f}s")
        print(f"  P99 Latency: {stable_metrics.latency_p99:.3f}s")

        # Decision criteria
        if canary_metrics.error_rate > self.error_threshold:
            print(f"\n❌ ROLLBACK: Error rate {canary_metrics.error_rate:.2%} exceeds threshold")
            return False

        latency_ratio = canary_metrics.latency_p95 / stable_metrics.latency_p95
        if latency_ratio > self.latency_threshold:
            print(f"\n❌ ROLLBACK: Latency increased by {latency_ratio:.1f}x")
            return False

        print(f"\n✅ PROMOTE: Canary metrics within acceptable range")
        return True

# Usage
if __name__ == "__main__":
    monitor = CanaryMonitor(
        canary_endpoint="http://ml-model-canary:8000",
        stable_endpoint="http://ml-model-stable:8000"
    )

    should_promote = monitor.evaluate_canary(duration=600)  # 10 minutes

    if should_promote:
        print("Promoting canary to production...")
        # Run kubectl commands to promote
    else:
        print("Rolling back canary...")
        # Run kubectl commands to rollback
```

### 3. Blue-Green Deployment

```yaml
# blue-green-deployment.yaml

# Blue deployment (current production)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-blue
spec:
  replicas: 5
  selector:
    matchLabels:
      app: ml-model
      slot: blue
  template:
    metadata:
      labels:
        app: ml-model
        slot: blue
    spec:
      containers:
      - name: ml-model
        image: myregistry/ml-model:v3.2.0

---
# Green deployment (new version)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-green
spec:
  replicas: 5
  selector:
    matchLabels:
      app: ml-model
      slot: green
  template:
    metadata:
      labels:
        app: ml-model
        slot: green
    spec:
      containers:
      - name: ml-model
        image: myregistry/ml-model:v3.2.1

---
# Service points to blue initially
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
    slot: blue  # Change to 'green' to switch traffic
  ports:
  - port: 80
    targetPort: 8000
```

**Blue-Green Switch Script:**

```python
from kubernetes import client, config
import time

class BlueGreenDeployment:
    """Manage blue-green deployments in Kubernetes."""

    def __init__(self, service_name: str, namespace: str = "default"):
        config.load_kube_config()
        self.v1 = client.CoreV1Api()
        self.service_name = service_name
        self.namespace = namespace

    def get_active_slot(self) -> str:
        """Get currently active slot (blue or green)."""
        service = self.v1.read_namespaced_service(
            name=self.service_name,
            namespace=self.namespace
        )
        return service.spec.selector.get('slot', 'blue')

    def switch_traffic(self, target_slot: str) -> None:
        """Switch traffic to target slot."""
        # Read current service
        service = self.v1.read_namespaced_service(
            name=self.service_name,
            namespace=self.namespace
        )

        # Update selector
        service.spec.selector['slot'] = target_slot

        # Patch service
        self.v1.patch_namespaced_service(
            name=self.service_name,
            namespace=self.namespace,
            body=service
        )

        print(f"✅ Traffic switched to {target_slot} slot")

    def deploy_to_inactive_slot(self, new_image: str) -> str:
        """Deploy new version to inactive slot."""
        active = self.get_active_slot()
        inactive = 'green' if active == 'blue' else 'blue'

        print(f"Deploying {new_image} to {inactive} slot...")

        # Update deployment (simplified)
        apps_v1 = client.AppsV1Api()
        deployment_name = f"ml-model-{inactive}"

        deployment = apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace=self.namespace
        )

        deployment.spec.template.spec.containers[0].image = new_image

        apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=self.namespace,
            body=deployment
        )

        # Wait for rollout
        print("Waiting for rollout to complete...")
        time.sleep(30)  # In production, properly wait for ready status

        print(f"✅ Deployment to {inactive} slot complete")
        return inactive

    def rollback(self) -> None:
        """Rollback to previous slot."""
        current = self.get_active_slot()
        previous = 'green' if current == 'blue' else 'blue'

        print(f"Rolling back from {current} to {previous}...")
        self.switch_traffic(previous)

# Usage
if __name__ == "__main__":
    deployer = BlueGreenDeployment(service_name="ml-model-service")

    # Deploy new version to inactive slot
    inactive_slot = deployer.deploy_to_inactive_slot("myregistry/ml-model:v3.2.1")

    # Run tests on inactive slot
    print("Running smoke tests on inactive slot...")
    # ... test logic ...

    # Switch traffic
    deployer.switch_traffic(inactive_slot)

    # Monitor for issues
    print("Monitoring for 5 minutes...")
    time.sleep(300)

    # If issues detected:
    # deployer.rollback()
```

---

## Model Versioning and Registry

```python
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Optional
import hashlib
import json

class ModelRegistry:
    """Production model registry with versioning and metadata."""

    def __init__(self, tracking_uri: str = "http://mlflow:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(
        self,
        model_path: str,
        model_name: str,
        metrics: Dict[str, float],
        params: Dict[str, any],
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register model with metadata."""

        # Create model signature
        with open(model_path, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.pytorch.log_model(
                model_path,
                artifact_path="model",
                registered_model_name=model_name
            )

            # Log additional metadata
            mlflow.set_tags({
                "model_hash": model_hash,
                "deployment_ready": "false",
                **(tags or {})
            })

            run_id = mlflow.active_run().info.run_id

        print(f"✅ Model registered with run_id: {run_id}")
        return run_id

    def promote_model(
        self,
        model_name: str,
        version: int,
        stage: str = "Production"
    ) -> None:
        """Promote model to production stage."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True  # Archive previous production
        )

        print(f"✅ Model {model_name} v{version} promoted to {stage}")

    def get_production_model(self, model_name: str) -> str:
        """Get current production model URI."""
        model_version = self.client.get_latest_versions(
            model_name,
            stages=["Production"]
        )[0]

        return f"models:/{model_name}/{model_version.version}"

    def compare_models(
        self,
        model_name: str,
        versions: list,
        metric: str = "accuracy"
    ) -> Dict:
        """Compare different model versions."""
        results = {}

        for version in versions:
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)

            results[f"v{version}"] = {
                "metric_value": run.data.metrics.get(metric),
                "stage": model_version.current_stage,
                "created_at": model_version.creation_timestamp
            }

        return results

# Usage
registry = ModelRegistry(tracking_uri="http://mlflow-server:5000")

# Register new model
run_id = registry.register_model(
    model_path="/models/trained/model_v4.pt",
    model_name="fraud_detection",
    metrics={"accuracy": 0.95, "auc": 0.98, "f1": 0.94},
    params={"learning_rate": 0.001, "epochs": 100},
    tags={"framework": "pytorch", "task": "classification"}
)

# Compare models
comparison = registry.compare_models(
    model_name="fraud_detection",
    versions=[3, 4],
    metric="auc"
)
print(comparison)

# Promote to production if better
if comparison["v4"]["metric_value"] > comparison["v3"]["metric_value"]:
    registry.promote_model("fraud_detection", version=4, stage="Production")
```

---

## Best Practices

### 1. Model Artifacts

```python
class ModelArtifacts:
    """Package all necessary artifacts for deployment."""

    @staticmethod
    def save_deployment_package(
        model: torch.nn.Module,
        preprocessor: object,
        metadata: dict,
        output_dir: str
    ):
        """Save complete deployment package."""
        import joblib
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save model as TorchScript
        model.eval()
        dummy_input = torch.randn(1, metadata['input_dim'])
        traced_model = torch.jit.trace(model, dummy_input)
        torch.jit.save(traced_model, output_path / "model.pt")

        # Save preprocessor
        joblib.dump(preprocessor, output_path / "preprocessor.pkl")

        # Save metadata
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save requirements
        requirements = [
            f"torch=={torch.__version__}",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0"
        ]

        with open(output_path / "requirements.txt", 'w') as f:
            f.write("\n".join(requirements))

        print(f"✅ Deployment package saved to {output_dir}")
```

### 2. Configuration Management

```python
from pydantic import BaseSettings, validator
from typing import Optional

class DeploymentConfig(BaseSettings):
    """Production deployment configuration."""

    # Model settings
    model_path: str
    model_version: str
    device: str = "cuda"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 60

    # Performance settings
    batch_size: int = 32
    max_batch_time: float = 0.1
    enable_caching: bool = True
    cache_ttl: int = 300

    # Monitoring
    enable_prometheus: bool = True
    log_level: str = "INFO"

    # Resource limits
    max_memory_mb: Optional[int] = None
    max_requests_per_second: Optional[int] = None

    @validator('device')
    def validate_device(cls, v):
        if v not in ['cuda', 'cpu', 'mps']:
            raise ValueError('Device must be cuda, cpu, or mps')
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Usage
config = DeploymentConfig()
print(f"Deploying model {config.model_version} on {config.device}")
```

### 3. Error Handling

```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging

class ModelError(Exception):
    """Base exception for model errors."""
    pass

class InputValidationError(ModelError):
    """Invalid input data."""
    pass

class ModelInferenceError(ModelError):
    """Error during model inference."""
    pass

@app.exception_handler(InputValidationError)
async def input_validation_handler(request: Request, exc: InputValidationError):
    """Handle input validation errors."""
    logging.error(f"Input validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "message": str(exc),
            "request_id": request.headers.get("X-Request-ID")
        }
    )

@app.exception_handler(ModelInferenceError)
async def model_inference_handler(request: Request, exc: ModelInferenceError):
    """Handle model inference errors."""
    logging.error(f"Model inference error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Model inference failed",
            "message": "Please try again later",
            "request_id": request.headers.get("X-Request-ID")
        }
    )
```

---

## Summary

Model deployment in 2025 requires:

1. **Strategy Selection:** Choose deployment pattern based on latency, cost, and scale requirements
2. **Robust Infrastructure:** Containerization, orchestration, and auto-scaling
3. **CI/CD Automation:** Automated testing, building, and deployment
4. **Advanced Patterns:** A/B testing, canary, blue-green for safe rollouts
5. **Monitoring:** Comprehensive metrics, logging, and alerting
6. **Version Control:** Proper model versioning and artifact management

**Key Metrics:**
- 60% faster deployment with MLOps
- 40% reduction in production incidents
- Sub-100ms latency for real-time systems
- 99.9% uptime with proper deployment patterns

**Next Steps:**
- Implement monitoring (Section 37)
- Set up pipeline orchestration (Section 38)
- Enable experiment tracking (Section 40)
