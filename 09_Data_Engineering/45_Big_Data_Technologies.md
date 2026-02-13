# Big Data Technologies for Machine Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Apache Hadoop](#apache-hadoop)
3. [Apache Spark](#apache-spark)
4. [Dask - 2025 Recommended](#dask---2025-recommended)
5. [Distributed ML Frameworks](#distributed-ml-frameworks)
6. [Cloud Platforms (2025 Trends)](#cloud-platforms-2025-trends)
7. [Kubernetes Deployment](#kubernetes-deployment)
8. [When to Use Each Technology](#when-to-use-each-technology)
9. [Complete Code Examples](#complete-code-examples)

---

## Introduction

Modern machine learning requires processing datasets that don't fit in memory. Big data technologies enable distributed computing across clusters of machines, making it possible to train models on terabytes or petabytes of data.

### What is "Big Data"?

**Traditional definition:** Data that exceeds the memory capacity of a single machine.

**Practical thresholds:**
- **Small data:** < 10 GB (use pandas, single machine)
- **Medium data:** 10 GB - 1 TB (use Dask or Spark)
- **Big data:** > 1 TB (use Spark, cloud-native solutions)

### The 3 V's of Big Data

1. **Volume:** Amount of data (terabytes to petabytes)
2. **Velocity:** Speed of data generation and processing
3. **Variety:** Different data types (structured, unstructured, streaming)

### 2025 Landscape

**Key Trend:** Shift from on-premise Hadoop clusters to cloud-native solutions and Python-first frameworks.

**Performance Benchmarks (from RESEARCH_SUMMARY_2025.md):**
- **Hadoop MapReduce:** Baseline (1x)
- **Spark:** 10-100x faster than Hadoop MapReduce
- **Dask:** 50% faster than Spark on standard benchmarks

---

## Apache Hadoop

### Overview

**Apache Hadoop** is the foundational big data framework, consisting of:
1. **HDFS** (Hadoop Distributed File System) - distributed storage
2. **MapReduce** - distributed processing
3. **YARN** (Yet Another Resource Negotiator) - resource management

### HDFS (Hadoop Distributed File System)

**Key Features:**
- **Distributed storage** across cluster nodes
- **Fault tolerance** via replication (default 3x)
- **Optimized for large files** (64MB+ blocks)
- **Write-once, read-many** pattern

#### HDFS Architecture

```
NameNode (Master)
+---- Manages file system metadata
+---- Tracks block locations
+---- Handles client requests

DataNodes (Workers)
+---- Store actual data blocks
+---- Report to NameNode (heartbeats)
+---- Handle read/write operations
```

#### Working with HDFS

```bash
# Start Hadoop services
start-dfs.sh
start-yarn.sh

# Create directory
hdfs dfs -mkdir /user/data

# Upload file to HDFS
hdfs dfs -put local_data.csv /user/data/

# List files
hdfs dfs -ls /user/data/

# Download file from HDFS
hdfs dfs -get /user/data/output.csv ./

# Check file replication
hdfs dfs -stat %r /user/data/local_data.csv

# Delete file
hdfs dfs -rm /user/data/local_data.csv
```

#### Python HDFS Client

```python
from hdfs import InsecureClient

# Connect to HDFS
client = InsecureClient('http://namenode:50070', user='hadoop')

# Upload file
client.upload('/user/data/dataset.csv', 'local_dataset.csv')

# Download file
client.download('/user/data/output.csv', 'local_output.csv')

# List directory
files = client.list('/user/data/')
print(f"Files in HDFS: {files}")

# Read file content
with client.read('/user/data/dataset.csv', encoding='utf-8') as reader:
    content = reader.read()
    print(content[:100])

# Write file
with client.write('/user/data/new_file.txt', encoding='utf-8') as writer:
    writer.write('Hello HDFS!')

# Delete file
client.delete('/user/data/old_file.csv')
```

### MapReduce

**MapReduce** is a programming model for parallel processing:

1. **Map Phase:** Transform input data into key-value pairs
2. **Shuffle Phase:** Group values by key
3. **Reduce Phase:** Aggregate values for each key

#### MapReduce Example: Word Count

```python
from mrjob.job import MRJob
from mrjob.step import MRStep

class WordCount(MRJob):
    """
    Classic MapReduce example: Count word frequencies.

    Input: Text files
    Output: (word, count) pairs
    """

    def mapper(self, _, line):
        """
        Map phase: Emit (word, 1) for each word.

        Parameters:
        -----------
        _ : ignored key
        line : str
            Input line

        Yields:
        -------
        (word, 1) pairs
        """
        for word in line.split():
            yield (word.lower(), 1)

    def reducer(self, word, counts):
        """
        Reduce phase: Sum counts for each word.

        Parameters:
        -----------
        word : str
            Word key
        counts : iterator
            Iterator of 1s

        Yields:
        -------
        (word, total_count) pairs
        """
        yield (word, sum(counts))

if __name__ == '__main__':
    WordCount.run()

# Run on Hadoop
# python word_count.py -r hadoop hdfs:///user/data/text_files/*.txt
```

#### MapReduce for ML: Feature Extraction

```python
from mrjob.job import MRJob
import json

class FeatureExtractor(MRJob):
    """
    Extract features from log data using MapReduce.

    Input: JSON log files
    Output: User features (total_clicks, avg_duration, etc.)
    """

    def mapper(self, _, line):
        """Extract user actions from logs."""
        try:
            log = json.loads(line)
            user_id = log['user_id']
            action = log['action']
            duration = log['duration']

            yield (user_id, {
                'click': 1 if action == 'click' else 0,
                'duration': duration
            })
        except:
            pass

    def reducer(self, user_id, records):
        """Aggregate features per user."""
        records = list(records)

        total_clicks = sum(r['click'] for r in records)
        total_duration = sum(r['duration'] for r in records)
        avg_duration = total_duration / len(records)

        yield (user_id, {
            'total_clicks': total_clicks,
            'avg_duration': avg_duration,
            'num_sessions': len(records)
        })

if __name__ == '__main__':
    FeatureExtractor.run()
```

### When to Use Hadoop

**Pros:**
-  Mature ecosystem (15+ years)
-  Fault tolerance and reliability
-  Handles petabyte-scale data
-  Cheap storage (commodity hardware)

**Cons:**
-  Slow (disk-based, writes intermediate results)
-  Not suitable for iterative algorithms (ML)
-  High latency
-  Complex setup and maintenance

**Use Cases:**
- Massive batch processing (ETL pipelines)
- Archival storage with occasional access
- When cost > speed priority
- Legacy systems already on Hadoop

---

## Apache Spark

### Overview

**Apache Spark** is a unified analytics engine for big data processing, designed to overcome Hadoop MapReduce limitations.

**Key Innovation:** **In-memory processing** (10-100x faster than MapReduce)

### Spark Architecture

```
Driver Program
+---- SparkContext (entry point)
+---- DAG Scheduler (optimize execution)
+---- Task Scheduler (assign tasks)

Cluster Manager (YARN, Mesos, or Standalone)
+---- Resource allocation
+---- Worker node management

Worker Nodes
+---- Executor (JVM process)
|   +---- Cache (in-memory storage)
|   +---- Tasks (parallel operations)
+---- Multiple per node
```

### Spark Components

1. **Spark Core:** RDDs, distributed task execution
2. **Spark SQL:** Structured data processing (DataFrames)
3. **MLlib:** Machine learning algorithms
4. **GraphX:** Graph processing
5. **Spark Streaming:** Real-time stream processing

### PySpark Setup

```python
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("ML Pipeline") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Check Spark version
print(f"Spark version: {spark.version}")

# Access SparkContext
sc = spark.sparkContext
print(f"Master: {sc.master}")
print(f"App name: {sc.appName}")
```

### Spark DataFrames

```python
# Read CSV
df = spark.read.csv(
    'hdfs:///user/data/dataset.csv',
    header=True,
    inferSchema=True
)

# Show schema
df.printSchema()

# Show first 10 rows
df.show(10)

# Select columns
df.select('age', 'income', 'target').show()

# Filter rows
df_filtered = df.filter(df['age'] > 30)

# Group and aggregate
df_agg = df.groupBy('category').agg(
    {'price': 'mean', 'quantity': 'sum'}
)

# SQL queries
df.createOrReplaceTempView("data")
result = spark.sql("""
    SELECT category, AVG(price) as avg_price
    FROM data
    WHERE age > 25
    GROUP BY category
""")
result.show()

# Write to Parquet (columnar format, much faster than CSV)
df.write.parquet('hdfs:///user/data/output.parquet', mode='overwrite')

# Read Parquet
df_parquet = spark.read.parquet('hdfs:///user/data/output.parquet')
```

### Spark RDDs (Resilient Distributed Datasets)

```python
# Create RDD from list
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rdd = sc.parallelize(data, numSlices=4)

# Transformations (lazy evaluation)
rdd_squared = rdd.map(lambda x: x ** 2)
rdd_filtered = rdd_squared.filter(lambda x: x > 20)

# Actions (trigger computation)
result = rdd_filtered.collect()
print(f"Result: {result}")

count = rdd_filtered.count()
print(f"Count: {count}")

# Read text file
text_rdd = sc.textFile('hdfs:///user/data/text_files/*.txt')

# Word count with RDD
word_counts = text_rdd \
    .flatMap(lambda line: line.split()) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b)

# Get top 10 words
top_words = word_counts.takeOrdered(10, key=lambda x: -x[1])
print(f"Top 10 words: {top_words}")
```

### Spark MLlib

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Load data
df = spark.read.csv('hdfs:///user/data/train.csv', header=True, inferSchema=True)

# Prepare features
feature_cols = ['age', 'income', 'credit_score', 'balance']

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features_raw'
)

scaler = StandardScaler(
    inputCol='features_raw',
    outputCol='features',
    withStd=True,
    withMean=True
)

# Random Forest
rf = RandomForestClassifier(
    featuresCol='features',
    labelCol='target',
    numTrees=100,
    maxDepth=10
)

# Create pipeline
pipeline = Pipeline(stages=[assembler, scaler, rf])

# Split data
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Train
model = pipeline.fit(train)

# Predict
predictions = model.transform(test)

# Evaluate
evaluator = MulticlassClassificationEvaluator(
    labelCol='target',
    predictionCol='prediction',
    metricName='accuracy'
)

accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.4f}")

# Save model
model.write().overwrite().save('hdfs:///user/models/rf_model')

# Load model
from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load('hdfs:///user/models/rf_model')
```

### Spark Performance Optimization

```python
# 1. Persist/Cache frequently used DataFrames
df_cached = df.cache()  # Store in memory
df_persisted = df.persist()  # Store in memory + disk

# 2. Repartition for parallel processing
df_repartitioned = df.repartition(200)  # 200 partitions

# 3. Broadcast small DataFrames (< 10MB)
from pyspark.sql.functions import broadcast
result = large_df.join(broadcast(small_df), 'key')

# 4. Use Parquet instead of CSV
df.write.parquet('output.parquet')  # Columnar, compressed, fast

# 5. Avoid collect() on large DataFrames
# BAD: data = df.collect()  # Brings all data to driver
# GOOD: df.show(10)  # Shows sample

# 6. Use SQL for complex queries (optimized)
df.createOrReplaceTempView('data')
result = spark.sql("SELECT * FROM data WHERE age > 30")

# 7. Coalesce to reduce partitions (after filtering)
df_filtered.coalesce(10).write.parquet('output.parquet')
```

### When to Use Spark

**Pros:**
-  10-100x faster than Hadoop MapReduce
-  In-memory processing
-  Unified framework (batch, streaming, ML, graph)
-  Supports SQL, Python, Scala, Java, R
-  Mature ML library (MLlib)

**Cons:**
-  Requires significant memory
-  Complex setup and tuning
-  JVM overhead for Python (PySpark)
-  Slower than Dask for Python-native workflows

**Use Cases:**
- Large-scale ETL and data processing
- Iterative ML algorithms on big data
- Real-time streaming analytics
- Graph analytics
- When you need mature ecosystem and support

---

## Dask - 2025 Recommended

### Overview

**Dask** is a flexible parallel computing library for Python that scales pandas, numpy, and scikit-learn to multi-core and distributed systems.

**2025 Benchmark:** **Dask is 50% faster than Spark** on standard benchmarks (from RESEARCH_SUMMARY_2025.md)

### Why Dask?

**Advantages over Spark:**
-  **Python-native** (no JVM overhead)
-  **50% faster** on Python ML workloads
-  **Seamless pandas/numpy/scikit-learn integration**
-  **Lightweight** (easy setup)
-  **Dynamic task scheduling** (better for irregular workloads)
-  **Lower latency**

**When to use Dask over Spark:**
- Python-centric ML workflows
- Need to scale pandas/numpy/scikit-learn
- Medium-sized data (10GB - 1TB)
- Rapid prototyping and iteration
- Lower operational overhead

### Dask Installation and Setup

```bash
# Install Dask
pip install "dask[complete]"

# Optional: Install distributed scheduler
pip install dask distributed

# Optional: Install ML integrations
pip install dask-ml xgboost lightgbm
```

### Dask DataFrames (Parallel Pandas)

```python
import dask.dataframe as dd
import pandas as pd

# Read large CSV (larger than memory)
df = dd.read_csv('s3://bucket/data/*.csv')

# Or from multiple files
df = dd.read_csv('data/part-*.csv')

# Dask DataFrame API is nearly identical to pandas
df_filtered = df[df['age'] > 30]
df_grouped = df.groupby('category')['price'].mean()

# Lazy evaluation (define computation graph)
result = df.groupby('region').agg({
    'sales': 'sum',
    'customers': 'count',
    'revenue': 'mean'
})

# Trigger computation
result_computed = result.compute()  # Returns pandas DataFrame

# Alternative: Persist in memory for reuse
df_cached = df.persist()

# Convert to pandas (if result fits in memory)
df_pandas = result.compute()

# Save to Parquet (distributed write)
df.to_parquet('output.parquet')

# Read Parquet (much faster than CSV)
df_parquet = dd.read_parquet('output.parquet')
```

### Dask Arrays (Parallel NumPy)

```python
import dask.array as da
import numpy as np

# Create large array (chunked)
x = da.random.random((10000, 10000), chunks=(1000, 1000))

# NumPy-like operations
y = x + x.T
z = y.mean(axis=0)

# Compute result
result = z.compute()  # Returns numpy array

# Matrix operations
a = da.random.random((10000, 1000), chunks=(1000, 100))
b = da.random.random((1000, 500), chunks=(100, 100))
c = da.dot(a, b)

# SVD on large matrix
u, s, v = da.linalg.svd(a)
s_computed = s.compute()
```

### Dask Machine Learning

```python
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler
from dask_ml.linear_model import LogisticRegression
import dask.dataframe as dd

# Load large dataset
df = dd.read_parquet('s3://bucket/data.parquet')

# Prepare data
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

# Evaluate
score = clf.score(X_test_scaled, y_test)
print(f"Accuracy: {score:.4f}")
```

### Dask + XGBoost (Distributed Training)

```python
import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.model_selection import train_test_split
import xgboost as xgb

# Start Dask cluster
client = Client(n_workers=4, threads_per_worker=2)
print(f"Dashboard: {client.dashboard_link}")

# Load data
df = dd.read_parquet('large_dataset.parquet')

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to DMatrix (XGBoost format)
dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)

# Train distributed XGBoost
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'tree_method': 'hist'  # Fast histogram-based algorithm
}

output = xgb.dask.train(
    client,
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtest, 'test')]
)

# Get model
model = output['booster']

# Predict
predictions = xgb.dask.predict(client, model, X_test)
predictions_computed = predictions.compute()

print(f"Predictions shape: {predictions_computed.shape}")

# Close client
client.close()
```

### Dask + LightGBM

```python
import dask.dataframe as dd
from dask.distributed import Client
import lightgbm as lgb

# Start Dask cluster
client = Client()

# Load data
df = dd.read_parquet('data.parquet')

X = df.drop('target', axis=1).to_dask_array(lengths=True)
y = df['target'].to_dask_array(lengths=True)

# Create LightGBM Dask classifier
clf = lgb.DaskLGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31
)

# Fit
clf.fit(X, y)

# Predict
predictions = clf.predict(X)

# Close client
client.close()
```

### Dask + Scikit-Learn

```python
from dask_ml.wrappers import ParallelPostFit
from sklearn.ensemble import RandomForestClassifier
import dask.dataframe as dd

# Wrap sklearn model for parallel prediction
rf = RandomForestClassifier(n_estimators=100)
parallel_rf = ParallelPostFit(rf)

# Train on in-memory data
parallel_rf.fit(X_train_pandas, y_train_pandas)

# Parallel prediction on Dask DataFrame
df_large = dd.read_parquet('large_test_data.parquet')
predictions = parallel_rf.predict(df_large)

# Compute predictions in parallel
predictions_computed = predictions.compute()
```

### Dask Distributed Cluster

```python
from dask.distributed import Client, LocalCluster

# Local cluster (single machine, multiple cores)
cluster = LocalCluster(n_workers=4, threads_per_worker=2)
client = Client(cluster)

print(f"Dashboard: {client.dashboard_link}")

# Distributed cluster (multiple machines)
# On scheduler machine:
# dask-scheduler --port 8786

# On worker machines:
# dask-worker scheduler-address:8786

# In Python:
client = Client('scheduler-address:8786')

# Check cluster status
print(client)

# Submit computation
future = client.submit(lambda x: x ** 2, 10)
result = future.result()

# Map function across data
futures = client.map(lambda x: x ** 2, range(100))
results = client.gather(futures)

# Close client
client.close()
```

### Dask Performance Optimization

```python
import dask
import dask.dataframe as dd

# 1. Optimize chunk size (target: 100-200 MB per partition)
df = dd.read_csv('data.csv', blocksize='100MB')

# 2. Persist frequently used DataFrames
df_cached = df.persist()

# 3. Repartition to optimize parallelism
df_repartitioned = df.repartition(npartitions=100)

# 4. Use Parquet (columnar, compressed)
df.to_parquet('output.parquet', compression='snappy')

# 5. Avoid computing intermediate results
# BAD:
# result1 = df.groupby('A').sum().compute()
# result2 = df.groupby('B').mean().compute()

# GOOD (single computation graph):
# result = dask.compute(
#     df.groupby('A').sum(),
#     df.groupby('B').mean()
# )

# 6. Set number of workers
dask.config.set(num_workers=4)

# 7. Use appropriate scheduler
# Single machine: 'threads' (default)
# CPU-bound: 'processes'
# Distributed: Client

with dask.config.set(scheduler='threads'):
    result = df.compute()
```

### When to Use Dask

**Pros:**
-  **50% faster than Spark** (2025 benchmark)
-  Python-native (no JVM)
-  Familiar pandas/numpy/scikit-learn API
-  Lightweight and easy setup
-  Scales XGBoost, LightGBM seamlessly

**Cons:**
-  Less mature than Spark
-  Smaller ecosystem
-  Not ideal for streaming (use Spark Streaming)

**Use Cases:**
- Python ML workflows on medium-to-large data
- Scaling pandas/numpy operations
- Distributed XGBoost/LightGBM training
- Rapid prototyping and iteration
- When you want minimal operational overhead

---

## Distributed ML Frameworks

### Ray

**Ray** is a general-purpose distributed computing framework for Python.

#### Ray Core

```python
import ray

# Initialize Ray
ray.init(num_cpus=4)

# Remote functions (run in parallel)
@ray.remote
def compute_square(x):
    return x ** 2

# Submit tasks
futures = [compute_square.remote(i) for i in range(100)]

# Get results
results = ray.get(futures)

print(f"Sum of squares: {sum(results)}")

# Shutdown Ray
ray.shutdown()
```

#### Ray Train (Distributed Training)

```python
from ray import train
from ray.train import ScalingConfig
from ray.train.xgboost import XGBoostTrainer
import pandas as pd

# Prepare data
df = pd.read_parquet('data.parquet')

# Create Ray dataset
dataset = ray.data.from_pandas(df)

# Split data
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2)

# Configure distributed training
scaling_config = ScalingConfig(
    num_workers=4,
    use_gpu=False
)

# XGBoost trainer
trainer = XGBoostTrainer(
    scaling_config=scaling_config,
    label_column='target',
    params={
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1
    },
    datasets={'train': train_dataset, 'test': test_dataset}
)

# Train
result = trainer.fit()

print(f"Training complete!")
```

#### Ray Tune (Hyperparameter Tuning)

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_model(config):
    """Training function with hyperparameters."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    clf = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth']
    )

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    # Report metric to Ray Tune
    tune.report(accuracy=accuracy)

# Define search space
search_space = {
    'n_estimators': tune.choice([50, 100, 200]),
    'max_depth': tune.randint(3, 15)
}

# Run hyperparameter search
analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=20,
    scheduler=ASHAScheduler(metric='accuracy', mode='max')
)

# Best config
best_config = analysis.get_best_config(metric='accuracy', mode='max')
print(f"Best config: {best_config}")
```

### Horovod (Distributed Deep Learning)

**Horovod** enables distributed training of deep learning models with near-linear scaling.

#### Horovod + PyTorch

```python
import torch
import torch.nn as nn
import horovod.torch as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to local rank
torch.cuda.set_device(hvd.local_rank())

# Build model
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
).cuda()

# Wrap optimizer with Horovod DistributedOptimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters()
)

# Broadcast initial parameters from rank 0
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Training loop
for epoch in range(100):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = nn.BCELoss()(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if hvd.rank() == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save model (only on rank 0)
if hvd.rank() == 0:
    torch.save(model.state_dict(), 'model.pth')
```

#### Running Horovod

```bash
# Single machine, 4 GPUs
horovodrun -np 4 python train.py

# Multiple machines, 16 GPUs (4 machines x 4 GPUs)
horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py
```

### When to Use Ray vs Horovod

**Ray:**
- General-purpose distributed computing
- Hyperparameter tuning
- Reinforcement learning
- Mixed CPU/GPU workloads

**Horovod:**
- Distributed deep learning training
- Multi-GPU, multi-node scaling
- When you need near-linear scaling (1000+ GPUs)

---

## Cloud Platforms (2025 Trends)

### AWS (Amazon Web Services)

#### EMR (Elastic MapReduce)

```python
import boto3

# Create EMR client
emr = boto3.client('emr', region_name='us-east-1')

# Launch Spark cluster
response = emr.run_job_flow(
    Name='ML-Spark-Cluster',
    ReleaseLabel='emr-6.10.0',
    Instances={
        'InstanceGroups': [
            {
                'Name': 'Master',
                'InstanceRole': 'MASTER',
                'InstanceType': 'm5.xlarge',
                'InstanceCount': 1
            },
            {
                'Name': 'Workers',
                'InstanceRole': 'CORE',
                'InstanceType': 'm5.xlarge',
                'InstanceCount': 4
            }
        ],
        'KeepJobFlowAliveWhenNoSteps': True,
        'TerminationProtected': False
    },
    Applications=[
        {'Name': 'Spark'},
        {'Name': 'Hadoop'}
    ],
    BootstrapActions=[
        {
            'Name': 'Install Python packages',
            'ScriptBootstrapAction': {
                'Path': 's3://bucket/bootstrap.sh'
            }
        }
    ],
    JobFlowRole='EMR_EC2_DefaultRole',
    ServiceRole='EMR_DefaultRole'
)

cluster_id = response['JobFlowId']
print(f"Cluster ID: {cluster_id}")
```

#### SageMaker (Managed ML)

```python
import sagemaker
from sagemaker.sklearn import SKLearn

# SageMaker session
session = sagemaker.Session()

# Upload training data to S3
train_input = session.upload_data('train.csv', key_prefix='data/train')

# Define training script
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role='SageMakerRole',
    instance_type='ml.m5.xlarge',
    framework_version='1.0-1',
    py_version='py3'
)

# Train
sklearn_estimator.fit({'train': train_input})

# Deploy
predictor = sklearn_estimator.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1
)

# Predict
predictions = predictor.predict(test_data)
```

### Azure

#### HDInsight (Managed Hadoop/Spark)

```python
from azure.mgmt.hdinsight import HDInsightManagementClient
from azure.identity import DefaultAzureCredential

# Authenticate
credential = DefaultAzureCredential()
client = HDInsightManagementClient(credential, subscription_id)

# Create Spark cluster
cluster_params = {
    'location': 'eastus',
    'properties': {
        'clusterVersion': '4.0',
        'osType': 'Linux',
        'tier': 'Standard',
        'clusterDefinition': {
            'kind': 'spark',
            'configurations': {
                'gateway': {
                    'restAuthCredential.isEnabled': True,
                    'restAuthCredential.username': 'admin',
                    'restAuthCredential.password': 'password'
                }
            }
        },
        'computeProfile': {
            'roles': [
                {
                    'name': 'headnode',
                    'targetInstanceCount': 2,
                    'hardwareProfile': {'vmSize': 'Standard_D12_v2'}
                },
                {
                    'name': 'workernode',
                    'targetInstanceCount': 4,
                    'hardwareProfile': {'vmSize': 'Standard_D13_v2'}
                }
            ]
        }
    }
}

client.clusters.create('resource-group', 'cluster-name', cluster_params)
```

#### Databricks (Unified Analytics)

```python
# Databricks REST API
import requests

DATABRICKS_INSTANCE = 'https://your-instance.azuredatabricks.net'
TOKEN = 'your-token'

headers = {'Authorization': f'Bearer {TOKEN}'}

# Create cluster
cluster_config = {
    'cluster_name': 'ML-Cluster',
    'spark_version': '11.3.x-scala2.12',
    'node_type_id': 'Standard_DS3_v2',
    'num_workers': 4,
    'autoscale': {
        'min_workers': 2,
        'max_workers': 8
    }
}

response = requests.post(
    f'{DATABRICKS_INSTANCE}/api/2.0/clusters/create',
    headers=headers,
    json=cluster_config
)

cluster_id = response.json()['cluster_id']
print(f"Cluster ID: {cluster_id}")
```

### GCP (Google Cloud Platform)

#### Dataproc (Managed Hadoop/Spark)

```python
from google.cloud import dataproc_v1

# Create client
client = dataproc_v1.ClusterControllerClient(
    client_options={'api_endpoint': 'us-central1-dataproc.googleapis.com:443'}
)

# Define cluster
cluster = {
    'project_id': 'your-project',
    'cluster_name': 'ml-cluster',
    'config': {
        'master_config': {
            'num_instances': 1,
            'machine_type_uri': 'n1-standard-4'
        },
        'worker_config': {
            'num_instances': 4,
            'machine_type_uri': 'n1-standard-4'
        },
        'software_config': {
            'image_version': '2.0-debian10'
        }
    }
}

# Create cluster
operation = client.create_cluster(
    request={'project_id': 'your-project', 'region': 'us-central1', 'cluster': cluster}
)

result = operation.result()
print(f"Cluster created: {result.cluster_name}")
```

#### BigQuery ML

```sql
-- Create model in BigQuery
CREATE OR REPLACE MODEL `dataset.logistic_model`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['target']
) AS
SELECT
  feature1,
  feature2,
  feature3,
  target
FROM
  `dataset.training_data`;

-- Predict
SELECT
  *
FROM
  ML.PREDICT(MODEL `dataset.logistic_model`,
    (SELECT * FROM `dataset.test_data`));

-- Evaluate
SELECT
  *
FROM
  ML.EVALUATE(MODEL `dataset.logistic_model`,
    (SELECT * FROM `dataset.test_data`));
```

### 2025 Cloud Trends

**Key Trends from RESEARCH_SUMMARY_2025.md:**

1. **Kubernetes deployment** for all frameworks (standard)
2. **Serverless big data** (AWS Athena, BigQuery)
3. **Auto-scaling** clusters (cost optimization)
4. **ML platform consolidation** (Databricks, SageMaker)
5. **Edge computing** for real-time inference

---

## Kubernetes Deployment

### Deploy Spark on Kubernetes

```yaml
# spark-application.yaml
apiVersion: "sparkoperator.k8s.io/v1beta2"
kind: SparkApplication
metadata:
  name: spark-ml-job
  namespace: default
spec:
  type: Python
  pythonVersion: "3"
  mode: cluster
  image: "gcr.io/spark-operator/spark-py:v3.1.1"
  imagePullPolicy: Always
  mainApplicationFile: "s3a://bucket/train.py"
  sparkVersion: "3.1.1"
  restartPolicy:
    type: Never
  driver:
    cores: 1
    coreLimit: "1200m"
    memory: "4g"
    labels:
      version: "3.1.1"
    serviceAccount: spark
  executor:
    cores: 2
    instances: 4
    memory: "8g"
    labels:
      version: "3.1.1"
```

```bash
# Deploy to Kubernetes
kubectl apply -f spark-application.yaml

# Check status
kubectl get sparkapplication spark-ml-job

# View logs
kubectl logs spark-ml-job-driver
```

### Deploy Dask on Kubernetes

```yaml
# dask-cluster.yaml
apiVersion: kubernetes.dask.org/v1
kind: DaskCluster
metadata:
  name: dask-cluster
spec:
  worker:
    replicas: 4
    spec:
      containers:
      - name: worker
        image: daskdev/dask:latest
        args: [dask-worker, --nthreads, "4", --memory-limit, 8GB]
        resources:
          limits:
            memory: 8Gi
            cpu: 4
  scheduler:
    spec:
      containers:
      - name: scheduler
        image: daskdev/dask:latest
        args: [dask-scheduler]
        resources:
          limits:
            memory: 4Gi
            cpu: 2
```

```bash
# Deploy Dask cluster
kubectl apply -f dask-cluster.yaml

# Connect from Python
from dask_kubernetes import KubeCluster
from dask.distributed import Client

cluster = KubeCluster.from_yaml('dask-cluster.yaml')
client = Client(cluster)

# Scale workers
cluster.scale(10)
```

---

## When to Use Each Technology

### Decision Tree

```
Data size < 10 GB?
+--- Yes --> Use pandas (single machine)
+--- No --> Data size < 1 TB?
    +--- Yes --> Python-centric ML workflow?
    |   +--- Yes --> Use Dask (50% faster than Spark for Python)
    |   +--- No --> Use Spark (mature ecosystem)
    +--- No --> Data size > 1 TB?
        +--- Use cloud-native (BigQuery, Snowflake, Databricks)
        +--- Or Spark with cloud storage (S3, GCS, ADLS)

Streaming data?
+--- Use Spark Structured Streaming or Apache Flink

Deep learning at scale?
+--- Use Horovod (multi-GPU/multi-node)

General distributed computing?
+--- Use Ray (reinforcement learning, hyperparameter tuning)

Batch ETL on budget?
+--- Use Hadoop MapReduce (slow but cheap storage)
```

### Technology Comparison Table

| Technology | Speed | Best For | When to Use |
|------------|-------|----------|-------------|
| **Pandas** | Fast (single core) | < 10 GB | Exploratory analysis, prototyping |
| **Dask** | Fastest for Python (50% > Spark) | 10GB - 1TB | Python ML workflows, scaling pandas/scikit-learn |
| **Spark** | 10-100x > Hadoop | > 100 GB | Large-scale ETL, iterative ML, mature ecosystem |
| **Hadoop** | Baseline | > 1 TB | Batch processing, archival storage, cost priority |
| **Ray** | Fast | Variable | Hyperparameter tuning, RL, distributed Python |
| **Horovod** | Near-linear scaling | Deep learning | Multi-GPU/multi-node deep learning |
| **BigQuery** | Very fast | > 1 TB | SQL analytics, serverless, no cluster management |

---

## Complete Code Examples

### End-to-End ML Pipeline with Dask

```python
"""
Complete ML pipeline with Dask for medium-to-large data.
"""

import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler
from dask_ml.linear_model import LogisticRegression
from dask_ml.metrics import accuracy_score
import xgboost as xgb

# 1. Start Dask cluster
print("Starting Dask cluster...")
client = Client(n_workers=4, threads_per_worker=2, memory_limit='8GB')
print(f"Dashboard: {client.dashboard_link}")

# 2. Load large dataset
print("\nLoading data...")
df = dd.read_parquet('s3://bucket/large_dataset.parquet')

print(f"Dataset shape: {df.shape[0].compute()} rows, {len(df.columns)} columns")

# 3. Data preprocessing
print("\nPreprocessing data...")

# Select features
feature_cols = [col for col in df.columns if col not in ['id', 'target']]
X = df[feature_cols]
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)

lr_score = lr.score(X_test_scaled, y_test)
print(f"Logistic Regression Accuracy: {lr_score:.4f}")

# 5. Train XGBoost (distributed)
print("\nTraining XGBoost (distributed)...")

dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
dtest = xgb.dask.DaskDMatrix(client, X_test, y_test)

params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'tree_method': 'hist'
}

output = xgb.dask.train(
    client,
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtest, 'test')]
)

xgb_model = output['booster']

# 6. Evaluate XGBoost
print("\nEvaluating XGBoost...")
predictions = xgb.dask.predict(client, xgb_model, X_test)

# Convert to pandas for sklearn metrics
y_test_pd = y_test.compute()
predictions_pd = predictions.compute()

from sklearn.metrics import classification_report
print("\nXGBoost Classification Report:")
print(classification_report(y_test_pd, (predictions_pd > 0.5).astype(int)))

# 7. Save model
print("\nSaving model...")
import joblib
joblib.dump(xgb_model, 'xgboost_model.joblib')

# 8. Close client
print("\nClosing Dask cluster...")
client.close()

print("\nPipeline complete!")
```

### End-to-End ML Pipeline with Spark

```python
"""
Complete ML pipeline with PySpark.
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# 1. Create Spark session
print("Creating Spark session...")
spark = SparkSession.builder \
    .appName("ML Pipeline") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# 2. Load data
print("\nLoading data...")
df = spark.read.parquet('hdfs:///user/data/large_dataset.parquet')

print(f"Dataset rows: {df.count()}")
df.printSchema()

# 3. Prepare features
print("\nPreparing features...")

feature_cols = [col for col in df.columns if col not in ['id', 'target']]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features_raw'
)

scaler = StandardScaler(
    inputCol='features_raw',
    outputCol='features'
)

# 4. Split data
train, test = df.randomSplit([0.8, 0.2], seed=42)

# 5. Train Random Forest
print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    featuresCol='features',
    labelCol='target',
    numTrees=100,
    maxDepth=10
)

rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
rf_model = rf_pipeline.fit(train)

# Evaluate
rf_predictions = rf_model.transform(test)

evaluator = BinaryClassificationEvaluator(
    labelCol='target',
    rawPredictionCol='rawPrediction',
    metricName='areaUnderROC'
)

rf_auc = evaluator.evaluate(rf_predictions)
print(f"Random Forest AUC-ROC: {rf_auc:.4f}")

# 6. Train Gradient Boosting
print("\nTraining Gradient Boosting...")

gbt = GBTClassifier(
    featuresCol='features',
    labelCol='target',
    maxIter=100,
    maxDepth=6
)

gbt_pipeline = Pipeline(stages=[assembler, scaler, gbt])
gbt_model = gbt_pipeline.fit(train)

# Evaluate
gbt_predictions = gbt_model.transform(test)
gbt_auc = evaluator.evaluate(gbt_predictions)
print(f"Gradient Boosting AUC-ROC: {gbt_auc:.4f}")

# 7. Save model
print("\nSaving model...")
gbt_model.write().overwrite().save('hdfs:///user/models/gbt_model')

# 8. Stop Spark
print("\nStopping Spark...")
spark.stop()

print("\nPipeline complete!")
```

---

## Summary

This comprehensive guide covered big data technologies for machine learning in 2025:

**Key Technologies:**

1. **Hadoop:** Foundational distributed storage (HDFS) and batch processing (MapReduce)
2. **Spark:** 10-100x faster than Hadoop, in-memory processing, mature ML library (MLlib)
3. **Dask:** 50% faster than Spark for Python workflows, seamless pandas/scikit-learn integration
4. **Ray:** General-purpose distributed computing, hyperparameter tuning, reinforcement learning
5. **Horovod:** Near-linear scaling for distributed deep learning

**2025 Benchmarks (from RESEARCH_SUMMARY_2025.md):**
- **Spark:** 10-100x faster than Hadoop MapReduce
- **Dask:** 50% faster than Spark on standard benchmarks
- **Horovod:** Near-linear scaling to 1000+ GPUs

**Key Recommendations:**

- **Data < 10GB:** Use pandas (single machine)
- **Data 10GB-1TB (Python ML):** Use Dask (faster, Python-native)
- **Data 10GB-1TB (General):** Use Spark (mature ecosystem)
- **Data > 1TB:** Use Spark or cloud-native (BigQuery, Databricks)
- **Deep Learning at Scale:** Use Horovod
- **Distributed Python:** Use Ray

**Cloud Trends:**
- Kubernetes deployment standard for all frameworks
- Serverless big data (BigQuery, Athena)
- ML platform consolidation (Databricks, SageMaker)
- Edge computing for real-time inference

Master these technologies to build scalable ML systems that handle big data efficiently!
