# ML Encyclopedia

Personal reference library covering machine learning and data science topics I've needed while building production systems. Started this while working on my healthcare AI capstone and kept adding to it as I learned new techniques.

**86 markdown files | 121K+ lines | Foundations through 2025 methods**

Designed to work as a RAG knowledge base -- every file is dense with specific model names, hyperparameters, production-ready code, and concrete techniques.

---

## Table of Contents

- [What's In Here](#whats-in-here)
- [Quick Start by Task](#quick-start-by-task)
- [Directory Structure](#directory-structure)
- [Section Details](#section-details)
- [Tech Stack Covered](#tech-stack-covered)
- [Useful Sections by Role](#useful-sections-by-role)
- [How I Use This](#how-i-use-this)
- [Best Practices](#best-practices)

---

## What's In Here

### Foundations
- **Data Analysis** (4 files) - EDA, data cleaning, feature scaling/encoding, advanced feature engineering
- **Statistics** (7 files) - Probability distributions, inference, hypothesis testing, Bayesian stats, A/B testing, causal inference, statistical tests guide
- **Classical ML** (7 files) - Linear regression, trees, SVMs, ensembles, clustering, dimensionality reduction, anomaly detection

### Deep Learning
- **Fundamentals** (5 files) - Neural network basics, activation functions, optimization, regularization, training strategies
- **Computer Vision** (4 files) - CNN fundamentals, architectures (ResNet/EfficientNet/ViT), object detection and segmentation, 3D vision and medical imaging
- **NLP** (8 files) - NLP fundamentals, embeddings, RNNs, attention, transformers, LLMs, complete 2025 guide, information retrieval and RAG
- **Generative Models** (5 files) - Autoencoders, VAEs, GANs, diffusion models, complete 2025 guide

### Production and Advanced
- **Advanced Topics** (7 files) - Transfer learning, meta-learning, reinforcement learning, graph NNs, time series, deep anomaly detection, multi-task and continual learning
- **MLOps** (5 files) - Deployment, monitoring/drift, pipeline orchestration, AutoML/NAS, experiment tracking
- **Data Engineering** (4 files) - Preprocessing, augmentation, imbalanced data, big data technologies
- **Model Evaluation** (4 files) - Metrics, cross-validation, interpretability (SHAP/LIME), fairness and bias

### Competition and Specialized
- **Kaggle and Competitions** (6 files) - Winning strategies, cross-validation, CV/NLP/time series/tabular competition solutions with winning architectures
- **Cutting Edge 2025** (7 files) - Multimodal LLMs, agentic AI, small language models, edge AI, AutoML, prompt engineering and LLM APIs
- **ARC and Problem Solving** (5 files) - ARC challenge, test-time training, LLM reasoning, hybrid ensembles, problem solving strategies
- **Ethics and Governance** (4 files) - AI regulations (EU AI Act), bias detection, privacy-preserving ML, explainability

### Reference Guides
- **Master Guide** (3 files) - Model selection decision tree, Kaggle competition playbook, production ML checklist

---

## Quick Start by Task

| I need to... | Start here |
|---|---|
| Classify images | `04_Computer_Vision_CNNs/` then `11_Kaggle/03_CV_Competition_Solutions.md` |
| Fine-tune an LLM | `05_NLP/26_Large_Language_Models.md` then `12_Cutting_Edge/06_Prompt_Engineering.md` |
| Build a RAG system | `05_NLP/28_Information_Retrieval_and_RAG.md` |
| Win a Kaggle competition | `00_Master_Guide/02_Kaggle_Playbook.md` then the competition-specific file |
| Deploy a model | `08_MLOps/36_Model_Deployment_2025.md` then `00_Master_Guide/03_Production_ML_Checklist.md` |
| Handle tabular data | `02_Classical_ML/09_Ensemble_Methods.md` then `11_Kaggle/06_Tabular_Solutions.md` |
| Detect anomalies | `02_Classical_ML/12_Anomaly_Detection.md` then `07_Advanced/36_Deep_Anomaly_Detection.md` |
| Choose the right model | `00_Master_Guide/12_Model_Selection_Decision_Tree.md` |
| Run A/B tests | `01_Statistics/05_AB_Testing.md` |
| Understand transformers | `05_NLP/25_Transformer_Architecture.md` |
| Work with medical images | `04_Computer_Vision_CNNs/04_3D_Vision_and_Medical_Imaging.md` |
| Train RL agents | `07_Advanced/33_Reinforcement_Learning.md` |

---

## Directory Structure

```
Personal_ML_Encyclopedia_Reference_Guide/
+-- 00_Data_Analysis_Fundamentals/
|   +-- 01_Exploratory_Data_Analysis.md
|   +-- 02_Data_Cleaning.md
|   +-- 03_Feature_Scaling_and_Encoding.md
|   +-- 04_Advanced_Feature_Engineering.md
|
+-- 00_Master_Guide/
|   +-- 02_Kaggle_Playbook.md
|   +-- 03_Production_ML_Checklist.md
|   +-- 12_Model_Selection_Decision_Tree.md
|
+-- 01_Statistical_Foundations/
|   +-- 01_Probability_Distributions.md
|   +-- 02_Statistical_Inference.md
|   +-- 03_Hypothesis_Testing.md
|   +-- 04_Bayesian_Statistics.md
|   +-- 05_AB_Testing.md
|   +-- 06_Causal_Inference.md
|   +-- 07_Statistical_Tests_Guide.md
|
+-- 02_Classical_Machine_Learning/
|   +-- 06_Linear_Regression.md
|   +-- 07_Tree_Based_Models.md
|   +-- 08_Linear_Models_and_SVMs.md
|   +-- 09_Ensemble_Methods.md
|   +-- 10_Clustering.md
|   +-- 11_Dimensionality_Reduction.md
|   +-- 12_Anomaly_Detection.md
|
+-- 03_Deep_Learning_Fundamentals/
|   +-- 09_Deep_Learning_When_To_Use.md
|   +-- 13_Neural_Network_Basics.md
|   +-- 14_Activation_Functions.md
|   +-- 15_Optimization.md
|   +-- 16_Regularization.md
|   +-- 17_Training_Strategies.md
|
+-- 04_Computer_Vision_CNNs/
|   +-- 01_CNN_Fundamentals.md
|   +-- 02_CNN_Architectures.md
|   +-- 03_Object_Detection_Segmentation.md
|   +-- 04_3D_Vision_and_Medical_Imaging.md
|
+-- 05_NLP_and_Transformers/
|   +-- 21_NLP_Fundamentals.md
|   +-- 22_Word_Embeddings.md
|   +-- 23_Recurrent_Neural_Networks.md
|   +-- 24_Attention_Mechanisms.md
|   +-- 25_Transformer_Architecture.md
|   +-- 26_Large_Language_Models.md
|   +-- 27_Complete_NLP_Transformers_Guide_2025.md
|   +-- 28_Information_Retrieval_and_RAG.md
|
+-- 06_Generative_Models/
|   +-- 27_Autoencoders.md
|   +-- 28_VAE.md
|   +-- 29_GANs.md
|   +-- 30_Diffusion_Models.md
|   +-- 31_Complete_Generative_Models_Guide_2025.md
|
+-- 07_Advanced_Topics/
|   +-- 31_Transfer_Learning.md
|   +-- 32_Meta_Learning.md
|   +-- 33_Reinforcement_Learning.md
|   +-- 34_Graph_Neural_Networks.md
|   +-- 35_Time_Series_Deep_Learning.md
|   +-- 36_Anomaly_Detection_Deep_Learning.md
|   +-- 37_Multi_Task_and_Continual_Learning.md
|
+-- 08_MLOps_and_Production/
|   +-- 36_Model_Deployment_2025.md
|   +-- 37_Monitoring_and_Drift_Detection.md
|   +-- 38_ML_Pipeline_Orchestration.md
|   +-- 39_AutoML_NAS.md
|   +-- 40_Experiment_Tracking.md
|
+-- 09_Data_Engineering/
|   +-- 41_Data_Preprocessing.md
|   +-- 43_Data_Augmentation.md
|   +-- 44_Handling_Imbalanced_Data.md
|   +-- 45_Big_Data_Technologies.md
|
+-- 10_Model_Evaluation/
|   +-- 46_Evaluation_Metrics.md
|   +-- 47_Cross_Validation.md
|   +-- 48_Model_Interpretability.md
|   +-- 49_Fairness_and_Bias.md
|
+-- 11_Kaggle_and_Competitions/
|   +-- 01_Kaggle_Winning_Strategies.md
|   +-- 02_Cross_Validation_and_Leakage.md
|   +-- 03_Computer_Vision_Competition_Solutions.md
|   +-- 04_NLP_Competition_Solutions.md
|   +-- 05_Time_Series_Competition_Solutions.md
|   +-- 06_Tabular_Competition_Solutions.md
|
+-- 12_Cutting_Edge_2025/
|   +-- 01_Multimodal_LLMs_2025.md
|   +-- 02_Agentic_AI.md
|   +-- 03_Small_Language_Models.md
|   +-- 04_Edge_AI.md
|   +-- 05_AutoML_and_Neural_Architecture_Search.md
|   +-- 06_Prompt_Engineering_and_LLM_APIs.md
|   +-- Advanced_Pseudo_Labeling.md
|
+-- 13_ARC_and_Problem_Solving/
|   +-- 01_ARC_Challenge_Overview.md
|   +-- 02_Test_Time_Training.md
|   +-- 03_LLM_Based_Reasoning.md
|   +-- 04_Hybrid_Ensembles.md
|   +-- 05_General_Problem_Solving_Strategies.md
|
+-- 14_AI_Ethics_and_Governance/
    +-- 01_AI_Regulations_and_Compliance.md
    +-- 02_Bias_Detection_and_Fairness_Metrics.md
    +-- 03_Privacy_Preserving_ML.md
    +-- 04_Model_Explainability.md
```

---

## Section Details

### 00 - Data Analysis Fundamentals
EDA workflows, missing value strategies, encoding categorical variables (target encoding, frequency encoding, one-hot), feature scaling (StandardScaler, RobustScaler, MinMax), and advanced feature engineering (polynomial features, interaction terms, aggregation features).

### 01 - Statistical Foundations
Probability distributions with scipy implementations, confidence intervals, p-values, Bayesian inference with PyMC, proper A/B test design with sample size calculations, causal inference (DiD, IV, RDD), and a complete statistical tests selection guide.

### 02 - Classical Machine Learning
Linear/logistic regression with regularization, decision trees and random forests, SVMs with kernel tricks, gradient boosting (XGBoost, LightGBM, CatBoost with tuning recipes), clustering (K-Means, DBSCAN, hierarchical), PCA/t-SNE/UMAP, and anomaly detection (Isolation Forest, LOF, One-Class SVM, statistical methods).

### 03 - Deep Learning Fundamentals
When to use deep learning, neural network architecture design, all activation functions compared, optimization (SGD, Adam, AdamW, learning rate scheduling), regularization (dropout, batch norm, weight decay, early stopping), and training strategies (mixed precision, gradient accumulation, distributed training).

### 04 - Computer Vision
CNN operations and architectures (LeNet through ConvNeXt and ViT), object detection (YOLO, Faster R-CNN, DETR), semantic/instance/panoptic segmentation (U-Net, Mask R-CNN, DeepLabv3+), 3D vision and medical imaging (DICOM/NIfTI processing, 3D U-Net, nnU-Net, MONAI, PointNet, radiomics, FDA regulatory considerations).

### 05 - NLP and Transformers
Text preprocessing, Word2Vec/GloVe/FastText, LSTM/GRU architectures, self-attention and multi-head attention, transformer architecture from scratch, LLM landscape (GPT, BERT, T5, LLaMA), fine-tuning with LoRA/QLoRA, and a complete guide to information retrieval, RAG architecture, vector databases, and chunking strategies.

### 06 - Generative Models
Autoencoders (vanilla, denoising, sparse), VAEs with ELBO derivation, GANs (DCGAN, StyleGAN, conditional GANs), and diffusion models (DDPM, score-based, stable diffusion).

### 07 - Advanced Topics
Transfer learning strategies, meta-learning (MAML, prototypical networks), comprehensive reinforcement learning (MDP through PPO/SAC/DPO, RLHF for LLM alignment, multi-agent RL, offline RL), graph neural networks (GCN, GAT, GraphSAGE), time series deep learning (N-BEATS, TFT, PatchTST), deep anomaly detection (autoencoders, GANomaly, transformers for AD), and multi-task/continual learning (EWC, replay methods, progressive networks).

### 08 - MLOps and Production
Model deployment with FastAPI/Docker/Kubernetes, monitoring and drift detection (PSI, KS test, Evidently), ML pipeline orchestration (Airflow, Prefect, Kubeflow), AutoML and neural architecture search, and experiment tracking (MLflow, W&B).

### 09 - Data Engineering
Data preprocessing pipelines, augmentation techniques (image, text, tabular), handling imbalanced data (SMOTE, class weights, focal loss), and big data technologies (Spark, Dask, distributed computing).

### 10 - Model Evaluation
Classification/regression/ranking metrics, cross-validation strategies (stratified, group, time series), model interpretability (SHAP, LIME, Grad-CAM, feature importance), and fairness metrics (demographic parity, equalized odds, calibration).

### 11 - Kaggle and Competitions
General winning strategies, cross-validation and leakage prevention, plus detailed competition solutions with winning architectures and code for computer vision (EfficientNet ensembles, ArcFace, TTA), NLP (DeBERTa fine-tuning, AWP, pseudo-labeling), time series (LightGBM lag features, hierarchical forecasting, financial tricks), and tabular (XGBoost/LightGBM/CatBoost tuning, stacking, feature engineering, winning solution breakdowns).

### 12 - Cutting Edge 2025
Multimodal LLMs (GPT-4V, Gemini, Claude vision), agentic AI patterns (ReAct, tool use, planning), small language models for edge deployment, edge AI optimization (quantization, pruning, distillation), AutoML/NAS, and prompt engineering with LLM API patterns (Claude, OpenAI, Gemini APIs with production code).

### 13 - ARC and Problem Solving
ARC challenge approaches, test-time training, LLM-based reasoning strategies, hybrid ensemble methods, and general algorithmic problem-solving frameworks.

### 14 - AI Ethics and Governance
EU AI Act compliance, bias detection and fairness metrics, privacy-preserving ML (federated learning, differential privacy), and model explainability requirements.

---

## Tech Stack Covered

**ML Frameworks:** PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM, CatBoost
**Deep Learning:** Hugging Face Transformers, timm, segmentation-models-pytorch
**Data:** pandas, NumPy, Dask, PySpark, Polars
**NLP:** sentence-transformers, tokenizers, FAISS, ChromaDB
**MLOps:** Docker, FastAPI, MLflow, Weights & Biases, Airflow
**Deployment:** Kubernetes, ONNX Runtime, TensorRT, vLLM
**Visualization:** Matplotlib, Seaborn, Plotly

---

## Useful Sections by Role

**Data Analyst:**
- Start with `00_Data_Analysis_Fundamentals/`
- Move to `01_Statistical_Foundations/` (A/B testing, hypothesis tests)
- Check `10_Model_Evaluation/` for metrics

**ML Engineer:**
- Focus on `08_MLOps_and_Production/`
- Review `00_Master_Guide/03_Production_ML_Checklist.md`
- Check `09_Data_Engineering/` for pipelines

**Data Scientist:**
- Start with `02_Classical_Machine_Learning/`
- Move to `03_Deep_Learning_Fundamentals/`
- Use `00_Master_Guide/12_Model_Selection_Decision_Tree.md` for model choice

**Competition Player:**
- Start with `00_Master_Guide/02_Kaggle_Playbook.md`
- Pick domain: `11_Kaggle/03_CV.md`, `04_NLP.md`, or `05_Time_Series.md`
- Review `10_Model_Evaluation/47_Cross_Validation.md`

**Researcher:**
- Explore `12_Cutting_Edge_2025/`
- Check `07_Advanced_Topics/`
- Review `14_AI_Ethics_and_Governance/`

---

## How I Use This

When starting a new project, I reference the relevant sections instead of Googling the same stuff over and over. For example:
- Building Apollo Healthcare -> Computer Vision + MLOps + Ethics
- RSNA Kaggle -> CV Competition Solutions + Model Evaluation + Kaggle Playbook
- Business Analytics Platform -> Data Engineering + Production Deployment
- RAG System -> Information Retrieval and RAG + Prompt Engineering

---

## Best Practices

From experience building production systems:
- Use AdamW for optimization (not vanilla Adam)
- Enable mixed precision training (FP16/BF16)
- Handle imbalanced data with SMOTE + proper metrics (PR-AUC, not just accuracy)
- Track experiments with MLflow or W&B
- Deploy with Docker containers behind FastAPI
- Monitor for data drift with PSI and KS tests
- Use stratified cross-validation, never random splits
- Ensemble diverse models for competitions
- Start simple, add complexity only when needed

---

## Why Public?

Making this public because:
1. Might help others learning ML
2. Shows how I organize knowledge
3. Forces me to keep it clean and updated
4. Good reference when I'm on different machines
5. Serves as a knowledge base for RAG systems

Feel free to use anything here. If you find errors or have suggestions, let me know.

---

**Built while working on:** Apollo Healthcare AI, RSNA Kaggle Competition, Business Analytics Platform, and various consulting projects.
