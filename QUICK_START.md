# Quick Start

## How to Use This

Pick what you need based on what you're doing:

### Building a New Project

**Computer Vision:**
1. Start: `04_Computer_Vision_CNNs/`
2. Then: `09_Data_Engineering/` (augmentation)
3. Deploy: `08_MLOps_and_Production/`

**NLP/Text:**
1. Start: `05_NLP_and_Transformers/`
2. Fine-tuning: Check LoRA/QLoRA sections
3. Deploy: `08_MLOps_and_Production/`

**Classical ML:**
1. Start: `02_Classical_Machine_Learning/`
2. Evaluation: `10_Model_Evaluation/`
3. Tuning: Check hyperparameter sections

### Kaggle Competition

1. **Strategy:** `11_Kaggle_and_Competitions/`
2. **Techniques:**
   - Ensembling methods
   - Cross-validation strategies
   - Pseudo-labeling (if limited labels)
3. **Evaluation:** `10_Model_Evaluation/` for metrics

### Production Deployment

1. **MLOps:** `08_MLOps_and_Production/`
2. **Monitoring:** Drift detection, model evaluation
3. **Ethics:** `14_AI_Ethics_and_Governance/` for compliance

### Learning Path

**Beginner:**
- Data Analysis Fundamentals
- Statistical Foundations
- Classical ML

**Intermediate:**
- Deep Learning Fundamentals
- Computer Vision OR NLP (pick one)
- Model Evaluation

**Advanced:**
- Advanced Topics (RL, Graph NNs, etc.)
- Cutting Edge 2025
- MLOps

## Quick Reference by Problem

| Problem | Check Here |
|---------|-----------|
| Class imbalance | `09_Data_Engineering/` (SMOTE, focal loss) |
| Limited training data | `12_Cutting_Edge_2025/` (pseudo-labeling) |
| Model too slow | `08_MLOps_and_Production/` (optimization) |
| Need explanations | `10_Model_Evaluation/` (SHAP, LIME) |
| A/B testing | `01_Statistical_Foundations/` |
| Time series | `07_Advanced_Topics/` |
| Object detection | `04_Computer_Vision_CNNs/` |
| Text classification | `05_NLP_and_Transformers/` |

## Folder Overview

```
00_Data_Analysis_Fundamentals/  → EDA, cleaning, features
01_Statistical_Foundations/      → Stats, A/B testing, causal
02_Classical_Machine_Learning/   → Regression, trees, ensembles
03_Deep_Learning_Fundamentals/   → Neural nets, optimization
04_Computer_Vision_CNNs/         → CNNs, detection, segmentation
05_NLP_and_Transformers/         → NLP, LLMs, transformers
06_Generative_Models/            → VAE, GAN, diffusion
07_Advanced_Topics/              → Transfer, RL, graphs, time series
08_MLOps_and_Production/         → Deploy, monitor, pipelines
09_Data_Engineering/             → Preprocessing, big data
10_Model_Evaluation/             → Metrics, interpretability
11_Kaggle_and_Competitions/      → Competition strategies
12_Cutting_Edge_2025/            → Latest techniques
13_ARC_and_Problem_Solving/      → Abstract reasoning
14_AI_Ethics_and_Governance/     → Compliance, fairness
```

## Tech Stack

**Primary:** Python, PyTorch, scikit-learn, pandas, NumPy
**Specialized:** Hugging Face, XGBoost, Docker, MLflow, FastAPI

Check specific folders for detailed library usage.

---

**Pro tip:** Use Ctrl+F to search across files for specific topics.
