# ML Encyclopedia

Personal reference library covering machine learning and data science topics I've needed while building production systems. Started this while working on my healthcare AI capstone and kept adding to it as I learned new techniques.

**~100 markdown files covering foundations through 2025 methods**

## What's In Here

### Core Stuff
- **Data Analysis** - EDA, cleaning, feature engineering basics
- **Statistics** - Probability, A/B testing, causal inference
- **Classical ML** - Regression, trees, ensembles, clustering
- **Model Evaluation** - Metrics, cross-validation, SHAP/LIME

### Deep Learning
- **Fundamentals** - Neural networks, backprop, optimization
- **Computer Vision** - CNNs, ResNets, ViTs, object detection
- **NLP** - Transformers, LLMs, fine-tuning with LoRA
- **Generative Models** - VAEs, GANs, diffusion models

### Production & Advanced
- **MLOps** - Deployment, monitoring, Docker, FastAPI
- **Data Engineering** - Big data, imbalanced data, preprocessing
- **Advanced Topics** - Transfer learning, RL, Graph NNs, time series
- **2025 Stuff** - Multimodal LLMs, agentic AI, edge deployment

### Specialized
- **Kaggle Strategies** - Competition techniques I used for RSNA
- **Ethics & Compliance** - EU AI Act, fairness, privacy
- **Quick References** - Cheat sheets and decision trees

## How I Use This

When starting a new project, I reference the relevant sections instead of Googling the same stuff over and over. For example:
- Building Apollo Healthcare → Computer Vision + MLOps + Ethics
- RSNA Kaggle → Classical ML + Model Evaluation + Kaggle Strategies
- Business Analytics Platform → Data Engineering + Production Deployment

## Organization

```
ML_Encyclopedia_Complete_2025/
├── 00_Data_Analysis_Fundamentals/
├── 01_Statistical_Foundations/
├── 02_Classical_Machine_Learning/
├── 03_Deep_Learning_Fundamentals/
├── 04_Computer_Vision_CNNs/
├── 05_NLP_and_Transformers/
├── 06_Generative_Models/
├── 07_Advanced_Topics/
├── 08_MLOps_and_Production/
├── 09_Data_Engineering/
├── 10_Model_Evaluation/
├── 11_Kaggle_and_Competitions/
├── 12_Cutting_Edge_2025/
├── 13_ARC_and_Problem_Solving/
└── 14_AI_Ethics_and_Governance/
```

## Tech Stack Covered

**Frameworks:** PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM, CatBoost
**Libraries:** Hugging Face Transformers, pandas, NumPy, Dask
**Tools:** Docker, FastAPI, MLflow, Weights & Biases
**Deployment:** Kubernetes, Apache Airflow, cloud platforms

## Useful Sections by Role

**Data Analyst:**
- Start with Data Analysis Fundamentals
- Move to Statistical Foundations (A/B testing)
- Check Model Evaluation for metrics

**ML Engineer:**
- Focus on MLOps and Production
- Review Model Evaluation (monitoring)
- Check Data Engineering for pipelines

**Researcher:**
- Explore Cutting Edge 2025
- Check Advanced Topics
- Review Ethics and Governance

## Best Practices I Follow

From experience building production systems:
- Use AdamW for optimization (not vanilla Adam)
- Enable mixed precision training (FP16)
- Handle imbalanced data with SMOTE + proper metrics
- Track experiments (MLflow or W&B)
- Deploy with Docker containers
- Monitor for data drift

## Why Public?

Making this public because:
1. Might help others learning ML
2. Shows how I organize knowledge
3. Forces me to keep it clean and updated
4. Good reference when I'm on different machines

Feel free to use anything here. If you find errors or have suggestions, let me know.

---

**Built while working on:** Apollo Healthcare AI, RSNA Kaggle Competition, Business Analytics Platform, and various consulting projects.
