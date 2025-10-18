# Model Evaluation

## Overview

Model evaluation is a critical component of the machine learning lifecycle that determines whether a model is fit for deployment and real-world use. This section covers comprehensive evaluation strategies, from selecting appropriate metrics to ensuring fairness and interpretability in production systems.

**2025 Context:** With the EU AI Act and increasing regulatory requirements, model evaluation now extends beyond accuracy to include fairness, bias detection, interpretability, and compliance documentation.

---

## Section Contents

### 1. **46_Evaluation_Metrics.md**
Comprehensive guide to evaluation metrics across all ML domains:
- **Classification Metrics:** Accuracy, precision, recall, F1, ROC-AUC, PR curves
- **Regression Metrics:** MSE, RMSE, MAE, R², MAPE, quantile loss
- **Ranking Metrics:** NDCG, MRR, MAP
- **NLP Metrics:** BLEU, ROUGE, METEOR, perplexity
- **Computer Vision Metrics:** IoU, mAP, Dice coefficient
- Complete scikit-learn implementations
- When to use each metric

### 2. **47_Cross_Validation.md**
All cross-validation strategies for robust model evaluation:
- K-fold and Stratified K-fold
- Time series cross-validation (rolling/expanding window)
- Leave-one-out (LOO)
- Nested cross-validation for hyperparameter tuning
- Group K-fold for grouped data
- Custom CV strategies
- Statistical significance testing
- Best practices for different data types

### 3. **48_Model_Interpretability.md**
Making black-box models explainable:
- **Intrinsic Interpretability:** Linear models, decision trees
- **Post-hoc Methods:** SHAP, LIME, feature importance
- **Visualization:** Partial Dependence Plots, ICE plots
- **Deep Learning:** Grad-CAM, attention visualization, Integrated Gradients
- Regulatory compliance (GDPR right to explanation, EU AI Act)
- Production-ready implementations

### 4. **49_Fairness_and_Bias.md**
Detecting and mitigating bias in ML systems:
- Bias types and fairness metrics
- Demographic parity, equal opportunity, equalized odds
- Disparate impact analysis
- Pre-processing, in-processing, post-processing mitigation
- Fairlearn, AIF360, What-If Tool
- EU AI Act compliance (2025)
- Mandatory bias monitoring and documentation

---

## Why Model Evaluation Matters

### 1. **Model Selection**
- Compare multiple models objectively
- Select the best performer for your specific use case
- Understand trade-offs (accuracy vs. interpretability vs. latency)

### 2. **Generalization Assessment**
- Ensure model performs well on unseen data
- Detect overfitting and underfitting
- Validate robustness across data distributions

### 3. **Business Impact**
- Translate ML metrics to business KPIs
- Justify investment in ML systems
- Track improvement over time

### 4. **Regulatory Compliance (2025)**
- **EU AI Act:** High-risk AI systems require rigorous evaluation
- **GDPR Article 22:** Right to explanation for automated decisions
- **FDA/Medical:** Validation requirements for ML in healthcare
- **Financial Services:** Fair lending and discrimination laws

### 5. **Trust and Safety**
- Detect and mitigate bias
- Ensure fairness across demographic groups
- Build trustworthy AI systems

---

## The Complete Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────┐
│              1. Define Success Metrics                  │
│  • Business KPIs → ML metrics mapping                   │
│  • Stakeholder alignment                                │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              2. Choose Evaluation Strategy              │
│  • Cross-validation method                              │
│  • Train/validation/test split                          │
│  • Stratification for imbalanced data                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│           3. Compute Performance Metrics                │
│  • Classification: precision, recall, F1, ROC-AUC       │
│  • Regression: RMSE, MAE, R²                            │
│  • Domain-specific: BLEU, IoU, etc.                     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│            4. Assess Model Interpretability             │
│  • Feature importance (global)                          │
│  • SHAP values (local + global)                         │
│  • Partial dependence plots                             │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              5. Evaluate Fairness and Bias              │
│  • Demographic parity check                             │
│  • Equal opportunity assessment                         │
│  • Disparate impact analysis                            │
│  • Mitigation if needed                                 │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│             6. Statistical Significance                 │
│  • Confidence intervals (bootstrap)                     │
│  • Hypothesis testing (paired t-test)                   │
│  • Multiple comparison correction                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                7. Documentation & Reporting             │
│  • Model cards (Mitchell et al., 2019)                  │
│  • Datasheets for datasets                              │
│  • EU AI Act technical documentation                    │
└─────────────────────────────────────────────────────────┘
```

---

## Key Principles for 2025

### 1. **Beyond Accuracy**
Accuracy alone is insufficient and often misleading:
- Use precision/recall for imbalanced data
- Consider false positive vs. false negative costs
- Evaluate performance per subgroup (fairness)

### 2. **Robustness Testing**
Test model performance under adversarial conditions:
- Distribution shift (train vs. production)
- Edge cases and outliers
- Adversarial examples
- Data quality degradation

### 3. **Interpretability by Default**
- Start with interpretable models when possible
- Add complexity only when necessary
- Always provide explanations for predictions
- Document decision-making process

### 4. **Continuous Evaluation**
Models degrade over time due to data drift:
- Monitor performance in production
- Track prediction distribution shifts
- Set up automated retraining pipelines
- A/B test model updates

### 5. **Fairness as a Requirement**
Not optional in 2025:
- Mandatory for high-risk AI systems (EU AI Act)
- Legal requirements in finance, healthcare, hiring
- Reputational risk if bias discovered
- Include in initial requirements, not post-hoc

---

## Common Pitfalls

### 1. **Train/Test Leakage**
```python
# WRONG: Scaling before split
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# CORRECT: Fit on train, transform test
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform
```

### 2. **Wrong Metric for the Problem**
- Accuracy with 99% imbalanced data (meaningless)
- MSE when you care about outliers (use MAE or quantile loss)
- Macro-average when classes have different importance (use weighted)

### 3. **Ignoring Uncertainty**
- Point predictions without confidence intervals
- No calibration for probability estimates
- Overconfidence in edge cases

### 4. **Cherry-Picking Metrics**
- Reporting only favorable metrics
- Multiple testing without correction
- Overfitting to validation set

### 5. **Neglecting Fairness**
- Not checking for demographic disparities
- Assuming "fairness through unawareness"
- Ignoring intersectional bias

---

## Industry Standards (2025)

### Medical AI
- **FDA Validation:** Independent test set, subgroup analysis
- **Dice Score:** >0.85 for segmentation
- **Sensitivity:** >0.95 for cancer detection
- **Calibration:** Essential for clinical decision support

### Financial Services
- **Fair Lending:** Equal opportunity across protected groups
- **Model Risk Management:** SR 11-7 guidance (US)
- **Explainability:** Required for credit decisions
- **Stress Testing:** Performance under economic scenarios

### Autonomous Vehicles
- **Safety Metrics:** Mean time between failures
- **IoU:** >0.7 for object detection
- **Latency:** <100ms for real-time decisions
- **Edge Case Coverage:** 99.999% reliability

### NLP/LLMs
- **Toxicity Detection:** <1% false negative rate
- **Hallucination Rate:** Measured on benchmark datasets
- **Bias Testing:** Winogender, StereoSet, BBQ
- **Human Alignment:** RLHF with human preferences

---

## Tools and Libraries

### Core Evaluation
```python
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, mean_squared_error, r2_score
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, TimeSeriesSplit
)
```

### Interpretability
```python
import shap                    # SHAP values (state-of-the-art)
import lime                    # Local explanations
from sklearn.inspection import (
    permutation_importance,
    PartialDependenceDisplay
)
```

### Fairness
```python
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
```

### Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay
)
```

---

## Quick Start Example

```python
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
import shap

# 1. Define evaluation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Choose appropriate metric
scorer = make_scorer(f1_score, average='weighted')

# 3. Cross-validate
model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer)
print(f"F1 Score: {scores.mean():.3f} ± {scores.std():.3f}")

# 4. Train final model
model.fit(X_train, y_train)

# 5. Evaluate on test set
y_pred = model.predict(X_test)
test_f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Test F1 Score: {test_f1:.3f}")

# 6. Explain predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# 7. Check fairness
from fairlearn.metrics import demographic_parity_difference
dpd = demographic_parity_difference(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_feature
)
print(f"Demographic Parity Difference: {dpd:.3f}")
```

---

## Learning Path

### Beginner
1. Start with **46_Evaluation_Metrics.md** - understand all metrics
2. Practice **47_Cross_Validation.md** - proper validation techniques
3. Learn basic interpretability (feature importance, linear models)

### Intermediate
1. Master **48_Model_Interpretability.md** - SHAP, LIME
2. Implement custom CV strategies
3. Compare models with statistical tests

### Advanced
1. **49_Fairness_and_Bias.md** - comprehensive fairness evaluation
2. Multi-objective optimization (accuracy + fairness + latency)
3. Causal evaluation frameworks
4. Production monitoring pipelines

---

## Recent Advances (2025)

### Conformal Prediction
- Distribution-free uncertainty quantification
- Guaranteed coverage probabilities
- Works with any model (black-box)

### Causal Evaluation
- Beyond correlation to causation
- Counterfactual fairness
- Interventional robustness

### Foundation Model Evaluation
- Prompt-based evaluation for LLMs
- Few-shot and zero-shot metrics
- Human preference alignment (RLHF)

### Continuous Evaluation
- Real-time monitoring in production
- Automated drift detection
- Adaptive model updating

---

## References

### Foundational Papers
- Provost, F., & Fawcett, T. (2013). Data Science for Business
- Mitchell, M., et al. (2019). Model Cards for Model Reporting
- Mehrabi, N., et al. (2021). A Survey on Bias and Fairness in Machine Learning

### 2025 Best Practices
- EU AI Act (2024) - Technical Documentation Requirements
- NIST AI Risk Management Framework (2023)
- Google's Responsible AI Practices
- Microsoft's HAX Toolkit

### Libraries Documentation
- scikit-learn Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- SHAP: https://shap.readthedocs.io/
- Fairlearn: https://fairlearn.org/
- AIF360: https://aif360.readthedocs.io/

---

## Next Steps

After completing this section, you will be able to:
- ✅ Select appropriate metrics for any ML problem
- ✅ Implement robust cross-validation strategies
- ✅ Explain model predictions with SHAP/LIME
- ✅ Detect and mitigate bias in ML systems
- ✅ Comply with 2025 regulatory requirements
- ✅ Document models for production deployment

**Recommended Next:** 08_MLOps_and_Production for deploying evaluated models to production.

---

**Last Updated:** 2025-10-14
**Status:** Complete - PhD-level content with 2025 best practices
