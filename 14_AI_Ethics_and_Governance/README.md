# AI Ethics & Governance - Complete Compliance Guide

## Overview

Comprehensive guide to AI ethics, regulations, and compliance. **Legally required** under EU AI Act (2024) and GDPR.

**Coverage:**
- AI regulations (EU AI Act, GDPR, global)
- Bias detection and fairness metrics (20+ metrics)
- Privacy-preserving ML (Federated Learning, Differential Privacy)
- Model explainability (SHAP, LIME, counterfactuals)

---

## Files in This Folder

### 1. **01_AI_Regulations_and_Compliance.md**
- EU AI Act (full compliance by Aug 2, 2026)
- Risk-based classification (prohibited, high-risk, limited, minimal)
- GDPR integration (Article 5 principles, automated decisions)
- US regulations (AI Bill of Rights, state laws)
- China regulations (Algorithm Recommendation, Deep Synthesis, Generative AI)
- Compliance checklists for high-risk AI
- Maximum fines: EUR35M or 7% global turnover

### 2. **02_Bias_Detection_and_Fairness_Metrics.md**
- Types of bias (historical, representation, measurement, aggregation, label, deployment)
- 20+ fairness metrics with code:
  - Statistical Parity
  - Equal Opportunity
  - Equalized Odds
  - Calibration
  - Predictive Parity
  - Individual Fairness
- Comprehensive FairnessAuditor class
- Mitigation techniques (pre/in/post-processing)
- Production monitoring

### 3. **03_Privacy_Preserving_ML.md**
- Differential Privacy (epsilon-DP, Laplace/Gaussian mechanisms)
- DP-SGD for neural networks
- Federated Learning (FedAvg, secure aggregation)
- DP-FL (96.1% accuracy with epsilon=1.9 - 2024 research)
- Adaptive Local DP (ALDP-FL - 2025 SOTA)
- Homomorphic Encryption for encrypted inference
- Privacy-utility trade-off analysis

### 4. **04_Model_Explainability.md**
- SHAP (game theory-based, production implementation)
- LIME (model-agnostic explanations)
- Counterfactual explanations ("what if" scenarios)
- Attention visualizations (transformers)
- Grad-CAM (image saliency maps)
- EU AI Act Article 13 compliance (transparency requirements)
- Natural language explanation generation

---

## Quick Start

### For Compliance Officers

**High-Risk AI Checklist:**
1. Read `01_AI_Regulations_and_Compliance.md` (Article 9-17 requirements)
2. Implement risk management system
3. Ensure data governance (Article 10: bias-free, representative data)
4. Create technical documentation (Article 11)
5. Implement human oversight (Article 14)
6. Register in EU database

**Deadlines:**
- Feb 2, 2025: Prohibitions active
- Aug 2, 2025: GPAI obligations
- Aug 2, 2026: Full compliance

---

### For Data Scientists

**Fairness Implementation:**
```python
# 1. Detect bias
from fairness_auditor import FairnessAuditor
auditor = FairnessAuditor(y_true, y_pred, y_prob, protected_attributes)
report = auditor.generate_report()

# 2. Mitigate bias
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
mitigator = ExponentiatedGradient(base_model, DemographicParity())
mitigator.fit(X, y, sensitive_features=protected_attr)

# 3. Monitor in production
monitor = FairnessMonitor(thresholds)
violations = monitor.check_fairness(y_true, y_pred, y_prob, protected_attrs)
```

**Privacy Implementation:**
```python
# Differential Privacy
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(l2_norm_clip=1.0, noise_multiplier=1.1)

# Federated Learning with DP
dp_fl = DPFederatedLearning(global_model, epsilon=1.9, delta=1e-5)
global_model = dp_fl.dp_federated_averaging(client_models, clip_norm=1.0)
```

**Explainability:**
```python
# SHAP
import shap
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])

# EU AI Act compliant explanations
transparency_report = AITransparencyReport(model, explainer)
report = transparency_report.generate_user_explanation(instance, prediction)
```

---

### For ML Engineers

**Production Pipeline:**
1. **Training:** Apply DP-SGD or Federated Learning
2. **Evaluation:** Audit fairness across all protected attributes
3. **Deployment:** Enable SHAP/LIME explanations for all predictions
4. **Monitoring:** Continuous fairness and privacy monitoring
5. **Documentation:** Maintain EU AI Act Article 11 technical docs

---

## Key Regulations

### EU AI Act (2024)

**Risk Levels:**
- **Prohibited** (banned): Social scoring, real-time biometric ID, manipulation
- **High-Risk** (strict rules): Employment, credit, healthcare, law enforcement
- **Limited Risk** (transparency): Chatbots, deepfakes
- **Minimal Risk** (no rules): Video games, spam filters

**High-Risk Requirements:**
- Risk management system
- Data governance (bias-free, representative)
- Technical documentation
- Human oversight
- Accuracy, robustness, cybersecurity
- CE marking and registration

**Fines:**
- Prohibited AI: EUR35M or 7% turnover
- Non-compliance: EUR15M or 3% turnover

---

### GDPR

**Key Principles for AI:**
- **Lawfulness:** Legal basis for processing (Article 6)
- **Fairness:** No discrimination
- **Transparency:** Clear information to users (Article 13/14)
- **Data Minimization:** Only necessary data
- **Accuracy:** Ensure data and model accuracy
- **Automated Decisions:** Article 22 safeguards (right to human intervention)

---

## Fairness Metrics Comparison

| Metric | Definition | When to Use | Threshold |
|--------|------------|-------------|-----------|
| Statistical Parity | Equal positive rate | Equal outcomes desired | <10% diff |
| Equal Opportunity | Equal TPR | False negatives costly | <10% diff |
| Equalized Odds | Equal TPR & FPR | Both errors matter | <10% diff |
| Calibration | Predicted = actual | Probabilities important | <5% ECE diff |
| Predictive Parity | Equal precision | False positives costly | <10% diff |

**No single metric is perfect** - choose based on use case

---

## Privacy-Utility Trade-off

| Privacy Level | Epsilon (epsilon) | Typical Accuracy | Use Case |
|---------------|-------------|------------------|----------|
| Very Strong | epsilon < 0.5 | ~75-80% | Medical records |
| Strong | epsilon = 1.0 | ~88-92% | **Recommended** |
| Moderate | epsilon = 3-5 | ~93-95% | Financial data |
| Weak | epsilon > 10 | ~95-96% | Low sensitivity |

**2025 SOTA:** ALDP-FL achieves **96.1% accuracy with epsilon=1.9** (strong privacy)

---

## Explainability Methods Comparison

| Method | Type | Model Scope | Speed | Output |
|--------|------|-------------|-------|--------|
| SHAP | Post-hoc | Any | Slow | Feature importance |
| LIME | Post-hoc | Any | Fast | Local approximation |
| Attention | Built-in | Transformers | Fast | Token importance |
| Grad-CAM | Post-hoc | CNNs | Fast | Heatmap |
| Counterfactuals | Post-hoc | Any | Medium | "What if" scenarios |

**Best practice:** Use multiple methods for robust explanations

---

## Compliance Checklist

### EU AI Act (High-Risk AI)
- [ ] Risk management system (Article 9)
- [ ] Data governance - bias examination (Article 10)
- [ ] Technical documentation (Article 11)
- [ ] Record-keeping - automatic logs (Article 12)
- [ ] Transparency - user information (Article 13)
- [ ] Human oversight (Article 14)
- [ ] Accuracy, robustness, cybersecurity (Article 15)
- [ ] Quality management system (Article 17)
- [ ] Conformity assessment
- [ ] CE marking
- [ ] EU database registration
- [ ] Post-market monitoring (Article 61)

### GDPR (AI-Specific)
- [ ] Legal basis identified (Article 6)
- [ ] Data minimization applied
- [ ] Privacy notice provided (Article 13/14)
- [ ] Data subject rights enabled (Article 15-22)
- [ ] Special category data safeguards (Article 9)
- [ ] Automated decision safeguards (Article 22)
- [ ] DPIA conducted if high-risk (Article 35)
- [ ] Security measures (Article 32)

### Fairness
- [ ] Protected attributes identified
- [ ] Fairness metrics calculated
- [ ] Bias mitigation applied
- [ ] Continuous monitoring in place
- [ ] Documentation of fairness measures

### Privacy
- [ ] Privacy technique selected (DP/FL/HE)
- [ ] Privacy budget set (epsilon, delta)
- [ ] Privacy-utility trade-off evaluated
- [ ] Privacy monitoring implemented

### Explainability
- [ ] Explanation method implemented
- [ ] User-facing explanations available
- [ ] Natural language generation for non-technical users
- [ ] Human review option provided

---

## Resources

**Regulations:**
- EU AI Act: https://artificialintelligenceact.eu/
- GDPR: https://gdpr-info.eu/
- IAPP AI Governance: https://iapp.org/

**Tools:**
- Fairness: Fairlearn, AI Fairness 360
- Privacy: TensorFlow Privacy, PySyft, TenSEAL
- Explainability: SHAP, LIME, DiCE, Captum

**Papers (2024-2025):**
- ALDP-FL: Adaptive Local Differential Privacy
- Hybrid DP-FL: 96.1% accuracy with epsilon=1.9
- EU AI Act Technical Report (2024)

---

## Key Takeaways

1. **Compliance is mandatory** - EU AI Act + GDPR are enforceable law
2. **High-risk AI has strict requirements** - Documentation, oversight, fairness
3. **Multiple fairness metrics needed** - No single metric captures all aspects
4. **Privacy-utility trade-off exists** - epsilon=1.0 is sweet spot (strong privacy, good accuracy)
5. **Explainability is legally required** - GDPR Article 15, EU AI Act Article 13
6. **Continuous monitoring essential** - Fairness and privacy degrade over time
7. **Documentation is critical** - Prove compliance with detailed records

**Status:** Complete guide to AI ethics and governance with 2025 state-of-the-art methods
