# Model Explainability & Interpretability - SHAP, LIME & Beyond

## Overview

**Explainability is legally required:** GDPR Article 15 (right to explanation), EU AI Act Article 13 (transparency), and regulatory scrutiny demand interpretable AI.

---

## SHAP (SHapley Additive exPlanations)

### Game Theory-Based Explanations

```python
import shap
import xgboost

# Train model
model = xgboost.XGBClassifier().fit(X_train, y_train)

# Create explainer
explainer = shap.Explainer(model)

# Compute SHAP values
shap_values = explainer(X_test)

# Visualize
shap.plots.waterfall(shap_values[0])  # Single prediction
shap.plots.beeswarm(shap_values)  # All predictions
shap.summary_plot(shap_values, X_test)  # Feature importance
```

### Production Implementation

```python
class SHAPExplainer:
    """Production-ready SHAP explanations"""

    def __init__(self, model, background_data):
        self.model = model
        self.explainer = shap.Explainer(model, background_data)

    def explain_prediction(self, instance):
        """Explain single prediction"""
        shap_values = self.explainer(instance)

        explanation = {
            'prediction': self.model.predict(instance)[0],
            'base_value': self.explainer.expected_value,
            'feature_contributions': dict(zip(
                instance.columns,
                shap_values.values[0]
            )),
            'top_3_features': self.get_top_features(shap_values, n=3)
        }

        return explanation

    def get_top_features(self, shap_values, n=3):
        """Get most important features"""
        feature_importance = abs(shap_values.values[0])
        top_indices = feature_importance.argsort()[-n:][::-1]

        return [
            {
                'feature': shap_values.feature_names[i],
                'shap_value': shap_values.values[0][i],
                'feature_value': shap_values.data[0][i]
            }
            for i in top_indices
        ]

# Usage
explainer = SHAPExplainer(model, X_train.sample(100))
explanation = explainer.explain_prediction(X_test.iloc[[0]])
print(f"Prediction: {explanation['prediction']}")
print(f"Top 3 features: {explanation['top_3_features']}")
```

---

## LIME (Local Interpretable Model-agnostic Explanations)

### Explain Any Model

```python
from lime import lime_tabular

# Create explainer
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['Negative', 'Positive'],
    mode='classification'
)

# Explain prediction
exp = lime_explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

exp.show_in_notebook()  # Visualization
exp.as_list()  # Feature contributions
```

---

## Model-Specific Interpretability

### Decision Trees (Inherently Interpretable)

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

dt = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X_train.columns, class_names=['0', '1'], filled=True)
plt.show()

# Extract rules
from sklearn.tree import export_text
rules = export_text(dt, feature_names=list(X_train.columns))
print(rules)
```

### Linear Models (Coefficient Interpretation)

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(X_train, y_train)

# Coefficients = feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print(feature_importance)
```

---

## Counterfactual Explanations

### "What would need to change for a different outcome?"

```python
from dice_ml import Data, Model, Dice

# Setup
d = Data(dataframe=df, continuous_features=cont_features, outcome_name='target')
m = Model(model=model, backend='sklearn')
exp = Dice(d, m)

# Generate counterfactuals
cf = exp.generate_counterfactuals(
    query_instance=X_test.iloc[0],
    total_CFs=3,
    desired_class='opposite'
)

cf.visualize_as_dataframe()

"""
Original: Income=$50K → Loan Denied
Counterfactual: If income was $65K (+$15K), loan would be approved
"""
```

---

## Attention Visualizations (For Deep Learning)

### Transformer Attention

```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('bert-base-uncased', output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer("This model explains itself", return_tensors='pt')
outputs = model(**inputs)

# Get attention weights
attention = outputs.attentions  # List of attention matrices per layer

# Visualize attention from last layer
import seaborn as sns
sns.heatmap(attention[-1][0][0].detach().numpy(), xticklabels=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]), yticklabels=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
```

---

## Saliency Maps (For Images)

### Grad-CAM

```python
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# Usage
model = ResNet50(weights='imagenet')
heatmap = make_gradcam_heatmap(img_array, model, 'conv5_block3_out')
```

---

## EU AI Act Compliance

### Article 13: Transparency Requirements

```python
class AITransparencyReport:
    """Generate EU AI Act compliant explanations"""

    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer

    def generate_user_explanation(self, instance, prediction):
        """Article 13(3)(b): Information to users"""

        # Get SHAP explanation
        shap_values = self.explainer(instance)
        top_features = self.get_top_features(shap_values, n=3)

        report = {
            'decision': 'Approved' if prediction == 1 else 'Denied',
            'confidence': self.model.predict_proba(instance)[0][prediction],
            'main_factors': [
                {
                    'factor': f['feature'],
                    'contribution': 'Positive' if f['shap_value'] > 0 else 'Negative',
                    'magnitude': abs(f['shap_value'])
                }
                for f in top_features
            ],
            'human_review_available': True,
            'contact': 'ai-review@company.com'
        }

        # Generate natural language explanation
        explanation_text = self.generate_natural_language(report)

        report['explanation'] = explanation_text
        return report

    def generate_natural_language(self, report):
        """Convert technical explanation to human-readable text"""

        decision = report['decision']
        main_factor = report['main_factors'][0]

        text = f"Your application was {decision}. "
        text += f"The main factor in this decision was your {main_factor['factor']}, "
        text += f"which had a {main_factor['contribution'].lower()} impact. "
        text += f"You have the right to request human review by contacting {report['contact']}."

        return text

# Usage
transparency_report = AITransparencyReport(model, shap_explainer)
report = transparency_report.generate_user_explanation(instance, prediction)
print(report['explanation'])
```

---

## Best Practices

1. **Use multiple methods** - SHAP + LIME + domain-specific
2. **Validate explanations** - Ensure they make sense
3. **Consider audience** - Technical vs non-technical explanations
4. **Document limitations** - Explanations are approximations
5. **Enable human review** - Always provide override option
6. **Monitor explanation quality** - Track user feedback

**Key Takeaway:** Explainability is not optional - it's a legal requirement under EU AI Act and GDPR.
