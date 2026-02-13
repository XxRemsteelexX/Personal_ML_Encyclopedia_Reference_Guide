# Classical Machine Learning

## Overview

This section covers fundamental machine learning algorithms that form the foundation of modern ML practice. These methods remain essential in 2025 for their interpretability, efficiency, and strong performance on structured/tabular data.

## Contents

### 1. Linear Models
- **06_Linear_Regression.md** - Linear models for regression tasks
  - Simple and multiple linear regression
  - OLS solution and statistical assumptions
  - Regularization techniques (Ridge, Lasso, Elastic Net)
  - Model diagnostics and validation

### 2. Ensemble Methods
- **09_Ensemble_Methods.md** - Combining multiple models for superior performance
  - Bagging and Random Forests
  - Gradient Boosting (XGBoost, LightGBM, CatBoost)
  - Stacking and blending strategies
  - 2025 benchmark comparisons

### 3. Unsupervised Learning
- **10_Clustering.md** - Discovering patterns in unlabeled data
  - K-Means and variants
  - Hierarchical clustering
  - Density-based methods (DBSCAN)
  - Gaussian Mixture Models

- **11_Dimensionality_Reduction.md** - Reducing feature space complexity
  - PCA for linear reduction
  - t-SNE and UMAP for visualization
  - Comparison of modern techniques

## Learning Path

### For Beginners
1. Start with **06_Linear_Regression.md** to understand fundamental ML concepts
2. Move to **10_Clustering.md** for unsupervised learning basics
3. Study **11_Dimensionality_Reduction.md** for data preprocessing
4. Master **09_Ensemble_Methods.md** for competition-grade models

### For Practitioners
1. Review **09_Ensemble_Methods.md** for 2025 SOTA tabular methods
2. Use **11_Dimensionality_Reduction.md** for feature engineering
3. Apply **10_Clustering.md** for customer segmentation and anomaly detection
4. Reference **06_Linear_Regression.md** for baseline models and interpretability

### For Researchers
1. Deep dive into theoretical foundations in each file
2. Compare mathematical derivations with modern optimizations
3. Benchmark implementations against 2025 research findings
4. Explore connections to deep learning methods

## Key Concepts

### Supervised Learning
- **Regression**: Predicting continuous values
- **Classification**: Predicting discrete categories (covered in other sections)
- **Regularization**: Preventing overfitting through penalty terms
- **Ensemble Methods**: Combining models for better performance

### Unsupervised Learning
- **Clustering**: Grouping similar data points
- **Dimensionality Reduction**: Compressing high-dimensional data
- **Anomaly Detection**: Identifying outliers and unusual patterns

## 2025 Best Practices

### Model Selection for Tabular Data
1. **Baseline**: Start with linear regression or logistic regression
2. **Production**: Use LightGBM or CatBoost for best speed/accuracy tradeoff
3. **Competitions**: Stack XGBoost, LightGBM, and CatBoost models
4. **Interpretability**: Use linear models with SHAP explanations

### Preprocessing Pipeline
1. **Missing Values**: Handle before dimensionality reduction
2. **Scaling**: Required for PCA, optional for tree-based models
3. **Encoding**: Target encoding outperforms one-hot for high-cardinality features
4. **Feature Engineering**: Domain knowledge > automatic feature generation

### Validation Strategy
1. **Time Series**: Always use time-based splits
2. **Small Datasets (<10K rows)**: Stratified K-fold cross-validation
3. **Large Datasets (>1M rows)**: Single train/validation/test split
4. **Imbalanced Data**: Stratified folds + appropriate metrics (F1, AUC-ROC)

## Performance Benchmarks (2025)

### Tabular Data Competitions
- **CatBoost**: +20% accuracy improvement, 30-60x faster prediction
- **LightGBM**: 7x faster training than XGBoost
- **Random Forest**: Still competitive for feature importance analysis
- **Linear Models**: 100x faster training, essential for real-time systems

### Dimensionality Reduction
- **UMAP**: Preferred over t-SNE for datasets >10K samples
- **PCA**: Still fastest for linear reduction, use for preprocessing
- **Autoencoders**: Best for non-linear reduction with reconstruction needs

## Integration with Other Sections

### Prerequisites
- **00_Data_Analysis_Fundamentals**: Data cleaning and EDA
- **01_Statistical_Foundations**: Hypothesis testing and distributions

### Next Steps
- **03_Deep_Learning_Fundamentals**: Neural networks for complex patterns
- **04_Computer_Vision**: CNNs for image data
- **05_NLP_and_Transformers**: Transformers for text data

### Related Topics
- **Feature Engineering** (Section 00): Critical for classical ML performance
- **A/B Testing** (Section 01): Validating model improvements in production
- **AutoML** (Section 08): Automated hyperparameter tuning

## Tools and Libraries

### Essential Python Libraries
```python
# Core ML
import sklearn
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Modern Gradient Boosting
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Dimensionality Reduction
import umap
from sklearn.manifold import TSNE

# Validation and Metrics
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
```

### Recommended Versions (2025)
- scikit-learn >= 1.4.0
- xgboost >= 2.0.0
- lightgbm >= 4.1.0
- catboost >= 1.2.0
- umap-learn >= 0.5.5

## Common Pitfalls

### Linear Regression
- Ignoring multicollinearity (use VIF analysis)
- Not checking LINE assumptions
- Using R^2 alone for model evaluation
- Extrapolating beyond training data range

### Ensemble Methods
- Overfitting on small datasets (use strong regularization)
- Not tuning learning rate and tree depth together
- Ignoring class imbalance in classification
- Using default hyperparameters

### Clustering
- Not scaling features before K-Means
- Choosing K without validation metrics
- Assuming spherical clusters when using K-Means
- Not handling outliers before clustering

### Dimensionality Reduction
- Using t-SNE for anything except visualization
- Not centering/scaling data before PCA
- Interpreting PCA components without domain knowledge
- Losing too much variance in reduction

## Production Considerations

### Model Deployment
1. **Serialization**: Use joblib for sklearn, native save for XGBoost/LightGBM/CatBoost
2. **Versioning**: Track model version, training data version, and hyperparameters
3. **Monitoring**: Log prediction distributions and feature drift
4. **Fallback**: Always have a simple baseline model as backup

### Computational Resources
- **Linear Models**: CPU-only, <1GB RAM for most datasets
- **Random Forest**: Benefits from multi-core, 2-8GB RAM
- **Gradient Boosting**: GPU optional, 4-16GB RAM recommended
- **Clustering**: RAM = 5-10x dataset size for hierarchical methods

### Real-Time Inference
- **Fastest**: Linear regression, logistic regression (<1ms)
- **Fast**: LightGBM, CatBoost (1-10ms)
- **Moderate**: XGBoost, Random Forest (10-50ms)
- **Slow**: Deep learning models (50-500ms)

## Further Reading

### Books
- "The Elements of Statistical Learning" (Hastie, Tibshirani, Friedman)
- "Pattern Recognition and Machine Learning" (Bishop)
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Geron)

### Papers
- XGBoost: Chen & Guestrin (2016)
- LightGBM: Ke et al. (2017)
- CatBoost: Prokhorenkova et al. (2018)
- UMAP: McInnes et al. (2018)

### Online Resources
- scikit-learn documentation and user guide
- Kaggle competitions for practical applications
- distill.pub for visual explanations

## Contributing

When adding content to this section:
1. Include mathematical derivations with intuitive explanations
2. Provide production-ready code with error handling
3. Reference 2025 benchmarks from RESEARCH_SUMMARY_2025.md
4. Add real-world use cases and when to use each method
5. Follow the existing format for consistency

---

**Last Updated**: 2025-10-14
**Maintainer**: ML Encyclopedia Project
**Status**: Complete and Current
