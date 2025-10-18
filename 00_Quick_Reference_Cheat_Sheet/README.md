# Machine Learning & Analytics Visual Cheat Sheet Guide

A comprehensive, print-ready reference guide for machine learning practitioners covering model selection, feature engineering, statistical tests, and best practices.

## 📋 Contents

### Core Fundamentals
- **ML Workflow Overview** - Complete end-to-end process
- **Model Selection Master Tree** - Decision framework for choosing algorithms (NOW WITH EASY-TO-READ TABLES!)
- **Data Type → Model Mapping** - Quick reference tables

### Feature Engineering
- **Categorical Encoding Guide** - When to use each encoding method
- **Numerical Transformations** - Scaling, normalization, and transformations
- **Feature Engineering Tips** - Domain-specific best practices

### Statistical Tests
- **Test Selection Flowchart** - Choose the right statistical test
- **A/B Testing Guide** - Complete framework with sample size calculations

### Deep Learning
- **When to Use Deep Learning** - Decision criteria and data requirements
- **Architecture Selection** - CNN, RNN, Transformer guidance
- **Training Best Practices** - Hyperparameters, regularization, monitoring

### Model Evaluation & Deployment
- **Metrics Selection Guide** - Classification and regression metrics
- **Cross-Validation Strategies** - Proper validation techniques
- **Hyperparameter Tuning** - Grid search, random search, Bayesian optimization

### Quick Reference
- **Algorithm Comparison Tables** - Side-by-side feature comparisons
- **Common Pitfalls** - What to avoid
- **Best Practices Checklist** - Essential guidelines

## 🎨 Design Features

- **Color-Coded Sections** - Easy visual navigation
- **Easy-to-Read Tables** - Clear, structured model recommendations (UPDATED!)
- **Comparison Tables** - Side-by-side algorithm comparisons
- **Practical Tips** - Highlighted best practices and warnings
- **Print-Ready Format** - 8.5x11 inches, optimized for printing/laminating

## 📄 Files

- `ml_analytics_guide.html` - Main guide (open in browser) - **UPDATED with readable tables!**
- `ml_analytics_guide.pdf` - Print-ready PDF version - **UPDATED!**
- `README.md` - This file

## 🚀 Usage

1. **View in Browser**: Open `ml_analytics_guide.html` in any web browser
2. **Print**: Use the PDF version for best print quality
3. **Laminate**: Perfect size for desk reference or wall poster

## 💡 Key Highlights

- **Start Simple**: Always baseline with simple models first
- **XGBoost Usually Wins**: For tabular data, gradient boosting beats neural networks 90% of the time
- **Cross-Validate Everything**: Never rely on a single train/test split
- **Domain Knowledge Matters**: Feature engineering requires understanding your data

## 📊 Model Selection Quick Guide (Now in Easy Tables!)

### 📊 Tabular Data
| Sample Size | Recommended Models | Why |
|-------------|-------------------|-----|
| < 1K | Logistic Regression OR Random Forest | Simple models generalize better |
| 1K-10K | **XGBoost (best)** OR Random Forest | Handles mixed types, missing values |
| 10K-100K | **XGBoost / LightGBM** (winner 90%) | Beats neural networks consistently |
| 100K+ | **LightGBM (fastest)** OR Deep Learning | Optimized for large datasets |

### 🖼️ Images
| Dataset Size | Approach | Models |
|-------------|----------|--------|
| < 10K | **Transfer Learning** | ResNet, EfficientNet, ViT |
| 10K-100K | Fine-tune pretrained CNN | ResNet50, EfficientNet-B0 |
| 100K+ | Train from scratch OR fine-tune | EfficientNet-B7, Custom CNNs |

### 📝 Text / NLP
| Sample Size | Recommended Models | Notes |
|-------------|-------------------|-------|
| < 1K | TF-IDF + Logistic Reg OR Few-shot LLM | Simple baseline works well |
| 1K-100K | **Fine-tune BERT/RoBERTa** | Pretrained + task-specific tuning |
| 100K+ | Fine-tune large Transformer (GPT, T5) | Enough data for larger models |

### 📈 Time Series
| Data Points | Recommended Models | Why |
|-------------|-------------------|-----|
| < 100 | ARIMA / Exponential Smoothing | Statistical methods for limited data |
| 100-10K | Prophet OR SARIMA | Prophet good for business data |
| 10K+ | LSTM OR XGBoost with lag features | DL for complex patterns |

---

## ✨ Recent Updates

**October 13, 2025**
- ✅ Replaced hard-to-read tree structure with clear, formatted tables
- ✅ Added color-coded section headers for each data type
- ✅ Improved readability and visual hierarchy
- ✅ Updated PDF with new table format

---

**Created**: October 2025  
**Last Updated**: October 13, 2025  
**Format**: HTML + PDF  
**Print Size**: 8.5" x 11" (Letter)  
**Pages**: 18
