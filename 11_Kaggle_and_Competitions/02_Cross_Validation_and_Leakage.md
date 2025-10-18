# Cross-Validation and Data Leakage Prevention

## The Most Important Rule

**"A good CV score predicts good LB score. A bad CV doesn't guarantee bad LB, but trust your CV."**

---

## Cross-Validation Strategies

### 1. Standard K-Fold
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
```
**Use when:** IID data, no time/group structure

### 2. Stratified K-Fold
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
**Use when:** Classification with imbalanced classes

### 3. Group K-Fold
```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
# groups = df['customer_id'] or df['patient_id'], etc.
```
**Use when:** Multiple rows per entity (customers, patients), prevent leakage

### 4. Time Series Split
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
```
**Use when:** Temporal data, predicting future

### 5. Adversarial Validation
```python
# Check if CV mimics train/test split
combined = pd.concat([train.assign(is_train=1), test.assign(is_train=0)])
cv_score = cross_val_score(model, combined.drop('is_train', axis=1), combined['is_train'])
print(f"AUC: {cv_score.mean():.3f}")  # Should be ~0.5 for good CV
```

---

## Data Leakage - The Silent Killer

### Types of Leakage

**1. Target Leakage**
```python
# ❌ WRONG - Future information in features
df['days_until_churn'] = (df['churn_date'] - df['current_date']).dt.days

# ✅ CORRECT - Only past information
df['days_since_signup'] = (df['current_date'] - df['signup_date']).dt.days
```

**2. Train-Test Contamination**
```python
# ❌ WRONG - Scaling before split
X_scaled = scaler.fit_transform(X)  # Uses test info!
X_train, X_test = train_test_split(X_scaled)

# ✅ CORRECT - Split first
X_train, X_test = train_test_split(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**3. Temporal Leakage**
```python
# ❌ WRONG - Using future aggregations
df['user_total_purchases'] = df.groupby('user_id')['purchase'].transform('sum')

# ✅ CORRECT - Only past purchases
df['user_total_purchases_before'] = df.groupby('user_id')['purchase'].cumsum().shift(1)
```

**4. Group Leakage**
```python
# ❌ WRONG - Same customer in train/val
kf = KFold(n_splits=5)  # Might split same customer

# ✅ CORRECT - Customer-level split
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=df['customer_id']):
    ...
```

---

## Leakage Detection Checklist

- [ ] Check feature importance - Suspiciously high importance (>0.8) might indicate leakage
- [ ] Validate CV vs LB correlation - If CV improves but LB degrades → leakage
- [ ] Adversarial validation - Train/test should be indistinguishable (AUC ~0.5)
- [ ] Temporal consistency - Features use only past information
- [ ] Group independence - No information sharing across groups

---

## Best Practices

1. **Always use same CV strategy for feature selection, tuning, and final evaluation**
2. **Create time-based holdout if data has temporal component**
3. **Use GroupKFold for hierarchical data** (customers, hospitals, etc.)
4. **Trust your CV** - If CV and LB diverge, investigate leakage
5. **Adversarial validation** - Regularly check train/test similarity
