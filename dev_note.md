# Regression Model Statistics Reference

## 1. Average Coefficients (`mean_beta`)

**What it is:**
- Average of all regression coefficients across 100 runs
- Includes the intercept (bias term) as the first value, followed by coefficients for each feature

**What it signifies:**
- Each coefficient shows the expected change in target (house price) per unit increase of that feature
- **Positive coefficient** → increasing feature increases predicted price
- **Negative coefficient** → increasing feature decreases predicted price
- Averaging over 100 runs reduces randomness and provides a stable estimate of feature importance

---

## 2. Coefficients Standard Deviation (`std_beta`)

**What it is:**
- Standard deviation of each coefficient across the 100 runs

**What it signifies:**
- Measures how much each coefficient varies due to random train/test splits
- **Small std** → coefficient is stable across splits → feature is reliably important
- **Large std** → coefficient is unstable → feature's effect may be inconsistent

---

## 3. Mean Absolute Error (`mean_error`)

**What it is:**
- Average of all absolute differences between predicted and actual house prices across all runs

**What it signifies:**
- Represents the typical size of prediction error in the model
- **Example:** `mean_error = 3.2` → on average, predicted price is off by 3.2 units (e.g., $3,200 if target is in thousands)

---

## 4. Standard Deviation of Absolute Error (`std_error`)

**What it is:**
- Standard deviation of absolute prediction errors

**What it signifies:**
- Measures variability of prediction error across predictions
- **Small std** → predictions are consistently close to actual values
- **Large std** → some predictions are much worse than others

---

## 5. RMSE Mean (`rmse_mean`)

**What it is:**
- Root Mean Squared Error averaged over all 100 runs

**What it signifies:**
- Provides a penalized measure of prediction error (larger mistakes are weighted more heavily)
- Commonly used in regression to report model accuracy

---

## 6. RMSE Standard Deviation (`rmse_std`)

**What it is:**
- Standard deviation of RMSE across all runs

**What it signifies:**
- Shows how much model performance varies with different random train/test splits
- **Smaller RMSE std** → model's accuracy is consistent

---

## 7. 95% Prediction Interval of Errors (`95pi`)

**What it is:**
- A range containing 95% of the absolute prediction errors
- **Example:** `(1.0, 7.5)` → 95% of predicted prices are within 1.0–7.5 units of the true price

**What it signifies:**
- Provides a statistical range of likely prediction errors
- Makes model performance trustworthy and interpretable
- Can be reported as: *"The model predicts house prices with 95% of errors falling between $1,000 and $7,500."*

---

## Summary: What You Can Report

Use these statistics to describe your model in a trustworthy way:

| Metric | Definition |
|--------|-----------|
| **Feature Importance** | `mean_beta ± std_beta` |
| **Prediction Accuracy** | `mean_error ± std_error` |
| **Typical Error Magnitude** | `rmse_mean ± rmse_std` |
| **Reliability of Predictions** | `95% prediction interval` |
