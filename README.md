# MLF_Lab2
# Diabetes Progression Prediction

This project uses the Scikit-Learn Diabetes dataset to predict how a patient’s diabetes may progress over one year. Two types of models are built:

- **Part 2:** Using only the **BMI** feature (Univariate Polynomial Regression)
- **Part 3:** Using **all features** (Multivariate Models)

---

## Dataset Info

- **Features (X):** age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
- **Target (y):** Disease progression after one year
- All features are already standardized

---

## Part 2: BMI-Only Models

- Trained polynomial regression models with degrees **0 to 5**
- Best model: **Degree 1 (Linear Regression)**
- Equation: `y = 153.14 + 999.06 * BMI`
- Prediction example: For BMI = 0.05, prediction = **203.09**
- Test R² = 0.20, MAPE ≈ 49%
- Simple model, but not very accurate since it uses only BMI

---

## Part 3: All Features Models

Tested the following models using all features:

- **Polynomial Regression** (Degree 2 and 3)
- **Decision Trees** (Depth 3 and 5)
- **kNN Regressor** (k = 3 and 5)

**Best overall:** Polynomial Regression (Degree 2)

- It gave the best balance between accuracy and generalization
- Other models (like Degree 3 or deep trees) showed overfitting

---

## Files

- `bmi_model.ipynb` – Univariate models using BMI
- `all_features_models.ipynb` – Multivariate models using all features

---

## 🔧 Requirements

```bash
pip install scikit-learn pandas matplotlib numpy
