# Credit Risk Analyzer

An end-to-end machine learning project for predicting loan default risk based on applicant credit profiles, financial details, and loan characteristics.

---

## 📌 Project Overview

Credit risk assessment is a critical component of retail banking and lending institutions. This project analyzes loan applicant data, performs data cleaning and exploratory data analysis (EDA), builds modular scikit-learn preprocessing pipelines, evaluates machine learning models to identify high-risk applicants, performs hyperparameter tuning, calibrates risk probabilities, optimizes decision thresholds, and analyzes model explainability with SHAP.

---

## 📊 Dataset

The dataset ([`credit_risk_dataset.csv`](credit_risk_dataset.csv)) contains historical loan applications with borrower financial attributes:

| Feature Name | Description |
| :--- | :--- |
| `person_age` | Age of borrower |
| `person_income` | Annual income of borrower |
| `person_home_ownership` | Home ownership status (`RENT`, `OWN`, `MORTGAGE`, `OTHER`) |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Intent of the loan (`PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION`) |
| `loan_grade` | Assigned loan grade (`A` to `G`) |
| `loan_amnt` | Total loan amount requested |
| `loan_int_rate` | Loan interest rate |
| `loan_status` | **Target variable**: `0` = Non-default, `1` = Default |
| `loan_percent_income` | Loan amount as a percentage of borrower's income |
| `cb_person_default_on_file`| Historical default indicator (`Y` / `N`) |
| `cb_person_cred_hist_length`| Length of borrower's credit history in years |

---

## 🛠️ Methodology & Modeling Pipeline

1. **Data Cleaning & Filtering**:
   - Deduplicated raw records.
   - Filtered out unrealistic ages and inconsistent employment lengths (`person_emp_length <= person_age`).
   - Filtered non-positive loan amounts.

2. **Feature Engineering & Preprocessing**:
   - **Numerical Features**: Median imputation (`SimpleImputer`) and scaling (`StandardScaler` for linear models; unscaled for tree-based models).
   - **Categorical Features**: Constant imputation (`missing_value`) and One-Hot Encoding (`OneHotEncoder`).
   - Integrated end-to-end using scikit-learn `Pipeline` and `ColumnTransformer`.

3. **Handling Imbalanced Classes**:
   - Utilized `scale_pos_weight` for XGBoost to penalize false negatives on defaults.
   - Applied `class_weight='balanced'` for Logistic Regression baseline.

4. **Model Validation & Hyperparameter Tuning**:
   - Evaluated models using **Stratified 5-Fold Cross-Validation**.
   - Tuned XGBoost with **`RandomizedSearchCV`** over 50 iterations, optimizing for `roc_auc` across tree depth, learning rate, subsample ratio, colsample by tree, min child weight, and estimator counts.
   - Tuned model achieved a cross-validated ROC-AUC score of **0.94**.

5. **Decision Threshold Optimization**:
   - Analyzed the **Precision-Recall Curve** across varying probability thresholds.
   - Selected the optimal threshold maximizing the F1-score, boosting precision to **93%** and F1-score to **0.82**.

6. **Probability Calibration**:
   - In lending, raw model outputs must correspond to true risk probabilities.
   - Applied **`CalibratedClassifierCV`** (Sigmoid / Platt Scaling) to evaluate and plot calibration curves comparing uncalibrated vs. calibrated model probabilities against a perfectly calibrated baseline.

7. **Model Interpretability (SHAP)**:
   - Implemented **SHAP (SHapley Additive exPlanations)** with `shap.TreeExplainer` on the transformed feature space to uncover key drivers of credit default risk and ensure transparent decision-making.

---

## 📈 Model Performance & Comparison

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.85 | 77% | 48% | 77% | 0.59 |
| **XGBoost Classifier (Threshold = 0.50)** | **0.94** | 91% | 77% | **80%** | 0.78 |
| **XGBoost Classifier (Optimized Threshold)** | **0.94** | **93%** | **93%** | 73% | **0.82** |

> **Key Takeaway**: 
> - XGBoost substantially outperformed Logistic Regression across all classification metrics.
> - Tuning the decision threshold via Precision-Recall trade-off optimization improved Precision from **77% to 93%** and F1-score from **0.78 to 0.82**, dramatically decreasing false default predictions while capturing high-risk loans.

---

## 📁 Repository Structure

```text
├── Credit_Risk.ipynb          # Jupyter notebook with EDA, preprocessing, tuning, calibration, and SHAP
├── credit_risk_dataset.csv    # Historical loan applicant dataset
├── README.md                  # Project overview and documentation
└── .gitignore                 # Git ignore configuration
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed along with the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost scipy shap jupyter
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone git@github.com:Maanit5117/credit-risk-analyzer.git
   cd credit-risk-analyzer
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Credit_Risk.ipynb
   ```
