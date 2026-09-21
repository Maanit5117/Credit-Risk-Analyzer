# Credit Risk Analyzer

An end-to-end machine learning project for predicting loan default risk based on applicant credit profile, financial details, and loan characteristics.

---

## 📌 Project Overview

Credit risk assessment is a critical component of retail banking and lending institutions. This project analyzes loan applicant data, performs data cleaning and exploratory data analysis (EDA), builds preprocessing pipelines, and evaluates machine learning models to identify high-risk applicants while minimizing false defaults.

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

## 🛠️ Data Preprocessing & Methodology

1. **Data Cleaning & Filtering**:
   - Dropped duplicate records.
   - Filtered out unrealistic ages and inconsistent employment lengths (`person_emp_length <= person_age`).
   - Filtered non-positive loan amounts.

2. **Feature Engineering & Imputation**:
   - Numerical columns: Median imputation (`SimpleImputer`), followed by standard scaling (`StandardScaler`) for linear models.
   - Categorical columns: Constant imputation and One-Hot Encoding (`OneHotEncoder`).

3. **Handling Imbalanced Classes**:
   - Class distribution weighting using `scale_pos_weight` for XGBoost and `class_weight='balanced'` for Logistic Regression.

4. **Model Validation**:
   - Evaluated using **Stratified 5-Fold Cross-Validation** with multiple metrics (`ROC-AUC`, `Accuracy`, `Precision`, `Recall`, `F1-Score`).

---

## 📈 Model Performance & Comparison

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.8499 | 77.04% | 48.04% | 76.82% | 0.5911 |
| **XGBoost Classifier** | **0.9389** | **90.36%** | **77.02%** | **78.92%** | **0.7795** |

> **Key Takeaway**: XGBoost significantly outperformed the baseline Logistic Regression model, achieving an ROC-AUC of **0.939** and reducing false positives while maintaining a high recall rate (~79%) on defaults.

---

## 📁 Repository Structure

```text
├── Credit_Risk.ipynb          # Jupyter notebook with EDA, preprocessing, and model training
├── credit_risk_dataset.csv    # Loan applicant dataset
├── README.md                  # Project overview and instructions
└── .gitignore                 # Git ignore file
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed along with the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/Maanit5117/credit-risk-analyzer.git
   cd credit-risk-analyzer
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Credit_Risk.ipynb
   ```
