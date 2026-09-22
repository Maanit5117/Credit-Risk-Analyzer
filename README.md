# Axiom | Credit Risk Analyzer

An end-to-end machine learning system and institutional underwriting web application for predicting loan default risk based on applicant credit profiles, financial capacity, and facility characteristics.

---

## Project Overview

Credit risk assessment is a foundational component of retail banking and institutional lending. **Axiom** combines exploratory data analysis, scikit-learn preprocessing pipelines, XGBoost modeling with hyperparameter tuning, Platt probability calibration, F1-optimal decision thresholding, and TreeSHAP explainability into an intuitive, production-ready web application ready to deploy on Render.

---

## Dataset

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

## Methodology & Modeling Pipeline

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

## Model Performance & Comparison

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.85 | 77% | 48% | 77% | 0.59 |
| **XGBoost Classifier (Threshold = 0.50)** | **0.94** | 91% | 77% | **80%** | 0.78 |
| **XGBoost Classifier (Optimized Threshold)** | **0.94** | **93%** | **93%** | 73% | **0.82** |

> **Key Takeaway**: 
> - XGBoost substantially outperformed Logistic Regression across all classification metrics.
> - Tuning the decision threshold via Precision-Recall trade-off optimization improved Precision from **77% to 93%** and F1-score from **0.78 to 0.82**, dramatically decreasing false default predictions while capturing high-risk loans.

---

## Repository Structure

```text
├── Credit_Risk.ipynb          # Jupyter notebook with EDA, preprocessing, tuning, calibration, and SHAP
├── credit_risk_dataset.csv    # Historical loan applicant dataset (32,581 records)
├── credit_risk_model.pkl      # Production CalibratedClassifierCV (XGBoost 5-fold ensemble)
├── best_threshold.pkl         # F1-optimal calibrated decision cutoff threshold (0.6765)
├── main.py                    # Production FastAPI application serving API & UI
├── requirements.txt           # Production dependency specifications
├── runtime.txt                # Python runtime specification for Render (python-3.11.9)
├── Procfile                   # Process configuration for PaaS/Render deployment
├── render.yaml                # Infrastructure-as-code blueprint for Render
├── static/
│   ├── index.html             # Institutional credit decisioning interface
│   ├── css/
│   │   └── styles.css         # Financial-grade styling, SVG dials, tables, grid
│   └── js/
│       └── app.js             # Real-time inference, DTI calculation, TreeSHAP chart, registry
├── README.md                  # Comprehensive system documentation
└── .gitignore                 # Git ignore configuration
```

---

## Axiom Web Application & Underwriting Suite

The project includes an interactive, financial-grade internal web dashboard built for credit risk officers and underwriting teams:

1. **Axiom Boot Preloader**:
   - Institutional splash screen with glowing brand crest and animated status indicator during system initialization.
2. **Streamlined Underwriting Workspace**:
   - Form inputs for borrower demographics, requested facility, and bureau records with real-time inline validation.
   - Live automated Debt-to-Income (DTI) ratio calculation with policy benchmark indicators.
   - One-click template demo buttons (`Prime Low Risk` and `Subprime High Risk`) for instant evaluation.
3. **Calibrated Decisioning Engine**:
   - Computes Probability of Default (PD) calibrated via 5-fold sigmoid Platt scaling against the **67.65%** policy cutoff.
   - Basel Synthetic Credit Risk Score (scale 300 - 850).
   - Clear underwriting directives: **Approved (STP)**, **Conditional (Senior Review)**, or **Declined (Policy Breached)**.
4. **Precision Risk Spectrum Dial**:
   - Custom SVG semicircular gauge with needle animation pointing to the applicant's exact default probability against the calibrated cutoff line.
5. **Local TreeSHAP Factor Decomposition**:
   - Evaluates the individual borrower's log-odds contributions across model trees.
   - Renders a clean horizontal bar chart displaying mitigating factors (reducing risk in green) and adverse drivers (increasing risk in red).
6. **Audit Registry & Historical Ledger**:
   - Searchable and sortable ledger of all evaluated credit applications with a 1-click **Load into Form** action.
   - Export full assessment history to CSV.

---

## Deployment on Render

This repository is pre-configured with `render.yaml`, `runtime.txt`, and `Procfile` for seamless zero-configuration deployment to [Render](https://render.com).

### Option A: Using the Render Dashboard (Manual Setup)

1. Push your repository to GitHub.
2. Sign in to [Render](https://dashboard.render.com/) and click **New + > Web Service**.
3. Connect your GitHub repository (`credit-risk-analyzer`).
4. Configure the service settings:
   - **Name**: `credit-risk-analyzer` (or your preferred name)
   - **Environment**: `Python 3`
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
5. Under **Advanced**, ensure the Python version is picked up from [`runtime.txt`](runtime.txt) (`python-3.11.9`), or add an environment variable:
   - Key: `PYTHON_VERSION`
   - Value: `3.11.9`
6. Click **Create Web Service**. Render will automatically build the wheels and launch your service with live HTTPS.

### Option B: Using Render Blueprint (`render.yaml`)

1. In Render, select **New + > Blueprint**.
2. Select your repository. Render will automatically detect [`render.yaml`](render.yaml) and configure the build and start commands with zero manual typing.
3. Click **Apply**.
