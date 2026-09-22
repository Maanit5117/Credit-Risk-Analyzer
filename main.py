import os
import json
import time
import uuid
import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Global ML model and artifact container
ml_state: Dict[str, Any] = {
    "model": None,
    "threshold": 0.6765,
    "global_importance": [],
    "history": []
}

# Friendly name mapping for model features
FEATURE_NAME_MAP = {
    "num__person_age": "Applicant Age",
    "num__person_emp_length": "Employment Duration",
    "num__loan_amnt": "Requested Loan Amount",
    "num__loan_int_rate": "Contractual Interest Rate",
    "num__loan_percent_income": "Loan-to-Income Ratio (DTI)",
    "num__cb_person_cred_hist_length": "Credit History Length",
    "num__person_income": "Annual Gross Income",
    "cat__person_home_ownership_RENT": "Housing: Renting",
    "cat__person_home_ownership_OWN": "Housing: Home Owner",
    "cat__person_home_ownership_MORTGAGE": "Housing: Mortgaged",
    "cat__person_home_ownership_OTHER": "Housing: Other",
    "cat__loan_intent_PERSONAL": "Purpose: Personal",
    "cat__loan_intent_EDUCATION": "Purpose: Education",
    "cat__loan_intent_MEDICAL": "Purpose: Medical Expense",
    "cat__loan_intent_VENTURE": "Purpose: Business Venture",
    "cat__loan_intent_HOMEIMPROVEMENT": "Purpose: Home Improvement",
    "cat__loan_intent_DEBTCONSOLIDATION": "Purpose: Debt Consolidation",
    "cat__cb_person_default_on_file_N": "Bureau Default: Clean History",
    "cat__cb_person_default_on_file_Y": "Bureau Default: Historical Default",
}

def compute_factor_attribution(model, input_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Computes local feature contributions across the calibrated ensemble estimators
    using XGBoost's native TreeSHAP implementation (pred_contribs).
    """
    all_contribs = []
    names = None
    for cal_clf in model.calibrated_classifiers_:
        base = cal_clf.estimator
        prep = base.named_steps["Preprocessor"]
        clf = base.named_steps["Classifier"]
        X_trans = prep.transform(input_df)
        names = list(prep.get_feature_names_out())
        dmat = xgb.DMatrix(X_trans, feature_names=names)
        c = clf.get_booster().predict(dmat, pred_contribs=True)
        # Exclude last element which is the model bias/intercept
        all_contribs.append(c[0][:-1])
    
    avg_contribs = np.mean(all_contribs, axis=0)
    factors = []
    for col, val in zip(names, avg_contribs):
        clean_name = FEATURE_NAME_MAP.get(col, col.replace("num__", "").replace("cat__", ""))
        val_f = float(round(val, 4))
        factors.append({
            "raw_name": col,
            "feature": clean_name,
            "impact": val_f,
            "direction": "Increases Risk" if val_f > 0 else "Reduces Risk",
            "abs_impact": abs(val_f)
        })
    
    # Sort by magnitude of contribution
    factors.sort(key=lambda x: x["abs_impact"], reverse=True)
    return factors[:8]

def extract_global_importance(model) -> List[Dict[str, Any]]:
    """
    Extracts normalized global feature importance from the base ensemble tree models.
    """
    base = model.calibrated_classifiers_[0].estimator
    prep = base.named_steps["Preprocessor"]
    clf = base.named_steps["Classifier"]
    importances = clf.feature_importances_
    feature_names = prep.get_feature_names_out()

    global_imp = []
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        clean = FEATURE_NAME_MAP.get(name, name)
        global_imp.append({
            "name": clean,
            "importance": round(float(imp), 4),
            "percentage": round(float(imp) * 100, 2)
        })
    return global_imp

def seed_initial_history(model, threshold: float):
    """
    Pre-populates the assessment history registry with realistic anonymized records
    from the dataset so the history table is populated on initial launch.
    """
    csv_path = "credit_risk_dataset.csv"
    if not os.path.exists(csv_path):
        return

    try:
        df = pd.read_csv(csv_path)
        # Sample representative records across different profiles
        sample_indices = [15, 34, 102, 250, 412, 630, 890, 1120, 1500, 2045]
        sample_df = df.iloc[sample_indices].copy()
        
        feature_cols = [
            "person_age", "person_income", "person_home_ownership", "person_emp_length",
            "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income",
            "cb_person_default_on_file", "cb_person_cred_hist_length"
        ]
        
        # Clean null values
        sample_df["person_emp_length"] = sample_df["person_emp_length"].fillna(3).astype(float)
        sample_df["loan_int_rate"] = sample_df["loan_int_rate"].fillna(11.0).astype(float)

        probs = model.predict_proba(sample_df[feature_cols])[:, 1]
        
        now = datetime.datetime.now(datetime.timezone.utc)
        
        for idx, (_, row) in enumerate(sample_df.iterrows()):
            prob = float(probs[idx])
            pred_class = 1 if prob >= threshold else 0
            
            if prob >= threshold:
                tier = "High Risk"
                recommendation = "DECLINED"
            elif prob >= 0.25:
                tier = "Moderate Risk"
                recommendation = "MANUAL REVIEW"
            else:
                tier = "Low Risk"
                recommendation = "APPROVED"

            credit_score = int(round(300 + (1.0 - min(max(prob, 0.0), 1.0)) * 550))
            time_offset = datetime.timedelta(hours=(len(sample_indices) - idx) * 3.5)
            record_time = (now - time_offset).strftime("%Y-%m-%d %H:%M UTC")
            
            ref_id = f"APP-{84000 + idx * 137}"
            ml_state["history"].append({
                "assessment_id": ref_id,
                "timestamp": record_time,
                "person_age": int(row["person_age"]),
                "person_income": float(row["person_income"]),
                "person_home_ownership": str(row["person_home_ownership"]),
                "person_emp_length": float(row["person_emp_length"]),
                "loan_intent": str(row["loan_intent"]),
                "loan_grade": str(row["loan_grade"]),
                "loan_amnt": float(row["loan_amnt"]),
                "loan_int_rate": float(row["loan_int_rate"]),
                "loan_percent_income": float(row["loan_percent_income"]),
                "cb_person_default_on_file": str(row["cb_person_default_on_file"]),
                "cb_person_cred_hist_length": int(row["cb_person_cred_hist_length"]),
                "default_probability": round(prob, 4),
                "probability_percent": round(prob * 100, 2),
                "threshold": round(threshold, 4),
                "default_prediction": pred_class,
                "risk_tier": tier,
                "credit_score": credit_score,
                "recommendation": recommendation,
                "Result": "High Risk" if pred_class == 1 else "Low Risk",
                "policy_flags": generate_policy_flags(row)
            })
    except Exception as e:
        print(f"Notice: Initial history seed skipped ({e})")

def generate_policy_flags(data: Any) -> List[Dict[str, str]]:
    """
    Evaluates rule-based underwriting institutional policy checks.
    """
    flags = []
    
    # Handle dict or Series
    d = data if isinstance(data, dict) else data.to_dict()
    
    dti = float(d.get("loan_percent_income", 0.0))
    if dti > 0.35:
        flags.append({
            "severity": "high",
            "code": "POL-DTI-01",
            "message": f"Loan-to-income ratio ({dti*100:.1f}%) exceeds internal policy ceiling of 35%."
        })
    elif dti > 0.25:
        flags.append({
            "severity": "medium",
            "code": "POL-DTI-02",
            "message": f"Elevated debt service ratio ({dti*100:.1f}%)."
        })
        
    if str(d.get("cb_person_default_on_file", "N")).upper() == "Y":
        flags.append({
            "severity": "high",
            "code": "POL-CB-01",
            "message": "Prior credit bureau default record on file within past reporting cycle."
        })
        
    grade = str(d.get("loan_grade", "C")).upper()
    if grade in ["E", "F", "G"]:
        flags.append({
            "severity": "high",
            "code": "POL-GRD-01",
            "message": f"Subprime loan grade rating ({grade}) assigned by credit scoring engine."
        })
        
    int_rate = float(d.get("loan_int_rate", 0.0))
    if int_rate >= 16.0:
        flags.append({
            "severity": "medium",
            "code": "POL-RAT-01",
            "message": f"High facility interest rate ({int_rate:.2f}%) indicates adverse risk pricing tier."
        })
        
    cred_len = int(d.get("cb_person_cred_hist_length", 0))
    if cred_len < 3:
        flags.append({
            "severity": "low",
            "code": "POL-EXP-01",
            "message": f"Thin bureau tenure ({cred_len} years). Secondary documentation recommended."
        })
        
    return flags


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML Model and Calibrated Cutoff Threshold
    model_path = "credit_risk_model.pkl"
    thresh_path = "best_threshold.pkl"
    
    if os.path.exists(model_path):
        ml_state["model"] = joblib.load(model_path)
    else:
        raise RuntimeError(f"Model file {model_path} not found.")
        
    if os.path.exists(thresh_path):
        loaded_thresh = joblib.load(thresh_path)
        ml_state["threshold"] = float(loaded_thresh)
    else:
        ml_state["threshold"] = 0.6765
        
    ml_state["global_importance"] = extract_global_importance(ml_state["model"])
    seed_initial_history(ml_state["model"], ml_state["threshold"])
    
    yield
    
    ml_state.clear()


app = FastAPI(
    title="Axiom Credit Risk Engine",
    description="Institutional credit underwriting and calibrated probability of default system.",
    version="2.4.0",
    lifespan=lifespan
)

# Input data schema
class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    person_income: float = Field(..., gt=0, description="Annual gross personal income in USD")
    person_home_ownership: str = Field(..., description="Housing status: RENT, OWN, MORTGAGE, OTHER")
    person_emp_length: float = Field(..., ge=0, le=60, description="Years of continuous employment")
    loan_intent: str = Field(..., description="Loan purpose: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION")
    loan_grade: str = Field(..., description="Internal risk grade: A, B, C, D, E, F, G")
    loan_amnt: float = Field(..., gt=0, description="Requested principal facility amount")
    loan_int_rate: float = Field(..., ge=0.5, le=40.0, description="Interest rate percentage")
    loan_percent_income: float = Field(..., ge=0.0, le=1.5, description="Ratio of loan amount to personal income")
    cb_person_default_on_file: str = Field(..., description="Historical default on bureau file: Y or N")
    cb_person_cred_hist_length: int = Field(..., ge=0, le=60, description="Credit history file age in years")

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 28,
                "person_income": 65000.0,
                "person_home_ownership": "MORTGAGE",
                "person_emp_length": 5.0,
                "loan_intent": "DEBTCONSOLIDATION",
                "loan_grade": "B",
                "loan_amnt": 10000.0,
                "loan_int_rate": 10.25,
                "loan_percent_income": 0.1538,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 6
            }
        }
    }


def execute_inference(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core prediction, factor attribution, and risk scoring logic.
    """
    # Normalize inputs
    data_dict["person_home_ownership"] = str(data_dict["person_home_ownership"]).upper()
    data_dict["loan_intent"] = str(data_dict["loan_intent"]).upper()
    data_dict["loan_grade"] = str(data_dict["loan_grade"]).upper()
    data_dict["cb_person_default_on_file"] = str(data_dict["cb_person_default_on_file"]).upper()
    
    input_df = pd.DataFrame([data_dict])
    
    model = ml_state["model"]
    threshold = ml_state["threshold"]
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    proba_array = model.predict_proba(input_df)[:, 1]
    prob = float(proba_array[0])
    pred = 1 if prob >= threshold else 0
    
    # Tier and Decision logic
    if prob >= threshold:
        tier = "High Risk"
        recommendation = "DECLINED"
        decision_notes = (
            f"Calibrated probability of default ({prob*100:.2f}%) exceeds the institutional "
            f"risk threshold of {threshold*100:.2f}%. Loan request is declined under policy cutoff."
        )
    elif prob >= 0.25:
        tier = "Moderate Risk"
        recommendation = "MANUAL REVIEW"
        decision_notes = (
            f"Calibrated probability of default ({prob*100:.2f}%) falls within secondary review range. "
            f"Senior underwriter approval and secondary asset verification required."
        )
    else:
        tier = "Low Risk"
        recommendation = "APPROVED"
        decision_notes = (
            f"Calibrated probability of default ({prob*100:.2f}%) satisfies tier-1 prime underwriting "
            f"criteria. Eligible for automated STP processing and direct booking."
        )
        
    credit_score = int(round(300 + (1.0 - min(max(prob, 0.0), 1.0)) * 550))
    factors = compute_factor_attribution(model, input_df)
    policy_flags = generate_policy_flags(data_dict)
    
    assessment_id = f"APP-{int(time.time()*1000) % 1000000:06d}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    
    result = {
        "assessment_id": assessment_id,
        "timestamp": timestamp,
        "default_probability": round(prob, 4),
        "probability_percent": round(prob * 100, 2),
        "default_prediction": pred,
        "threshold": round(threshold, 4),
        "threshold_percent": round(threshold * 100, 2),
        "Result": "High Risk" if pred == 1 else "Low Risk",
        "risk_tier": tier,
        "credit_score": credit_score,
        "recommendation": recommendation,
        "decision_notes": decision_notes,
        "key_drivers": factors,
        "policy_flags": policy_flags,
        "inputs": data_dict
    }
    
    # Store in history registry
    history_record = {
        "assessment_id": assessment_id,
        "timestamp": timestamp,
        **data_dict,
        "default_probability": round(prob, 4),
        "probability_percent": round(prob * 100, 2),
        "default_prediction": pred,
        "threshold": round(threshold, 4),
        "Result": result["Result"],
        "risk_tier": tier,
        "credit_score": credit_score,
        "recommendation": recommendation,
        "policy_flags": policy_flags
    }
    ml_state["history"].insert(0, history_record)
    # Cap history at 100 entries
    if len(ml_state["history"]) > 100:
        ml_state["history"].pop()
        
    return result


@app.post("/predict")
def predict(data: LoanApplication):
    """
    Primary prediction endpoint.
    Maintains 100% backward compatibility with original output keys while returning rich metadata.
    """
    try:
        # Pydantic v2 model_dump with fallback to dict
        data_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        return execute_inference(data_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
def api_predict(data: LoanApplication):
    return predict(data)

@app.get("/api/history")
def get_history():
    """
    Returns historical assessments recorded during the session.
    """
    return {
        "total": len(ml_state["history"]),
        "records": ml_state["history"]
    }

@app.get("/api/model-info")
def get_model_info():
    """
    Returns governance metadata, calibration parameters, and global feature importance.
    """
    return {
        "model_architecture": "CalibratedClassifierCV (Base: XGBoost Classifier, Sigmoid, 5-Fold)",
        "decision_threshold": ml_state["threshold"],
        "decision_threshold_percent": round(ml_state["threshold"] * 100, 2),
        "training_dataset_size": 32581,
        "validation_metric": "F1-Score Maximization on Precision-Recall Curve",
        "global_importance": ml_state["global_importance"]
    }

@app.get("/api/export-history")
def export_history_csv():
    """
    Generates a CSV export of all assessments in the registry.
    """
    records = ml_state["history"]
    if not records:
        return Response(content="assessment_id,timestamp,recommendation\n", media_type="text/csv")
        
    export_rows = []
    for r in records:
        export_rows.append({
            "assessment_id": r.get("assessment_id"),
            "timestamp": r.get("timestamp"),
            "age": r.get("person_age"),
            "income": r.get("person_income"),
            "home_ownership": r.get("person_home_ownership"),
            "emp_length": r.get("person_emp_length"),
            "loan_intent": r.get("loan_intent"),
            "loan_grade": r.get("loan_grade"),
            "loan_amnt": r.get("loan_amnt"),
            "interest_rate": r.get("loan_int_rate"),
            "dti_ratio": r.get("loan_percent_income"),
            "prior_default": r.get("cb_person_default_on_file"),
            "cred_hist_len": r.get("cb_person_cred_hist_length"),
            "pd_probability": r.get("default_probability"),
            "credit_score": r.get("credit_score"),
            "risk_tier": r.get("risk_tier"),
            "recommendation": r.get("recommendation")
        })
        
    df_exp = pd.DataFrame(export_rows)
    csv_content = df_exp.to_csv(index=False)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=credit_assessments_export.csv"}
    )

@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "service": "axiom-credit-risk",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

# Mount static directory and serve index.html at root
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Axiom Credit Risk Engine API Active. Access /docs for API documentation."}
