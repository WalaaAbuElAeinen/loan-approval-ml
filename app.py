import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = "/content/model_random_forest.joblib"   
mdl = joblib.load(MODEL_PATH)

def get_feature_cols(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return []

FEATURE_COLS = get_feature_cols(mdl)
if not FEATURE_COLS:
    st.error("Model is missing feature_names_in_. Refit the model using a pandas DataFrame.")
    st.stop()

# Top features 
TOP_FEATURES = [
    "Credit_History",
    "ApplicantIncome",
    "LoanAmount",
    "CoapplicantIncome",
    "Property_Area_Semiurban",
    "Loan_Amount_Term",
    "Gender",
    "Dependents_3+",
    "Education",
    "Married",
    "Self_Employed",
    "Dependents_1",
    "Dependents_2",
    "Property_Area_Urban"
    
]

# UI
st.set_page_config(page_title="Loan Approval (Top Features)", page_icon="💳", layout="centered")
st.title("💳 Loan Approval Predictor — Top Features")
st.caption("Inputs limited to the strongest predictors. Other model features default to 0.")

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_history = st.selectbox("Credit_History", [1.0, 0.0], index=0)
        applicant_income = st.number_input("ApplicantIncome", min_value=0.0, step=100.0)
        loan_amount = st.number_input("LoanAmount", min_value=0.0, step=1.0)
        coapplicant_income = st.number_input("CoapplicantIncome", min_value=0.0, step=100.0)
        loan_amount_term = st.number_input("Loan_Amount_Term (months)", min_value=0.0, step=6.0)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self_Employed", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

    property_area = st.selectbox("Property_Area", ["Rural", "Semiurban", "Urban"], index=1)
    submitted = st.form_submit_button("Predict")

# Build aligned feature vector from top-features UI 
def build_row_from_inputs():
    # Start with zeros for ALL model features
    row = pd.Series(0, index=FEATURE_COLS, dtype="float64")

    # Direct numeric features
    if "Credit_History" in row.index:        row["Credit_History"] = float(credit_history)
    if "ApplicantIncome" in row.index:       row["ApplicantIncome"] = float(applicant_income)
    if "LoanAmount" in row.index:            row["LoanAmount"] = float(loan_amount)
    if "CoapplicantIncome" in row.index:     row["CoapplicantIncome"] = float(coapplicant_income)
    if "Loan_Amount_Term" in row.index:      row["Loan_Amount_Term"] = float(loan_amount_term)

    # Binary maps 
    if "Gender" in row.index:
        row["Gender"] = 1.0 if gender in ["Male", "M"] else 0.0
    if "Married" in row.index:
        row["Married"] = 1.0 if married == "Yes" else 0.0
    if "Education" in row.index:
        row["Education"] = 1.0 if education == "Graduate" else 0.0
    if "Self_Employed" in row.index:
        row["Self_Employed"] = 1.0 if self_employed == "Yes" else 0.0

    # Dependents dummies: base = "0"
    # Expected columns from your training: Dependents_1, Dependents_2, Dependents_3+
    dep_map = {"1": "Dependents_1", "2": "Dependents_2", "3+": "Dependents_3+"}
    if dependents in dep_map and dep_map[dependents] in row.index:
        row[dep_map[dependents]] = 1.0
    # else "0" case: keep all zeros

    # Property_Area dummies: base = "Rural"
    # Expected columns: Property_Area_Semiurban, Property_Area_Urban
    if property_area == "Semiurban" and "Property_Area_Semiurban" in row.index:
        row["Property_Area_Semiurban"] = 1.0
    elif property_area == "Urban" and "Property_Area_Urban" in row.index:
        row["Property_Area_Urban"] = 1.0
    # else Rural: keep zeros

    return pd.DataFrame([row])

# Predict
if submitted:
    try:
        X = build_row_from_inputs()
        # predict_proba if available
        if hasattr(mdl, "predict_proba"):
            proba = float(mdl.predict_proba(X)[:, 1][0])
        else:
            score = float(getattr(mdl, "decision_function", mdl.predict)(X)[0])
            proba = 1 / (1 + np.exp(-score)) if not (0 <= score <= 1) else score

        label = "Approved" if proba >= 0.5 else "Not Approved"
        st.subheader(label)
        st.metric("Approval Probability", f"{proba:.3f}")

        with st.expander("Debug: model feature vector"):
            st.write(X.T.rename(columns={0: "value"}))

        # Show top features for transparency if model is RF
        if hasattr(mdl, "feature_importances_"):
            st.subheader("Top Features by Importance")
            importances = pd.Series(mdl.feature_importances_, index=FEATURE_COLS)
            top = importances.sort_values(ascending=False).head(15)
            st.bar_chart(top.sort_values())  # Streamlit bar chart (horizontal via sort)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
