README.md
# Loan Approval Prediction — Machine Learning in Action

## Overview
This project predicts whether a loan application will be approved or rejected using Machine Learning models — **Logistic Regression** and **Random Forest Classifier** — and an interactive **Streamlit app** deployed using **ngrok**.

## Steps
1. Data cleaning and encoding (converting Y/N to 1/0)
2. Feature engineering (applicant income, credit history, property area)
3. Model training and evaluation (accuracy, ROC AUC, precision, recall)
4. Streamlit app for real-time loan approval prediction

## Results
- **Logistic Regression:** 85.37% accuracy, ROC AUC 0.8477  
- **Random Forest:** 86.18% accuracy, ROC AUC 0.8592  
- Top features: Credit History, Applicant Income, Loan Amount

## Live Deployment
- Built with: Streamlit + ngrok
- Run locally:
  ```bash
  streamlit run app.py
ngrok http 8501
pip install -r requirements.txt


