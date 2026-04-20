from fastapi import FastAPI
import joblib
import pandas as pd
import sklearn

app = FastAPI()

# 1. Load the model
saved_model = joblib.load('credit_model.pkl')
model = saved_model['model']

@app.post("/predict")
def predict_risk(data: dict):
    # 2. Convert incoming JSON into a DataFrame
    column_order = ['Credit amount', 'Checking account', 'Duration','Purpose', 'Housing']
    df = pd.DataFrame([data])
    df = df[column_order]
    
    # 3. Get Probability from the Random Forest
    probability = model.predict_proba(df)[0][1]
    
    # 4. Apply Risk Tiers
    if probability < 0.20:
        tier = "Tier 1: Low Risk"
        action = "Auto-Approve"
    elif probability < 0.45:
        tier = "Tier 2: Medium Risk"
        action = "Conditional Approval (Manual Review)"
    else:
        tier = "Tier 3: High Risk"
        action = "Decline"
        
    return {
        "probability_of_default": round(float(probability), 4),
        "risk_tier": tier,
        "recommendation": action
    }