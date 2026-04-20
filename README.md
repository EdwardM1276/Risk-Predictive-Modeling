## Automated Credit Risk Decisioning System
This project transforms a predictive machine learning model into a live financial service. It uses a hyperparameter-tuned Random Forest to assess loan default risk and classifies applicants into actionable risk tiers via a FastAPI deployment.

Project Overview
Traditional credit risk assessment often relies on simple demographic filters. This system uses machine learning to identify hidden risk factors, specifically the relationship between loan duration, liquidity, and borrower behavior. The final model achieves an 0.8025 AUC, providing high-resolution predictive power.

Key Features
Predictive Engine: A Random Forest model optimized for stability and accuracy in financial contexts.

Automated Tiering: A business logic layer that translates raw probabilities into three risk tiers: Low Risk (Auto-Approve), Medium Risk (Manual Review), and High Risk (Decline).

Live API: A production-ready FastAPI service that allows external applications to request credit decisions in real time.

Serialization: Model persistence using Joblib for instant loading and prediction.

Tech Stack
Language: Python

Analysis: Pandas, Scikit-Learn, SciPy

Deployment: FastAPI, Uvicorn

Model: Random Forest Classifier

Installation and Setup
Install dependencies: pip install fastapi uvicorn scikit-learn pandas joblib scipy

Ensure the model file credit_model.pkl is in the project directory.

Launch the API: uvicorn main:app --reload

Access the interactive documentation at http://127.0.0.1:8000/docs

Results
The project successfully moved a baseline model from 0.77 to 0.8025 AUC. By deploying this as an API, the system can now provide instant risk classifications rather than requiring manual batch processing of data.
