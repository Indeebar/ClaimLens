import shap
import joblib
import numpy as np
import os

_explainer = None
_feature_cols = None

_default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgb_fraud_model.pkl')

def load_explainer(model_path=None):
    global _explainer, _feature_cols
    if model_path is None:
        model_path = _default_path
    artifact = joblib.load(model_path)
    _explainer = shap.TreeExplainer(artifact['model'])
    _feature_cols = artifact['feature_cols']

def explain(features_dict: dict) -> dict:
    import pandas as pd
    row = pd.DataFrame([features_dict])[_feature_cols].fillna(0)
    shap_vals = _explainer.shap_values(row)[0]
    top_3 = sorted(zip(_feature_cols, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:3]
    return {
        'top_factors': [{'feature': k, 'impact': round(float(v), 4)} for k, v in top_3],
        'base_value': round(float(_explainer.expected_value), 4)
    }
