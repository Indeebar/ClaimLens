import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from models.fraud_classifier.feature_eng import engineer_features

# Try to find the data file
_this_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(_this_dir, '../../data/raw/claims_tabular/insurance_claims.csv')
MODEL_PATH = os.path.join(_this_dir, 'xgb_fraud_model.pkl')

FEATURE_COLS = [
    'months_as_customer', 'age', 'policy_annual_premium',
    'umbrella_limit', 'capital-gains', 'capital-loss',
    'incident_hour_of_day', 'number_of_vehicles_involved',
    'bodily_injuries', 'witnesses', 'injury_claim',
    'property_claim', 'vehicle_claim', 'total_claim_amount',
    # India-specific engineered features:
    'city_tier', 'is_festival_season', 'policy_age_days',
    'incident_hour_bin', 'claim_to_value_ratio'
]

def train():
    df = pd.read_csv(DATA_PATH)
    print(f'Columns: {df.columns.tolist()}')

    # Normalize fraud column name
    fraud_col = None
    for col in df.columns:
        if 'fraud' in col.lower():
            fraud_col = col
            break
    if fraud_col is None:
        raise ValueError('No fraud column found in dataset')
    if fraud_col != 'fraud_reported':
        df = df.rename(columns={fraud_col: 'fraud_reported'})

    df = engineer_features(df)
    df['label'] = (df['fraud_reported'] == 'Y').astype(int)

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    y = df['label']

    print(f'Dataset: {len(X)} rows | Fraud rate: {y.mean():.2%}')
    print(f'Using features: {available}')

    # SMOTE for class imbalance (fraud is rare ~15-20%)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f'After SMOTE: {len(X_res)} rows | Fraud rate: {y_res.mean():.2%}')

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='auc', random_state=42, n_jobs=-1
    )

    # CV on original data for honest estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f'CV AUC: {scores.mean():.4f} +/- {scores.std():.4f}')

    # Train final model on SMOTE-balanced data
    model.fit(X_res, y_res)
    joblib.dump({'model': model, 'feature_cols': available}, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

if __name__ == '__main__':
    train()
