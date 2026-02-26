from fastapi import APIRouter, UploadFile, File, Form
from fastapi.exceptions import HTTPException
from PIL import Image
import io, json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from api.schemas import ClaimInput, FraudPredictionResponse
from models.damage_classifier.predict import predict_damage
from models.claim_nlp.anomaly_score import score_text
from models.fraud_classifier.feature_eng import engineer_features
from models.fraud_classifier.shap_explain import explain
import joblib
import pandas as pd

router = APIRouter()
_fraud_artifact = None

def load_fraud_model(path=None):
    global _fraud_artifact
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '../../models/fraud_classifier/xgb_fraud_model.pkl')
    _fraud_artifact = joblib.load(path)

@router.post('/predict/fraud', response_model=FraudPredictionResponse)
async def predict_fraud(
    image: UploadFile = File(...),
    claim_data: str = Form(...)   # JSON string
):
    try:
        claim_dict = json.loads(claim_data)
        claim = ClaimInput(**claim_dict)
    except Exception as e:
        raise HTTPException(400, f'Invalid claim_data: {e}')

    # 1. DL: damage severity from image
    img_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(img_bytes))
    damage_result = predict_damage(pil_img)

    # 2. NLP: anomaly score from text
    nlp_result = {'anomaly_score': 0.0, 'triggered_keywords': []}
    if claim.incident_description:
        nlp_result = score_text(claim.incident_description)

    # 3. ML: feature engineering + XGBoost
    df = pd.DataFrame([claim.model_dump()])
    df = engineer_features(df, damage_preds=[damage_result])
    df['nlp_anomaly_score'] = nlp_result.get('anomaly_score', 0.0)

    feat_cols = _fraud_artifact['feature_cols']
    avail = [c for c in feat_cols if c in df.columns]
    X = df[avail].fillna(0)
    fraud_prob = float(_fraud_artifact['model'].predict_proba(X)[0][1])

    # 4. SHAP explanation
    shap_result = explain(df[avail].iloc[0].to_dict())

    # 5. Risk level + recommendation
    if fraud_prob >= 0.7:
        risk, rec = 'HIGH',   'Flag for manual investigation immediately.'
    elif fraud_prob >= 0.4:
        risk, rec = 'MEDIUM', 'Assign to senior adjuster for review.'
    else:
        risk, rec = 'LOW',    'Proceed with standard claim processing.'

    return FraudPredictionResponse(
        fraud_probability=round(fraud_prob, 4),
        fraud_flag=fraud_prob >= 0.5,
        risk_level=risk,
        damage_severity=damage_result['severity'],
        damage_confidence=damage_result['confidence'],
        anomaly_score=nlp_result.get('anomaly_score'),
        triggered_keywords=nlp_result.get('triggered_keywords', []),
        top_shap_factors=shap_result['top_factors'],
        recommendation=rec
    )
