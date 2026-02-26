from pydantic import BaseModel
from typing import Optional, List

class ClaimInput(BaseModel):
    months_as_customer: float = 0
    age: float = 35
    policy_annual_premium: float = 0
    total_claim_amount: float = 0
    vehicle_claim: float = 0
    injury_claim: float = 0
    property_claim: float = 0
    incident_hour_of_day: int = 12
    witnesses: int = 1
    number_of_vehicles_involved: int = 1
    incident_city: str = 'mumbai'
    incident_date: str = '2024-01-15'
    policy_bind_date: str = '2022-01-01'
    incident_description: Optional[str] = None

class SHAPFactor(BaseModel):
    feature: str
    impact: float

class FraudPredictionResponse(BaseModel):
    fraud_probability: float
    fraud_flag: bool
    risk_level: str                        # LOW / MEDIUM / HIGH
    damage_severity: str
    damage_confidence: float
    anomaly_score: Optional[float] = None
    triggered_keywords: Optional[List[str]] = None
    top_shap_factors: List[SHAPFactor]
    recommendation: str
