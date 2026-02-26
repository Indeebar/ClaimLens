import pandas as pd
import numpy as np

TIER1_CITIES = {'mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'kolkata', 'pune', 'ahmedabad'}
TIER2_CITIES = {'jaipur', 'lucknow', 'surat', 'nagpur', 'indore', 'bhopal', 'visakhapatnam', 'patna'}
FESTIVAL_MONTHS = {10, 11}  # Oct-Nov: Diwali/Navratri

def get_city_tier(city: str) -> int:
    c = str(city).lower().strip()
    if c in TIER1_CITIES: return 1
    if c in TIER2_CITIES: return 2
    return 3

def get_incident_hour_bin(hour: int) -> int:
    if 2 <= hour <= 4:  return 0  # highest risk
    if 22 <= hour or hour <= 1: return 1  # night
    return 2  # day

def engineer_features(df: pd.DataFrame, damage_preds: list = None) -> pd.DataFrame:
    df = df.copy()

    if 'incident_city' in df.columns:
        df['city_tier'] = df['incident_city'].apply(get_city_tier)

    if 'incident_date' in df.columns:
        df['incident_month'] = pd.to_datetime(df['incident_date'], errors='coerce').dt.month
        df['is_festival_season'] = df['incident_month'].isin(FESTIVAL_MONTHS).astype(int)

    if 'policy_bind_date' in df.columns and 'incident_date' in df.columns:
        df['policy_age_days'] = (
            pd.to_datetime(df['incident_date'], errors='coerce') -
            pd.to_datetime(df['policy_bind_date'], errors='coerce')
        ).dt.days.fillna(365)

    if 'incident_hour_of_day' in df.columns:
        df['incident_hour_bin'] = df['incident_hour_of_day'].apply(get_incident_hour_bin)

    if 'total_claim_amount' in df.columns and 'vehicle_claim' in df.columns:
        df['claim_to_value_ratio'] = (df['total_claim_amount'] / (df['vehicle_claim'] + 1)).clip(0, 10)

    # DL fusion: damage-claim mismatch (key multi-modal feature)
    if damage_preds is not None:
        df['damage_severity_idx'] = [p['severity_idx'] for p in damage_preds]
        df['damage_confidence']   = [p['confidence']   for p in damage_preds]
        if 'total_claim_amount' in df.columns:
            claim_norm = df['total_claim_amount'] / df['total_claim_amount'].max()
            df['damage_claim_mismatch'] = ((1 - df['damage_severity_idx'] / 2) * claim_norm).clip(0, 1)

    return df
