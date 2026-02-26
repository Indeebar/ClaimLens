import sys
import os
sys.path.insert(0, r'f:\ClaimLens')

from models.claim_nlp.embed import load_nlp_model
from models.claim_nlp.anomaly_score import score_text

print("Loading NLP model...")
load_nlp_model()

test1 = 'car caught fire no witnesses'
test2 = 'minor scratch on rear bumper'

res1 = score_text(test1)
res2 = score_text(test2)

print(f"\nTest 1: '{test1}'")
print(f"  anomaly_score = {res1['anomaly_score']} (expected > 0.6)")
print(f"  result detail: {res1}")

print(f"\nTest 2: '{test2}'")
print(f"  anomaly_score = {res2['anomaly_score']} (expected < 0.2)")
print(f"  result detail: {res2}")

if res1['anomaly_score'] > 0.6 and res2['anomaly_score'] < 0.2:
    print("\n✅ Phase 3 verification PASSED")
else:
    print("\n❌ Phase 3 verification FAILED")
