import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routers.claim import router as claim_router, load_fraud_model
from models.damage_classifier.predict import load_model as load_dl_model
from models.claim_nlp.embed import load_nlp_model
from models.fraud_classifier.shap_explain import load_explainer

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading all models...')
    load_dl_model()
    load_nlp_model()
    load_fraud_model()
    load_explainer()
    print('All models loaded. API ready.')
    yield

app = FastAPI(
    title='ClaimLens API',
    description='Motor insurance fraud detection for India',
    version='1.0.0',
    lifespan=lifespan
)
app.include_router(claim_router, prefix='/api/v1', tags=['Fraud Detection'])

@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'claimlens'}
