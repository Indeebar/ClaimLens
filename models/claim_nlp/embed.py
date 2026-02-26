from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

_model = None
_pattern_embeddings = None
_patterns = None

def load_nlp_model(patterns_path=None):
    global _model, _pattern_embeddings, _patterns
    if patterns_path is None:
        patterns_path = os.path.join(os.path.dirname(__file__), 'fraud_patterns.json')
    _model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, fast
    with open(patterns_path) as f:
        data = json.load(f)
    _patterns = data['high_risk_patterns']
    _pattern_embeddings = _model.encode(_patterns, normalize_embeddings=True)
    print(f'[NLP] Loaded {len(_patterns)} fraud patterns')

def embed(text: str) -> np.ndarray:
    return _model.encode([text], normalize_embeddings=True)[0]
