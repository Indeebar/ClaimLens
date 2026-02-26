import numpy as np
from models.claim_nlp.embed import embed, _pattern_embeddings, _patterns

KEYWORD_WEIGHTS = {
    'total loss': 0.4, 'fire': 0.3, 'stolen': 0.3, 'no witnesses': 0.35,
    'no cctv': 0.4, 'fled': 0.25, 'remote': 0.2, 'deserted': 0.2,
    'overnight': 0.15, 'basement': 0.2, '3am': 0.25, 'no cameras': 0.35
}

def score_text(incident_text: str) -> dict:
    if not incident_text or len(incident_text.strip()) < 5:
        return {'anomaly_score': 0.0, 'method': 'empty', 'top_match': None}

    text_lower = incident_text.lower()

    # Layer 1: Semantic similarity
    text_emb = embed(incident_text)
    sims = np.dot(_pattern_embeddings, text_emb)  # cosine (normalized vectors)
    max_sim = float(sims.max())
    top_pattern_idx = int(sims.argmax())

    # Layer 2: Keyword scan
    keyword_score = 0.0
    triggered = []
    for kw, weight in KEYWORD_WEIGHTS.items():
        if kw in text_lower:
            keyword_score = min(1.0, keyword_score + weight)
            triggered.append(kw)

    # Combine: semantic 60% + keyword 40%
    combined = min(1.0, (max_sim * 0.6) + (keyword_score * 0.4))

    return {
        'anomaly_score':    round(combined, 4),
        'semantic_score':   round(max_sim, 4),
        'keyword_score':    round(keyword_score, 4),
        'triggered_keywords': triggered,
        'top_fraud_pattern': _patterns[top_pattern_idx] if max_sim > 0.4 else None
    }
