import torch
from PIL import Image
from dataset import VAL_TRANSFORMS, CLASSES
from model import DamageClassifier

_model = None  # Loaded once at API startup

def load_model(path='models/damage_classifier/best_model.pt'):
    global _model
    _model = DamageClassifier.load(path)
    print(f'[DL] Damage classifier loaded from {path}')

def predict_damage(image_input):
    # image_input: file path (str) or PIL.Image
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')

    tensor = VAL_TRANSFORMS(img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()

    return {
        'severity':     CLASSES[pred],
        'severity_idx': pred,
        'confidence':   round(probs[pred].item(), 4),
        'all_probs':    {c: round(probs[i].item(), 4) for i, c in enumerate(CLASSES)}
    }
