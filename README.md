claimlens/
├── data/
│   ├── raw/                          # Downloaded datasets (gitignored)
│   └── processed/
├── models/
│   ├── damage_classifier/
│   │   ├── train.py                  # EfficientNet training loop
│   │   ├── model.py                  # EfficientNet-B0 wrapper
│   │   ├── dataset.py                # PyTorch Dataset + DataLoader
│   │   ├── evaluate.py               # Confusion matrix + metrics
│   │   └── predict.py                # Single-image inference
│   ├── claim_nlp/
│   │   ├── embed.py                  # DistilBERT sentence embeddings
│   │   ├── anomaly_score.py          # Cosine sim + rule-based fallback
│   │   └── fraud_patterns.json       # Known fraud text patterns
│   └── fraud_classifier/
│       ├── feature_eng.py            # India-specific feature engineering
│       ├── train.py                  # XGBoost + SMOTE + CV
│       ├── shap_explain.py           # SHAP explanation generator
│       └── predict.py                # Inference with explanation
├── api/
│   ├── main.py                       # FastAPI app entry point
│   ├── routers/
│   │   └── claim.py                  # POST /predict/fraud endpoint
│   └── schemas.py                    # Pydantic request/response models
├── scripts/
│   ├── download_data.sh              # One-command dataset download
│   └── download_models.sh            # Pull trained models from S3
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD
├── Dockerfile
├── entrypoint.sh
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
