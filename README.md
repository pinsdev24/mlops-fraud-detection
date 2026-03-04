# Fraud Detection MLOps — M2 IA

> **Industrialisation d'un modèle de détection d'anomalies bancaires**  
> Stack : Python 3.11 · XGBoost · MLflow · FastAPI · Docker · GitLab CI/CD

---

## ⚡ Démarrage rapide (machine vierge)

```bash
# 1. Cloner le dépôt
git clone <URL_DU_DEPOT> && cd mlops-fraud-detection

# 2. Télécharger le dataset
#    Placer creditcard.csv dans data/raw/
#    Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 3. Démarrer MLflow (UI sur http://localhost:5000)
docker compose -f docker/docker-compose.yml up mlflow -d

# 4. Lancer un entraînement complet
docker compose -f docker/docker-compose.yml run --rm training

# 5. Consulter les résultats dans MLflow UI
open http://localhost:5000
```

---

## 🗂️ Structure du projet

```
mlops-fraud-detection/
├── data/
│   ├── raw/            # creditcard.csv (non versionné — voir .gitignore)
│   └── processed/      # données nettoyées (générées par le pipeline)
├── src/
│   ├── data/           # loader.py · quality_gate.py (CI-3)
│   ├── preprocessing/  # cleaner.py · features.py
│   ├── models/         # train.py (CI-4) · compare.py (CI-5)
│   └── evaluation/     # metrics.py · plots.py · drift_detector.py (CI-9)
├── api/                # main.py (CI-7/CI-8) · schemas.py
├── docker/             # Dockerfile.train · Dockerfile.api · docker-compose.yml
├── configs/            # train_config.yaml (hyperparamètres versionnés)
├── notebooks/          # 01_eda.ipynb
├── tests/              # CI-1 : 42 tests unitaires
├── artifacts/          # figures, rapports (générés — non versionnés)
├── .gitlab-ci.yml      # Pipeline CI/CD complet (CI-1 à CI-9)
└── README.md
```

---

## 🔧 Commandes détaillées

### Environnement local (sans Docker)

```bash
# Installer uv (gestionnaire de dépendances)
pip install uv

# Créer le venv et installer les dépendances
uv sync

# Lancer la CI-3 (data quality gate)
uv run python -m src.data.quality_gate --data data/raw/creditcard.csv

# Lancer un entraînement (MLflow local SQLite)
MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db \
  uv run python -m src.models.train --config configs/train_config.yaml

# Voir le leaderboard des runs
uv run python -m src.models.compare --experiment fraud-detection-v1 --leaderboard

# Lancer les tests unitaires (CI-1)
uv run pytest tests/ -v

# Vérifier la qualité du code (CI-2)
uv run black --check src/ tests/ api/
uv run flake8 src/ tests/ api/ --max-line-length=100
```

### Docker

```bash
# Construire l'image d'entraînement
docker build -f docker/Dockerfile.train -t fraud-trainer:latest .

# Démarrer MLflow seul
docker compose -f docker/docker-compose.yml up mlflow -d

# Lancer un entraînement dans Docker
docker compose -f docker/docker-compose.yml run --rm training

# Démarrer l'API de prédiction (bonus)
docker compose -f docker/docker-compose.yml --profile api up api -d

# Arrêter tout
docker compose -f docker/docker-compose.yml down
```

### API de prédiction

```bash
# Tester l'API (une fois l'API démarrée sur le port 8080)
curl http://localhost:8080/health

# Prédire sur une transaction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [{
      "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
      "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
      "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
      "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
      "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
      "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
      "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
      "Amount": 149.62,
      "Time": 0.0
    }]
  }'

# Documentation interactive Swagger
open http://localhost:8080/docs
```

---

## 📊 Résultats (run champion — XGBoost)

| Métrique | Valeur |
|---|---|
| **PR-AUC** (métrique principale) | **0.8126** |
| ROC-AUC | 0.9765 |
| F1-score (fraude) | 0.8293 |
| Précision (fraude) | 98.6% |
| Recall (fraude) | 71.6% |
| Seuil optimal | 0.9907 |

> Le seuil est calibré sur le validation set avec la stratégie F1-optimal.

---

## ✅ Check-list de remise

- [x] `docker-compose up mlflow -d` — MLflow UI accessible sur port 5000
- [x] `docker compose run --rm training` — entraînement + run MLflow visible
- [x] Modèle et artefacts visibles dans MLflow (metrics + figures + SHAP)
- [x] `pytest tests/ -v` — 42 tests passent
- [x] `black --check src/` et `flake8 src/` — 0 erreur
- [x] `.gitlab-ci.yml` complet (CI-1 à CI-9)
- [x] API FastAPI avec endpoint `/predict` documenté
- [ ] Rapport `.docx` (section 8)
- [ ] Présentation orale 20 min

---

## 🔒 Aspects sécuritaires

- **Réseau Docker isolé** : bridge interne `fraud-net`, seuls les ports 5000 et 8080 exposés
- **Non-root** : les conteneurs s'exécutent sous l'utilisateur `appuser`
- **Secrets** : aucun credential hardcodé — utiliser `.env` (gitignored) ou variables CI/CD
- **Données** : `data/` est dans `.gitignore` — jamais commitées
- **Images** : base `python:3.11-slim` (surface d'attaque minimale)
- **Dépendances** : `uv.lock` assure la reproductibilité exacte des versions
- **Validation inputs** : Pydantic valide chaque requête API (types, bornes)
- **API key** : `API_SECRET_KEY` via variable d'environnement

---

## 🧪 Pipeline CI/CD GitLab

| Stage | Job | CI |
|---|---|---|
| lint | black + flake8 | CI-2 |
| test | pytest + coverage | CI-1 |
| data-quality | quality gate (exit 1 si >5% missing) | CI-3 |
| train | entraînement auto + comparaison champion | CI-4, CI-5 |
| build | image Docker versionnée (Git hash + MLflow ID) | CI-6 |
| deploy | API + rollback | CI-7 |
| monitor | détection drift (Evidently) | CI-8, CI-9 |
