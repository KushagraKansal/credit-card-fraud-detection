# 💳 Credit Card Fraud Detection System

An end-to-end machine learning system that detects fraudulent credit card transactions in real time, with built-in explainability so every prediction comes with a clear reason — not just a label.

Built on the **IEEE-CIS Fraud Detection dataset** (590,540 transactions, 223 engineered features), using a **Random Forest** classifier and **SHAP** for explainability, deployed as an interactive **Streamlit** web app.

## 🎯 Key Results

| Metric | Value |
|---|---|
| Model | Random Forest (100 estimators) |
| AUC Score | 0.9212 |
| Fraud Recall | 62% |
| Decision Threshold | 0.25 |

## ✨ Features

- **Real-time prediction** — enter transaction details and get an instant fraud/legitimate verdict with a risk score
- **Quick Demo mode** — one-click buttons load real fraud/legitimate transactions from the test set for guaranteed, reproducible results
- **SHAP explainability** — every prediction comes with a chart showing exactly which features pushed it toward fraud or legitimate
- **Class imbalance handling** — trained using SMOTE to address the extreme rarity of fraud in the dataset (<3.5%)

## 🛠️ Tech Stack

- **Python**, **scikit-learn** (Random Forest)
- **imbalanced-learn** (SMOTE)
- **SHAP** (model explainability)
- **Streamlit** (web app)
- **Pandas**, **NumPy**, **Matplotlib**

## 🚀 Running Locally

```bash
git clone https://github.com/KushagraKansal/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 📁 Project Structure

```
├── app.py                   # Streamlit application
├── fraud_detection.ipynb    # Model training & experimentation notebook
├── fraud_model.pkl          # Trained Random Forest model (tracked via Git LFS)
├── model_config.json        # Feature list & decision threshold
├── demo_examples.json        # Real fraud/legitimate examples for the Quick Demo
└── requirements.txt          # Python dependencies
```

## 📊 Dataset

[IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection) — a real-world benchmark dataset for fraud detection, provided by Vesta Corporation via a Kaggle competition.
