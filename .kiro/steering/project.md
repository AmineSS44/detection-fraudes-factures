---
title: Invoice Fraud Detection System
inclusion: always
---

# Contexte projet
Système de détection de fraude sur factures pour un bureau comptable marocain.

## Stack technique
- Backend: Python 3.11 + FastAPI
- Frontend: Streamlit
- OCR: YOLOv8 + Tesseract
- ML: scikit-learn (Isolation Forest, One-Class SVM, Random Forest)
- DB: SQLite
- Auth: JWT

## Structure des fichiers
- app.py → Dashboard Streamlit principal
- backend/api.py → Routes FastAPI
- ml/fraud_detector.py → Modèles ML
- ml/feature_engineering.py → Features
- data/generate_dataset.py → Dataset synthétique

## Conventions
- Langue du code: anglais, commentaires: français
- Monnaie: MAD (Dirham marocain)
- Toujours retourner fraud_score (0-1) + fraud_label + fraud_reason