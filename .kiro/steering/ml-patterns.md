---
title: ML Patterns
inclusion: auto
description: Utiliser quand on crée ou modifie des modèles ML, features, ou détection de fraude
---

## Modèles à comparer obligatoirement
1. Isolation Forest (non supervisé, principal)
2. One-Class SVM (non supervisé, comparaison)
3. Random Forest (supervisé, si labels dispo)

## Métriques de performance à logger
- F1 Score, Precision, Recall, AUC-ROC
- Matrice de confusion
- Sauvegarder dans results/model_comparison.json

## Features engineering obligatoires
- amount_zscore, tax_inconsistency, duplicate_flag
- weekend_flag, round_amount_flag, vendor_deviation