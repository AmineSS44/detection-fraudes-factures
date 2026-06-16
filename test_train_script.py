"""Script temporaire pour tester le pipeline d'entraînement."""
from ml.fraud_detector import FraudDetector

d = FraudDetector()
results = d.train_models("data/invoices.csv")
print("Training OK")
print(f"Models: {len(results['models'])}")
for m in results["models"]:
    print(f"  {m['name']}: F1={m['f1_score']}, Precision={m['precision']}, Recall={m['recall']}, AUC={m['auc_roc']}")
print()

# Tester la prédiction
from ml.feature_engineering import FeatureVector

features = FeatureVector(
    amount_zscore=2.5,
    tax_inconsistency=True,
    duplicate_flag=True,
    weekend_flag=True,
    round_amount_flag=False,
    vendor_deviation=0.8,
)
result = d.predict(features)
print(f"Prediction: score={result.fraud_score:.4f}, label={result.fraud_label}, reason={result.fraud_reason}")
