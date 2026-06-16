"""
Script d'entraînement et d'évaluation des modèles ML.
Pipeline complet: génération du dataset → entraînement de 3 modèles → évaluation → sauvegarde.

Exécution: python ml/train.py
"""

import json
import os
import sys

# Ajouter le répertoire racine au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_dataset import DatasetGenerator
from ml.fraud_detector import FraudDetector, COMPARISON_PATH, MODEL_PATH


def main():
    """Pipeline complet d'entraînement et d'évaluation des modèles ML."""

    print("=" * 60)
    print("  Pipeline d'entraînement ML - Détection de fraude")
    print("=" * 60)

    # Étape 1: Génération du dataset synthétique
    print("\n[1/4] Génération du dataset synthétique...")
    generator = DatasetGenerator(seed=42)
    dataset = generator.generate()

    # Statistiques du dataset
    normal_count = len(dataset[dataset["label"] == "Normal"])
    fraud_count = len(dataset[dataset["label"] == "Frauduleux"])
    print(f"  ✓ {len(dataset)} factures générées ({normal_count} normales, {fraud_count} frauduleuses)")

    # Export vers CSV pour l'entraînement
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "invoices.csv")
    generator.export_csv(dataset, csv_path)
    print(f"  ✓ Dataset exporté vers {csv_path}")

    # Export vers SQLite
    db_path = os.path.join(data_dir, "invoices.db")
    generator.export_sqlite(dataset, db_path)
    print(f"  ✓ Dataset exporté vers {db_path}")

    # Étape 2: Entraînement des 3 modèles
    print("\n[2/4] Entraînement des modèles ML...")
    print("  - Isolation Forest (modèle principal)")
    print("  - One-Class SVM")
    print("  - Random Forest")

    # Chemin absolu du modèle
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, MODEL_PATH)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    detector = FraudDetector(model_path=model_path)
    results = detector.train_models(csv_path)
    print("  ✓ Entraînement terminé")

    # Étape 3: Évaluation et métriques
    print("\n[3/4] Résultats de l'évaluation:")
    print("-" * 50)
    print(f"{'Modèle':<20} {'F1':>8} {'Précision':>10} {'Rappel':>8} {'AUC-ROC':>8}")
    print("-" * 50)

    for model_info in results["models"]:
        print(
            f"  {model_info['name']:<18} "
            f"{model_info['f1_score']:>6.4f} "
            f"{model_info['precision']:>8.4f} "
            f"{model_info['recall']:>8.4f} "
            f"{model_info['auc_roc']:>8.4f}"
        )

    print("-" * 50)

    # Afficher les matrices de confusion
    print("\n  Matrices de confusion (format [[TP, FP], [FN, TN]]):")
    for model_info in results["models"]:
        cm = model_info["confusion_matrix"]
        print(f"    {model_info['name']}: {cm}")

    # Étape 4: Sauvegarde des résultats
    print("\n[4/4] Sauvegarde des résultats...")

    # Sauvegarder model_comparison.json
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    comparison_path = os.path.join(project_root, COMPARISON_PATH)

    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Résultats sauvegardés dans {comparison_path}")

    # Vérifier que le modèle Isolation Forest est bien persisté
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / 1024
        print(f"  ✓ Modèle Isolation Forest sauvegardé dans {model_path} ({model_size:.1f} KB)")
    else:
        print(f"  ✗ ERREUR: Modèle non trouvé à {model_path}")
        sys.exit(1)

    # Résumé final
    print("\n" + "=" * 60)
    print("  Pipeline terminé avec succès!")
    print("=" * 60)
    print(f"\n  Fichiers générés:")
    print(f"    - {csv_path}")
    print(f"    - {db_path}")
    print(f"    - {model_path}")
    print(f"    - {comparison_path}")

    return results


if __name__ == "__main__":
    main()
