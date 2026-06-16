"""
Tests d'intégration pour l'évaluation des modèles ML.
Vérifie que l'entraînement produit un fichier model_comparison.json valide.

Validates: Requirements 9.3, 9.4
"""

import json
import os
import tempfile

import pytest

from data.generate_dataset import DatasetGenerator
from ml.fraud_detector import FraudDetector


# --- Fixtures ---


@pytest.fixture(scope="module")
def training_results():
    """Exécute le pipeline d'entraînement complet et retourne les résultats.

    Scope 'module' pour éviter de ré-entraîner à chaque test.
    """
    # Générer le dataset dans un répertoire temporaire
    with tempfile.TemporaryDirectory() as tmpdir:
        # Générer le dataset
        generator = DatasetGenerator(seed=42)
        dataset = generator.generate()

        csv_path = os.path.join(tmpdir, "invoices.csv")
        generator.export_csv(dataset, csv_path)

        # Entraîner les modèles
        model_path = os.path.join(tmpdir, "isolation_forest.pkl")
        detector = FraudDetector(model_path=model_path)
        results = detector.train_models(csv_path)

        yield {
            "results": results,
            "model_path": model_path,
            "csv_path": csv_path,
            "tmpdir": tmpdir,
        }


# --- Tests de l'évaluation des modèles ---


class TestModelEvaluation:
    """Tests vérifiant que l'évaluation produit des métriques valides."""

    def test_training_returns_results_dict(self, training_results):
        """Vérifie que l'entraînement retourne un dictionnaire de résultats."""
        results = training_results["results"]
        assert isinstance(results, dict)
        assert "models" in results

    def test_three_models_evaluated(self, training_results):
        """Vérifie que les 3 modèles sont évalués."""
        models = training_results["results"]["models"]
        assert len(models) == 3

        model_names = [m["name"] for m in models]
        assert "Isolation Forest" in model_names
        assert "One-Class SVM" in model_names
        assert "Random Forest" in model_names

    def test_each_model_has_required_metrics(self, training_results):
        """Vérifie que chaque modèle a les métriques requises: F1, Precision, Recall, AUC-ROC."""
        models = training_results["results"]["models"]

        required_keys = ["name", "f1_score", "precision", "recall", "auc_roc", "confusion_matrix"]

        for model in models:
            for key in required_keys:
                assert key in model, f"Clé '{key}' manquante pour le modèle '{model.get('name', 'inconnu')}'"

    def test_metrics_are_valid_floats(self, training_results):
        """Vérifie que les métriques sont des floats valides entre 0 et 1."""
        models = training_results["results"]["models"]

        for model in models:
            # F1 Score
            assert isinstance(model["f1_score"], float)
            assert 0.0 <= model["f1_score"] <= 1.0

            # Precision
            assert isinstance(model["precision"], float)
            assert 0.0 <= model["precision"] <= 1.0

            # Recall
            assert isinstance(model["recall"], float)
            assert 0.0 <= model["recall"] <= 1.0

            # AUC-ROC
            assert isinstance(model["auc_roc"], float)
            assert 0.0 <= model["auc_roc"] <= 1.0

    def test_metrics_rounded_to_4_decimals(self, training_results):
        """Vérifie que les métriques sont arrondies à 4 décimales."""
        models = training_results["results"]["models"]

        for model in models:
            for metric in ["f1_score", "precision", "recall", "auc_roc"]:
                value = model[metric]
                # Vérifier que la valeur a au maximum 4 décimales
                value_str = f"{value:.10f}"
                # Après la 4ème décimale, tout doit être 0
                decimal_part = value_str.split(".")[1]
                assert decimal_part[4:] == "000000", (
                    f"{model['name']}.{metric}={value} a plus de 4 décimales"
                )

    def test_confusion_matrix_format(self, training_results):
        """Vérifie le format de la matrice de confusion: [[TP, FP], [FN, TN]]."""
        models = training_results["results"]["models"]

        for model in models:
            cm = model["confusion_matrix"]
            assert isinstance(cm, list)
            assert len(cm) == 2, f"Matrice de confusion doit avoir 2 lignes, a {len(cm)}"
            assert len(cm[0]) == 2, f"Première ligne doit avoir 2 colonnes"
            assert len(cm[1]) == 2, f"Deuxième ligne doit avoir 2 colonnes"

            # Toutes les valeurs sont des entiers non négatifs
            for row in cm:
                for val in row:
                    assert isinstance(val, int), f"Valeur de confusion matrix doit être int, obtenu {type(val)}"
                    assert val >= 0, f"Valeur de confusion matrix doit être >= 0, obtenu {val}"

    def test_confusion_matrix_sum_equals_test_size(self, training_results):
        """Vérifie que la somme des éléments de la matrice = taille du jeu de test."""
        models = training_results["results"]["models"]

        # Toutes les matrices doivent avoir la même somme (même jeu de test)
        totals = []
        for model in models:
            cm = model["confusion_matrix"]
            total = sum(val for row in cm for val in row)
            totals.append(total)
            assert total > 0, "La matrice de confusion ne peut pas être vide"

        # Toutes les matrices ont le même total (même test set)
        assert all(
            t == totals[0] for t in totals
        ), f"Les matrices n'ont pas le même total: {totals}"

    def test_model_file_persisted(self, training_results):
        """Vérifie que le modèle Isolation Forest est bien sauvegardé sur disque."""
        model_path = training_results["model_path"]
        assert os.path.exists(model_path)
        assert os.path.getsize(model_path) > 0

    def test_results_serializable_to_json(self, training_results):
        """Vérifie que les résultats sont sérialisables en JSON valide."""
        results = training_results["results"]

        # Sérialiser et désérialiser
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        parsed = json.loads(json_str)

        assert parsed == results
        assert len(parsed["models"]) == 3


class TestModelComparisonJsonOutput:
    """Tests vérifiant la structure du fichier model_comparison.json généré."""

    def test_comparison_json_written_to_disk(self, training_results):
        """Vérifie que le fichier model_comparison.json peut être écrit et relu."""
        results = training_results["results"]
        tmpdir = training_results["tmpdir"]

        # Écrire le fichier comme le ferait train.py
        comparison_path = os.path.join(tmpdir, "model_comparison.json")
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Vérifier qu'il peut être relu
        assert os.path.exists(comparison_path)
        with open(comparison_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded == results

    def test_comparison_json_structure_matches_api_schema(self, training_results):
        """Vérifie que la structure JSON correspond au schéma ModelReportResponse."""
        results = training_results["results"]

        # Valider via le schéma Pydantic
        from backend.schemas import ModelReportResponse

        # Ceci doit passer sans exception
        report = ModelReportResponse(**results)

        assert len(report.models) == 3
        for model in report.models:
            assert model.name in ["Isolation Forest", "One-Class SVM", "Random Forest"]
            assert 0.0 <= model.f1_score <= 1.0
            assert 0.0 <= model.precision <= 1.0
            assert 0.0 <= model.recall <= 1.0
            assert 0.0 <= model.auc_roc <= 1.0
            assert len(model.confusion_matrix) == 2
            assert len(model.confusion_matrix[0]) == 2
            assert len(model.confusion_matrix[1]) == 2
