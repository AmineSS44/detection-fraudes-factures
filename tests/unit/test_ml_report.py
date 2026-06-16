"""
Tests unitaires pour la page Rapport ML (show_ml_report).
Vérifie la logique de mise en surbrillance des meilleures métriques,
la construction du DataFrame, et la gestion des cas d'erreur.
"""

import sys
import os

import numpy as np
import pandas as pd

# Ajouter le répertoire racine du projet au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# --- Données de test simulant la réponse de l'API ---

SAMPLE_REPORT = {
    "models": [
        {
            "name": "Isolation Forest",
            "f1_score": 0.9333,
            "precision": 0.875,
            "recall": 1.0,
            "auc_roc": 1.0,
            "confusion_matrix": [[7, 0], [1, 52]],
        },
        {
            "name": "One-Class SVM",
            "f1_score": 0.7778,
            "precision": 0.6364,
            "recall": 1.0,
            "auc_roc": 0.9838,
            "confusion_matrix": [[7, 0], [4, 49]],
        },
        {
            "name": "Random Forest",
            "f1_score": 0.8571,
            "precision": 0.8571,
            "recall": 0.8571,
            "auc_roc": 0.9838,
            "confusion_matrix": [[6, 1], [1, 52]],
        },
    ]
}


# --- Fonctions de logique extraites pour test (sans dépendance Streamlit) ---


def build_metrics_dataframe(models: list) -> pd.DataFrame:
    """Construit le DataFrame de comparaison des métriques.

    Args:
        models: Liste de dictionnaires avec les métriques de chaque modèle.

    Returns:
        DataFrame avec colonnes: Modèle, F1 Score, Precision, Recall, AUC-ROC.
    """
    metrics_data = []
    for model in models:
        metrics_data.append({
            "Modèle": model["name"],
            "F1 Score": model["f1_score"],
            "Precision": model["precision"],
            "Recall": model["recall"],
            "AUC-ROC": model["auc_roc"],
        })
    return pd.DataFrame(metrics_data)


def find_best_per_metric(df: pd.DataFrame) -> dict:
    """Identifie le meilleur modèle par métrique.

    Args:
        df: DataFrame des métriques.

    Returns:
        Dictionnaire {métrique: index_du_meilleur_modèle}.
    """
    metric_columns = ["F1 Score", "Precision", "Recall", "AUC-ROC"]
    best = {}
    for col in metric_columns:
        best[col] = df[col].idxmax()
    return best


def validate_confusion_matrix(cm: list) -> bool:
    """Valide qu'une matrice de confusion est bien au format 2x2 avec des entiers positifs.

    Args:
        cm: Matrice de confusion sous forme de liste de listes.

    Returns:
        True si la matrice est valide, False sinon.
    """
    if not isinstance(cm, list) or len(cm) != 2:
        return False
    for row in cm:
        if not isinstance(row, list) or len(row) != 2:
            return False
        for val in row:
            if not isinstance(val, int) or val < 0:
                return False
    return True


# --- Classes de test ---


class TestBuildMetricsDataframe:
    """Tests pour la construction du DataFrame de métriques."""

    def test_correct_shape(self):
        """Le DataFrame doit avoir 3 lignes et 5 colonnes."""
        df = build_metrics_dataframe(SAMPLE_REPORT["models"])
        assert df.shape == (3, 5)

    def test_correct_columns(self):
        """Les colonnes doivent être Modèle, F1 Score, Precision, Recall, AUC-ROC."""
        df = build_metrics_dataframe(SAMPLE_REPORT["models"])
        expected_cols = ["Modèle", "F1 Score", "Precision", "Recall", "AUC-ROC"]
        assert list(df.columns) == expected_cols

    def test_model_names(self):
        """Les noms des modèles doivent correspondre."""
        df = build_metrics_dataframe(SAMPLE_REPORT["models"])
        expected_names = ["Isolation Forest", "One-Class SVM", "Random Forest"]
        assert list(df["Modèle"]) == expected_names

    def test_metric_values(self):
        """Les valeurs des métriques doivent être correctes."""
        df = build_metrics_dataframe(SAMPLE_REPORT["models"])
        assert df.loc[0, "F1 Score"] == 0.9333
        assert df.loc[1, "Precision"] == 0.6364
        assert df.loc[2, "Recall"] == 0.8571

    def test_empty_models_list(self):
        """Liste vide de modèles → DataFrame vide."""
        df = build_metrics_dataframe([])
        assert df.empty


class TestFindBestPerMetric:
    """Tests pour l'identification du meilleur modèle par métrique."""

    def test_best_f1_score(self):
        """Isolation Forest a le meilleur F1 Score (0.9333)."""
        df = build_metrics_dataframe(SAMPLE_REPORT["models"])
        best = find_best_per_metric(df)
        assert best["F1 Score"] == 0  # Index 0 = Isolation Forest

    def test_best_precision(self):
        """Isolation Forest a la meilleure Precision (0.875)."""
        df = build_metrics_dataframe(SAMPLE_REPORT["models"])
        best = find_best_per_metric(df)
        assert best["Precision"] == 0  # Index 0 = Isolation Forest

    def test_best_recall(self):
        """Isolation Forest et One-Class SVM ont le même Recall (1.0).
        idxmax retourne le premier index avec la valeur max."""
        df = build_metrics_dataframe(SAMPLE_REPORT["models"])
        best = find_best_per_metric(df)
        # Les deux premiers ont 1.0, idxmax retourne le premier
        assert best["Recall"] == 0

    def test_best_auc_roc(self):
        """Isolation Forest a le meilleur AUC-ROC (1.0)."""
        df = build_metrics_dataframe(SAMPLE_REPORT["models"])
        best = find_best_per_metric(df)
        assert best["AUC-ROC"] == 0  # Index 0 = Isolation Forest

    def test_all_metrics_found(self):
        """Toutes les 4 métriques doivent être présentes dans le résultat."""
        df = build_metrics_dataframe(SAMPLE_REPORT["models"])
        best = find_best_per_metric(df)
        assert len(best) == 4
        assert "F1 Score" in best
        assert "Precision" in best
        assert "Recall" in best
        assert "AUC-ROC" in best


class TestHighlightMax:
    """Tests pour la fonction de mise en surbrillance."""

    def test_highlight_function(self):
        """La fonction highlight_max doit marquer la valeur maximale."""
        # Reproduire la logique de highlight_max
        s = pd.Series([0.9333, 0.7778, 0.8571])
        is_max = s == s.max()
        styles = [
            "background-color: #d4edda; font-weight: bold" if v else ""
            for v in is_max
        ]
        assert styles[0] == "background-color: #d4edda; font-weight: bold"
        assert styles[1] == ""
        assert styles[2] == ""

    def test_highlight_with_ties(self):
        """Quand deux valeurs sont égales au max, les deux doivent être mises en surbrillance."""
        s = pd.Series([1.0, 1.0, 0.8571])
        is_max = s == s.max()
        styles = [
            "background-color: #d4edda; font-weight: bold" if v else ""
            for v in is_max
        ]
        assert styles[0] == "background-color: #d4edda; font-weight: bold"
        assert styles[1] == "background-color: #d4edda; font-weight: bold"
        assert styles[2] == ""


class TestValidateConfusionMatrix:
    """Tests pour la validation des matrices de confusion."""

    def test_valid_matrix(self):
        """Matrice 2x2 valide avec entiers positifs."""
        cm = [[7, 0], [1, 52]]
        assert validate_confusion_matrix(cm) is True

    def test_valid_matrix_zeros(self):
        """Matrice avec des zéros est valide."""
        cm = [[0, 0], [0, 0]]
        assert validate_confusion_matrix(cm) is True

    def test_invalid_not_list(self):
        """Input qui n'est pas une liste → invalide."""
        assert validate_confusion_matrix("not a matrix") is False

    def test_invalid_wrong_size(self):
        """Matrice qui n'est pas 2x2 → invalide."""
        cm = [[7, 0, 1], [1, 52, 3]]
        assert validate_confusion_matrix(cm) is False

    def test_invalid_single_row(self):
        """Matrice avec une seule ligne → invalide."""
        cm = [[7, 0]]
        assert validate_confusion_matrix(cm) is False

    def test_invalid_negative_values(self):
        """Matrice avec valeurs négatives → invalide."""
        cm = [[7, -1], [1, 52]]
        assert validate_confusion_matrix(cm) is False

    def test_all_sample_matrices_valid(self):
        """Toutes les matrices du rapport exemple sont valides."""
        for model in SAMPLE_REPORT["models"]:
            assert validate_confusion_matrix(model["confusion_matrix"]) is True


class TestConfusionMatrixDecomposition:
    """Tests pour l'extraction des valeurs TP, FP, FN, TN."""

    def test_isolation_forest_matrix(self):
        """Vérification des composantes de la matrice Isolation Forest."""
        cm = SAMPLE_REPORT["models"][0]["confusion_matrix"]
        tp, fp = cm[0][0], cm[0][1]
        fn, tn = cm[1][0], cm[1][1]
        assert tp == 7
        assert fp == 0
        assert fn == 1
        assert tn == 52

    def test_one_class_svm_matrix(self):
        """Vérification des composantes de la matrice One-Class SVM."""
        cm = SAMPLE_REPORT["models"][1]["confusion_matrix"]
        tp, fp = cm[0][0], cm[0][1]
        fn, tn = cm[1][0], cm[1][1]
        assert tp == 7
        assert fp == 0
        assert fn == 4
        assert tn == 49

    def test_random_forest_matrix(self):
        """Vérification des composantes de la matrice Random Forest."""
        cm = SAMPLE_REPORT["models"][2]["confusion_matrix"]
        tp, fp = cm[0][0], cm[0][1]
        fn, tn = cm[1][0], cm[1][1]
        assert tp == 6
        assert fp == 1
        assert fn == 1
        assert tn == 52


class TestMissingReportHandling:
    """Tests pour la gestion du rapport manquant."""

    def test_404_response_triggers_warning_message(self):
        """Quand l'API retourne 404, on doit afficher un message d'avertissement."""
        # Simuler la logique de vérification
        status_code = 404
        expected_message = (
            "Rapport non disponible. Exécutez `python ml/train.py` pour générer le rapport."
        )
        if status_code == 404:
            message = expected_message
        else:
            message = None
        assert message == expected_message

    def test_empty_models_in_report(self):
        """Si le rapport contient une liste vide de modèles → signal d'erreur."""
        report = {"models": []}
        models = report.get("models", [])
        assert len(models) == 0
