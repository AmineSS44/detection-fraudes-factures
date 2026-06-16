"""
Tests unitaires pour le module ml/fraud_detector.py.
Vérifie le scoring, le labeling, la génération de raisons et la gestion d'erreurs.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from ml.feature_engineering import FeatureVector
from ml.fraud_detector import (
    COMPARISON_PATH,
    FEATURE_NAMES,
    MODEL_PATH,
    FraudDetector,
    FraudResult,
)


class TestFraudResult:
    """Tests pour la dataclass FraudResult."""

    def test_creation_normal(self):
        """Vérifie la création d'un résultat normal."""
        result = FraudResult(
            fraud_score=0.15,
            fraud_label="Normal",
            fraud_reason="Top 3 features: amount_zscore=0.5, tax_inconsistency=False, duplicate_flag=False",
        )
        assert result.fraud_score == 0.15
        assert result.fraud_label == "Normal"
        assert "amount_zscore" in result.fraud_reason

    def test_creation_error(self):
        """Vérifie la création d'un résultat d'erreur."""
        result = FraudResult(
            fraud_score=-1.0,
            fraud_label="Erreur",
            fraud_reason="Modèle non disponible",
        )
        assert result.fraud_score == -1.0
        assert result.fraud_label == "Erreur"


class TestAssignLabel:
    """Tests pour la méthode _assign_label."""

    def setup_method(self):
        """Initialise le détecteur pour les tests."""
        self.detector = FraudDetector(model_path="nonexistent.pkl")

    def test_normal_low(self):
        """Score 0.0 → Normal."""
        assert self.detector._assign_label(0.0) == "Normal"

    def test_normal_boundary(self):
        """Score 0.29 → Normal (juste avant la frontière)."""
        assert self.detector._assign_label(0.29) == "Normal"

    def test_suspect_boundary_lower(self):
        """Score 0.3 → Suspect (frontière basse incluse)."""
        assert self.detector._assign_label(0.3) == "Suspect"

    def test_suspect_middle(self):
        """Score 0.5 → Suspect."""
        assert self.detector._assign_label(0.5) == "Suspect"

    def test_suspect_boundary_upper(self):
        """Score 0.69 → Suspect (juste avant la frontière haute)."""
        assert self.detector._assign_label(0.69) == "Suspect"

    def test_frauduleux_boundary(self):
        """Score 0.7 → Frauduleux (frontière incluse)."""
        assert self.detector._assign_label(0.7) == "Frauduleux"

    def test_frauduleux_high(self):
        """Score 1.0 → Frauduleux."""
        assert self.detector._assign_label(1.0) == "Frauduleux"


class TestGenerateReason:
    """Tests pour la méthode _generate_reason."""

    def setup_method(self):
        """Initialise le détecteur pour les tests."""
        self.detector = FraudDetector(model_path="nonexistent.pkl")

    def test_top_3_features_included(self):
        """Vérifie que 3 features sont mentionnées dans la raison."""
        features = FeatureVector(
            amount_zscore=2.5,
            tax_inconsistency=True,
            duplicate_flag=False,
            weekend_flag=True,
            round_amount_flag=False,
            vendor_deviation=0.8,
        )
        importances = [0.5, 0.3, 0.05, 0.1, 0.02, 0.03]

        reason = self.detector._generate_reason(features, importances)

        assert "Top 3 features:" in reason
        # Doit contenir exactement 3 features (séparées par des virgules)
        parts = reason.replace("Top 3 features: ", "").split(", ")
        assert len(parts) == 3

    def test_highest_importance_first(self):
        """Vérifie que les features sont triées par importance décroissante."""
        features = FeatureVector(
            amount_zscore=3.0,
            tax_inconsistency=True,
            duplicate_flag=False,
            weekend_flag=False,
            round_amount_flag=False,
            vendor_deviation=0.1,
        )
        # amount_zscore a la plus haute importance
        importances = [0.9, 0.1, 0.05, 0.02, 0.01, 0.03]

        reason = self.detector._generate_reason(features, importances)

        # La première feature mentionnée devrait être amount_zscore
        assert reason.startswith("Top 3 features: amount_zscore=")


class TestValidateFeatureVector:
    """Tests pour la validation du vecteur de features."""

    def setup_method(self):
        """Initialise le détecteur pour les tests."""
        self.detector = FraudDetector(model_path="nonexistent.pkl")

    def test_valid_vector(self):
        """Un vecteur valide ne lève pas d'exception."""
        features = FeatureVector(
            amount_zscore=1.0,
            tax_inconsistency=True,
            duplicate_flag=False,
            weekend_flag=False,
            round_amount_flag=False,
            vendor_deviation=0.5,
        )
        # Ne devrait pas lever d'exception
        self.detector._validate_feature_vector(features)

    def test_missing_feature_raises(self):
        """Un vecteur avec feature None lève ValueError."""
        features = FeatureVector(
            amount_zscore=None,  # type: ignore
            tax_inconsistency=True,
            duplicate_flag=False,
            weekend_flag=False,
            round_amount_flag=False,
            vendor_deviation=0.5,
        )
        with pytest.raises(ValueError, match="manquantes"):
            self.detector._validate_feature_vector(features)


class TestPredictModelUnavailable:
    """Tests pour le cas où le modèle est indisponible."""

    def test_model_not_found(self):
        """Retourne Erreur si le fichier modèle n'existe pas."""
        detector = FraudDetector(model_path="nonexistent_model.pkl")
        features = FeatureVector(
            amount_zscore=1.0,
            tax_inconsistency=False,
            duplicate_flag=False,
            weekend_flag=False,
            round_amount_flag=False,
            vendor_deviation=0.3,
        )

        result = detector.predict(features)

        assert result.fraud_score == -1.0
        assert result.fraud_label == "Erreur"
        assert "non disponible" in result.fraud_reason.lower()


class TestNormalizeScore:
    """Tests pour la normalisation du score."""

    def setup_method(self):
        """Initialise le détecteur pour les tests."""
        self.detector = FraudDetector(model_path="nonexistent.pkl")

    def test_score_in_range(self):
        """Le score normalisé est toujours entre 0 et 1."""
        for raw in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            normalized = self.detector._normalize_score(raw)
            assert 0.0 <= normalized <= 1.0

    def test_negative_score_high_fraud(self):
        """Un score brut négatif donne un score de fraude élevé."""
        score = self.detector._normalize_score(-0.5)
        assert score > 0.5

    def test_positive_score_low_fraud(self):
        """Un score brut positif donne un score de fraude bas."""
        score = self.detector._normalize_score(0.5)
        assert score < 0.5


class TestFeatureVectorToArray:
    """Tests pour la conversion FeatureVector → numpy array."""

    def setup_method(self):
        """Initialise le détecteur pour les tests."""
        self.detector = FraudDetector(model_path="nonexistent.pkl")

    def test_correct_conversion(self):
        """Vérifie la conversion correcte des 6 features."""
        features = FeatureVector(
            amount_zscore=2.5,
            tax_inconsistency=True,
            duplicate_flag=False,
            weekend_flag=True,
            round_amount_flag=False,
            vendor_deviation=0.8,
        )

        arr = self.detector._feature_vector_to_array(features)

        assert arr.shape == (6,)
        assert arr[0] == 2.5  # amount_zscore
        assert arr[1] == 1.0  # tax_inconsistency (True → 1.0)
        assert arr[2] == 0.0  # duplicate_flag (False → 0.0)
        assert arr[3] == 1.0  # weekend_flag (True → 1.0)
        assert arr[4] == 0.0  # round_amount_flag (False → 0.0)
        assert arr[5] == 0.8  # vendor_deviation
