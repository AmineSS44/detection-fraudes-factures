# Feature: invoice-fraud-detection, Property 15: Fraud score bounds and label assignment
"""
Tests de propriétés pour le module ml/fraud_detector.py.
Vérifie les invariants via Hypothesis:
- Property 15: Fraud score bounds and label assignment

Pour tout score de fraude valide dans [0.0, 1.0]:
- fraud_label = "Normal" pour scores dans [0.0, 0.3[
- fraud_label = "Suspect" pour scores dans [0.3, 0.7[
- fraud_label = "Frauduleux" pour scores dans [0.7, 1.0]
"""

from unittest.mock import MagicMock, patch

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ml.feature_engineering import FeatureVector
from ml.fraud_detector import FraudDetector, FraudResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector() -> FraudDetector:
    """Crée un détecteur sans modèle (pour tester _assign_label directement)."""
    return FraudDetector(model_path="nonexistent.pkl")


# Stratégie: scores de fraude valides dans [0.0, 1.0]
fraud_scores = st.floats(
    min_value=0.0, max_value=1.0,
    allow_nan=False, allow_infinity=False,
)

# Stratégie: feature vectors valides avec des valeurs numériques réalistes
valid_feature_vectors = st.builds(
    FeatureVector,
    amount_zscore=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    tax_inconsistency=st.booleans(),
    duplicate_flag=st.booleans(),
    weekend_flag=st.booleans(),
    round_amount_flag=st.booleans(),
    vendor_deviation=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


# =============================================================================
# Property 15: Fraud score bounds and label assignment
# Validates: Requirements 6.2, 6.3
# =============================================================================


class TestFraudScoreBoundsAndLabelAssignment:
    """Property 15: Fraud score bounds and label assignment.

    **Validates: Requirements 6.2, 6.3**

    Pour tout score de fraude valide dans [0.0, 1.0], _assign_label doit
    retourner le label correct selon les seuils définis:
    - [0.0, 0.3[ → "Normal"
    - [0.3, 0.7[ → "Suspect"
    - [0.7, 1.0] → "Frauduleux"
    """

    # -------------------------------------------------------------------------
    # Test 1: Pour tout score dans [0.0, 1.0], le label est l'un des trois attendus
    # -------------------------------------------------------------------------

    @given(score=fraud_scores)
    @settings(max_examples=200)
    def test_label_always_valid_for_any_score(self, score: float):
        """Pour tout score dans [0.0, 1.0], le label retourné est parmi les 3 valides.

        **Validates: Requirements 6.3**
        """
        detector = _make_detector()
        label = detector._assign_label(score)

        assert label in ("Normal", "Suspect", "Frauduleux"), (
            f"Label invalide '{label}' pour score={score}. "
            f"Attendu: 'Normal', 'Suspect' ou 'Frauduleux'"
        )

    # -------------------------------------------------------------------------
    # Test 2: Scores dans [0.0, 0.3[ → "Normal"
    # -------------------------------------------------------------------------

    @given(score=st.floats(min_value=0.0, max_value=0.3, exclude_max=True,
                           allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_normal_label_for_scores_below_03(self, score: float):
        """Tout score dans [0.0, 0.3[ doit produire le label "Normal".

        **Validates: Requirements 6.3**
        """
        detector = _make_detector()
        label = detector._assign_label(score)

        assert label == "Normal", (
            f"Pour score={score} (dans [0.0, 0.3[), attendu 'Normal', obtenu '{label}'"
        )

    # -------------------------------------------------------------------------
    # Test 3: Scores dans [0.3, 0.7[ → "Suspect"
    # -------------------------------------------------------------------------

    @given(score=st.floats(min_value=0.3, max_value=0.7, exclude_max=True,
                           allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_suspect_label_for_scores_03_to_07(self, score: float):
        """Tout score dans [0.3, 0.7[ doit produire le label "Suspect".

        **Validates: Requirements 6.3**
        """
        detector = _make_detector()
        label = detector._assign_label(score)

        assert label == "Suspect", (
            f"Pour score={score} (dans [0.3, 0.7[), attendu 'Suspect', obtenu '{label}'"
        )

    # -------------------------------------------------------------------------
    # Test 4: Scores dans [0.7, 1.0] → "Frauduleux"
    # -------------------------------------------------------------------------

    @given(score=st.floats(min_value=0.7, max_value=1.0,
                           allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_frauduleux_label_for_scores_07_to_10(self, score: float):
        """Tout score dans [0.7, 1.0] doit produire le label "Frauduleux".

        **Validates: Requirements 6.3**
        """
        detector = _make_detector()
        label = detector._assign_label(score)

        assert label == "Frauduleux", (
            f"Pour score={score} (dans [0.7, 1.0]), attendu 'Frauduleux', obtenu '{label}'"
        )

    # -------------------------------------------------------------------------
    # Test 5: Conditions aux limites exactes (0.0, 0.3, 0.7, 1.0)
    # -------------------------------------------------------------------------

    @given(boundary=st.sampled_from([0.0, 0.3, 0.7, 1.0]))
    @settings(max_examples=100)
    def test_boundary_values(self, boundary: float):
        """Vérifie le comportement exact aux frontières 0.0, 0.3, 0.7, 1.0.

        - 0.0 → "Normal" (borne basse incluse)
        - 0.3 → "Suspect" (borne basse incluse dans [0.3, 0.7[)
        - 0.7 → "Frauduleux" (borne basse incluse dans [0.7, 1.0])
        - 1.0 → "Frauduleux" (borne haute incluse dans [0.7, 1.0])

        **Validates: Requirements 6.3**
        """
        detector = _make_detector()
        label = detector._assign_label(boundary)

        expected_map = {
            0.0: "Normal",
            0.3: "Suspect",
            0.7: "Frauduleux",
            1.0: "Frauduleux",
        }

        expected = expected_map[boundary]
        assert label == expected, (
            f"Pour la frontière score={boundary}, attendu '{expected}', obtenu '{label}'"
        )

    # -------------------------------------------------------------------------
    # Test 6: predict() retourne toujours un fraud_score dans [0.0, 1.0]
    #          quand le modèle est disponible (mocké)
    # -------------------------------------------------------------------------

    @given(feature_vector=valid_feature_vectors)
    @settings(max_examples=150, deadline=None)
    def test_predict_score_always_in_bounds(self, feature_vector: FeatureVector):
        """Pour tout vecteur de features valide, predict() retourne un fraud_score dans [0.0, 1.0].

        On mocke le modèle Isolation Forest pour simuler un modèle chargé.

        **Validates: Requirements 6.2**
        """
        detector = _make_detector()

        # Simuler un modèle chargé avec un score de décision aléatoire
        mock_model = MagicMock()
        # decision_function retourne des valeurs typiquement dans [-0.5, 0.5]
        mock_model.decision_function.return_value = np.array([np.random.uniform(-1.0, 1.0)])
        mock_model.estimators_ = []  # Pas d'arbres → fallback importances

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.5]])

        detector._model = mock_model
        detector._scaler = mock_scaler

        result = detector.predict(feature_vector)

        # Le résultat ne doit pas être une erreur
        assert result.fraud_score != -1.0, (
            f"predict() a retourné une erreur pour un vecteur valide: {result.fraud_reason}"
        )

        # Le score doit être dans [0.0, 1.0]
        assert 0.0 <= result.fraud_score <= 1.0, (
            f"fraud_score hors bornes: {result.fraud_score} pour feature_vector={feature_vector}"
        )

    # -------------------------------------------------------------------------
    # Test 7: predict() avec modèle mocké → le label correspond au score
    # -------------------------------------------------------------------------

    @given(feature_vector=valid_feature_vectors)
    @settings(max_examples=150, deadline=None)
    def test_predict_label_consistent_with_score(self, feature_vector: FeatureVector):
        """Pour tout vecteur valide, le label retourné par predict() est cohérent avec le score.

        **Validates: Requirements 6.2, 6.3**
        """
        detector = _make_detector()

        # Simuler un modèle chargé
        mock_model = MagicMock()
        mock_model.decision_function.return_value = np.array([np.random.uniform(-1.0, 1.0)])
        mock_model.estimators_ = []

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.5]])

        detector._model = mock_model
        detector._scaler = mock_scaler

        result = detector.predict(feature_vector)

        # Vérifier la cohérence label/score
        if result.fraud_score == -1.0:
            # Cas d'erreur — on skip
            return

        score = result.fraud_score
        label = result.fraud_label

        if score < 0.3:
            assert label == "Normal", (
                f"Score={score} < 0.3 mais label='{label}' (attendu 'Normal')"
            )
        elif score < 0.7:
            assert label == "Suspect", (
                f"Score={score} dans [0.3, 0.7[ mais label='{label}' (attendu 'Suspect')"
            )
        else:
            assert label == "Frauduleux", (
                f"Score={score} >= 0.7 mais label='{label}' (attendu 'Frauduleux')"
            )

    # -------------------------------------------------------------------------
    # Test 8: _normalize_score produit toujours des valeurs dans [0.0, 1.0]
    # -------------------------------------------------------------------------

    @given(raw_score=st.floats(min_value=-10.0, max_value=10.0,
                               allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_normalize_score_always_in_bounds(self, raw_score: float):
        """Pour tout score brut, _normalize_score retourne une valeur dans [0.0, 1.0].

        **Validates: Requirements 6.2**
        """
        detector = _make_detector()
        normalized = detector._normalize_score(raw_score)

        assert 0.0 <= normalized <= 1.0, (
            f"Score normalisé hors bornes: {normalized} pour raw_score={raw_score}"
        )


# =============================================================================
# Feature: invoice-fraud-detection, Property 16: Fraud reason references top 3 features
# Validates: Requirements 6.4
# =============================================================================

import re

from ml.fraud_detector import FEATURE_NAMES


# Stratégie: liste de 6 importances (valeurs positives pour simuler les poids)
_importances_strategy = st.lists(
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=6,
    max_size=6,
)


class TestFraudReasonTop3Features:
    """Property 16: Fraud reason references top 3 features.

    **Validates: Requirements 6.4**

    Pour tout vecteur de features valide scoré par le Fraud_Detector,
    la fraud_reason doit référencer exactement 3 features parmi
    {amount_zscore, tax_inconsistency, duplicate_flag, weekend_flag,
    round_amount_flag, vendor_deviation} avec leurs valeurs respectives.
    """

    # -------------------------------------------------------------------------
    # Test 1: _generate_reason référence exactement 3 features de FEATURE_NAMES
    # -------------------------------------------------------------------------

    @given(features=valid_feature_vectors, importances=_importances_strategy)
    @settings(max_examples=150)
    def test_generate_reason_references_exactly_3_features(
        self, features: FeatureVector, importances: list
    ):
        """Pour tout vecteur de features valide, _generate_reason doit référencer
        exactement 3 features parmi FEATURE_NAMES.

        **Validates: Requirements 6.4**
        """
        detector = _make_detector()
        reason = detector._generate_reason(features, importances)

        # Compter combien de features de FEATURE_NAMES sont présentes dans la raison
        mentioned_features = [
            name for name in FEATURE_NAMES if name in reason
        ]
        assert len(mentioned_features) == 3, (
            f"La raison doit référencer exactement 3 features, "
            f"mais en a trouvé {len(mentioned_features)}: {mentioned_features}. "
            f"Raison: '{reason}'"
        )

    # -------------------------------------------------------------------------
    # Test 2: Les 3 features sont incluses avec leurs valeurs respectives
    # -------------------------------------------------------------------------

    @given(features=valid_feature_vectors, importances=_importances_strategy)
    @settings(max_examples=150)
    def test_generate_reason_includes_feature_values(
        self, features: FeatureVector, importances: list
    ):
        """Pour tout vecteur de features valide, les 3 features référencées doivent
        inclure leurs valeurs respectives dans la raison.

        **Validates: Requirements 6.4**
        """
        detector = _make_detector()
        reason = detector._generate_reason(features, importances)

        # Construire le dictionnaire des valeurs réelles
        feature_dict = {
            "amount_zscore": features.amount_zscore,
            "tax_inconsistency": features.tax_inconsistency,
            "duplicate_flag": features.duplicate_flag,
            "weekend_flag": features.weekend_flag,
            "round_amount_flag": features.round_amount_flag,
            "vendor_deviation": features.vendor_deviation,
        }

        # Vérifier que chaque feature mentionnée a sa valeur correcte
        mentioned_features = [
            name for name in FEATURE_NAMES if name in reason
        ]

        for name in mentioned_features:
            expected_value = feature_dict[name]
            # La raison doit contenir "name=value"
            expected_fragment = f"{name}={expected_value}"
            assert expected_fragment in reason, (
                f"La feature '{name}' devrait apparaître comme '{expected_fragment}' "
                f"dans la raison, mais la raison est: '{reason}'"
            )


# =============================================================================
# Feature: invoice-fraud-detection, Property 17: Invalid feature vector rejection
# Validates: Requirements 6.7
# =============================================================================

import pytest
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Helpers pour Property 17
# ---------------------------------------------------------------------------

# Les 6 features requises dans l'ordre
_ALL_FEATURES = [
    "amount_zscore",
    "tax_inconsistency",
    "duplicate_flag",
    "weekend_flag",
    "round_amount_flag",
    "vendor_deviation",
]


def _make_feature_vector_with_none(fields_to_nullify: list):
    """Crée un objet simulant un FeatureVector avec certaines features à None."""
    @dataclass
    class MockFeatureVector:
        amount_zscore: Any = 0.5
        tax_inconsistency: Any = False
        duplicate_flag: Any = True
        weekend_flag: Any = False
        round_amount_flag: Any = False
        vendor_deviation: Any = 0.3

    fv = MockFeatureVector()
    for field in fields_to_nullify:
        setattr(fv, field, None)
    return fv


def _make_feature_vector_with_strings(fields_to_corrupt: list):
    """Crée un objet simulant un FeatureVector avec des valeurs non numériques (strings)."""
    @dataclass
    class MockFeatureVector:
        amount_zscore: Any = 0.5
        tax_inconsistency: Any = False
        duplicate_flag: Any = True
        weekend_flag: Any = False
        round_amount_flag: Any = False
        vendor_deviation: Any = 0.3

    fv = MockFeatureVector()
    for field in fields_to_corrupt:
        setattr(fv, field, "invalid_string")
    return fv


# Instancier le détecteur (pas besoin de modèle pour la validation)
_detector_p17 = FraudDetector(model_path="nonexistent_model.pkl")


class TestInvalidFeatureVectorRejection:
    """Property 17: Invalid feature vector rejection.

    **Validates: Requirements 6.7**

    Pour tout vecteur de features manquant une ou plusieurs des 6 features requises
    ou contenant des valeurs non numériques, le Fraud_Detector doit rejeter l'entrée
    et retourner une erreur spécifiant quelles features sont invalides ou manquantes.
    """

    @given(
        fields_to_nullify=st.lists(
            st.sampled_from(_ALL_FEATURES),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    @settings(max_examples=150)
    def test_missing_features_raises_valueerror(self, fields_to_nullify: list):
        """
        Quand un ou plusieurs features sont None (manquantes),
        _validate_feature_vector doit lever ValueError.

        **Validates: Requirements 6.7**
        """
        fv = _make_feature_vector_with_none(fields_to_nullify)

        with pytest.raises(ValueError) as exc_info:
            _detector_p17._validate_feature_vector(fv)

        error_message = str(exc_info.value)

        # Vérifier que chaque feature manquante est mentionnée dans l'erreur
        for field in fields_to_nullify:
            assert field in error_message, (
                f"La feature manquante '{field}' n'est pas mentionnée dans "
                f"l'erreur: '{error_message}'"
            )

    @given(
        fields_to_corrupt=st.lists(
            st.sampled_from(_ALL_FEATURES),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    @settings(max_examples=150)
    def test_non_numeric_features_raises_valueerror(self, fields_to_corrupt: list):
        """
        Quand un ou plusieurs features contiennent des valeurs non numériques
        (strings), _validate_feature_vector doit lever ValueError.

        **Validates: Requirements 6.7**
        """
        fv = _make_feature_vector_with_strings(fields_to_corrupt)

        with pytest.raises(ValueError) as exc_info:
            _detector_p17._validate_feature_vector(fv)

        error_message = str(exc_info.value)

        # Vérifier que chaque feature invalide est mentionnée dans l'erreur
        for field in fields_to_corrupt:
            assert field in error_message, (
                f"La feature non numérique '{field}' n'est pas mentionnée dans "
                f"l'erreur: '{error_message}'"
            )

    @given(
        fields_to_nullify=st.lists(
            st.sampled_from(_ALL_FEATURES),
            min_size=1,
            max_size=3,
            unique=True,
        ),
        fields_to_corrupt=st.lists(
            st.sampled_from(_ALL_FEATURES),
            min_size=1,
            max_size=3,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_mixed_missing_and_non_numeric_raises_valueerror(
        self, fields_to_nullify: list, fields_to_corrupt: list
    ):
        """
        Quand un vecteur contient à la fois des features manquantes (None)
        et des features non numériques, _validate_feature_vector doit lever
        ValueError mentionnant toutes les features problématiques.

        **Validates: Requirements 6.7**
        """
        # S'assurer qu'il n'y a pas de chevauchement entre les deux listes
        fields_to_corrupt_final = [
            f for f in fields_to_corrupt if f not in fields_to_nullify
        ]
        assume(len(fields_to_corrupt_final) > 0)

        @dataclass
        class MockFeatureVector:
            amount_zscore: Any = 0.5
            tax_inconsistency: Any = False
            duplicate_flag: Any = True
            weekend_flag: Any = False
            round_amount_flag: Any = False
            vendor_deviation: Any = 0.3

        fv = MockFeatureVector()
        # Mettre certains champs à None
        for field in fields_to_nullify:
            setattr(fv, field, None)
        # Mettre d'autres champs à des strings
        for field in fields_to_corrupt_final:
            setattr(fv, field, "not_a_number")

        with pytest.raises(ValueError) as exc_info:
            _detector_p17._validate_feature_vector(fv)

        error_message = str(exc_info.value)

        # Vérifier que chaque feature problématique est mentionnée
        for field in fields_to_nullify:
            assert field in error_message, (
                f"La feature manquante '{field}' n'est pas mentionnée dans "
                f"l'erreur: '{error_message}'"
            )
        for field in fields_to_corrupt_final:
            assert field in error_message, (
                f"La feature non numérique '{field}' n'est pas mentionnée dans "
                f"l'erreur: '{error_message}'"
            )

    @given(
        amount_zscore=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        tax_inconsistency=st.booleans(),
        duplicate_flag=st.booleans(),
        weekend_flag=st.booleans(),
        round_amount_flag=st.booleans(),
        vendor_deviation=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_valid_feature_vector_does_not_raise(
        self,
        amount_zscore: float,
        tax_inconsistency: bool,
        duplicate_flag: bool,
        weekend_flag: bool,
        round_amount_flag: bool,
        vendor_deviation: float,
    ):
        """
        Un vecteur de features valide (toutes présentes et numériques/booléennes)
        ne doit PAS lever d'exception lors de la validation.

        **Validates: Requirements 6.7**
        """
        fv = FeatureVector(
            amount_zscore=amount_zscore,
            tax_inconsistency=tax_inconsistency,
            duplicate_flag=duplicate_flag,
            weekend_flag=weekend_flag,
            round_amount_flag=round_amount_flag,
            vendor_deviation=vendor_deviation,
        )

        # Ne doit pas lever d'exception
        _detector_p17._validate_feature_vector(fv)
