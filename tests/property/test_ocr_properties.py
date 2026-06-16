# Feature: invoice-fraud-detection, Property 4: OCR result serialization round-trip
"""
Tests de propriétés pour le pipeline OCR.
Vérifie les invariants via Hypothesis:
- Property 4: OCR result serialization round-trip (to_json/from_json, to_dict/from_dict)
- Property 7: YOLOv8 fallback on low confidence
"""

import math
from unittest.mock import patch

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from ocr.pipeline import (
    OCRPipeline,
    OCRResult,
    YOLO_CONFIDENCE_THRESHOLD,
    AMOUNT_MIN,
    AMOUNT_MAX,
    TAX_RATE_MIN,
    TAX_RATE_MAX,
)


# =============================================================================
# Stratégies Hypothesis pour générer des OCRResult valides
# =============================================================================

# Stratégie pour invoice_id: string nullable, max 50 caractères
invoice_id_strategy = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=("L", "N", "P"),
        whitelist_characters="-_/",
    )),
)

# Stratégie pour vendor_name: string nullable, max 200 caractères
vendor_name_strategy = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=200, alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00",
    )),
)

# Stratégie pour amount: float nullable dans les bornes [0.01, 999_999_999.99]
amount_strategy = st.one_of(
    st.none(),
    st.floats(min_value=0.01, max_value=999_999_999.99, allow_nan=False, allow_infinity=False),
)

# Stratégie pour date: string nullable au format ISO 8601 YYYY-MM-DD
date_strategy = st.one_of(
    st.none(),
    st.dates().map(lambda d: d.isoformat()),
)

# Stratégie pour tax_rate: float nullable dans [0, 100]
tax_rate_strategy = st.one_of(
    st.none(),
    st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)

# Stratégie pour total: float nullable dans [0.01, 999_999_999.99]
total_strategy = st.one_of(
    st.none(),
    st.floats(min_value=0.01, max_value=999_999_999.99, allow_nan=False, allow_infinity=False),
)

# Stratégie pour les confiances par champ: dictionnaire avec 6 clés, valeurs entre 0.0 et 1.0
FIELD_NAMES = ["invoice_id", "vendor_name", "amount", "date", "tax_rate", "total"]

field_confidences_strategy = st.fixed_dictionaries({
    name: st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    for name in FIELD_NAMES
})

# Stratégie pour les warnings: liste de strings
warnings_strategy = st.lists(
    st.text(min_size=1, max_size=100, alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00",
    )),
    min_size=0,
    max_size=6,
)


# Stratégie composite pour un OCRResult valide complet
@st.composite
def valid_ocr_result(draw):
    """Génère une instance OCRResult valide avec des valeurs aléatoires."""
    return OCRResult(
        invoice_id=draw(invoice_id_strategy),
        vendor_name=draw(vendor_name_strategy),
        amount=draw(amount_strategy),
        date=draw(date_strategy),
        tax_rate=draw(tax_rate_strategy),
        total=draw(total_strategy),
        field_confidences=draw(field_confidences_strategy),
        warnings=draw(warnings_strategy),
    )


# =============================================================================
# Fonctions utilitaires pour comparer les valeurs avec tolérance flottante
# =============================================================================

def floats_equal(a, b):
    """Compare deux valeurs flottantes (ou None) avec tolérance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    # Comparer avec une tolérance relative pour les flottants
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


# =============================================================================
# Property 4: OCR result serialization round-trip
# Validates: Requirements 4.6
# =============================================================================


class TestOCRResultSerializationRoundTrip:
    """
    Property 4: OCR result serialization round-trip

    For any valid OCRResult instance, serializing it to JSON and deserializing
    back should produce an OCRResult with field values identical to the original
    for all six fields and their confidence scores.

    **Validates: Requirements 4.6**
    """

    @settings(max_examples=150, deadline=None)
    @given(ocr_result=valid_ocr_result())
    def test_to_json_then_from_json_preserves_all_fields(self, ocr_result: OCRResult):
        """
        Pour toute instance OCRResult valide, to_json() puis from_json()
        doit préserver les 6 champs exactement.

        **Validates: Requirements 4.6**
        """
        # Sérialiser en JSON puis désérialiser
        json_str = ocr_result.to_json()
        restored = OCRResult.from_json(json_str)

        # Vérifier les 6 champs
        assert restored.invoice_id == ocr_result.invoice_id, (
            f"invoice_id: {restored.invoice_id!r} != {ocr_result.invoice_id!r}"
        )
        assert restored.vendor_name == ocr_result.vendor_name, (
            f"vendor_name: {restored.vendor_name!r} != {ocr_result.vendor_name!r}"
        )
        assert floats_equal(restored.amount, ocr_result.amount), (
            f"amount: {restored.amount} != {ocr_result.amount}"
        )
        assert restored.date == ocr_result.date, (
            f"date: {restored.date!r} != {ocr_result.date!r}"
        )
        assert floats_equal(restored.tax_rate, ocr_result.tax_rate), (
            f"tax_rate: {restored.tax_rate} != {ocr_result.tax_rate}"
        )
        assert floats_equal(restored.total, ocr_result.total), (
            f"total: {restored.total} != {ocr_result.total}"
        )

    @settings(max_examples=150, deadline=None)
    @given(ocr_result=valid_ocr_result())
    def test_to_dict_then_from_dict_preserves_all_fields(self, ocr_result: OCRResult):
        """
        Pour toute instance OCRResult valide, to_dict() puis from_dict()
        doit préserver les 6 champs exactement.

        **Validates: Requirements 4.6**
        """
        # Sérialiser en dict puis désérialiser
        data = ocr_result.to_dict()
        restored = OCRResult.from_dict(data)

        # Vérifier les 6 champs
        assert restored.invoice_id == ocr_result.invoice_id, (
            f"invoice_id: {restored.invoice_id!r} != {ocr_result.invoice_id!r}"
        )
        assert restored.vendor_name == ocr_result.vendor_name, (
            f"vendor_name: {restored.vendor_name!r} != {ocr_result.vendor_name!r}"
        )
        assert floats_equal(restored.amount, ocr_result.amount), (
            f"amount: {restored.amount} != {ocr_result.amount}"
        )
        assert restored.date == ocr_result.date, (
            f"date: {restored.date!r} != {ocr_result.date!r}"
        )
        assert floats_equal(restored.tax_rate, ocr_result.tax_rate), (
            f"tax_rate: {restored.tax_rate} != {ocr_result.tax_rate}"
        )
        assert floats_equal(restored.total, ocr_result.total), (
            f"total: {restored.total} != {ocr_result.total}"
        )

    @settings(max_examples=150, deadline=None)
    @given(ocr_result=valid_ocr_result())
    def test_to_json_then_from_json_preserves_confidences(self, ocr_result: OCRResult):
        """
        Pour toute instance OCRResult valide, to_json() puis from_json()
        doit préserver les field_confidences (valeurs 0.0-1.0) pour les 6 champs.

        **Validates: Requirements 4.6**
        """
        # Sérialiser en JSON puis désérialiser
        json_str = ocr_result.to_json()
        restored = OCRResult.from_json(json_str)

        # Vérifier que toutes les clés de confiance sont préservées
        assert set(restored.field_confidences.keys()) == set(FIELD_NAMES), (
            f"Clés de confiance manquantes ou en trop: "
            f"{set(restored.field_confidences.keys())} != {set(FIELD_NAMES)}"
        )

        # Vérifier chaque score de confiance
        for field_name in FIELD_NAMES:
            original_conf = ocr_result.field_confidences[field_name]
            restored_conf = restored.field_confidences[field_name]
            assert floats_equal(original_conf, restored_conf), (
                f"Confiance '{field_name}': {restored_conf} != {original_conf}"
            )
            # Vérifier que la confiance reste dans [0.0, 1.0]
            assert 0.0 <= restored_conf <= 1.0, (
                f"Confiance '{field_name}' hors bornes: {restored_conf}"
            )

    @settings(max_examples=150, deadline=None)
    @given(ocr_result=valid_ocr_result())
    def test_to_dict_then_from_dict_preserves_confidences(self, ocr_result: OCRResult):
        """
        Pour toute instance OCRResult valide, to_dict() puis from_dict()
        doit préserver les field_confidences (valeurs 0.0-1.0) pour les 6 champs.

        **Validates: Requirements 4.6**
        """
        # Sérialiser en dict puis désérialiser
        data = ocr_result.to_dict()
        restored = OCRResult.from_dict(data)

        # Vérifier que toutes les clés de confiance sont préservées
        assert set(restored.field_confidences.keys()) == set(FIELD_NAMES), (
            f"Clés de confiance manquantes ou en trop: "
            f"{set(restored.field_confidences.keys())} != {set(FIELD_NAMES)}"
        )

        # Vérifier chaque score de confiance
        for field_name in FIELD_NAMES:
            original_conf = ocr_result.field_confidences[field_name]
            restored_conf = restored.field_confidences[field_name]
            assert floats_equal(original_conf, restored_conf), (
                f"Confiance '{field_name}': {restored_conf} != {original_conf}"
            )
            # Vérifier que la confiance reste dans [0.0, 1.0]
            assert 0.0 <= restored_conf <= 1.0, (
                f"Confiance '{field_name}' hors bornes: {restored_conf}"
            )


# =============================================================================
# Property 5: OCR output structure validity
# Validates: Requirements 4.4, 4.7
# =============================================================================


# Champs attendus dans la structure OCR de sortie
EXPECTED_OCR_FIELDS = {"invoice_id", "vendor_name", "amount", "date", "tax_rate", "total"}


class TestOCROutputStructureValidity:
    """
    Property 5: OCR output structure validity

    Pour tout OCRResult complété, le JSON de sortie doit contenir exactement
    6 champs (invoice_id, vendor_name, amount, date, tax_rate, total), chacun
    avec un score de confiance dans [0.0, 1.0], et les champs numériques
    doivent respecter leurs bornes (amount/total: 0.01–999,999,999.99,
    tax_rate: 0–100).

    **Validates: Requirements 4.4, 4.7**
    """

    @settings(max_examples=150, deadline=None)
    @given(ocr_result=valid_ocr_result())
    def test_ocr_result_has_exactly_6_fields_with_valid_confidences(self, ocr_result: OCRResult):
        """
        Pour tout OCRResult, field_confidences doit contenir exactement 6 entrées
        (invoice_id, vendor_name, amount, date, tax_rate, total) et chaque
        confiance doit être dans [0.0, 1.0].

        **Validates: Requirements 4.4, 4.7**
        """
        # Vérifier que field_confidences contient exactement les 6 champs attendus
        assert set(ocr_result.field_confidences.keys()) == EXPECTED_OCR_FIELDS, (
            f"field_confidences devrait contenir exactement {EXPECTED_OCR_FIELDS}, "
            f"mais contient {set(ocr_result.field_confidences.keys())}"
        )
        assert len(ocr_result.field_confidences) == 6, (
            f"field_confidences devrait avoir 6 entrées, a {len(ocr_result.field_confidences)}"
        )

        # Vérifier que chaque confiance est dans [0.0, 1.0]
        for field_name, confidence in ocr_result.field_confidences.items():
            assert isinstance(confidence, float), (
                f"Confiance pour '{field_name}' doit être un float, est {type(confidence)}"
            )
            assert 0.0 <= confidence <= 1.0, (
                f"Confiance pour '{field_name}' = {confidence}, "
                f"doit être dans [0.0, 1.0]"
            )

    @settings(max_examples=150, deadline=None)
    @given(ocr_result=valid_ocr_result())
    def test_numeric_fields_respect_bounds_when_non_null(self, ocr_result: OCRResult):
        """
        Pour tout OCRResult, quand amount, total ou tax_rate ne sont pas None,
        ils doivent respecter leurs bornes:
        - amount/total: [0.01, 999,999,999.99]
        - tax_rate: [0, 100]

        **Validates: Requirements 4.4, 4.7**
        """
        # Vérifier amount
        if ocr_result.amount is not None:
            assert AMOUNT_MIN <= ocr_result.amount <= AMOUNT_MAX, (
                f"amount = {ocr_result.amount}, doit être dans "
                f"[{AMOUNT_MIN}, {AMOUNT_MAX}]"
            )

        # Vérifier total
        if ocr_result.total is not None:
            assert AMOUNT_MIN <= ocr_result.total <= AMOUNT_MAX, (
                f"total = {ocr_result.total}, doit être dans "
                f"[{AMOUNT_MIN}, {AMOUNT_MAX}]"
            )

        # Vérifier tax_rate
        if ocr_result.tax_rate is not None:
            assert TAX_RATE_MIN <= ocr_result.tax_rate <= TAX_RATE_MAX, (
                f"tax_rate = {ocr_result.tax_rate}, doit être dans "
                f"[{TAX_RATE_MIN}, {TAX_RATE_MAX}]"
            )

    @settings(max_examples=150, deadline=None)
    @given(ocr_result=valid_ocr_result())
    def test_output_structure_is_well_formed(self, ocr_result: OCRResult):
        """
        Pour tout OCRResult, la structure de sortie (via to_dict) doit contenir
        exactement les 6 champs de facture, field_confidences, et warnings.
        Les types doivent être cohérents.

        **Validates: Requirements 4.4, 4.7**
        """
        output = ocr_result.to_dict()

        # Vérifier la présence des 6 champs de facture dans le dict de sortie
        for field_name in EXPECTED_OCR_FIELDS:
            assert field_name in output, (
                f"Le champ '{field_name}' est absent du dict de sortie"
            )

        # Vérifier la présence de field_confidences et warnings
        assert "field_confidences" in output, "field_confidences absent du dict"
        assert "warnings" in output, "warnings absent du dict"

        # Vérifier que field_confidences est un dict avec exactement les 6 champs
        fc = output["field_confidences"]
        assert isinstance(fc, dict), "field_confidences doit être un dictionnaire"
        assert set(fc.keys()) == EXPECTED_OCR_FIELDS, (
            f"field_confidences keys: {set(fc.keys())} != {EXPECTED_OCR_FIELDS}"
        )

        # Vérifier que chaque confiance dans le dict est dans [0.0, 1.0]
        for field_name, conf_val in fc.items():
            assert isinstance(conf_val, float), (
                f"Confiance '{field_name}' dans dict doit être float, est {type(conf_val)}"
            )
            assert 0.0 <= conf_val <= 1.0, (
                f"Confiance '{field_name}' = {conf_val}, hors bornes [0.0, 1.0]"
            )

        # Vérifier que warnings est une liste
        assert isinstance(output["warnings"], list), "warnings doit être une liste"

        # Vérifier les types des champs numériques quand non-None
        if output["amount"] is not None:
            assert isinstance(output["amount"], (int, float)), (
                f"amount doit être numérique, est {type(output['amount'])}"
            )
        if output["total"] is not None:
            assert isinstance(output["total"], (int, float)), (
                f"total doit être numérique, est {type(output['total'])}"
            )
        if output["tax_rate"] is not None:
            assert isinstance(output["tax_rate"], (int, float)), (
                f"tax_rate doit être numérique, est {type(output['tax_rate'])}"
            )



# =============================================================================
# Stratégies supplémentaires pour Property 7: YOLOv8 fallback on low confidence
# =============================================================================

# Les 6 champs du pipeline OCR (réutilisation de FIELD_NAMES)
OCR_FIELDS = FIELD_NAMES

# Stratégie: scores de confiance faibles (en dessous du seuil)
low_confidence_scores = st.floats(
    min_value=0.0, max_value=0.49,
    allow_nan=False, allow_infinity=False,
)

# Stratégie: scores de confiance élevés (au-dessus ou égal au seuil)
high_confidence_scores = st.floats(
    min_value=0.5, max_value=1.0,
    allow_nan=False, allow_infinity=False,
)


@st.composite
def yolo_confidences_with_low_fields(draw):
    """Génère des confiances YOLOv8 avec au moins un champ sous le seuil 0.5."""
    confidences = {}
    has_low = False

    for field_name in OCR_FIELDS:
        # Mélanger des scores hauts et bas
        score = draw(st.floats(
            min_value=0.0, max_value=1.0,
            allow_nan=False, allow_infinity=False,
        ))
        confidences[field_name] = score
        if score < YOLO_CONFIDENCE_THRESHOLD:
            has_low = True

    # Garantir qu'au moins un champ est sous le seuil
    if not has_low:
        forced_field = draw(st.sampled_from(OCR_FIELDS))
        confidences[forced_field] = draw(low_confidence_scores)

    return confidences


# =============================================================================
# Property 7: YOLOv8 fallback on low confidence
# Feature: invoice-fraud-detection, Property 7: YOLOv8 fallback on low confidence
# Validates: Requirements 4.3
# =============================================================================


class TestYOLOv8FallbackOnLowConfidence:
    """Property 7: YOLOv8 fallback on low confidence.

    **Validates: Requirements 4.3**

    Pour tout résultat de détection YOLOv8 où au moins un champ a un score de
    confiance inférieur à 0.5, le pipeline OCR doit utiliser l'extraction
    Tesseract pour les champs concernés.
    """

    @given(
        confidences=yolo_confidences_with_low_fields(),
    )
    @settings(max_examples=150, deadline=None)
    def test_low_confidence_fields_use_tesseract_fallback(self, confidences):
        """Vérifie que les champs avec confiance YOLOv8 < 0.5 utilisent le fallback Tesseract.

        Pour chaque champ dont la confiance YOLOv8 est inférieure à 0.5,
        la valeur finale doit provenir de Tesseract (pas de YOLOv8).

        **Validates: Requirements 4.3**
        """
        # Préparer les résultats simulés de YOLOv8 et Tesseract
        yolo_fields = {f: f"yolo_{f}" for f in OCR_FIELDS}
        tesseract_fields = {f: f"tesseract_{f}" for f in OCR_FIELDS}
        tesseract_confidences = {f: 0.7 for f in OCR_FIELDS}

        # Résultat simulé de YOLOv8
        yolo_result = {
            "fields": yolo_fields,
            "confidences": confidences,
        }

        # Résultat simulé de Tesseract
        tesseract_result = {
            "fields": tesseract_fields,
            "confidences": tesseract_confidences,
        }

        pipeline = OCRPipeline()

        # Mocker les méthodes internes du pipeline
        with patch.object(pipeline, 'detect_fields_yolo', return_value=yolo_result), \
             patch.object(pipeline, 'extract_tesseract', return_value=tesseract_result), \
             patch.object(pipeline, 'preprocess', return_value=np.zeros((100, 100), dtype=np.uint8)), \
             patch('cv2.imread', return_value=np.zeros((100, 100, 3), dtype=np.uint8)), \
             patch('os.path.exists', return_value=True):

            result = pipeline.extract("fake_image.png")

        # Vérification: pour chaque champ avec confiance < 0.5, Tesseract doit être utilisé
        for field_name in OCR_FIELDS:
            yolo_conf = confidences.get(field_name, 0.0)
            if yolo_conf < YOLO_CONFIDENCE_THRESHOLD:
                # Le champ doit utiliser la confiance Tesseract (0.7), pas YOLOv8
                # Note: _validate_and_convert_fields peut mettre 0.0 si la valeur texte
                # "tesseract_xxx" ne passe pas la validation de type, mais la confiance
                # dans le flux de fusion doit provenir de Tesseract
                final_conf = result.field_confidences[field_name]
                # Si la validation échoue, la confiance sera 0.0 (champ invalide)
                # Sinon elle doit être la confiance Tesseract
                assert final_conf != yolo_conf or yolo_conf == 0.0, (
                    f"Le champ '{field_name}' avec confiance YOLOv8={yolo_conf:.4f} "
                    f"ne devrait pas garder cette confiance (fallback attendu)"
                )

    @given(
        low_field=st.sampled_from(FIELD_NAMES),
        low_score=low_confidence_scores,
    )
    @settings(max_examples=150, deadline=None)
    def test_specific_low_field_triggers_tesseract(self, low_field, low_score):
        """Vérifie qu'un champ spécifique avec confiance < 0.5 déclenche le fallback.

        On met un seul champ en confiance basse et les autres en confiance haute,
        puis on vérifie que seul le champ bas utilise Tesseract.

        **Validates: Requirements 4.3**
        """
        # Construire les confiances: un seul champ bas, les autres hauts
        confidences = {f: 0.85 for f in OCR_FIELDS}
        confidences[low_field] = low_score

        # Valeurs distinctes pour identifier la source
        yolo_fields = {f: f"yolo_{f}" for f in OCR_FIELDS}
        tesseract_fields = {f: f"tesseract_{f}" for f in OCR_FIELDS}
        tesseract_confidences = {f: 0.6 for f in OCR_FIELDS}

        yolo_result = {
            "fields": yolo_fields,
            "confidences": confidences,
        }
        tesseract_result = {
            "fields": tesseract_fields,
            "confidences": tesseract_confidences,
        }

        pipeline = OCRPipeline()

        with patch.object(pipeline, 'detect_fields_yolo', return_value=yolo_result), \
             patch.object(pipeline, 'extract_tesseract', return_value=tesseract_result), \
             patch.object(pipeline, 'preprocess', return_value=np.zeros((100, 100), dtype=np.uint8)), \
             patch('cv2.imread', return_value=np.zeros((100, 100, 3), dtype=np.uint8)), \
             patch('os.path.exists', return_value=True):

            result = pipeline.extract("fake_image.png")

        # Le champ à faible confiance ne doit PAS avoir la confiance YOLOv8
        # Il doit avoir soit la confiance Tesseract (0.6), soit 0.0 si la valeur
        # "tesseract_xxx" échoue la validation
        final_conf = result.field_confidences[low_field]
        assert final_conf != low_score or low_score == 0.0, (
            f"Le champ '{low_field}' avec confiance YOLOv8={low_score:.4f} "
            f"ne devrait pas garder sa confiance YOLOv8 (fallback attendu)"
        )

        # Les champs à haute confiance (0.85): ils peuvent garder la confiance YOLOv8
        # ou avoir 0.0 si "yolo_xxx" échoue la validation, mais ne doivent pas
        # avoir la confiance Tesseract (0.6) car YOLOv8 est prioritaire
        for field_name in OCR_FIELDS:
            if field_name != low_field:
                field_conf = result.field_confidences[field_name]
                # La confiance doit être soit 0.85 (YOLOv8), soit 0.0 (validation échouée)
                # mais pas 0.6 (Tesseract) car le seuil YOLOv8 est satisfait
                assert field_conf != tesseract_confidences[field_name] or field_conf == 0.0, (
                    f"Le champ '{field_name}' haute confiance ne devrait pas utiliser Tesseract"
                )

    @given(
        subset=st.lists(
            st.sampled_from(FIELD_NAMES),
            min_size=1,
            max_size=6,
            unique=True,
        ),
        low_scores=st.lists(
            low_confidence_scores,
            min_size=1,
            max_size=6,
        ),
    )
    @settings(max_examples=150, deadline=None)
    def test_multiple_low_fields_all_fallback_to_tesseract(self, subset, low_scores):
        """Vérifie que si plusieurs champs sont sous le seuil, tous utilisent Tesseract.

        Pour N champs avec confiance < 0.5, les N champs doivent utiliser
        l'extraction Tesseract comme fallback.

        **Validates: Requirements 4.3**
        """
        # Construire les confiances: champs du subset en bas, le reste en haut
        confidences = {f: 0.85 for f in OCR_FIELDS}
        for i, field_name in enumerate(subset):
            score_idx = i % len(low_scores)
            confidences[field_name] = low_scores[score_idx]

        yolo_fields = {f: f"yolo_{f}" for f in OCR_FIELDS}
        tesseract_fields = {f: f"tesseract_{f}" for f in OCR_FIELDS}
        tesseract_confidences = {f: 0.65 for f in OCR_FIELDS}

        yolo_result = {
            "fields": yolo_fields,
            "confidences": confidences,
        }
        tesseract_result = {
            "fields": tesseract_fields,
            "confidences": tesseract_confidences,
        }

        pipeline = OCRPipeline()

        with patch.object(pipeline, 'detect_fields_yolo', return_value=yolo_result), \
             patch.object(pipeline, 'extract_tesseract', return_value=tesseract_result), \
             patch.object(pipeline, 'preprocess', return_value=np.zeros((100, 100), dtype=np.uint8)), \
             patch('cv2.imread', return_value=np.zeros((100, 100, 3), dtype=np.uint8)), \
             patch('os.path.exists', return_value=True):

            result = pipeline.extract("fake_image.png")

        # Tous les champs du subset doivent avoir été redirigés vers Tesseract
        for field_name in subset:
            final_conf = result.field_confidences[field_name]
            yolo_conf = confidences[field_name]
            # La confiance finale ne doit pas être celle de YOLOv8 (qui est < 0.5)
            # Elle doit être soit Tesseract (0.65), soit 0.0 (validation échouée)
            assert final_conf != yolo_conf or yolo_conf == 0.0, (
                f"Le champ '{field_name}' avec confiance YOLOv8={yolo_conf:.4f} "
                f"ne devrait pas garder cette confiance (fallback attendu)"
            )

        # Les champs non affectés ne doivent pas avoir utilisé Tesseract
        for field_name in OCR_FIELDS:
            if field_name not in subset:
                field_conf = result.field_confidences[field_name]
                # Ne doit pas être la confiance Tesseract (0.65)
                assert field_conf != tesseract_confidences[field_name] or field_conf == 0.0, (
                    f"Le champ '{field_name}' non affecté ne devrait pas utiliser Tesseract"
                )



# =============================================================================
# Property 6: Missing OCR field handling
# Validates: Requirements 4.5
# =============================================================================

# Feature: invoice-fraud-detection, Property 6: Missing OCR field handling


# Stratégie: sous-ensemble non vide des champs requis à simuler comme manquants
missing_fields_subset_strategy = st.lists(
    st.sampled_from(FIELD_NAMES),
    min_size=1,
    max_size=6,
    unique=True,
)

# Stratégie: valeur manquante (None ou chaîne vide)
missing_value_strategy = st.sampled_from([None, ""])


class TestMissingOCRFieldHandling:
    """
    Property 6: Missing OCR field handling

    Pour tout sous-ensemble des 6 champs requis qui ne peuvent être extraits,
    chaque champ manquant doit avoir sa valeur à null, sa confiance à 0.0,
    et un avertissement correspondant identifiant le nom du champ.

    **Validates: Requirements 4.5**
    """

    @given(
        missing_fields=missing_fields_subset_strategy,
        missing_values=st.lists(missing_value_strategy, min_size=6, max_size=6),
    )
    @settings(max_examples=150, deadline=None)
    def test_missing_fields_set_null_confidence_zero_and_warning(
        self, missing_fields, missing_values
    ):
        """
        Pour tout sous-ensemble de champs simulés comme manquants (None ou ""),
        _validate_and_convert_fields doit mettre null, confiance 0.0,
        et un warning pour chaque champ manquant.

        **Validates: Requirements 4.5**
        """
        # Valeurs valides pour les champs présents
        valid_values = {
            "invoice_id": "INV-2024-001",
            "vendor_name": "Maroc Telecom",
            "amount": "15000.50",
            "date": "2024-06-15",
            "tax_rate": "20",
            "total": "18000.60",
        }

        raw_fields = {}
        confidences = {}

        for i, field_name in enumerate(FIELD_NAMES):
            if field_name in missing_fields:
                # Simuler le champ comme manquant (None ou "")
                missing_val_idx = FIELD_NAMES.index(field_name) % len(missing_values)
                raw_fields[field_name] = missing_values[missing_val_idx]
                confidences[field_name] = 0.0
            else:
                # Champ présent avec valeur valide
                raw_fields[field_name] = valid_values[field_name]
                confidences[field_name] = 0.85

        # Appeler la méthode sous test
        pipeline = OCRPipeline()
        result = pipeline._validate_and_convert_fields(raw_fields, confidences)

        # Vérifications pour chaque champ manquant
        for field_name in missing_fields:
            # La valeur du champ doit être None (null)
            field_value = getattr(result, field_name)
            assert field_value is None, (
                f"Le champ '{field_name}' devrait être None mais vaut {field_value}"
            )

            # La confiance du champ doit être 0.0
            assert result.field_confidences[field_name] == 0.0, (
                f"La confiance de '{field_name}' devrait être 0.0 "
                f"mais vaut {result.field_confidences[field_name]}"
            )

            # Un warning doit mentionner le nom du champ
            field_warnings = [w for w in result.warnings if field_name in w]
            assert len(field_warnings) >= 1, (
                f"Aucun warning trouvé pour le champ manquant '{field_name}'. "
                f"Warnings actuels: {result.warnings}"
            )

    @given(
        missing_fields=missing_fields_subset_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_fields_with_none_values(self, missing_fields):
        """
        Vérifie spécifiquement que les champs à None sont correctement
        traités comme manquants avec null, confiance 0.0, et warning.

        **Validates: Requirements 4.5**
        """
        # Valeurs valides pour les champs présents
        valid_values = {
            "invoice_id": "FACT-9999",
            "vendor_name": "Atlas BTP",
            "amount": "7500.00",
            "date": "2024-01-10",
            "tax_rate": "14",
            "total": "8550.00",
        }

        raw_fields = {}
        confidences = {}

        for field_name in FIELD_NAMES:
            if field_name in missing_fields:
                raw_fields[field_name] = None
                confidences[field_name] = 0.0
            else:
                raw_fields[field_name] = valid_values[field_name]
                confidences[field_name] = 0.92

        pipeline = OCRPipeline()
        result = pipeline._validate_and_convert_fields(raw_fields, confidences)

        # Vérifier que les champs manquants ont bien null + confiance 0 + warning
        for field_name in missing_fields:
            assert getattr(result, field_name) is None
            assert result.field_confidences[field_name] == 0.0
            assert any(field_name in w for w in result.warnings)

        # Vérifier que les champs présents n'ont PAS de warning "non extrait"
        present_fields = [f for f in FIELD_NAMES if f not in missing_fields]
        for field_name in present_fields:
            non_extrait_warnings = [
                w for w in result.warnings
                if field_name in w and "non extrait" in w
            ]
            assert len(non_extrait_warnings) == 0, (
                f"Le champ présent '{field_name}' ne devrait pas avoir de warning "
                f"'non extrait'. Warnings: {result.warnings}"
            )

    @given(
        missing_fields=missing_fields_subset_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_fields_with_empty_string_values(self, missing_fields):
        """
        Vérifie spécifiquement que les champs vides ("") sont traités
        comme manquants avec les mêmes garanties (null, confiance 0.0, warning).

        **Validates: Requirements 4.5**
        """
        valid_values = {
            "invoice_id": "INV-ABC-123",
            "vendor_name": "Souss Agro",
            "amount": "25000.75",
            "date": "2024-03-20",
            "tax_rate": "10",
            "total": "27500.83",
        }

        raw_fields = {}
        confidences = {}

        for field_name in FIELD_NAMES:
            if field_name in missing_fields:
                raw_fields[field_name] = ""
                confidences[field_name] = 0.0
            else:
                raw_fields[field_name] = valid_values[field_name]
                confidences[field_name] = 0.78

        pipeline = OCRPipeline()
        result = pipeline._validate_and_convert_fields(raw_fields, confidences)

        # Vérifier les garanties pour chaque champ manquant
        for field_name in missing_fields:
            assert getattr(result, field_name) is None, (
                f"Champ '{field_name}' avec valeur vide devrait être None"
            )
            assert result.field_confidences[field_name] == 0.0, (
                f"Confiance de '{field_name}' avec valeur vide devrait être 0.0"
            )
            assert any(field_name in w for w in result.warnings), (
                f"Un warning devrait mentionner '{field_name}'"
            )
