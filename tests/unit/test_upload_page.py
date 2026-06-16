"""
Tests unitaires pour la page d'upload et d'analyse de factures (app.py - show_upload).
Vérifie la logique de validation côté client et le comportement de l'affichage des résultats.
"""

import io
from unittest.mock import MagicMock, patch

import pytest


class TestFileValidation:
    """Tests pour la validation des fichiers côté client."""

    def test_max_file_size_constant(self):
        """Vérifie que la taille maximale est bien 10 Mo."""
        from app import MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB

        assert MAX_FILE_SIZE_MB == 10
        assert MAX_FILE_SIZE_BYTES == 10 * 1024 * 1024

    def test_accepted_types(self):
        """Vérifie que les types acceptés sont PDF, JPG, JPEG, PNG."""
        from app import ACCEPTED_TYPES

        assert "pdf" in ACCEPTED_TYPES
        assert "jpg" in ACCEPTED_TYPES
        assert "jpeg" in ACCEPTED_TYPES
        assert "png" in ACCEPTED_TYPES
        # Pas d'autres formats non supportés
        assert "doc" not in ACCEPTED_TYPES
        assert "xlsx" not in ACCEPTED_TYPES
        assert "gif" not in ACCEPTED_TYPES

    def test_api_base_url(self):
        """Vérifie l'URL de base de l'API."""
        from app import API_BASE_URL

        assert API_BASE_URL == "http://127.0.0.1:8000"


class TestAuthHeaders:
    """Tests pour la génération des headers d'authentification."""

    def test_get_auth_headers_with_token(self):
        """Vérifie que les headers contiennent le token Bearer."""
        import app
        with patch.object(app, "st") as mock_st:
            mock_st.session_state = {"token": "test_jwt_token_123"}
            headers = app.get_auth_headers()
            assert headers == {"Authorization": "Bearer test_jwt_token_123"}

    def test_get_auth_headers_without_token(self):
        """Vérifie le comportement sans token en session."""
        import app
        with patch.object(app, "st") as mock_st:
            mock_st.session_state = {}
            headers = app.get_auth_headers()
            assert headers == {"Authorization": "Bearer "}

    def test_is_authenticated_true(self):
        """Vérifie qu'un utilisateur avec token est authentifié."""
        import app
        with patch.object(app, "st") as mock_st:
            mock_st.session_state = {"token": "valid_token"}
            result = app.is_authenticated()
            assert result is True

    def test_is_authenticated_false_no_token(self):
        """Vérifie qu'un utilisateur sans token n'est pas authentifié."""
        import app
        # Directement tester la logique: un dict vide retourne "" via .get
        original_st = app.st
        try:
            mock_st = MagicMock()
            mock_st.session_state = {}
            app.st = mock_st
            result = app.is_authenticated()
            assert result is False
        finally:
            app.st = original_st

    def test_is_authenticated_false_empty_token(self):
        """Vérifie qu'un token vide n'est pas considéré comme authentifié."""
        import app
        original_st = app.st
        try:
            mock_st = MagicMock()
            mock_st.session_state = {"token": ""}
            app.st = mock_st
            result = app.is_authenticated()
            assert result is False
        finally:
            app.st = original_st


class TestDisplayPipelineError:
    """Tests pour l'affichage des erreurs de pipeline."""

    @patch("app.st")
    def test_extract_ocr_stage(self, mock_st):
        """Vérifie l'extraction du stage OCR depuis le message d'erreur."""
        from app import _display_pipeline_error

        _display_pipeline_error("Échec au stage 'Extraction OCR': fichier corrompu")

        # Vérifie que st.error a été appelé avec le bon stage
        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        assert "Extraction OCR" in error_msg

    @patch("app.st")
    def test_extract_feature_engineering_stage(self, mock_st):
        """Vérifie l'extraction du stage Feature Engineering."""
        from app import _display_pipeline_error

        _display_pipeline_error("Échec au stage 'Feature Engineering': champs manquants")

        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        assert "Feature Engineering" in error_msg

    @patch("app.st")
    def test_extract_fraud_detection_stage(self, mock_st):
        """Vérifie l'extraction du stage Détection de fraude."""
        from app import _display_pipeline_error

        _display_pipeline_error("Échec au stage 'Détection de fraude': modèle indisponible")

        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        assert "Détection de fraude" in error_msg

    @patch("app.st")
    def test_extract_unknown_stage(self, mock_st):
        """Vérifie le comportement quand le stage n'est pas identifiable."""
        from app import _display_pipeline_error

        _display_pipeline_error("Erreur inconnue")

        mock_st.error.assert_called_once()
        error_msg = mock_st.error.call_args[0][0]
        assert "Inconnu" in error_msg


class TestDisplayResults:
    """Tests pour l'affichage des résultats d'analyse."""

    def _make_mock_container(self, mock_st):
        """Crée un conteneur mock avec des colonnes fonctionnelles."""
        # Créer des mocks pour les colonnes qui supportent le context manager
        def make_col():
            col = MagicMock()
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
            return col

        # st.columns doit retourner le bon nombre de colonnes selon l'argument
        def mock_columns(n):
            return [make_col() for _ in range(n)]

        mock_st.columns.side_effect = mock_columns

        # Créer le conteneur avec context manager qui redirige vers mock_st
        container = MagicMock()
        container.__enter__ = MagicMock(return_value=mock_st)
        container.__exit__ = MagicMock(return_value=False)

        return container

    @patch("app.st")
    def test_display_results_normal(self, mock_st):
        """Vérifie l'affichage pour une facture normale."""
        from app import _display_results

        container = self._make_mock_container(mock_st)

        data = {
            "invoice_data": {
                "vendor_name": "Maroc Telecom",
                "invoice_id": "INV-001",
                "date": "2024-01-15",
                "tax_rate": 20.0,
                "amount_ht": 5000.00,
                "amount_ttc": 6000.00,
            },
            "fraud_score": 0.15,
            "fraud_label": "Normal",
            "fraud_reason": "Tous les indicateurs sont normaux",
        }

        # L'appel ne doit pas lever d'exception
        _display_results(container, data)

    @patch("app.st")
    def test_display_results_frauduleux(self, mock_st):
        """Vérifie l'affichage pour une facture frauduleuse."""
        from app import _display_results

        container = self._make_mock_container(mock_st)

        data = {
            "invoice_data": {
                "vendor_name": "Fournisseur Inconnu",
                "invoice_id": "INV-999",
                "date": "2024-03-16",
                "tax_rate": 25.0,
                "amount_ht": 750000.00,
                "amount_ttc": 937500.00,
            },
            "fraud_score": 0.85,
            "fraud_label": "Frauduleux",
            "fraud_reason": "tax_inconsistency=True, amount_zscore=3.5, vendor_deviation=1.0",
        }

        _display_results(container, data)

    @patch("app.st")
    def test_display_results_missing_fields(self, mock_st):
        """Vérifie l'affichage quand des champs sont absents (None)."""
        from app import _display_results

        container = self._make_mock_container(mock_st)

        data = {
            "invoice_data": {
                "vendor_name": None,
                "invoice_id": None,
                "date": None,
                "tax_rate": None,
                "amount_ht": None,
                "amount_ttc": None,
            },
            "fraud_score": -1.0,
            "fraud_label": "Erreur",
            "fraud_reason": "",
        }

        # Ne doit pas lever d'exception même avec des champs None
        _display_results(container, data)
