"""
Tests d'intégration end-to-end du pipeline complet.
Vérifie le flux: upload → OCR → features → ML → DB → réponse.

Validates: Requirements 3.2, 9.3, 9.4
"""

import io
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api import app
from backend.auth import hash_password, seed_demo_account
from backend.database import Base, Invoice, User, get_db
from ml.feature_engineering import FeatureVector
from ml.fraud_detector import FraudDetector, FraudResult
from ocr.pipeline import OCRResult


# --- Fixtures ---


@pytest.fixture(scope="function")
def integration_db():
    """Crée une base de données en mémoire pour les tests d'intégration."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )
    session = TestingSessionLocal()

    # Créer le compte démo
    seed_demo_account(session)

    yield session

    session.rollback()
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def integration_client(integration_db):
    """Client FastAPI utilisant la DB d'intégration."""

    def override_get_db():
        try:
            yield integration_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_token(integration_client):
    """Obtient un token JWT valide via le login."""
    response = integration_client.post(
        "/api/login",
        json={"username": "admin", "password": "admin123"},
    )
    assert response.status_code == 200
    return response.json()["token"]


@pytest.fixture
def auth_headers(auth_token):
    """Headers d'authentification avec le token Bearer."""
    return {"Authorization": f"Bearer {auth_token}"}


def _create_test_image():
    """Crée une image PNG minimale pour les tests (blanc 100x100)."""
    # Créer une image PNG valide minimale
    import struct
    import zlib

    width, height = 100, 100
    # Image blanche
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00" + b"\xff" * (width * 3)

    def create_png(width, height, raw_data):
        """Génère un fichier PNG valide."""

        def chunk(chunk_type, data):
            c = chunk_type + data
            crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack(">I", len(data)) + c + crc

        # Signature PNG
        signature = b"\x89PNG\r\n\x1a\n"
        # IHDR
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr = chunk(b"IHDR", ihdr_data)
        # IDAT
        compressed = zlib.compress(raw_data)
        idat = chunk(b"IDAT", compressed)
        # IEND
        iend = chunk(b"IEND", b"")

        return signature + ihdr + idat + iend

    return create_png(width, height, raw_data)


# --- Mock pour le pipeline OCR ---


def _mock_ocr_result():
    """Résultat OCR simulé pour les tests d'intégration."""
    return OCRResult(
        invoice_id="FAC-2024-001",
        vendor_name="Maroc Telecom",
        amount=15000.0,
        date="2024-06-15",
        tax_rate=20.0,
        total=18000.0,
        field_confidences={
            "invoice_id": 0.95,
            "vendor_name": 0.92,
            "amount": 0.88,
            "date": 0.91,
            "tax_rate": 0.85,
            "total": 0.87,
        },
        warnings=[],
    )


# --- Tests E2E du pipeline complet ---


class TestPipelineEndToEnd:
    """Tests d'intégration du pipeline complet upload → résultat."""

    def test_upload_triggers_full_pipeline(
        self, integration_client, auth_headers, integration_db
    ):
        """Vérifie que l'upload déclenche OCR → features → ML → DB → réponse.

        Le pipeline complet doit:
        1. Accepter le fichier
        2. Extraire les données OCR (mocké)
        3. Calculer les features
        4. Obtenir un score ML
        5. Persister en base
        6. Retourner une réponse complète
        """
        test_image = _create_test_image()

        # Mock du pipeline OCR pour éviter les dépendances YOLOv8/Tesseract
        with patch("ocr.pipeline.OCRPipeline") as MockOCRPipeline:
            mock_instance = MockOCRPipeline.return_value
            mock_instance.extract.return_value = _mock_ocr_result()

            # Mock du FraudDetector pour avoir un résultat déterministe
            with patch("ml.fraud_detector.FraudDetector") as MockDetector:
                mock_detector = MockDetector.return_value
                mock_detector.predict.return_value = FraudResult(
                    fraud_score=0.45,
                    fraud_label="Suspect",
                    fraud_reason="Top 3 features: amount_zscore=0.5, tax_inconsistency=False, vendor_deviation=0.3",
                )

                response = integration_client.post(
                    "/api/upload",
                    files={"file": ("facture.png", test_image, "image/png")},
                    headers=auth_headers,
                )

        # Vérifier la réponse HTTP
        assert response.status_code == 200
        data = response.json()

        # Vérifier la structure de la réponse
        assert "invoice_data" in data
        assert "fraud_score" in data
        assert "fraud_label" in data
        assert "fraud_reason" in data

        # Vérifier les valeurs du résultat ML
        assert data["fraud_score"] == 0.45
        assert data["fraud_label"] == "Suspect"
        assert "Top 3 features" in data["fraud_reason"]

        # Vérifier les données extraites
        assert data["invoice_data"]["vendor_name"] == "Maroc Telecom"
        assert data["invoice_data"]["amount_ht"] == 15000.0
        assert data["invoice_data"]["date"] == "2024-06-15"

    def test_upload_persists_to_database(
        self, integration_client, auth_headers, integration_db
    ):
        """Vérifie que le résultat d'analyse est bien persisté en base de données."""
        test_image = _create_test_image()

        with patch("ocr.pipeline.OCRPipeline") as MockOCRPipeline:
            mock_instance = MockOCRPipeline.return_value
            mock_instance.extract.return_value = _mock_ocr_result()

            with patch("ml.fraud_detector.FraudDetector") as MockDetector:
                mock_detector = MockDetector.return_value
                mock_detector.predict.return_value = FraudResult(
                    fraud_score=0.72,
                    fraud_label="Frauduleux",
                    fraud_reason="Top 3 features: amount_zscore=2.1, round_amount_flag=True, vendor_deviation=0.8",
                )

                response = integration_client.post(
                    "/api/upload",
                    files={"file": ("facture.png", test_image, "image/png")},
                    headers=auth_headers,
                )

        assert response.status_code == 200

        # Vérifier la persistance en DB
        invoice = (
            integration_db.query(Invoice)
            .filter(Invoice.invoice_id == "FAC-2024-001")
            .first()
        )
        assert invoice is not None
        assert invoice.vendor_name == "Maroc Telecom"
        assert invoice.amount_ht == 15000.0
        assert invoice.fraud_score == 0.72
        assert invoice.fraud_label == "Frauduleux"
        assert invoice.fraud_reason is not None
        assert invoice.analyzed_at is not None
        assert invoice.file_type == "png"

    def test_upload_invalid_format_rejected(
        self, integration_client, auth_headers
    ):
        """Vérifie que les fichiers avec un format non supporté sont rejetés."""
        fake_content = b"not a real file content"

        response = integration_client.post(
            "/api/upload",
            files={"file": ("document.txt", fake_content, "text/plain")},
            headers=auth_headers,
        )

        assert response.status_code == 400
        assert "Format non supporté" in response.json()["detail"]

    def test_upload_oversized_file_rejected(
        self, integration_client, auth_headers
    ):
        """Vérifie que les fichiers > 10 MB sont rejetés."""
        # Créer un contenu > 10 MB
        oversized_content = b"\x00" * (11 * 1024 * 1024)

        response = integration_client.post(
            "/api/upload",
            files={"file": ("large.png", oversized_content, "image/png")},
            headers=auth_headers,
        )

        assert response.status_code == 400
        assert "volumineux" in response.json()["detail"] or "10 MB" in response.json()["detail"]

    def test_upload_without_auth_returns_401(self, integration_client):
        """Vérifie que l'upload sans token est rejeté avec HTTP 401."""
        test_image = _create_test_image()

        response = integration_client.post(
            "/api/upload",
            files={"file": ("facture.png", test_image, "image/png")},
        )

        assert response.status_code == 401

    def test_invoices_visible_after_upload(
        self, integration_client, auth_headers, integration_db
    ):
        """Vérifie qu'une facture analysée apparaît dans GET /api/invoices."""
        test_image = _create_test_image()

        with patch("ocr.pipeline.OCRPipeline") as MockOCRPipeline:
            mock_instance = MockOCRPipeline.return_value
            mock_instance.extract.return_value = _mock_ocr_result()

            with patch("ml.fraud_detector.FraudDetector") as MockDetector:
                mock_detector = MockDetector.return_value
                mock_detector.predict.return_value = FraudResult(
                    fraud_score=0.15,
                    fraud_label="Normal",
                    fraud_reason="Top 3 features: amount_zscore=0.1, tax_inconsistency=False, weekend_flag=False",
                )

                # Upload une facture
                upload_response = integration_client.post(
                    "/api/upload",
                    files={"file": ("facture.png", test_image, "image/png")},
                    headers=auth_headers,
                )
                assert upload_response.status_code == 200

        # Vérifier que la facture apparaît dans la liste
        list_response = integration_client.get(
            "/api/invoices", headers=auth_headers
        )
        assert list_response.status_code == 200
        invoices = list_response.json()
        assert len(invoices) >= 1

        # Trouver notre facture
        our_invoice = next(
            (inv for inv in invoices if inv.get("invoice_id") == "FAC-2024-001"),
            None,
        )
        assert our_invoice is not None
        assert our_invoice["fraud_score"] == 0.15
        assert our_invoice["fraud_label"] == "Normal"

    def test_stats_updated_after_upload(
        self, integration_client, auth_headers, integration_db
    ):
        """Vérifie que les stats sont mises à jour après un upload."""
        test_image = _create_test_image()

        # Vérifier les stats avant (vide)
        stats_before = integration_client.get(
            "/api/stats", headers=auth_headers
        )
        assert stats_before.status_code == 200
        assert stats_before.json()["total_invoices"] == 0

        # Upload une facture
        with patch("ocr.pipeline.OCRPipeline") as MockOCRPipeline:
            mock_instance = MockOCRPipeline.return_value
            mock_instance.extract.return_value = _mock_ocr_result()

            with patch("ml.fraud_detector.FraudDetector") as MockDetector:
                mock_detector = MockDetector.return_value
                mock_detector.predict.return_value = FraudResult(
                    fraud_score=0.55,
                    fraud_label="Suspect",
                    fraud_reason="Top 3 features: amount_zscore=1.2, tax_inconsistency=True, duplicate_flag=False",
                )

                upload_response = integration_client.post(
                    "/api/upload",
                    files={"file": ("facture.png", test_image, "image/png")},
                    headers=auth_headers,
                )
                assert upload_response.status_code == 200

        # Vérifier les stats après
        stats_after = integration_client.get(
            "/api/stats", headers=auth_headers
        )
        assert stats_after.status_code == 200
        stats_data = stats_after.json()
        assert stats_data["total_invoices"] == 1
        assert stats_data["fraud_rate"] == 100.0  # 1 Suspect sur 1 total
        assert stats_data["total_amount"] == 18000.0  # amount_ttc
