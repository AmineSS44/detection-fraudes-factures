"""
Tests d'intégration du flux d'authentification API.
Vérifie: login → obtention token → accès aux routes protégées.

Validates: Requirements 9.3, 9.10
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api import app
from backend.auth import hash_password, seed_demo_account
from backend.database import Base, Invoice, User, get_db


# --- Fixtures ---


@pytest.fixture(scope="function")
def auth_db():
    """Base de données en mémoire pour les tests d'authentification."""
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
def auth_client(auth_db):
    """Client de test avec la DB d'authentification."""

    def override_get_db():
        try:
            yield auth_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


# --- Tests du flux d'authentification ---


class TestAuthenticationFlow:
    """Tests du flux complet d'authentification: login → token → routes protégées."""

    def test_login_with_valid_credentials_returns_token(self, auth_client):
        """Vérifie que le login avec identifiants valides retourne un JWT token."""
        response = auth_client.post(
            "/api/login",
            json={"username": "admin", "password": "admin123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert "expires_at" in data
        assert len(data["token"]) > 0

    def test_login_with_invalid_password_returns_401(self, auth_client):
        """Vérifie que des identifiants invalides retournent 401."""
        response = auth_client.post(
            "/api/login",
            json={"username": "admin", "password": "wrong_password"},
        )

        assert response.status_code == 401
        assert "Identifiants incorrects" in response.json()["detail"]

    def test_login_with_unknown_user_returns_401(self, auth_client):
        """Vérifie qu'un utilisateur inexistant retourne 401."""
        response = auth_client.post(
            "/api/login",
            json={"username": "unknown_user", "password": "any_password"},
        )

        assert response.status_code == 401

    def test_token_grants_access_to_invoices(self, auth_client):
        """Vérifie que le token obtenu donne accès à GET /api/invoices."""
        # Étape 1: Login
        login_response = auth_client.post(
            "/api/login",
            json={"username": "admin", "password": "admin123"},
        )
        assert login_response.status_code == 200
        token = login_response.json()["token"]

        # Étape 2: Accès à la route protégée avec le token
        headers = {"Authorization": f"Bearer {token}"}
        invoices_response = auth_client.get("/api/invoices", headers=headers)

        assert invoices_response.status_code == 200
        assert isinstance(invoices_response.json(), list)

    def test_token_grants_access_to_stats(self, auth_client):
        """Vérifie que le token obtenu donne accès à GET /api/stats."""
        # Login pour obtenir le token
        login_response = auth_client.post(
            "/api/login",
            json={"username": "admin", "password": "admin123"},
        )
        token = login_response.json()["token"]

        # Accès à /api/stats
        headers = {"Authorization": f"Bearer {token}"}
        stats_response = auth_client.get("/api/stats", headers=headers)

        assert stats_response.status_code == 200
        stats = stats_response.json()
        assert "total_invoices" in stats
        assert "fraud_rate" in stats
        assert "total_amount" in stats

    def test_token_grants_access_to_models_report(self, auth_client):
        """Vérifie que le token donne accès à GET /api/models/report.

        Le endpoint est protégé: avec un token valide on obtient 200 ou 404
        (selon que le fichier model_comparison.json existe ou non).
        L'important est que ce ne soit PAS un 401.
        """
        # Login
        login_response = auth_client.post(
            "/api/login",
            json={"username": "admin", "password": "admin123"},
        )
        token = login_response.json()["token"]

        # Accès à /api/models/report avec token valide
        headers = {"Authorization": f"Bearer {token}"}
        report_response = auth_client.get("/api/models/report", headers=headers)

        # Avec un token valide, on ne doit PAS avoir 401
        # Le résultat est soit 200 (fichier existe) soit 404 (fichier absent)
        assert report_response.status_code in (200, 404)
        assert report_response.status_code != 401


class TestProtectedEndpointsWithoutAuth:
    """Tests vérifiant que les routes protégées rejettent les requêtes sans token."""

    def test_invoices_without_token_returns_401(self, auth_client):
        """GET /api/invoices sans token → 401."""
        response = auth_client.get("/api/invoices")
        assert response.status_code == 401

    def test_stats_without_token_returns_401(self, auth_client):
        """GET /api/stats sans token → 401."""
        response = auth_client.get("/api/stats")
        assert response.status_code == 401

    def test_models_report_without_token_returns_401(self, auth_client):
        """GET /api/models/report sans token → 401."""
        response = auth_client.get("/api/models/report")
        assert response.status_code == 401

    def test_upload_without_token_returns_401(self, auth_client):
        """POST /api/upload sans token → 401."""
        response = auth_client.post(
            "/api/upload",
            files={"file": ("test.png", b"fake_content", "image/png")},
        )
        assert response.status_code == 401

    def test_invalid_token_returns_401(self, auth_client):
        """Vérifie qu'un token invalide est rejeté avec 401."""
        headers = {"Authorization": "Bearer invalid_token_value"}
        response = auth_client.get("/api/invoices", headers=headers)
        assert response.status_code == 401

    def test_expired_token_format_returns_401(self, auth_client):
        """Vérifie qu'un token mal formaté est rejeté."""
        # Token sans préfixe Bearer
        headers = {"Authorization": "just_a_token"}
        response = auth_client.get("/api/invoices", headers=headers)
        assert response.status_code == 401

    def test_empty_authorization_header_returns_401(self, auth_client):
        """Vérifie qu'un header Authorization vide est rejeté."""
        headers = {"Authorization": ""}
        response = auth_client.get("/api/invoices", headers=headers)
        assert response.status_code == 401


class TestAccountLockout:
    """Tests du mécanisme de verrouillage de compte."""

    def test_account_locks_after_5_failed_attempts(self, auth_client):
        """Vérifie que le compte se verrouille après 5 tentatives échouées."""
        # 5 tentatives avec mauvais mot de passe
        for _ in range(5):
            response = auth_client.post(
                "/api/login",
                json={"username": "admin", "password": "wrong"},
            )
            assert response.status_code == 401

        # La 6ème tentative (même avec bon mot de passe) devrait échouer
        response = auth_client.post(
            "/api/login",
            json={"username": "admin", "password": "admin123"},
        )
        assert response.status_code == 401
