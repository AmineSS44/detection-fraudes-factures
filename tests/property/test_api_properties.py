# Feature: invoice-fraud-detection, Property 18: Stats aggregation correctness
# Feature: invoice-fraud-detection, Property 19: File upload format validation
# Feature: invoice-fraud-detection, Property 20: Protected endpoint authentication enforcement
"""
Tests de propriétés pour les endpoints API.
Vérifie la validation d'upload de fichiers, l'enforcement d'auth et les stats via Hypothesis:
- Property 18: Stats aggregation correctness (total_invoices, fraud_rate, total_amount)
- Property 19: File upload format validation (format + taille)
- Property 20: Protected endpoint authentication enforcement

**Validates: Requirements 2.3, 3.5, 9.5, 9.7, 9.10**
"""

import io
import string
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api import app, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from backend.auth import SECRET_KEY, ALGORITHM, create_token
from backend.database import Base, get_db


# --- Helpers pour Property 19 ---


def _make_test_client():
    """Crée un client de test FastAPI avec une DB SQLite en mémoire."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    return client, session, engine


def _get_auth_header():
    """Génère un header Authorization avec un token JWT valide."""
    token = create_token("admin")
    return {"Authorization": f"Bearer {token}"}


# Stratégies Hypothesis pour Property 19

# Extensions invalides : chaînes alphanumériques qui ne sont pas dans ALLOWED_EXTENSIONS
_FORBIDDEN_EXTENSIONS = ALLOWED_EXTENSIONS  # {"pdf", "jpg", "jpeg", "png"}

invalid_extensions = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=10,
).filter(lambda ext: ext.lower() not in _FORBIDDEN_EXTENSIONS)

# Extensions valides pour le test de taille
valid_extensions = st.sampled_from(["pdf", "jpg", "jpeg", "png"])

# Taille de fichier trop grande (entre 10MB+1 byte et 15MB)
oversized_file_sizes = st.integers(
    min_value=MAX_FILE_SIZE + 1,
    max_value=MAX_FILE_SIZE + 5 * 1024 * 1024,  # max 15 MB
)


# =============================================================================
# Property 19: File upload format validation
# Validates: Requirements 3.5, 9.5
# =============================================================================


class TestFileUploadFormatValidation:
    """
    Property 19: File upload format validation

    Pour tout fichier avec une extension non comprise dans {pdf, jpg, png},
    la validation d'upload doit le rejeter.
    Pour tout fichier dépassant 10 MB, la validation d'upload doit le rejeter
    quel que soit le format.

    **Validates: Requirements 3.5, 9.5**
    """

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(extension=invalid_extensions)
    def test_invalid_extension_rejected(self, extension: str):
        """
        Pour toute extension non autorisée, l'API doit retourner 400.

        Génère des extensions aléatoires qui ne sont pas dans {pdf, jpg, jpeg, png}
        et vérifie que l'endpoint /api/upload les rejette.

        **Validates: Requirements 3.5, 9.5**
        """
        client, session, engine = _make_test_client()
        try:
            # Créer un fichier minimal avec l'extension invalide
            filename = f"test_invoice.{extension}"
            file_content = b"fake file content for testing"
            file_obj = io.BytesIO(file_content)

            response = client.post(
                "/api/upload",
                headers=_get_auth_header(),
                files={"file": (filename, file_obj, "application/octet-stream")},
            )

            # Doit être rejeté avec HTTP 400
            assert response.status_code == 400, (
                f"L'extension '.{extension}' devrait être rejetée (400), "
                f"mais a reçu {response.status_code}"
            )

            # Vérifier que le message d'erreur mentionne le format
            detail = response.json().get("detail", "")
            assert "format" in detail.lower() or "supporté" in detail.lower() or "Format" in detail, (
                f"Le message d'erreur devrait mentionner le format. Reçu: {detail}"
            )
        finally:
            app.dependency_overrides.clear()
            session.close()
            engine.dispose()

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])
    @given(extension=valid_extensions, file_size=oversized_file_sizes)
    def test_oversized_file_rejected(self, extension: str, file_size: int):
        """
        Pour tout fichier dépassant 10 MB, même avec une extension valide,
        l'API doit retourner 400.

        Génère des fichiers avec des extensions valides mais une taille > 10 MB.

        **Validates: Requirements 3.5, 9.5**
        """
        client, session, engine = _make_test_client()
        try:
            # Créer un fichier de la taille spécifiée avec une extension valide
            filename = f"test_invoice.{extension}"
            # Contenu de bytes nuls pour atteindre la taille voulue
            file_content = b"\x00" * file_size
            file_obj = io.BytesIO(file_content)

            response = client.post(
                "/api/upload",
                headers=_get_auth_header(),
                files={"file": (filename, file_obj, "application/octet-stream")},
            )

            # Doit être rejeté avec HTTP 400
            assert response.status_code == 400, (
                f"Un fichier de {file_size / (1024*1024):.1f} MB avec extension '.{extension}' "
                f"devrait être rejeté (400), mais a reçu {response.status_code}"
            )

            # Vérifier que le message d'erreur mentionne la taille
            detail = response.json().get("detail", "")
            assert "volumineux" in detail.lower() or "10" in detail or "taille" in detail.lower() or "MB" in detail, (
                f"Le message d'erreur devrait mentionner la taille. Reçu: {detail}"
            )
        finally:
            app.dependency_overrides.clear()
            session.close()
            engine.dispose()

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])
    @given(extension=invalid_extensions, file_size=oversized_file_sizes)
    def test_invalid_extension_and_oversized_rejected(self, extension: str, file_size: int):
        """
        Pour tout fichier avec extension invalide ET taille > 10 MB,
        l'API doit retourner 400 (le format est vérifié en premier).

        **Validates: Requirements 3.5, 9.5**
        """
        client, session, engine = _make_test_client()
        try:
            filename = f"test_invoice.{extension}"
            file_content = b"\x00" * file_size
            file_obj = io.BytesIO(file_content)

            response = client.post(
                "/api/upload",
                headers=_get_auth_header(),
                files={"file": (filename, file_obj, "application/octet-stream")},
            )

            # Doit être rejeté avec HTTP 400
            assert response.status_code == 400, (
                f"Un fichier .{extension} de {file_size / (1024*1024):.1f} MB "
                f"devrait être rejeté (400), mais a reçu {response.status_code}"
            )
        finally:
            app.dependency_overrides.clear()
            session.close()
            engine.dispose()


# --- Helpers pour Property 20 ---

# Client de test FastAPI
client = TestClient(app)

# Endpoints protégés à tester
PROTECTED_ENDPOINTS = [
    ("GET", "/api/invoices"),
    ("GET", "/api/stats"),
    ("POST", "/api/upload"),
    ("GET", "/api/models/report"),
]

# Stratégie pour générer des chaînes aléatoires ASCII (tokens invalides)
# Les headers HTTP n'acceptent que des caractères ASCII
random_strings = st.text(
    alphabet=string.ascii_letters + string.digits + string.punctuation,
    min_size=0,
    max_size=200,
)

# Stratégie pour sélectionner un endpoint protégé
endpoint_strategy = st.sampled_from(PROTECTED_ENDPOINTS)


def _make_request(method: str, path: str, headers: dict = None):
    """Effectue une requête HTTP vers l'API de test."""
    if method == "GET":
        return client.get(path, headers=headers)
    elif method == "POST":
        # Pour POST /api/upload, on envoie un fichier bidon
        if "/upload" in path:
            return client.post(
                path,
                headers=headers,
                files={"file": ("test.pdf", b"fake content", "application/pdf")},
            )
        return client.post(path, headers=headers)
    return None


# =============================================================================
# Property 20: Protected endpoint authentication enforcement
# Validates: Requirements 9.10
# =============================================================================


class TestProtectedEndpointAuthEnforcement:
    """
    Property 20: Protected endpoint authentication enforcement

    For any request to a protected endpoint (invoices, stats, upload,
    models/report) that does not include a valid, non-expired JWT token
    in the header, the API should return HTTP 401.

    **Validates: Requirements 9.10**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(endpoint=endpoint_strategy)
    def test_missing_authorization_header_returns_401(self, endpoint):
        """
        Sans header Authorization, tous les endpoints protégés
        doivent retourner HTTP 401.

        **Validates: Requirements 9.10**
        """
        method, path = endpoint
        response = _make_request(method, path, headers=None)
        assert response.status_code == 401, (
            f"{method} {path} sans Authorization header devrait retourner 401, "
            f"mais a retourné {response.status_code}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        endpoint=endpoint_strategy,
        invalid_token=random_strings,
    )
    def test_random_invalid_token_returns_401(self, endpoint, invalid_token):
        """
        Pour tout token aléatoire (chaîne quelconque), les endpoints protégés
        doivent retourner HTTP 401.

        **Validates: Requirements 9.10**
        """
        method, path = endpoint
        headers = {"Authorization": f"Bearer {invalid_token}"}
        response = _make_request(method, path, headers=headers)
        assert response.status_code == 401, (
            f"{method} {path} avec token invalide '{invalid_token[:50]}...' "
            f"devrait retourner 401, mais a retourné {response.status_code}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(endpoint=endpoint_strategy)
    def test_empty_authorization_header_returns_401(self, endpoint):
        """
        Avec un header Authorization vide, les endpoints protégés
        doivent retourner HTTP 401.

        **Validates: Requirements 9.10**
        """
        method, path = endpoint
        headers = {"Authorization": ""}
        response = _make_request(method, path, headers=headers)
        assert response.status_code == 401, (
            f"{method} {path} avec Authorization vide devrait retourner 401, "
            f"mais a retourné {response.status_code}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        endpoint=endpoint_strategy,
        malformed_prefix=st.text(
            alphabet=string.ascii_letters,
            min_size=1,
            max_size=20,
        ).filter(lambda s: s.lower() != "bearer"),
    )
    def test_malformed_authorization_format_returns_401(self, endpoint, malformed_prefix):
        """
        Pour tout format d'Authorization qui n'est pas 'Bearer <token>',
        les endpoints protégés doivent retourner HTTP 401.

        **Validates: Requirements 9.10**
        """
        method, path = endpoint
        # Utiliser un préfixe différent de "Bearer"
        headers = {"Authorization": f"{malformed_prefix} some-token-value"}
        response = _make_request(method, path, headers=headers)
        assert response.status_code == 401, (
            f"{method} {path} avec format malformé '{malformed_prefix} ...' "
            f"devrait retourner 401, mais a retourné {response.status_code}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        endpoint=endpoint_strategy,
        username=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=30,
        ),
    )
    def test_expired_token_returns_401(self, endpoint, username):
        """
        Pour tout token JWT expiré (même signé avec la bonne clé),
        les endpoints protégés doivent retourner HTTP 401.

        **Validates: Requirements 9.10**
        """
        method, path = endpoint

        # Créer un token expiré (expiration dans le passé)
        expired_payload = {
            "sub": username,
            "exp": datetime.utcnow() - timedelta(hours=1),
        }
        expired_token = jwt.encode(expired_payload, SECRET_KEY, algorithm=ALGORITHM)

        headers = {"Authorization": f"Bearer {expired_token}"}
        response = _make_request(method, path, headers=headers)
        assert response.status_code == 401, (
            f"{method} {path} avec token expiré pour '{username}' "
            f"devrait retourner 401, mais a retourné {response.status_code}"
        )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        endpoint=endpoint_strategy,
        wrong_key=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=10,
            max_size=50,
        ),
        username=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=30,
        ),
    )
    def test_token_signed_with_wrong_key_returns_401(self, endpoint, wrong_key, username):
        """
        Pour tout token signé avec une clé différente de la clé secrète,
        les endpoints protégés doivent retourner HTTP 401.

        **Validates: Requirements 9.10**
        """
        # S'assurer que la mauvaise clé est différente de la vraie
        assume(wrong_key != SECRET_KEY)

        method, path = endpoint

        # Créer un token valide mais signé avec la mauvaise clé
        payload = {
            "sub": username,
            "exp": datetime.utcnow() + timedelta(hours=24),
        }
        bad_token = jwt.encode(payload, wrong_key, algorithm=ALGORITHM)

        headers = {"Authorization": f"Bearer {bad_token}"}
        response = _make_request(method, path, headers=headers)
        assert response.status_code == 401, (
            f"{method} {path} avec token signé par mauvaise clé "
            f"devrait retourner 401, mais a retourné {response.status_code}"
        )


# =============================================================================
# Property 18: Stats aggregation correctness
# Feature: invoice-fraud-detection, Property 18: Stats aggregation correctness
# Validates: Requirements 2.3, 9.7
# =============================================================================

from backend.database import Invoice


# Stratégies pour la génération de factures
_fraud_labels = st.sampled_from(["Normal", "Suspect", "Frauduleux"])
_amounts_ttc = st.floats(min_value=0.01, max_value=999_999.99, allow_nan=False, allow_infinity=False)


def _invoice_strategy():
    """Stratégie pour générer une facture avec label et montant aléatoires."""
    return st.fixed_dictionaries({
        "fraud_label": _fraud_labels,
        "amount_ttc": _amounts_ttc,
        "fraud_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    })


# Liste de factures (entre 1 et 30 factures par test)
_invoice_lists = st.lists(_invoice_strategy(), min_size=1, max_size=30)


def _make_stats_db_and_client():
    """Crée une session SQLite en mémoire et un TestClient FastAPI configuré pour les tests stats."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()

    def override_get_db():
        """Surcharge la dépendance DB pour utiliser la session de test."""
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)

    return session, test_client


# Token JWT valide pour les requêtes authentifiées
_STATS_AUTH_TOKEN = create_token("stats_test_user")
_STATS_AUTH_HEADERS = {"Authorization": f"Bearer {_STATS_AUTH_TOKEN}"}


class TestStatsAggregationCorrectness:
    """
    Property 18: Stats aggregation correctness

    Pour toute collection de factures en base de données:
    - total_invoices doit être égal au nombre de records
    - fraud_rate doit être égal à (count Suspect + Frauduleux) / total * 100
    - total_amount doit être égal à la somme de tous les amount_ttc

    **Validates: Requirements 2.3, 9.7**
    """

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(invoices=_invoice_lists)
    def test_stats_aggregation_formulas(self, invoices):
        """
        Pour toute liste de factures insérées en base, GET /api/stats doit
        retourner des valeurs cohérentes avec les formules d'agrégation.

        **Validates: Requirements 2.3, 9.7**
        """
        session, test_client = _make_stats_db_and_client()
        try:
            # Insérer les factures générées en base
            for i, inv_data in enumerate(invoices):
                invoice = Invoice(
                    invoice_id=f"INV-STATS-{i}",
                    vendor_name="Test Vendor",
                    amount_ht=inv_data["amount_ttc"] / 1.2,
                    tax_rate=20.0,
                    amount_ttc=inv_data["amount_ttc"],
                    date="2024-01-15",
                    file_path="/tmp/test.pdf",
                    file_type="pdf",
                    fraud_score=inv_data["fraud_score"],
                    fraud_label=inv_data["fraud_label"],
                    fraud_reason="Test reason",
                    analyzed_at=datetime.utcnow(),
                    created_at=datetime.utcnow(),
                )
                session.add(invoice)
            session.commit()

            # Appeler GET /api/stats
            response = test_client.get("/api/stats", headers=_STATS_AUTH_HEADERS)
            assert response.status_code == 200, f"Statut inattendu: {response.status_code}"

            data = response.json()

            # --- Vérification: total_invoices = nombre de factures ---
            expected_total = len(invoices)
            assert data["total_invoices"] == expected_total, (
                f"total_invoices: attendu={expected_total}, reçu={data['total_invoices']}"
            )

            # --- Vérification: fraud_rate = (Suspect + Frauduleux) / total * 100 ---
            fraud_count = sum(
                1 for inv in invoices
                if inv["fraud_label"] in ("Suspect", "Frauduleux")
            )
            expected_fraud_rate = (fraud_count / expected_total) * 100
            # L'API arrondit à 1 décimale
            expected_fraud_rate_rounded = round(expected_fraud_rate, 1)
            assert data["fraud_rate"] == expected_fraud_rate_rounded, (
                f"fraud_rate: attendu={expected_fraud_rate_rounded}, reçu={data['fraud_rate']} "
                f"(fraud_count={fraud_count}, total={expected_total})"
            )

            # --- Vérification: total_amount = sum(amount_ttc) ---
            expected_total_amount = sum(inv["amount_ttc"] for inv in invoices)
            # L'API arrondit à 2 décimales
            expected_total_amount_rounded = round(expected_total_amount, 2)
            assert data["total_amount"] == expected_total_amount_rounded, (
                f"total_amount: attendu={expected_total_amount_rounded}, reçu={data['total_amount']}"
            )

        finally:
            session.close()
            app.dependency_overrides.clear()

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(invoices=_invoice_lists)
    def test_stats_fraud_rate_bounds_and_label_semantics(self, invoices):
        """
        Pour toute collection de factures, seuls les labels "Suspect" et
        "Frauduleux" comptent dans le calcul du fraud_rate. Le fraud_rate
        est toujours dans [0, 100].

        **Validates: Requirements 2.3, 9.7**
        """
        session, test_client = _make_stats_db_and_client()
        try:
            # Insérer les factures
            for i, inv_data in enumerate(invoices):
                invoice = Invoice(
                    invoice_id=f"INV-BOUNDS-{i}",
                    vendor_name="Test Vendor",
                    amount_ht=inv_data["amount_ttc"] / 1.2,
                    tax_rate=20.0,
                    amount_ttc=inv_data["amount_ttc"],
                    date="2024-01-15",
                    file_path="/tmp/test.pdf",
                    file_type="pdf",
                    fraud_score=inv_data["fraud_score"],
                    fraud_label=inv_data["fraud_label"],
                    fraud_reason="Test reason",
                    analyzed_at=datetime.utcnow(),
                    created_at=datetime.utcnow(),
                )
                session.add(invoice)
            session.commit()

            # Appeler GET /api/stats
            response = test_client.get("/api/stats", headers=_STATS_AUTH_HEADERS)
            assert response.status_code == 200

            data = response.json()

            # Vérifier que fraud_rate est entre 0 et 100
            assert 0.0 <= data["fraud_rate"] <= 100.0, (
                f"fraud_rate hors bornes: {data['fraud_rate']}"
            )

            # Si toutes les factures sont "Normal", fraud_rate = 0
            all_normal = all(inv["fraud_label"] == "Normal" for inv in invoices)
            if all_normal:
                assert data["fraud_rate"] == 0.0, (
                    f"fraud_rate devrait être 0.0 si toutes normales, reçu={data['fraud_rate']}"
                )

            # Si toutes les factures sont "Suspect" ou "Frauduleux", fraud_rate = 100.0
            all_fraud = all(
                inv["fraud_label"] in ("Suspect", "Frauduleux") for inv in invoices
            )
            if all_fraud:
                assert data["fraud_rate"] == 100.0, (
                    f"fraud_rate devrait être 100.0 si toutes suspectes/frauduleuses, "
                    f"reçu={data['fraud_rate']}"
                )

        finally:
            session.close()
            app.dependency_overrides.clear()
