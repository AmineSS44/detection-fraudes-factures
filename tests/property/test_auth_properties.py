# Feature: invoice-fraud-detection, Property 1: JWT token round-trip
# Feature: invoice-fraud-detection, Property 3: Account lockout threshold
"""
Tests de propriétés pour le module d'authentification.
Vérifie les invariants de sécurité via Hypothesis:
- Property 1: JWT token round-trip (create→verify, expiry 24h, expired token rejected)
- Property 3: Account lockout threshold (5 failures → lock)
"""

from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st
from hypothesis.strategies import integers
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.auth import (
    create_token,
    verify_token,
    authenticate,
    check_lockout,
    hash_password,
    seed_demo_account,
    SECRET_KEY,
    ALGORITHM,
    TOKEN_EXPIRY_HOURS,
    MAX_FAILED_ATTEMPTS,
    LOCKOUT_DURATION_MINUTES,
)
from backend.database import Base, User


# Stratégie pour des usernames valides (alphanumériques + underscore/tiret, 1-50 chars)
valid_usernames = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=50,
)


def _make_session():
    """Crée une session SQLite en mémoire fraîche pour chaque exemple."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Session()


# =============================================================================
# Property 1: JWT token round-trip
# Validates: Requirements 1.1, 1.4
# =============================================================================


class TestJWTTokenRoundTrip:
    """
    Property 1: JWT token round-trip

    For any valid username, creating a JWT token and then verifying it should
    return the same username claim, and the expiry should be exactly 24 hours
    from creation. Conversely, for any token with an expiry time in the past,
    verification should fail.

    **Validates: Requirements 1.1, 1.4**
    """

    @settings(max_examples=100)
    @given(username=valid_usernames)
    def test_create_then_verify_returns_same_username(self, username: str):
        """
        Pour tout username valide, créer un token puis le vérifier
        doit retourner le même username dans le claim 'sub'.

        **Validates: Requirements 1.1**
        """
        # Créer un token
        token = create_token(username)

        # Vérifier le token
        payload = verify_token(token)

        # Le claim 'sub' doit correspondre au username original
        assert payload["sub"] == username

    @settings(max_examples=100)
    @given(username=valid_usernames)
    def test_token_expiry_is_24_hours(self, username: str):
        """
        Pour tout username valide, le token créé doit avoir une expiration
        exactement 24 heures après la création (à la seconde de création près).

        **Validates: Requirements 1.4**
        """
        before = datetime.utcnow()
        token = create_token(username)
        after = datetime.utcnow()

        # Décoder le token sans vérifier l'expiration pour inspecter le claim
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # L'expiration doit être TOKEN_EXPIRY_HOURS (24h) après la création
        exp_timestamp = payload["exp"]
        exp_datetime = datetime.utcfromtimestamp(exp_timestamp)

        # Le claim exp est stocké en secondes entières (pas de microseconds),
        # donc on tronque 'before' à la seconde inférieure pour la comparaison.
        expected_min = before.replace(microsecond=0) + timedelta(hours=TOKEN_EXPIRY_HOURS)
        expected_max = after + timedelta(hours=TOKEN_EXPIRY_HOURS, seconds=1)

        assert expected_min <= exp_datetime <= expected_max

    @settings(max_examples=100)
    @given(username=valid_usernames)
    def test_expired_token_raises_error(self, username: str):
        """
        Pour tout token avec une expiration dans le passé,
        la vérification doit échouer avec une ValueError.

        **Validates: Requirements 1.4**
        """
        # Créer un token expiré manuellement (expiration dans le passé)
        expired_payload = {
            "sub": username,
            "exp": datetime.utcnow() - timedelta(hours=1),
        }
        expired_token = jwt.encode(expired_payload, SECRET_KEY, algorithm=ALGORITHM)

        # La vérification doit lever une ValueError
        with pytest.raises(ValueError, match="invalide ou expiré"):
            verify_token(expired_token)


# =============================================================================
# Property 3: Account lockout threshold
# Validates: Requirements 1.7
# =============================================================================


class TestAccountLockoutThreshold:
    """
    Property 3: Account lockout threshold

    For any user account, after exactly 5 consecutive failed authentication
    attempts, the account should be locked. For any number of consecutive
    failures less than 5, the account should remain unlocked.

    **Validates: Requirements 1.7**
    """

    @given(n_attempts=integers(min_value=1, max_value=4))
    @settings(max_examples=100, deadline=None)
    def test_account_not_locked_below_threshold(self, n_attempts):
        """Moins de 5 échecs consécutifs → compte non verrouillé."""
        db = _make_session()
        try:
            # Créer un utilisateur frais
            user = User(
                username="lockout_test_user",
                password_hash=hash_password("correct_password"),
                failed_attempts=0,
                locked_until=None,
            )
            db.add(user)
            db.commit()

            # Simuler n_attempts tentatives échouées
            for _ in range(n_attempts):
                result = authenticate(db, "lockout_test_user", "wrong_password")
                assert result is None  # Chaque tentative échoue

            # Vérifier que le compte n'est PAS verrouillé
            is_locked = check_lockout(db, "lockout_test_user")
            assert is_locked is False, (
                f"Le compte ne devrait pas être verrouillé après {n_attempts} échecs "
                f"(seuil = {MAX_FAILED_ATTEMPTS})"
            )

            # Vérifier le compteur de tentatives
            db.refresh(user)
            assert user.failed_attempts == n_attempts
            assert user.locked_until is None
        finally:
            db.close()

    @given(n_attempts=integers(min_value=5, max_value=10))
    @settings(max_examples=100, deadline=None)
    def test_account_locked_at_threshold(self, n_attempts):
        """5 échecs consécutifs ou plus → compte verrouillé."""
        db = _make_session()
        try:
            # Créer un utilisateur frais
            user = User(
                username="lockout_test_user",
                password_hash=hash_password("correct_password"),
                failed_attempts=0,
                locked_until=None,
            )
            db.add(user)
            db.commit()

            # Simuler n_attempts tentatives échouées
            for _ in range(n_attempts):
                authenticate(db, "lockout_test_user", "wrong_password")

            # Vérifier que le compte EST verrouillé
            is_locked = check_lockout(db, "lockout_test_user")
            assert is_locked is True, (
                f"Le compte devrait être verrouillé après {n_attempts} échecs "
                f"(seuil = {MAX_FAILED_ATTEMPTS})"
            )

            # Vérifier que locked_until est défini
            db.refresh(user)
            assert user.locked_until is not None
        finally:
            db.close()

    @given(n_attempts=integers(min_value=1, max_value=10))
    @settings(max_examples=100, deadline=None)
    def test_lockout_threshold_boundary(self, n_attempts):
        """Vérifie la frontière exacte: verrouillé ssi n >= MAX_FAILED_ATTEMPTS."""
        db = _make_session()
        try:
            # Créer un utilisateur frais
            user = User(
                username="lockout_boundary_user",
                password_hash=hash_password("correct_password"),
                failed_attempts=0,
                locked_until=None,
            )
            db.add(user)
            db.commit()

            # Simuler n_attempts tentatives échouées
            for _ in range(n_attempts):
                authenticate(db, "lockout_boundary_user", "wrong_password")

            # Propriété: verrouillé ssi n_attempts >= MAX_FAILED_ATTEMPTS
            is_locked = check_lockout(db, "lockout_boundary_user")
            expected_locked = n_attempts >= MAX_FAILED_ATTEMPTS

            assert is_locked == expected_locked, (
                f"Après {n_attempts} échecs: verrouillé={is_locked}, "
                f"attendu={expected_locked} (seuil={MAX_FAILED_ATTEMPTS})"
            )
        finally:
            db.close()


# =============================================================================
# Property 2: Invalid credentials rejection
# Validates: Requirements 1.2
# =============================================================================


class TestInvalidCredentialsRejection:
    """
    Property 2: Invalid credentials rejection

    For any username/password combination that does not match a registered
    user in the database, the authenticate function should return None
    (no token issued).

    **Validates: Requirements 1.2**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        username=st.text(min_size=1, max_size=50),
        password=st.text(min_size=1, max_size=128),
    )
    def test_non_matching_credentials_return_none(self, username: str, password: str):
        """authenticate() retourne None pour tout couple (username, password)
        qui ne correspond pas au compte démo admin/admin123."""
        # Exclure les credentials valides du compte démo
        assume(not (username == "admin" and password == "admin123"))

        # Utiliser une session fraîche pour chaque exemple
        db = _make_session()
        try:
            # Seed du compte démo pour avoir un utilisateur enregistré
            seed_demo_account(db)

            # Toute autre combinaison doit retourner None
            result = authenticate(db, username, password)
            assert result is None, (
                f"authenticate() devrait retourner None pour username={username!r}, "
                f"password={password!r}, mais a retourné {result!r}"
            )
        finally:
            db.close()
