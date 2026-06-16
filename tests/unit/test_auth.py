"""
Tests unitaires pour le module d'authentification JWT.
Couvre: création/vérification de tokens, authentification,
verrouillage de compte et seed du compte démo.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from jose import jwt

from backend.auth import (
    ALGORITHM,
    LOCKOUT_DURATION_MINUTES,
    MAX_FAILED_ATTEMPTS,
    SECRET_KEY,
    TOKEN_EXPIRY_HOURS,
    authenticate,
    check_lockout,
    create_token,
    hash_password,
    seed_demo_account,
    verify_password,
    verify_token,
)
from backend.database import User


class TestHashPassword:
    """Tests pour le hachage de mot de passe."""

    def test_hash_returns_bcrypt_string(self):
        """Le hash retourné commence par le préfixe bcrypt."""
        hashed = hash_password("test123")
        assert hashed.startswith("$2b$")

    def test_different_passwords_produce_different_hashes(self):
        """Deux mots de passe différents produisent des hashes différents."""
        h1 = hash_password("password1")
        h2 = hash_password("password2")
        assert h1 != h2


class TestVerifyPassword:
    """Tests pour la vérification de mot de passe."""

    def test_correct_password_returns_true(self):
        """Un mot de passe correct retourne True."""
        hashed = hash_password("secret")
        assert verify_password("secret", hashed) is True

    def test_incorrect_password_returns_false(self):
        """Un mot de passe incorrect retourne False."""
        hashed = hash_password("secret")
        assert verify_password("wrong", hashed) is False


class TestCreateToken:
    """Tests pour la création de JWT."""

    def test_token_contains_username_claim(self):
        """Le token contient le claim 'sub' avec le username."""
        token = create_token("admin")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "admin"

    def test_token_has_expiry_claim(self):
        """Le token contient un claim 'exp'."""
        token = create_token("admin")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "exp" in payload

    def test_token_expiry_is_24_hours(self):
        """Le token expire dans environ 24 heures."""
        before = datetime.utcnow()
        token = create_token("admin")
        after = datetime.utcnow()

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp = datetime.utcfromtimestamp(payload["exp"])

        # L'expiration doit être entre 24h avant et 24h après (avec marge)
        expected_min = before + timedelta(hours=TOKEN_EXPIRY_HOURS) - timedelta(seconds=5)
        expected_max = after + timedelta(hours=TOKEN_EXPIRY_HOURS) + timedelta(seconds=5)
        assert expected_min <= exp <= expected_max


class TestVerifyToken:
    """Tests pour la vérification de JWT."""

    def test_valid_token_returns_payload(self):
        """Un token valide retourne le payload décodé."""
        token = create_token("testuser")
        payload = verify_token(token)
        assert payload["sub"] == "testuser"

    def test_expired_token_raises(self):
        """Un token expiré lève une exception."""
        # Créer un token avec une date d'expiration passée
        expire = datetime.utcnow() - timedelta(hours=1)
        payload = {"sub": "testuser", "exp": expire}
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

        with pytest.raises(ValueError, match="invalide ou expiré"):
            verify_token(token)

    def test_invalid_token_raises(self):
        """Un token invalide lève une exception."""
        with pytest.raises(ValueError):
            verify_token("token.invalide.ici")

    def test_token_without_sub_raises(self):
        """Un token sans claim 'sub' lève une exception."""
        payload = {"exp": datetime.utcnow() + timedelta(hours=1)}
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

        with pytest.raises(ValueError, match="claim 'sub' manquant"):
            verify_token(token)


class TestCheckLockout:
    """Tests pour la vérification du verrouillage de compte."""

    def test_nonexistent_user_not_locked(self, db_session):
        """Un utilisateur inexistant n'est pas verrouillé."""
        assert check_lockout(db_session, "nobody") is False

    def test_user_without_lockout_not_locked(self, db_session):
        """Un utilisateur sans verrouillage n'est pas verrouillé."""
        user = User(
            username="testuser",
            password_hash=hash_password("pass"),
            failed_attempts=0,
        )
        db_session.add(user)
        db_session.commit()

        assert check_lockout(db_session, "testuser") is False

    def test_user_with_active_lockout_is_locked(self, db_session):
        """Un utilisateur avec verrouillage actif est verrouillé."""
        user = User(
            username="locked_user",
            password_hash=hash_password("pass"),
            failed_attempts=5,
            locked_until=datetime.utcnow() + timedelta(minutes=10),
        )
        db_session.add(user)
        db_session.commit()

        assert check_lockout(db_session, "locked_user") is True

    def test_expired_lockout_resets(self, db_session):
        """Un verrouillage expiré est réinitialisé."""
        user = User(
            username="expired_lock",
            password_hash=hash_password("pass"),
            failed_attempts=5,
            locked_until=datetime.utcnow() - timedelta(minutes=1),
        )
        db_session.add(user)
        db_session.commit()

        assert check_lockout(db_session, "expired_lock") is False
        # Vérifier que les compteurs sont réinitialisés
        db_session.refresh(user)
        assert user.failed_attempts == 0
        assert user.locked_until is None


class TestAuthenticate:
    """Tests pour l'authentification complète."""

    def test_valid_credentials_return_token(self, db_session):
        """Des identifiants valides retournent un token JWT."""
        user = User(
            username="admin",
            password_hash=hash_password("admin123"),
        )
        db_session.add(user)
        db_session.commit()

        token = authenticate(db_session, "admin", "admin123")
        assert token is not None
        # Vérifier que le token est valide
        payload = verify_token(token)
        assert payload["sub"] == "admin"

    def test_invalid_password_returns_none(self, db_session):
        """Un mot de passe incorrect retourne None."""
        user = User(
            username="admin",
            password_hash=hash_password("admin123"),
        )
        db_session.add(user)
        db_session.commit()

        result = authenticate(db_session, "admin", "wrong_password")
        assert result is None

    def test_nonexistent_user_returns_none(self, db_session):
        """Un utilisateur inexistant retourne None."""
        result = authenticate(db_session, "nobody", "any_password")
        assert result is None

    def test_failed_attempts_increment(self, db_session):
        """Les tentatives échouées sont comptées."""
        user = User(
            username="admin",
            password_hash=hash_password("admin123"),
        )
        db_session.add(user)
        db_session.commit()

        authenticate(db_session, "admin", "wrong")
        db_session.refresh(user)
        assert user.failed_attempts == 1

        authenticate(db_session, "admin", "wrong")
        db_session.refresh(user)
        assert user.failed_attempts == 2

    def test_account_locks_after_5_failures(self, db_session):
        """Le compte se verrouille après 5 échecs consécutifs."""
        user = User(
            username="admin",
            password_hash=hash_password("admin123"),
        )
        db_session.add(user)
        db_session.commit()

        for _ in range(MAX_FAILED_ATTEMPTS):
            authenticate(db_session, "admin", "wrong")

        db_session.refresh(user)
        assert user.failed_attempts == MAX_FAILED_ATTEMPTS
        assert user.locked_until is not None

    def test_locked_account_rejects_valid_credentials(self, db_session):
        """Un compte verrouillé rejette même les bons identifiants."""
        user = User(
            username="admin",
            password_hash=hash_password("admin123"),
            failed_attempts=5,
            locked_until=datetime.utcnow() + timedelta(minutes=10),
        )
        db_session.add(user)
        db_session.commit()

        result = authenticate(db_session, "admin", "admin123")
        assert result is None

    def test_successful_login_resets_failed_attempts(self, db_session):
        """Un login réussi réinitialise le compteur de tentatives."""
        user = User(
            username="admin",
            password_hash=hash_password("admin123"),
            failed_attempts=3,
        )
        db_session.add(user)
        db_session.commit()

        token = authenticate(db_session, "admin", "admin123")
        assert token is not None

        db_session.refresh(user)
        assert user.failed_attempts == 0


class TestSeedDemoAccount:
    """Tests pour le seed du compte démo."""

    def test_creates_demo_account(self, db_session):
        """Le seed crée le compte admin si absent."""
        seed_demo_account(db_session)

        user = db_session.query(User).filter(User.username == "admin").first()
        assert user is not None
        assert user.username == "admin"
        assert verify_password("admin123", user.password_hash)

    def test_does_not_duplicate_account(self, db_session):
        """Le seed ne duplique pas le compte s'il existe déjà."""
        seed_demo_account(db_session)
        seed_demo_account(db_session)

        count = db_session.query(User).filter(User.username == "admin").count()
        assert count == 1
