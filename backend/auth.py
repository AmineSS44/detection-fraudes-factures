"""
Module d'authentification JWT.
Gère la création/vérification de tokens, l'authentification
et le verrouillage de compte après tentatives échouées.
"""

from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.database import User

# Clé secrète pour signer les JWT (projet démo)
SECRET_KEY = "dev-secret-key-projet-stage-comptable-2024"
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24

# Seuils de verrouillage
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15

# Contexte de hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hache un mot de passe en bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe contre son hash bcrypt."""
    return pwd_context.verify(plain_password, hashed_password)


def create_token(username: str) -> str:
    """Génère un JWT token valide 24h avec claim d'expiration."""
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)
    payload = {
        "sub": username,
        "exp": expire,
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


def verify_token(token: str) -> dict:
    """Vérifie et décode un JWT token.

    Lève une exception si le token est invalide ou expiré.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise ValueError("Token invalide: claim 'sub' manquant")
        return payload
    except JWTError as e:
        raise ValueError(f"Token invalide ou expiré: {e}")


def check_lockout(db: Session, username: str) -> bool:
    """Vérifie si le compte est verrouillé (5 échecs → 15 min lock).

    Retourne True si le compte est actuellement verrouillé.
    """
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        return False

    if user.locked_until is not None:
        if datetime.utcnow() < user.locked_until:
            return True
        else:
            # Le verrouillage a expiré, on réinitialise
            user.locked_until = None
            user.failed_attempts = 0
            db.commit()
            return False

    return False


def _record_failed_attempt(db: Session, user: User) -> None:
    """Enregistre une tentative échouée et verrouille si nécessaire."""
    user.failed_attempts += 1
    if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
        user.locked_until = datetime.utcnow() + timedelta(
            minutes=LOCKOUT_DURATION_MINUTES
        )
    db.commit()


def _reset_failed_attempts(db: Session, user: User) -> None:
    """Réinitialise le compteur de tentatives après un login réussi."""
    user.failed_attempts = 0
    user.locked_until = None
    db.commit()


def authenticate(db: Session, username: str, password: str) -> Optional[str]:
    """Authentifie un utilisateur et retourne un token JWT ou None.

    Vérifie les identifiants, gère le verrouillage de compte,
    et retourne un token en cas de succès.
    """
    # Vérifier le verrouillage
    if check_lockout(db, username):
        return None

    # Chercher l'utilisateur
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        return None

    # Vérifier le mot de passe
    if not verify_password(password, user.password_hash):
        _record_failed_attempt(db, user)
        return None

    # Authentification réussie, réinitialiser les tentatives
    _reset_failed_attempts(db, user)
    return create_token(username)


def seed_demo_account(db: Session) -> None:
    """Crée le compte démo si absent (username='admin', password='admin123')."""
    existing = db.query(User).filter(User.username == "admin").first()
    if existing is None:
        demo_user = User(
            username="admin",
            password_hash=hash_password("admin123"),
        )
        db.add(demo_user)
        db.commit()
