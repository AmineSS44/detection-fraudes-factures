"""
Configuration de la base de données SQLAlchemy.
Fournit le moteur, la session factory et le modèle de base.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


# Chemin par défaut de la base SQLite
DATABASE_URL = "sqlite:///data/invoices.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Classe de base pour tous les modèles ORM."""

    pass


class User(Base):
    """Modèle utilisateur pour l'authentification."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    failed_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Invoice(Base):
    """Modèle facture avec résultats d'analyse de fraude."""

    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True)
    invoice_id = Column(String(50), nullable=True)
    vendor_name = Column(String(200), nullable=True)
    amount_ht = Column(Float, nullable=True)
    tax_rate = Column(Float, nullable=True)
    amount_ttc = Column(Float, nullable=True)
    date = Column(String(10), nullable=True)  # Format YYYY-MM-DD
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(10), nullable=False)
    fraud_score = Column(Float, nullable=True)
    fraud_label = Column(String(20), nullable=True)
    fraud_reason = Column(String(500), nullable=True)
    label = Column(String(20), nullable=True)  # Label pour le dataset d'entraînement
    fraud_type = Column(String(200), nullable=True)
    ocr_confidences = Column(JSON, nullable=True)
    feature_vector = Column(JSON, nullable=True)
    analyzed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)


def get_db():
    """Générateur de session DB pour l'injection de dépendances FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Crée toutes les tables dans la base de données."""
    Base.metadata.create_all(bind=engine)
