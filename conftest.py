"""
Configuration pytest partagée avec fixtures pour les tests.
Fournit une session SQLite en mémoire et un client de test FastAPI.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="function")
def db_engine():
    """Crée un moteur SQLite en mémoire pour les tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Fournit une session de base de données pour les tests.
    
    Crée les tables au début et fait un rollback à la fin.
    """
    # Import différé pour éviter les imports circulaires
    from backend.database import Base

    Base.metadata.create_all(bind=db_engine)
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=db_engine
    )
    session = TestingSessionLocal()

    yield session

    session.rollback()
    session.close()
    Base.metadata.drop_all(bind=db_engine)


@pytest.fixture(scope="function")
def test_client(db_session):
    """Fournit un client de test FastAPI avec la DB en mémoire injectée."""
    from fastapi.testclient import TestClient
    from backend.api import app
    from backend.database import get_db

    def override_get_db():
        """Surcharge la dépendance DB pour utiliser la session de test."""
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
