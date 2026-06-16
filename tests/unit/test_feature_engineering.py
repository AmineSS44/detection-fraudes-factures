"""
Tests unitaires pour le module de feature engineering.
Vérifie le calcul des 6 features et la validation des entrées.
"""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base, Invoice
from ml.feature_engineering import FeatureEngine, FeatureVector


@pytest.fixture
def test_session():
    """Crée une session SQLite en mémoire pour les tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestSession()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def engine(test_session):
    """Crée une instance FeatureEngine avec la session de test."""
    return FeatureEngine(session=test_session)


class TestTaxInconsistency:
    """Tests pour _compute_tax_inconsistency."""

    def test_valid_tax_rates(self, engine):
        """Les taux marocains valides retournent False."""
        for rate in [7, 10, 14, 20]:
            assert engine._compute_tax_inconsistency(rate) is False

    def test_invalid_tax_rate(self, engine):
        """Un taux non standard retourne True."""
        assert engine._compute_tax_inconsistency(15) is True
        assert engine._compute_tax_inconsistency(0) is True
        assert engine._compute_tax_inconsistency(25) is True
        assert engine._compute_tax_inconsistency(5.5) is True


class TestWeekendFlag:
    """Tests pour _compute_weekend_flag."""

    def test_weekday_returns_false(self, engine):
        """Lundi à vendredi retournent False."""
        # 2024-01-15 est un lundi
        assert engine._compute_weekend_flag("2024-01-15") is False
        # 2024-01-19 est un vendredi
        assert engine._compute_weekend_flag("2024-01-19") is False

    def test_saturday_returns_true(self, engine):
        """Samedi retourne True."""
        # 2024-01-20 est un samedi
        assert engine._compute_weekend_flag("2024-01-20") is True

    def test_sunday_returns_true(self, engine):
        """Dimanche retourne True."""
        # 2024-01-21 est un dimanche
        assert engine._compute_weekend_flag("2024-01-21") is True


class TestRoundAmountFlag:
    """Tests pour _compute_round_amount_flag."""

    def test_round_above_10000(self, engine):
        """Multiple de 1000 et > 10000 retourne True."""
        assert engine._compute_round_amount_flag(11000) is True
        assert engine._compute_round_amount_flag(50000) is True
        assert engine._compute_round_amount_flag(100000) is True

    def test_round_at_10000(self, engine):
        """10000 exactement retourne False (doit être strictement > 10000)."""
        assert engine._compute_round_amount_flag(10000) is False

    def test_round_below_10000(self, engine):
        """Multiple de 1000 mais <= 10000 retourne False."""
        assert engine._compute_round_amount_flag(5000) is False
        assert engine._compute_round_amount_flag(1000) is False

    def test_not_round(self, engine):
        """Montant non multiple de 1000 retourne False."""
        assert engine._compute_round_amount_flag(15500) is False
        assert engine._compute_round_amount_flag(11001) is False


class TestAmountZscore:
    """Tests pour _compute_amount_zscore."""

    def test_insufficient_history(self, engine, test_session):
        """Moins de 5 factures historiques retourne 0.0."""
        # Ajouter seulement 3 factures
        today = datetime.now()
        for i in range(3):
            inv = Invoice(
                vendor_name="TestVendor",
                amount_ht=1000.0 + i * 100,
                date=(today - timedelta(days=30 * i)).strftime("%Y-%m-%d"),
                file_path="test.pdf",
                file_type="pdf",
            )
            test_session.add(inv)
        test_session.commit()

        result = engine._compute_amount_zscore(1500.0, "TestVendor")
        assert result == 0.0

    def test_sufficient_history(self, engine, test_session):
        """Avec >= 5 factures, retourne un z-score non nul."""
        today = datetime.now()
        amounts = [1000, 2000, 3000, 4000, 5000]
        for i, amt in enumerate(amounts):
            inv = Invoice(
                vendor_name="VendorA",
                amount_ht=float(amt),
                date=(today - timedelta(days=30 * i)).strftime("%Y-%m-%d"),
                file_path="test.pdf",
                file_type="pdf",
            )
            test_session.add(inv)
        test_session.commit()

        # Montant moyen = 3000, std calculée
        mean = 3000.0
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        std = variance**0.5

        result = engine._compute_amount_zscore(6000.0, "VendorA")
        expected = (6000.0 - mean) / std
        assert abs(result - expected) < 0.001

    def test_unknown_vendor(self, engine):
        """Fournisseur sans historique retourne 0.0."""
        result = engine._compute_amount_zscore(5000.0, "UnknownVendor")
        assert result == 0.0


class TestDuplicateFlag:
    """Tests pour _compute_duplicate_flag."""

    def test_no_duplicate(self, engine, test_session):
        """Pas de doublon retourne False."""
        result = engine._compute_duplicate_flag(
            5000.0, "VendorX", "2024-06-15"
        )
        assert result is False

    def test_duplicate_within_30_days(self, engine, test_session):
        """Même montant+fournisseur dans les 30 jours retourne True."""
        inv = Invoice(
            vendor_name="VendorY",
            amount_ht=7500.0,
            date="2024-06-01",
            file_path="test.pdf",
            file_type="pdf",
        )
        test_session.add(inv)
        test_session.commit()

        result = engine._compute_duplicate_flag(
            7500.0, "VendorY", "2024-06-15"
        )
        assert result is True

    def test_duplicate_beyond_30_days(self, engine, test_session):
        """Même montant+fournisseur au-delà de 30 jours retourne False."""
        inv = Invoice(
            vendor_name="VendorZ",
            amount_ht=3000.0,
            date="2024-01-01",
            file_path="test.pdf",
            file_type="pdf",
        )
        test_session.add(inv)
        test_session.commit()

        result = engine._compute_duplicate_flag(
            3000.0, "VendorZ", "2024-06-15"
        )
        assert result is False

    def test_different_amount_same_vendor(self, engine, test_session):
        """Même fournisseur mais montant différent retourne False."""
        inv = Invoice(
            vendor_name="VendorW",
            amount_ht=5000.0,
            date="2024-06-10",
            file_path="test.pdf",
            file_type="pdf",
        )
        test_session.add(inv)
        test_session.commit()

        result = engine._compute_duplicate_flag(
            6000.0, "VendorW", "2024-06-15"
        )
        assert result is False


class TestVendorDeviation:
    """Tests pour _compute_vendor_deviation."""

    def test_unknown_vendor(self, engine):
        """Fournisseur sans historique retourne 1.0."""
        result = engine._compute_vendor_deviation("NewVendor")
        assert result == 1.0

    def test_most_frequent_vendor(self, engine, test_session):
        """Le fournisseur le plus fréquent retourne 0.0."""
        today = datetime.now()
        # Ajouter 10 factures pour VendorA (le plus fréquent)
        for i in range(10):
            inv = Invoice(
                vendor_name="VendorA",
                amount_ht=1000.0,
                date=(today - timedelta(days=i * 10)).strftime("%Y-%m-%d"),
                file_path="test.pdf",
                file_type="pdf",
            )
            test_session.add(inv)
        test_session.commit()

        # VendorA est le seul, et a le max → deviation = 1 - 10/10 = 0.0
        result = engine._compute_vendor_deviation("VendorA")
        assert result == 0.0

    def test_rare_vendor(self, engine, test_session):
        """Un fournisseur rare a un score élevé."""
        today = datetime.now()
        # VendorA: 10 factures
        for i in range(10):
            inv = Invoice(
                vendor_name="VendorFrequent",
                amount_ht=1000.0,
                date=(today - timedelta(days=i * 10)).strftime("%Y-%m-%d"),
                file_path="test.pdf",
                file_type="pdf",
            )
            test_session.add(inv)
        # VendorRare: 2 factures
        for i in range(2):
            inv = Invoice(
                vendor_name="VendorRare",
                amount_ht=2000.0,
                date=(today - timedelta(days=i * 10)).strftime("%Y-%m-%d"),
                file_path="test.pdf",
                file_type="pdf",
            )
            test_session.add(inv)
        test_session.commit()

        result = engine._compute_vendor_deviation("VendorRare")
        # deviation = 1 - (2 / 10) = 0.8
        assert abs(result - 0.8) < 0.001


class TestComputeFeatures:
    """Tests pour compute_features (orchestration)."""

    def test_valid_input(self, engine):
        """Entrée valide retourne un FeatureVector complet."""
        data = {
            "amount": 5000.0,
            "vendor": "TestVendor",
            "date": "2024-01-15",
            "tax_rate": 20,
        }
        result = engine.compute_features(data)
        assert isinstance(result, FeatureVector)
        assert isinstance(result.amount_zscore, float)
        assert isinstance(result.tax_inconsistency, bool)
        assert isinstance(result.duplicate_flag, bool)
        assert isinstance(result.weekend_flag, bool)
        assert isinstance(result.round_amount_flag, bool)
        assert isinstance(result.vendor_deviation, float)

    def test_missing_amount(self, engine):
        """Champ amount manquant lève ValueError."""
        data = {
            "vendor": "TestVendor",
            "date": "2024-01-15",
            "tax_rate": 20,
        }
        with pytest.raises(ValueError, match="amount"):
            engine.compute_features(data)

    def test_missing_vendor(self, engine):
        """Champ vendor manquant lève ValueError."""
        data = {
            "amount": 5000.0,
            "date": "2024-01-15",
            "tax_rate": 20,
        }
        with pytest.raises(ValueError, match="vendor"):
            engine.compute_features(data)

    def test_missing_date(self, engine):
        """Champ date manquant lève ValueError."""
        data = {
            "amount": 5000.0,
            "vendor": "TestVendor",
            "tax_rate": 20,
        }
        with pytest.raises(ValueError, match="date"):
            engine.compute_features(data)

    def test_missing_tax_rate(self, engine):
        """Champ tax_rate manquant lève ValueError."""
        data = {
            "amount": 5000.0,
            "vendor": "TestVendor",
            "date": "2024-01-15",
        }
        with pytest.raises(ValueError, match="tax_rate"):
            engine.compute_features(data)

    def test_multiple_missing_fields(self, engine):
        """Plusieurs champs manquants sont listés dans l'erreur."""
        data = {"amount": 5000.0}
        with pytest.raises(ValueError) as exc_info:
            engine.compute_features(data)
        error_msg = str(exc_info.value)
        assert "vendor" in error_msg
        assert "date" in error_msg
        assert "tax_rate" in error_msg

    def test_none_field_treated_as_missing(self, engine):
        """Un champ avec valeur None est traité comme manquant."""
        data = {
            "amount": 5000.0,
            "vendor": None,
            "date": "2024-01-15",
            "tax_rate": 20,
        }
        with pytest.raises(ValueError, match="vendor"):
            engine.compute_features(data)
