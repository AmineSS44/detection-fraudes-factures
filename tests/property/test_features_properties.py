# Feature: invoice-fraud-detection, Property 8: amount_zscore correctness
# Feature: invoice-fraud-detection, Property 12: round_amount_flag correctness
# Feature: invoice-fraud-detection, Property 13: vendor_deviation correctness
# Feature: invoice-fraud-detection, Property 10: duplicate_flag correctness
"""
Tests de propriétés pour le module de feature engineering.
Vérifie les invariants via Hypothesis:
- Property 8: amount_zscore correctness (z-score vs historique fournisseur)
- Property 10: duplicate_flag correctness (même vendor+montant dans 30 jours)
- Property 12: round_amount_flag correctness (multiple de 1000 ET > 10 000)
- Property 13: vendor_deviation correctness (score de rareté 0.0-1.0)
"""

from datetime import datetime, timedelta

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base, Invoice
from ml.feature_engineering import FeatureEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# Instancier le moteur sans session DB (pas besoin pour round_amount_flag)
_engine_no_db = FeatureEngine(session=None)


# ---------------------------------------------------------------------------
# Stratégies Hypothesis
# ---------------------------------------------------------------------------

# Noms de fournisseurs réalistes
vendor_names = st.sampled_from([
    "Maroc Telecom",
    "Atlas BTP",
    "Souss Agro",
    "Sahara Logistics",
    "Fès Textile",
    "Casablanca Import",
    "Rabat Services",
])

# Montants HT en MAD (positifs, avec 2 décimales)
amounts = st.floats(
    min_value=0.01, max_value=999_999_999.99,
    allow_nan=False, allow_infinity=False,
).map(lambda x: round(x, 2))

# Dates dans une fenêtre raisonnable
dates = st.dates(
    min_value=datetime(2023, 1, 1).date(),
    max_value=datetime(2025, 6, 30).date(),
)


# =============================================================================
# Property 12: round_amount_flag correctness
# Validates: Requirements 5.6
# =============================================================================


class TestRoundAmountFlagCorrectness:
    """Property 12: round_amount_flag correctness.

    **Validates: Requirements 5.6**

    Pour tout montant, round_amount_flag doit être True ssi le montant est
    un multiple de 1000 ET strictement supérieur à 10 000.
    """

    @given(amount=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_round_amount_flag_floats(self, amount: float):
        """Vérifie la propriété sur des floats arbitraires."""
        result = _engine_no_db._compute_round_amount_flag(amount)
        expected = (amount % 1000 == 0) and (amount > 10000)
        assert result == expected, (
            f"Pour amount={amount}: attendu {expected}, obtenu {result}"
        )

    @given(amount=st.integers(min_value=-1_000_000, max_value=1_000_000))
    @settings(max_examples=100)
    def test_round_amount_flag_integers(self, amount: int):
        """Vérifie la propriété sur des entiers arbitraires."""
        result = _engine_no_db._compute_round_amount_flag(float(amount))
        expected = (amount % 1000 == 0) and (amount > 10000)
        assert result == expected, (
            f"Pour amount={amount}: attendu {expected}, obtenu {result}"
        )

    @given(k=st.integers(min_value=11, max_value=1000))
    @settings(max_examples=100)
    def test_multiples_of_1000_above_10000_are_true(self, k: int):
        """Tout multiple de 1000 strictement > 10 000 doit retourner True."""
        amount = float(k * 1000)
        result = _engine_no_db._compute_round_amount_flag(amount)
        assert result is True, (
            f"Pour amount={amount} (k={k}): attendu True, obtenu {result}"
        )

    @given(k=st.integers(min_value=-100, max_value=10))
    @settings(max_examples=100)
    def test_multiples_of_1000_at_or_below_10000_are_false(self, k: int):
        """Tout multiple de 1000 avec valeur ≤ 10 000 doit retourner False."""
        amount = float(k * 1000)
        result = _engine_no_db._compute_round_amount_flag(amount)
        assert result is False, (
            f"Pour amount={amount} (k={k}): attendu False, obtenu {result}"
        )

    @given(
        k=st.integers(min_value=1, max_value=1000),
        offset=st.integers(min_value=1, max_value=999)
    )
    @settings(max_examples=100)
    def test_non_multiples_of_1000_are_always_false(self, k: int, offset: int):
        """Un montant qui n'est pas multiple de 1000 doit toujours retourner False."""
        amount = float(k * 1000 + offset)
        result = _engine_no_db._compute_round_amount_flag(amount)
        assert result is False, (
            f"Pour amount={amount}: attendu False (pas multiple de 1000), obtenu {result}"
        )


# =============================================================================
# Property 13: vendor_deviation correctness
# Validates: Requirements 5.7
# =============================================================================


class TestVendorDeviationCorrectness:
    """
    Property 13: vendor_deviation correctness

    For any vendor, vendor_deviation should equal 1.0 if the vendor has no prior
    invoices. Otherwise 1 - (vendor_invoice_count / max_vendor_invoice_count)
    over last 12 months. Result always in [0.0, 1.0].

    **Validates: Requirements 5.7**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        vendor_name=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
            min_size=1,
            max_size=50,
        )
    )
    def test_unknown_vendor_returns_one(self, vendor_name: str):
        """
        Quand le fournisseur n'a aucune facture antérieure dans la base,
        vendor_deviation doit être exactement 1.0.

        **Validates: Requirements 5.7**
        """
        assume(vendor_name.strip() != "")

        session = _make_session()
        try:
            engine = FeatureEngine(session=session)
            result = engine._compute_vendor_deviation(vendor_name)

            assert result == 1.0, (
                f"vendor_deviation devrait être 1.0 pour un fournisseur inconnu, "
                f"mais obtenu={result} pour vendor='{vendor_name}'"
            )
        finally:
            session.close()

    @settings(max_examples=100, deadline=None)
    @given(
        invoice_count=st.integers(min_value=1, max_value=20),
    )
    def test_vendor_with_highest_count_returns_zero(self, invoice_count: int):
        """
        Quand le fournisseur a le nombre max de factures parmi tous les
        fournisseurs (et est le seul fournisseur), vendor_deviation doit être 0.0.

        **Validates: Requirements 5.7**
        """
        session = _make_session()
        try:
            vendor = "TopVendor"
            base_date = datetime.now() - timedelta(days=30)
            for i in range(invoice_count):
                inv = Invoice(
                    invoice_id=f"INV-TOP-{i:04d}",
                    vendor_name=vendor,
                    amount_ht=1000.0 + i,
                    tax_rate=20.0,
                    amount_ttc=1200.0 + i,
                    date=(base_date - timedelta(days=i)).strftime("%Y-%m-%d"),
                    file_path=f"/tmp/top_{i}.pdf",
                    file_type="pdf",
                )
                session.add(inv)
            session.commit()

            engine = FeatureEngine(session=session)
            result = engine._compute_vendor_deviation(vendor)

            assert result == 0.0, (
                f"vendor_deviation devrait être 0.0 pour le fournisseur le plus fréquent, "
                f"mais obtenu={result} (invoice_count={invoice_count})"
            )
        finally:
            session.close()

    @settings(max_examples=100, deadline=None)
    @given(
        vendor_count=st.integers(min_value=1, max_value=10),
        max_vendor_count=st.integers(min_value=1, max_value=20),
    )
    def test_intermediate_deviation_values_correct(
        self, vendor_count: int, max_vendor_count: int
    ):
        """
        Pour un fournisseur intermédiaire, vendor_deviation doit être
        1 - (vendor_count / max_vendor_count), et toujours dans [0.0, 1.0].

        **Validates: Requirements 5.7**
        """
        assume(max_vendor_count >= vendor_count)

        session = _make_session()
        try:
            target_vendor = "TargetVendor"
            top_vendor = "TopVendor"
            base_date = datetime.now() - timedelta(days=30)

            for i in range(vendor_count):
                inv = Invoice(
                    invoice_id=f"INV-TGT-{i:04d}",
                    vendor_name=target_vendor,
                    amount_ht=2000.0 + i,
                    tax_rate=20.0,
                    amount_ttc=2400.0 + i,
                    date=(base_date - timedelta(days=i)).strftime("%Y-%m-%d"),
                    file_path=f"/tmp/target_{i}.pdf",
                    file_type="pdf",
                )
                session.add(inv)

            for i in range(max_vendor_count):
                inv = Invoice(
                    invoice_id=f"INV-MAX-{i:04d}",
                    vendor_name=top_vendor,
                    amount_ht=3000.0 + i,
                    tax_rate=20.0,
                    amount_ttc=3600.0 + i,
                    date=(base_date - timedelta(days=i)).strftime("%Y-%m-%d"),
                    file_path=f"/tmp/max_{i}.pdf",
                    file_type="pdf",
                )
                session.add(inv)
            session.commit()

            engine = FeatureEngine(session=session)
            result = engine._compute_vendor_deviation(target_vendor)

            expected = 1.0 - (vendor_count / max_vendor_count)

            assert abs(result - expected) < 1e-9, (
                f"vendor_deviation incorrect: obtenu={result}, attendu={expected}, "
                f"vendor_count={vendor_count}, max_vendor_count={max_vendor_count}"
            )

            assert 0.0 <= result <= 1.0, (
                f"vendor_deviation hors bornes [0.0, 1.0]: obtenu={result}"
            )
        finally:
            session.close()

    @settings(max_examples=100, deadline=None)
    @given(
        vendor_count=st.integers(min_value=1, max_value=15),
        other_vendor_count=st.integers(min_value=1, max_value=15),
    )
    def test_result_always_in_bounds(
        self, vendor_count: int, other_vendor_count: int
    ):
        """
        Pour n'importe quelle configuration de fournisseurs,
        vendor_deviation doit toujours être dans [0.0, 1.0].

        **Validates: Requirements 5.7**
        """
        session = _make_session()
        try:
            target_vendor = "VendorA"
            other_vendor = "VendorB"
            base_date = datetime.now() - timedelta(days=30)

            for i in range(vendor_count):
                inv = Invoice(
                    invoice_id=f"INV-A-{i:04d}",
                    vendor_name=target_vendor,
                    amount_ht=1500.0 + i,
                    tax_rate=14.0,
                    amount_ttc=1710.0 + i,
                    date=(base_date - timedelta(days=i)).strftime("%Y-%m-%d"),
                    file_path=f"/tmp/a_{i}.pdf",
                    file_type="pdf",
                )
                session.add(inv)

            for i in range(other_vendor_count):
                inv = Invoice(
                    invoice_id=f"INV-B-{i:04d}",
                    vendor_name=other_vendor,
                    amount_ht=2500.0 + i,
                    tax_rate=10.0,
                    amount_ttc=2750.0 + i,
                    date=(base_date - timedelta(days=i)).strftime("%Y-%m-%d"),
                    file_path=f"/tmp/b_{i}.pdf",
                    file_type="pdf",
                )
                session.add(inv)
            session.commit()

            engine = FeatureEngine(session=session)
            result = engine._compute_vendor_deviation(target_vendor)

            assert 0.0 <= result <= 1.0, (
                f"vendor_deviation hors bornes: obtenu={result}, "
                f"vendor_count={vendor_count}, other_vendor_count={other_vendor_count}"
            )
        finally:
            session.close()


# =============================================================================
# Property 10: duplicate_flag correctness
# Validates: Requirements 5.4
# =============================================================================


class TestDuplicateFlagCorrectness:
    """
    Property 10: duplicate_flag correctness

    For any invoice, duplicate_flag should be True if and only if there exists
    another invoice with the same vendor_name and exact same amount within the
    preceding 30 calendar days.

    **Validates: Requirements 5.4**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        vendor=vendor_names,
        amount=amounts,
        invoice_date=dates,
        prior_offset_days=st.integers(min_value=1, max_value=29),
    )
    def test_duplicate_flag_true_when_match_exists_within_30_days(
        self, vendor, amount, invoice_date, prior_offset_days
    ):
        """
        Quand une facture antérieure avec même vendor et même montant existe
        dans les 30 jours précédents, duplicate_flag doit être True.

        **Validates: Requirements 5.4**
        """
        session = _make_session()
        try:
            # Insérer une facture antérieure dans la fenêtre de 30 jours
            prior_date = invoice_date - timedelta(days=prior_offset_days)
            prior_invoice = Invoice(
                vendor_name=vendor,
                amount_ht=amount,
                date=prior_date.strftime("%Y-%m-%d"),
                file_path="test.pdf",
                file_type="pdf",
            )
            session.add(prior_invoice)
            session.commit()

            # Calculer le flag
            engine = FeatureEngine(session=session)
            result = engine._compute_duplicate_flag(
                amount, vendor, invoice_date.strftime("%Y-%m-%d")
            )

            assert result is True, (
                f"duplicate_flag devrait être True: même vendor={vendor!r}, "
                f"même amount={amount}, écart={prior_offset_days} jours (<30)"
            )
        finally:
            session.close()

    @settings(max_examples=100, deadline=None)
    @given(
        vendor=vendor_names,
        amount=amounts,
        invoice_date=dates,
    )
    def test_duplicate_flag_false_when_no_prior_invoices(
        self, vendor, amount, invoice_date
    ):
        """
        Quand aucune facture antérieure n'existe dans la DB,
        duplicate_flag doit être False.

        **Validates: Requirements 5.4**
        """
        session = _make_session()
        try:
            # DB vide — pas de factures antérieures
            engine = FeatureEngine(session=session)
            result = engine._compute_duplicate_flag(
                amount, vendor, invoice_date.strftime("%Y-%m-%d")
            )

            assert result is False, (
                "duplicate_flag devrait être False quand la DB est vide"
            )
        finally:
            session.close()

    @settings(max_examples=100, deadline=None)
    @given(
        vendor=vendor_names,
        amount=amounts,
        invoice_date=dates,
        prior_offset_days=st.integers(min_value=31, max_value=365),
    )
    def test_duplicate_flag_false_when_match_outside_30_days(
        self, vendor, amount, invoice_date, prior_offset_days
    ):
        """
        Quand une facture avec même vendor+montant existe mais au-delà
        de 30 jours avant, duplicate_flag doit être False.

        **Validates: Requirements 5.4**
        """
        session = _make_session()
        try:
            # Insérer une facture trop ancienne (> 30 jours avant)
            prior_date = invoice_date - timedelta(days=prior_offset_days)
            prior_invoice = Invoice(
                vendor_name=vendor,
                amount_ht=amount,
                date=prior_date.strftime("%Y-%m-%d"),
                file_path="test.pdf",
                file_type="pdf",
            )
            session.add(prior_invoice)
            session.commit()

            engine = FeatureEngine(session=session)
            result = engine._compute_duplicate_flag(
                amount, vendor, invoice_date.strftime("%Y-%m-%d")
            )

            assert result is False, (
                f"duplicate_flag devrait être False: écart={prior_offset_days} jours (>30)"
            )
        finally:
            session.close()

    @settings(max_examples=100, deadline=None)
    @given(
        vendor=vendor_names,
        amount=amounts,
        different_amount=amounts,
        invoice_date=dates,
        prior_offset_days=st.integers(min_value=1, max_value=29),
    )
    def test_duplicate_flag_false_when_different_amount(
        self, vendor, amount, different_amount, invoice_date, prior_offset_days
    ):
        """
        Quand une facture avec même vendor mais montant différent existe
        dans les 30 jours, duplicate_flag doit être False.

        **Validates: Requirements 5.4**
        """
        # S'assurer que les montants sont différents
        assume(amount != different_amount)

        session = _make_session()
        try:
            prior_date = invoice_date - timedelta(days=prior_offset_days)
            prior_invoice = Invoice(
                vendor_name=vendor,
                amount_ht=different_amount,  # montant différent
                date=prior_date.strftime("%Y-%m-%d"),
                file_path="test.pdf",
                file_type="pdf",
            )
            session.add(prior_invoice)
            session.commit()

            engine = FeatureEngine(session=session)
            result = engine._compute_duplicate_flag(
                amount, vendor, invoice_date.strftime("%Y-%m-%d")
            )

            assert result is False, (
                f"duplicate_flag devrait être False: montants différents "
                f"({amount} vs {different_amount})"
            )
        finally:
            session.close()

    @settings(max_examples=100, deadline=None)
    @given(
        vendor=vendor_names,
        different_vendor=vendor_names,
        amount=amounts,
        invoice_date=dates,
        prior_offset_days=st.integers(min_value=1, max_value=29),
    )
    def test_duplicate_flag_false_when_different_vendor(
        self, vendor, different_vendor, amount, invoice_date, prior_offset_days
    ):
        """
        Quand une facture avec même montant mais vendor différent existe
        dans les 30 jours, duplicate_flag doit être False.

        **Validates: Requirements 5.4**
        """
        # S'assurer que les vendors sont différents
        assume(vendor != different_vendor)

        session = _make_session()
        try:
            prior_date = invoice_date - timedelta(days=prior_offset_days)
            prior_invoice = Invoice(
                vendor_name=different_vendor,  # vendor différent
                amount_ht=amount,
                date=prior_date.strftime("%Y-%m-%d"),
                file_path="test.pdf",
                file_type="pdf",
            )
            session.add(prior_invoice)
            session.commit()

            engine = FeatureEngine(session=session)
            result = engine._compute_duplicate_flag(
                amount, vendor, invoice_date.strftime("%Y-%m-%d")
            )

            assert result is False, (
                f"duplicate_flag devrait être False: vendors différents "
                f"({vendor!r} vs {different_vendor!r})"
            )
        finally:
            session.close()

    @settings(max_examples=100, deadline=None)
    @given(
        vendor=vendor_names,
        amount=amounts,
        invoice_date=dates,
    )
    def test_duplicate_flag_boundary_exactly_30_days(
        self, vendor, amount, invoice_date
    ):
        """
        Quand une facture existe exactement à la frontière de 30 jours,
        vérifie le comportement de la borne. L'implémentation utilise:
        Invoice.date >= thirty_days_ago AND Invoice.date < date,
        où thirty_days_ago = invoice_date - 30 jours.
        Donc une facture datée exactement à thirty_days_ago est INCLUSE (>=).

        **Validates: Requirements 5.4**
        """
        session = _make_session()
        try:
            # Facture exactement à la frontière de 30 jours
            prior_date = invoice_date - timedelta(days=30)
            prior_invoice = Invoice(
                vendor_name=vendor,
                amount_ht=amount,
                date=prior_date.strftime("%Y-%m-%d"),
                file_path="test.pdf",
                file_type="pdf",
            )
            session.add(prior_invoice)
            session.commit()

            engine = FeatureEngine(session=session)
            result = engine._compute_duplicate_flag(
                amount, vendor, invoice_date.strftime("%Y-%m-%d")
            )

            # L'implémentation: Invoice.date >= thirty_days_ago AND Invoice.date < date
            # thirty_days_ago = (invoice_date - 30 days).strftime(...)
            # prior_date == thirty_days_ago → condition >= est satisfaite → inclus
            assert result is True, (
                f"duplicate_flag devrait être True: facture à exactement 30 jours "
                f"est incluse dans la fenêtre (>=)"
            )
        finally:
            session.close()


# Feature: invoice-fraud-detection, Property 14: Feature vector missing field rejection
class TestMissingFieldRejectionProperty:
    """
    **Validates: Requirements 5.9**

    Property: For any invoice data missing one or more required fields from
    {amount, vendor, date, tax_rate}, the Feature_Engine should reject the input
    and return an error that specifies exactly which fields are missing.
    """

    REQUIRED_FIELDS = ["amount", "vendor", "date", "tax_rate"]

    @given(
        fields_to_remove=st.lists(
            st.sampled_from(["amount", "vendor", "date", "tax_rate"]),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    @settings(max_examples=100)
    def test_missing_fields_raises_valueerror(self, fields_to_remove: list):
        """Retirer un sous-ensemble de champs requis doit lever ValueError."""
        # Construire un invoice_data complet puis retirer les champs sélectionnés
        complete_data = {
            "amount": 5000.0,
            "vendor": "Atlas BTP",
            "date": "2024-06-15",
            "tax_rate": 20.0,
        }

        invoice_data = {
            k: v for k, v in complete_data.items() if k not in fields_to_remove
        }

        engine = FeatureEngine(session=None)

        try:
            engine.compute_features(invoice_data)
            assert False, (
                f"compute_features aurait dû lever ValueError pour champs manquants: "
                f"{fields_to_remove}"
            )
        except ValueError as e:
            error_message = str(e)
            # Vérifier que chaque champ manquant est mentionné dans l'erreur
            for field in fields_to_remove:
                assert field in error_message, (
                    f"Le champ manquant '{field}' n'est pas mentionné dans "
                    f"l'erreur: '{error_message}'"
                )

    @given(
        fields_to_remove=st.lists(
            st.sampled_from(["amount", "vendor", "date", "tax_rate"]),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    @settings(max_examples=100)
    def test_missing_fields_via_none_values(self, fields_to_remove: list):
        """Champs présents mais à None doivent aussi être rejetés."""
        complete_data = {
            "amount": 5000.0,
            "vendor": "Atlas BTP",
            "date": "2024-06-15",
            "tax_rate": 20.0,
        }

        # Mettre les champs sélectionnés à None au lieu de les supprimer
        invoice_data = dict(complete_data)
        for field in fields_to_remove:
            invoice_data[field] = None

        engine = FeatureEngine(session=None)

        try:
            engine.compute_features(invoice_data)
            assert False, (
                f"compute_features aurait dû lever ValueError pour champs None: "
                f"{fields_to_remove}"
            )
        except ValueError as e:
            error_message = str(e)
            # Vérifier que chaque champ None est mentionné dans l'erreur
            for field in fields_to_remove:
                assert field in error_message, (
                    f"Le champ None '{field}' n'est pas mentionné dans "
                    f"l'erreur: '{error_message}'"
                )


# =============================================================================
# Feature: invoice-fraud-detection, Property 11: weekend_flag correctness
# Validates: Requirements 5.5
# =============================================================================


class TestWeekendFlagCorrectness:
    """
    Property 11: weekend_flag correctness

    For any date, weekend_flag should be True if and only if the date falls
    on a Saturday (weekday=5) or Sunday (weekday=6).

    **Validates: Requirements 5.5**
    """

    @settings(max_examples=100, deadline=None)
    @given(date=st.dates())
    def test_weekend_flag_matches_weekday(self, date):
        """
        Pour toute date générée aléatoirement, weekend_flag doit être True
        ssi la date tombe un samedi (weekday()=5) ou dimanche (weekday()=6).

        **Validates: Requirements 5.5**
        """
        date_str = date.strftime("%Y-%m-%d")

        engine = FeatureEngine(session=None)
        result = engine._compute_weekend_flag(date_str)

        expected = date.weekday() >= 5

        assert result == expected, (
            f"weekend_flag incorrect pour {date_str}: "
            f"obtenu={result}, attendu={expected}, weekday={date.weekday()}"
        )



# =============================================================================
# Property 8: amount_zscore correctness
# Validates: Requirements 5.1, 5.2
# =============================================================================


def _populate_vendor_history(session, vendor: str, amounts: list):
    """Insère des factures historiques pour un fournisseur dans la session.

    Les factures sont datées dans les 6 derniers mois (dans la fenêtre de 12 mois).
    """
    base_date = datetime.now() - timedelta(days=30)
    for i, amount in enumerate(amounts):
        invoice = Invoice(
            invoice_id=f"INV-ZSCORE-{i:04d}",
            vendor_name=vendor,
            amount_ht=amount,
            tax_rate=20.0,
            amount_ttc=amount * 1.2,
            date=(base_date - timedelta(days=i)).strftime("%Y-%m-%d"),
            file_path=f"/tmp/zscore_test_{i}.pdf",
            file_type="pdf",
        )
        session.add(invoice)
    session.commit()


class TestAmountZscoreCorrectness:
    """
    Property 8: amount_zscore correctness

    For any invoice where vendor has ≥5 historical invoices in last 12 months,
    amount_zscore = (amount - mean) / std. For fewer than 5, zscore = 0.0.

    **Validates: Requirements 5.1, 5.2**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        amount=st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
        historical_amounts=st.lists(
            st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=10,
        ),
    )
    def test_zscore_computed_correctly_with_sufficient_history(
        self, amount: float, historical_amounts: list
    ):
        """
        Quand le fournisseur a ≥5 factures historiques sur 12 mois,
        le z-score doit être (amount - mean) / std.

        **Validates: Requirements 5.1**
        """
        # Calculer mean et std attendus
        mean = sum(historical_amounts) / len(historical_amounts)
        variance = sum((x - mean) ** 2 for x in historical_amounts) / len(historical_amounts)
        std = variance ** 0.5
        # Éviter les cas dégénérés où std=0 (tous montants identiques)
        assume(std > 1e-9)

        session = _make_session()
        try:
            vendor = "ZscoreTestVendor"
            _populate_vendor_history(session, vendor, historical_amounts)

            fe = FeatureEngine(session=session)
            result = fe._compute_amount_zscore(amount, vendor)

            # Calcul attendu
            expected_zscore = (amount - mean) / std

            assert abs(result - expected_zscore) < 1e-6, (
                f"Z-score incorrect: obtenu={result}, attendu={expected_zscore}, "
                f"amount={amount}, mean={mean}, std={std}"
            )
        finally:
            session.close()

    @settings(max_examples=100, deadline=None)
    @given(
        amount=st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
        historical_amounts=st.lists(
            st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
            min_size=0,
            max_size=4,
        ),
    )
    def test_zscore_is_zero_with_insufficient_history(
        self, amount: float, historical_amounts: list
    ):
        """
        Quand le fournisseur a < 5 factures historiques sur 12 mois,
        le z-score doit être exactement 0.0.

        **Validates: Requirements 5.2**
        """
        session = _make_session()
        try:
            vendor = "ZscoreTestVendor"
            _populate_vendor_history(session, vendor, historical_amounts)

            fe = FeatureEngine(session=session)
            result = fe._compute_amount_zscore(amount, vendor)

            assert result == 0.0, (
                f"Z-score devrait être 0.0 avec {len(historical_amounts)} factures "
                f"historiques (< 5), mais obtenu={result}"
            )
        finally:
            session.close()

    @settings(max_examples=100, deadline=None)
    @given(amount=st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False))
    def test_zscore_is_zero_with_no_history(self, amount: float):
        """
        Quand le fournisseur n'a aucune facture historique,
        le z-score doit être exactement 0.0.

        **Validates: Requirements 5.2**
        """
        session = _make_session()
        try:
            vendor = "UnknownVendorZscore"
            # Pas de données historiques insérées

            fe = FeatureEngine(session=session)
            result = fe._compute_amount_zscore(amount, vendor)

            assert result == 0.0, (
                f"Z-score devrait être 0.0 sans historique, mais obtenu={result}"
            )
        finally:
            session.close()

    @settings(max_examples=100, deadline=None)
    @given(
        amount=st.floats(min_value=100.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
        n_invoices=st.integers(min_value=5, max_value=10),
    )
    def test_zscore_is_zero_when_all_amounts_equal(
        self, amount: float, n_invoices: int
    ):
        """
        Quand toutes les factures historiques ont le même montant (std=0),
        le z-score doit être 0.0 (éviter division par zéro).

        **Validates: Requirements 5.1**
        """
        # Tous les montants historiques sont identiques → std = 0
        historical_amounts = [5000.0] * n_invoices

        session = _make_session()
        try:
            vendor = "ZscoreTestVendor"
            _populate_vendor_history(session, vendor, historical_amounts)

            fe = FeatureEngine(session=session)
            result = fe._compute_amount_zscore(amount, vendor)

            assert result == 0.0, (
                f"Z-score devrait être 0.0 quand std=0 (montants tous identiques), "
                f"mais obtenu={result}"
            )
        finally:
            session.close()
