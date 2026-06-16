"""
Tests de propriétés pour le module de génération de dataset synthétique.
Vérifie les invariants des factures générées via Hypothesis.
"""

from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers

from data.generate_dataset import DatasetGenerator


# Feature: invoice-fraud-detection, Property 21: Normal invoice generation invariants
class TestNormalInvoiceInvariants:
    """
    Property 21: Normal invoice generation invariants

    For any invoice generated with label "Normal" by the Dataset_Generator,
    the vendor_name should be from the predefined vendor list, the amount_ht
    should be in [500.00, 500000.00] with 2 decimal places, the tax_rate
    should be in {7, 10, 14, 20}, the date should be a weekday (Monday-Friday)
    within the last 12 months, and all 8 required fields should be present.

    **Validates: Requirements 8.2, 8.3, 8.4, 8.5, 8.8**
    """

    REQUIRED_FIELDS = [
        "invoice_id",
        "vendor_name",
        "amount_ht",
        "tax_rate",
        "amount_ttc",
        "date",
        "label",
        "fraud_type",
    ]

    VALID_TAX_RATES = {7, 10, 14, 20}

    @given(seed=integers(min_value=0, max_value=10000))
    @settings(max_examples=100, deadline=None)
    def test_normal_invoices_have_all_required_fields(self, seed):
        """Toutes les factures normales contiennent les 8 champs requis."""
        generator = DatasetGenerator(seed=seed)
        df = generator.generate()

        normal_df = df[df["label"] == "Normal"]
        assert len(normal_df) > 0, "Aucune facture normale générée"

        for _, row in normal_df.iterrows():
            for field in self.REQUIRED_FIELDS:
                assert field in row.index, (
                    f"Champ requis '{field}' manquant dans la facture {row.get('invoice_id', '?')}"
                )
                # Vérifier que le champ n'est pas NaN (sauf fraud_type qui peut être vide)
                if field != "fraud_type":
                    assert row[field] is not None and str(row[field]) != "nan", (
                        f"Champ '{field}' est null/NaN pour la facture {row.get('invoice_id', '?')}"
                    )

    @given(seed=integers(min_value=0, max_value=10000))
    @settings(max_examples=100, deadline=None)
    def test_normal_invoices_vendor_in_predefined_list(self, seed):
        """Les factures normales utilisent un fournisseur de la liste prédéfinie."""
        generator = DatasetGenerator(seed=seed)
        df = generator.generate()

        normal_df = df[df["label"] == "Normal"]

        for _, row in normal_df.iterrows():
            assert row["vendor_name"] in generator.VENDORS, (
                f"Vendor '{row['vendor_name']}' n'est pas dans la liste prédéfinie "
                f"{generator.VENDORS} (facture {row['invoice_id']})"
            )

    @given(seed=integers(min_value=0, max_value=10000))
    @settings(max_examples=100, deadline=None)
    def test_normal_invoices_amount_in_valid_range(self, seed):
        """Les factures normales ont un montant HT entre 500.00 et 500,000.00 MAD avec 2 décimales."""
        generator = DatasetGenerator(seed=seed)
        df = generator.generate()

        normal_df = df[df["label"] == "Normal"]

        for _, row in normal_df.iterrows():
            amount = row["amount_ht"]

            # Vérifier la plage
            assert 500.00 <= amount <= 500000.00, (
                f"Montant HT {amount} hors plage [500.00, 500000.00] "
                f"(facture {row['invoice_id']})"
            )

            # Vérifier 2 décimales: round(amount, 2) == amount
            assert round(amount, 2) == amount, (
                f"Montant HT {amount} n'a pas exactement 2 décimales "
                f"(facture {row['invoice_id']})"
            )

    @given(seed=integers(min_value=0, max_value=10000))
    @settings(max_examples=100, deadline=None)
    def test_normal_invoices_tax_rate_valid(self, seed):
        """Les factures normales ont un taux de TVA parmi {7, 10, 14, 20}."""
        generator = DatasetGenerator(seed=seed)
        df = generator.generate()

        normal_df = df[df["label"] == "Normal"]

        for _, row in normal_df.iterrows():
            assert row["tax_rate"] in self.VALID_TAX_RATES, (
                f"Taux de TVA {row['tax_rate']} invalide pour facture normale "
                f"(attendu: {self.VALID_TAX_RATES}, facture {row['invoice_id']})"
            )

    @given(seed=integers(min_value=0, max_value=10000))
    @settings(max_examples=100, deadline=None)
    def test_normal_invoices_date_is_weekday_within_12_months(self, seed):
        """Les factures normales ont une date en semaine (lun-ven) dans les 12 derniers mois."""
        generator = DatasetGenerator(seed=seed)
        df = generator.generate()

        normal_df = df[df["label"] == "Normal"]
        today = datetime.now().date()
        twelve_months_ago = today - timedelta(days=365)

        for _, row in normal_df.iterrows():
            date = datetime.strptime(row["date"], "%Y-%m-%d").date()

            # Vérifier que c'est un jour de semaine (0=lundi, 4=vendredi)
            assert date.weekday() < 5, (
                f"Date {row['date']} tombe un weekend (jour {date.weekday()}) "
                f"pour facture normale {row['invoice_id']}"
            )

            # Vérifier que la date est dans les 12 derniers mois
            assert twelve_months_ago <= date <= today, (
                f"Date {row['date']} hors de la fenêtre de 12 mois "
                f"[{twelve_months_ago}, {today}] (facture {row['invoice_id']})"
            )


# Feature: invoice-fraud-detection, Property 23: Fraudulent invoice pattern presence
class TestFraudulentInvoicePatternPresence:
    """
    Property 23: Fraudulent invoice pattern presence

    For any invoice generated with label "Frauduleux" by the Dataset_Generator,
    at least one fraud pattern should be present: duplicated vendor-amount-date,
    tax rate not in {7, 10, 14, 20}, amount outside [500, 500000] MAD,
    or date on Saturday/Sunday.

    **Validates: Requirements 8.6**
    """

    VALID_TAX_RATES = {7, 10, 14, 20}
    MIN_AMOUNT = 500.00
    MAX_AMOUNT = 500000.00

    def _has_invalid_tax_rate(self, row) -> bool:
        """Vérifie si le taux de TVA n'est pas dans {7, 10, 14, 20}."""
        return row["tax_rate"] not in self.VALID_TAX_RATES

    def _has_out_of_range_amount(self, row) -> bool:
        """Vérifie si le montant HT est hors de [500, 500000] MAD."""
        return row["amount_ht"] < self.MIN_AMOUNT or row["amount_ht"] > self.MAX_AMOUNT

    def _has_weekend_date(self, row) -> bool:
        """Vérifie si la date tombe un samedi ou dimanche."""
        date = datetime.strptime(row["date"], "%Y-%m-%d").date()
        return date.weekday() >= 5

    def _has_duplicate_vendor_amount_date(self, row, df) -> bool:
        """Vérifie si le combo vendor-amount-date est dupliqué dans le dataset."""
        matches = df[
            (df["vendor_name"] == row["vendor_name"])
            & (df["amount_ht"] == row["amount_ht"])
            & (df["date"] == row["date"])
        ]
        # Dupliqué si plus d'une occurrence (la ligne elle-même + au moins une autre)
        return len(matches) > 1

    def _has_at_least_one_fraud_pattern(self, row, df) -> bool:
        """Vérifie qu'au moins un pattern de fraude est présent."""
        return (
            self._has_invalid_tax_rate(row)
            or self._has_out_of_range_amount(row)
            or self._has_weekend_date(row)
            or self._has_duplicate_vendor_amount_date(row, df)
        )

    @given(seed=integers(min_value=0, max_value=10000))
    @settings(max_examples=100, deadline=None)
    def test_fraudulent_invoices_have_at_least_one_pattern(self, seed: int):
        """Chaque facture frauduleuse doit exhiber au moins un pattern de fraude détectable."""
        # Générer le dataset avec le seed donné
        generator = DatasetGenerator(seed=seed)
        df = generator.generate()

        # Filtrer les factures frauduleuses
        fraudulent = df[df["label"] == "Frauduleux"]

        # Vérifier que chaque facture frauduleuse a au moins un pattern
        for idx, row in fraudulent.iterrows():
            assert self._has_at_least_one_fraud_pattern(row, df), (
                f"Facture frauduleuse (seed={seed}, index={idx}, invoice_id={row['invoice_id']}) "
                f"n'a aucun pattern de fraude détectable.\n"
                f"  vendor_name: {row['vendor_name']}\n"
                f"  amount_ht: {row['amount_ht']}\n"
                f"  tax_rate: {row['tax_rate']}\n"
                f"  date: {row['date']}\n"
                f"  fraud_type: {row['fraud_type']}"
            )


# Feature: invoice-fraud-detection, Property 22: Dataset size and distribution
class TestDatasetSizeAndDistribution:
    """
    Property 22: Dataset size and distribution

    For any execution of the Dataset_Generator, the output should contain
    exactly 200 records with exactly 175 labeled "Normal" and exactly 25
    labeled "Frauduleux".

    **Validates: Requirements 8.1**
    """

    @given(seed=integers(min_value=0, max_value=2**32 - 1))
    @settings(max_examples=100, deadline=None)
    def test_dataset_has_exactly_200_records(self, seed):
        """
        Pour tout seed aléatoire, le dataset généré doit contenir
        exactement 200 enregistrements.

        **Validates: Requirements 8.1**
        """
        generator = DatasetGenerator(seed=seed)
        df = generator.generate()

        assert len(df) == 200, (
            f"Le dataset contient {len(df)} enregistrements au lieu de 200 "
            f"(seed={seed})"
        )

    @given(seed=integers(min_value=0, max_value=2**32 - 1))
    @settings(max_examples=100, deadline=None)
    def test_dataset_has_exactly_175_normal(self, seed):
        """
        Pour tout seed aléatoire, le dataset généré doit contenir
        exactement 175 factures labellisées "Normal".

        **Validates: Requirements 8.1**
        """
        generator = DatasetGenerator(seed=seed)
        df = generator.generate()

        normal_count = len(df[df["label"] == "Normal"])
        assert normal_count == 175, (
            f"Le dataset contient {normal_count} factures normales au lieu de 175 "
            f"(seed={seed})"
        )

    @given(seed=integers(min_value=0, max_value=2**32 - 1))
    @settings(max_examples=100, deadline=None)
    def test_dataset_has_exactly_25_frauduleux(self, seed):
        """
        Pour tout seed aléatoire, le dataset généré doit contenir
        exactement 25 factures labellisées "Frauduleux".

        **Validates: Requirements 8.1**
        """
        generator = DatasetGenerator(seed=seed)
        df = generator.generate()

        fraud_count = len(df[df["label"] == "Frauduleux"])
        assert fraud_count == 25, (
            f"Le dataset contient {fraud_count} factures frauduleuses au lieu de 25 "
            f"(seed={seed})"
        )

    @given(seed=integers(min_value=0, max_value=2**32 - 1))
    @settings(max_examples=100, deadline=None)
    def test_dataset_labels_only_normal_or_frauduleux(self, seed):
        """
        Pour tout seed aléatoire, les seules valeurs possibles dans
        la colonne "label" sont "Normal" et "Frauduleux".

        **Validates: Requirements 8.1**
        """
        generator = DatasetGenerator(seed=seed)
        df = generator.generate()

        valid_labels = {"Normal", "Frauduleux"}
        actual_labels = set(df["label"].unique())
        assert actual_labels == valid_labels, (
            f"Labels inattendus trouvés: {actual_labels - valid_labels} "
            f"(seed={seed})"
        )
