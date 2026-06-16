"""
Tests unitaires pour le module de génération de dataset synthétique.
Vérifie la structure, les contraintes et les exports du DatasetGenerator.
"""

import os
import tempfile
from datetime import datetime

import pandas as pd
import pytest
import sqlalchemy

from data.generate_dataset import DatasetGenerator


@pytest.fixture
def generator():
    """Fournit un générateur avec seed fixe pour la reproductibilité."""
    return DatasetGenerator(seed=42)


@pytest.fixture
def dataset(generator):
    """Fournit un dataset généré pour les tests."""
    return generator.generate()


class TestDatasetGeneration:
    """Tests pour la méthode generate()."""

    def test_total_count(self, dataset):
        """Vérifie que le dataset contient exactement 200 factures."""
        assert len(dataset) == 200

    def test_normal_count(self, dataset):
        """Vérifie qu'il y a exactement 175 factures normales."""
        normal_count = len(dataset[dataset["label"] == "Normal"])
        assert normal_count == 175

    def test_fraud_count(self, dataset):
        """Vérifie qu'il y a exactement 25 factures frauduleuses."""
        fraud_count = len(dataset[dataset["label"] == "Frauduleux"])
        assert fraud_count == 25

    def test_required_columns(self, dataset):
        """Vérifie que toutes les colonnes requises sont présentes."""
        required_columns = [
            "invoice_id",
            "vendor_name",
            "amount_ht",
            "tax_rate",
            "amount_ttc",
            "date",
            "label",
            "fraud_type",
        ]
        for col in required_columns:
            assert col in dataset.columns, f"Colonne manquante: {col}"

    def test_labels_only_normal_or_frauduleux(self, dataset):
        """Vérifie que les labels sont uniquement Normal ou Frauduleux."""
        valid_labels = {"Normal", "Frauduleux"}
        assert set(dataset["label"].unique()) == valid_labels


class TestNormalInvoices:
    """Tests pour les factures normales."""

    @pytest.fixture
    def normal_invoices(self, dataset):
        """Filtre uniquement les factures normales."""
        return dataset[dataset["label"] == "Normal"]

    def test_vendors_from_list(self, normal_invoices):
        """Vérifie que les fournisseurs proviennent de la liste prédéfinie."""
        for vendor in normal_invoices["vendor_name"].unique():
            assert vendor in DatasetGenerator.VENDORS

    def test_amount_range(self, normal_invoices):
        """Vérifie que les montants HT sont entre 500 et 500,000 MAD."""
        assert normal_invoices["amount_ht"].min() >= 500.00
        assert normal_invoices["amount_ht"].max() <= 500000.00

    def test_valid_tax_rates(self, normal_invoices):
        """Vérifie que les taux de TVA sont valides (7, 10, 14, 20)."""
        valid_rates = set(DatasetGenerator.TAX_RATES)
        for rate in normal_invoices["tax_rate"].unique():
            assert rate in valid_rates

    def test_weekday_dates(self, normal_invoices):
        """Vérifie que les dates normales tombent en semaine."""
        for date_str in normal_invoices["date"]:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            assert date_obj.weekday() < 5, f"Date weekend trouvée: {date_str}"

    def test_amount_ttc_calculation(self, normal_invoices):
        """Vérifie que amount_ttc = amount_ht * (1 + tax_rate/100)."""
        for _, row in normal_invoices.iterrows():
            expected_ttc = round(row["amount_ht"] * (1 + row["tax_rate"] / 100), 2)
            assert abs(row["amount_ttc"] - expected_ttc) < 0.01

    def test_fraud_type_empty(self, normal_invoices):
        """Vérifie que fraud_type est vide pour les factures normales."""
        for fraud_type in normal_invoices["fraud_type"]:
            assert fraud_type == ""

    def test_dates_within_12_months(self, normal_invoices):
        """Vérifie que les dates sont dans les 12 derniers mois."""
        today = datetime.now().date()
        for date_str in normal_invoices["date"]:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            diff = (today - date_obj).days
            assert 0 <= diff <= 365

    def test_amount_two_decimals(self, normal_invoices):
        """Vérifie que les montants ont 2 décimales."""
        for amount in normal_invoices["amount_ht"]:
            # Vérifier que la valeur arrondie à 2 décimales est identique
            assert amount == round(amount, 2)


class TestFraudulentInvoices:
    """Tests pour les factures frauduleuses."""

    @pytest.fixture
    def fraud_invoices(self, dataset):
        """Filtre uniquement les factures frauduleuses."""
        return dataset[dataset["label"] == "Frauduleux"]

    def test_at_least_one_fraud_pattern(self, fraud_invoices):
        """Vérifie que chaque facture frauduleuse a au moins 1 pattern de fraude."""
        valid_patterns = {
            "duplicate_vendor_amount_date",
            "invalid_tax_rate",
            "out_of_range_amount",
            "weekend_date",
        }
        for _, row in fraud_invoices.iterrows():
            fraud_type = row["fraud_type"]
            assert fraud_type != "", f"fraud_type vide pour {row['invoice_id']}"
            patterns = [p.strip() for p in fraud_type.split(",")]
            for pattern in patterns:
                assert pattern in valid_patterns, f"Pattern inconnu: {pattern}"

    def test_fraud_type_not_empty(self, fraud_invoices):
        """Vérifie que fraud_type est renseigné pour les factures frauduleuses."""
        for fraud_type in fraud_invoices["fraud_type"]:
            assert fraud_type != ""
            assert len(fraud_type) > 0


class TestExport:
    """Tests pour les fonctions d'export."""

    def test_export_sqlite(self, generator, dataset):
        """Vérifie l'export SQLite."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        try:
            generator.export_sqlite(dataset, path)
            # Vérifier que le fichier existe et contient les données
            engine = sqlalchemy.create_engine(f"sqlite:///{path}")
            loaded = pd.read_sql("SELECT * FROM invoices", engine)
            engine.dispose()
            assert len(loaded) == 200
            assert "invoice_id" in loaded.columns
            assert "fraud_type" in loaded.columns
        finally:
            os.unlink(path)

    def test_export_csv(self, generator, dataset):
        """Vérifie l'export CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name

        try:
            generator.export_csv(dataset, path)
            # Vérifier que le fichier existe et contient les données
            loaded = pd.read_csv(path)
            assert len(loaded) == 200
            assert "invoice_id" in loaded.columns
            assert "fraud_type" in loaded.columns
        finally:
            os.unlink(path)


class TestInvoiceIds:
    """Tests pour les identifiants de factures."""

    def test_unique_ids(self, dataset):
        """Vérifie que les invoice_id sont uniques."""
        assert dataset["invoice_id"].nunique() == len(dataset)

    def test_id_format(self, dataset):
        """Vérifie le format des invoice_id (INV-XXXXXXXX)."""
        for inv_id in dataset["invoice_id"]:
            assert inv_id.startswith("INV-")
            assert len(inv_id) == 12  # INV- + 8 caractères hex
