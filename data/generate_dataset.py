"""
Génération du dataset synthétique de factures marocaines.
Produit 200 factures (175 normales + 25 frauduleuses) pour l'entraînement ML.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import sqlalchemy


class DatasetGenerator:
    """Générateur de factures synthétiques marocaines réalistes."""

    # Liste de fournisseurs marocains prédéfinis (≥5)
    VENDORS: List[str] = [
        "Maroc Telecom",
        "Atlas BTP",
        "Souss Agro",
        "Sahara Logistics",
        "Fès Textile",
        "Casablanca Import",
        "Rabat Services",
    ]

    # Taux de TVA marocains valides
    TAX_RATES: List[int] = [7, 10, 14, 20]

    # Constantes de génération
    TOTAL_INVOICES = 200
    NORMAL_COUNT = 175
    FRAUD_COUNT = 25

    # Plage de montants HT pour les factures normales (MAD)
    MIN_AMOUNT = 500.00
    MAX_AMOUNT = 500000.00

    def __init__(self, seed: int = None):
        """Initialise le générateur avec un seed optionnel pour la reproductibilité."""
        if seed is not None:
            random.seed(seed)

    def generate(self) -> pd.DataFrame:
        """
        Génère 200 factures: 175 normales + 25 frauduleuses.

        Returns:
            DataFrame avec colonnes: invoice_id, vendor_name, amount_ht,
            tax_rate, amount_ttc, date, label, fraud_type
        """
        invoices = []

        # Générer les factures normales
        for _ in range(self.NORMAL_COUNT):
            invoices.append(self._generate_normal_invoice())

        # Générer les factures frauduleuses
        for _ in range(self.FRAUD_COUNT):
            invoices.append(self._generate_fraudulent_invoice(invoices))

        # Mélanger l'ordre des factures
        random.shuffle(invoices)

        return pd.DataFrame(invoices)

    def _generate_normal_invoice(self) -> dict:
        """
        Génère une facture normale respectant toutes les contraintes:
        - Date en semaine (lundi-vendredi)
        - Taux de TVA valide (7, 10, 14, 20)
        - Montant HT entre 500 et 500,000 MAD
        - Montant avec 2 décimales

        Returns:
            Dictionnaire avec les 8 champs requis
        """
        invoice_id = f"INV-{uuid.uuid4().hex[:8].upper()}"
        vendor_name = random.choice(self.VENDORS)
        amount_ht = round(random.uniform(self.MIN_AMOUNT, self.MAX_AMOUNT), 2)
        tax_rate = random.choice(self.TAX_RATES)
        amount_ttc = round(amount_ht * (1 + tax_rate / 100), 2)
        date = self._generate_weekday_date()

        return {
            "invoice_id": invoice_id,
            "vendor_name": vendor_name,
            "amount_ht": amount_ht,
            "tax_rate": tax_rate,
            "amount_ttc": amount_ttc,
            "date": date,
            "label": "Normal",
            "fraud_type": "",
        }

    def _generate_fraudulent_invoice(self, existing_invoices: List[dict]) -> dict:
        """
        Génère une facture frauduleuse avec au moins 1 pattern de fraude:
        - Duplicata vendor-amount-date
        - Taux de TVA invalide
        - Montant hors plage [500, 500000]
        - Date le weekend (samedi/dimanche)

        Args:
            existing_invoices: Liste des factures déjà générées pour les duplicatas

        Returns:
            Dictionnaire avec les 8 champs requis + fraud_type décrit
        """
        # Choisir un ou plusieurs patterns de fraude
        fraud_patterns = [
            "duplicate_vendor_amount_date",
            "invalid_tax_rate",
            "out_of_range_amount",
            "weekend_date",
        ]
        # Choisir au moins 1 pattern (1 à 2 pour la variété)
        num_patterns = random.randint(1, 2)
        selected_patterns = random.sample(fraud_patterns, min(num_patterns, len(fraud_patterns)))

        # Valeurs de base
        invoice_id = f"INV-{uuid.uuid4().hex[:8].upper()}"
        vendor_name = random.choice(self.VENDORS)
        amount_ht = round(random.uniform(self.MIN_AMOUNT, self.MAX_AMOUNT), 2)
        tax_rate = random.choice(self.TAX_RATES)
        date = self._generate_weekday_date()
        fraud_descriptions = []

        # Appliquer les patterns de fraude sélectionnés
        for pattern in selected_patterns:
            if pattern == "duplicate_vendor_amount_date":
                # Dupliquer une facture existante (si possible)
                normal_invoices = [inv for inv in existing_invoices if inv["label"] == "Normal"]
                if normal_invoices:
                    source = random.choice(normal_invoices)
                    vendor_name = source["vendor_name"]
                    amount_ht = source["amount_ht"]
                    date = source["date"]
                    fraud_descriptions.append("duplicate_vendor_amount_date")

            elif pattern == "invalid_tax_rate":
                # Taux de TVA non standard
                invalid_rates = [5, 8, 12, 15, 18, 22, 25, 30]
                tax_rate = random.choice(invalid_rates)
                fraud_descriptions.append("invalid_tax_rate")

            elif pattern == "out_of_range_amount":
                # Montant hors de la plage normale
                if random.random() < 0.5:
                    # Montant trop bas
                    amount_ht = round(random.uniform(0.01, 499.99), 2)
                else:
                    # Montant trop élevé
                    amount_ht = round(random.uniform(500001.00, 2000000.00), 2)
                fraud_descriptions.append("out_of_range_amount")

            elif pattern == "weekend_date":
                # Date tombant un weekend
                date = self._generate_weekend_date()
                fraud_descriptions.append("weekend_date")

        # Calculer le TTC avec le taux (valide ou invalide)
        amount_ttc = round(amount_ht * (1 + tax_rate / 100), 2)

        return {
            "invoice_id": invoice_id,
            "vendor_name": vendor_name,
            "amount_ht": amount_ht,
            "tax_rate": tax_rate,
            "amount_ttc": amount_ttc,
            "date": date,
            "label": "Frauduleux",
            "fraud_type": ", ".join(fraud_descriptions) if fraud_descriptions else "unknown",
        }

    def _generate_weekday_date(self) -> str:
        """
        Génère une date aléatoire en semaine (lundi-vendredi)
        dans les 12 derniers mois au format ISO 8601.

        Returns:
            Date au format YYYY-MM-DD
        """
        today = datetime.now().date()
        start_date = today - timedelta(days=365)

        while True:
            random_days = random.randint(0, 365)
            candidate = start_date + timedelta(days=random_days)
            # 0=lundi, 4=vendredi → weekday
            if candidate.weekday() < 5:
                return candidate.isoformat()

    def _generate_weekend_date(self) -> str:
        """
        Génère une date aléatoire tombant un weekend (samedi/dimanche)
        dans les 12 derniers mois au format ISO 8601.

        Returns:
            Date au format YYYY-MM-DD
        """
        today = datetime.now().date()
        start_date = today - timedelta(days=365)

        while True:
            random_days = random.randint(0, 365)
            candidate = start_date + timedelta(days=random_days)
            # 5=samedi, 6=dimanche → weekend
            if candidate.weekday() >= 5:
                return candidate.isoformat()

    def export_sqlite(self, df: pd.DataFrame, path: str) -> None:
        """
        Exporte le DataFrame vers une base SQLite.

        Args:
            df: DataFrame contenant les factures générées
            path: Chemin du fichier SQLite (ex: data/invoices.db)
        """
        engine = sqlalchemy.create_engine(f"sqlite:///{path}")
        df.to_sql("invoices", engine, if_exists="replace", index=False)
        engine.dispose()

    def export_csv(self, df: pd.DataFrame, path: str) -> None:
        """
        Exporte le DataFrame vers un fichier CSV.

        Args:
            df: DataFrame contenant les factures générées
            path: Chemin du fichier CSV (ex: data/invoices.csv)
        """
        df.to_csv(path, index=False, encoding="utf-8")


if __name__ == "__main__":
    # Script exécutable directement pour générer le dataset
    generator = DatasetGenerator(seed=42)
    dataset = generator.generate()

    # Afficher un résumé
    print(f"Dataset généré: {len(dataset)} factures")
    print(f"  - Normales: {len(dataset[dataset['label'] == 'Normal'])}")
    print(f"  - Frauduleuses: {len(dataset[dataset['label'] == 'Frauduleux'])}")
    print(f"\nDistribution des types de fraude:")
    fraud_df = dataset[dataset["label"] == "Frauduleux"]
    print(fraud_df["fraud_type"].value_counts().to_string())

    # Export
    generator.export_sqlite(dataset, "data/invoices.db")
    generator.export_csv(dataset, "data/invoices.csv")
    print("\nExports réalisés:")
    print("  - data/invoices.db (SQLite)")
    print("  - data/invoices.csv (CSV)")
