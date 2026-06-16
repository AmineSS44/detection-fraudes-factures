"""
Moteur de feature engineering pour la détection de fraude sur factures.
Calcule 6 features discriminantes à partir des données extraites et de l'historique.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import Invoice, SessionLocal


@dataclass
class FeatureVector:
    """Vecteur de 6 features calculées pour une facture."""

    amount_zscore: float  # z-score vs historique fournisseur
    tax_inconsistency: bool  # taux de taxe hors {7, 10, 14, 20}
    duplicate_flag: bool  # même fournisseur+montant dans les 30 derniers jours
    weekend_flag: bool  # date un samedi ou dimanche
    round_amount_flag: bool  # multiple de 1000 ET > 10 000
    vendor_deviation: float  # score de rareté 0.0-1.0


class FeatureEngine:
    """Calcule les features de détection de fraude à partir des données facture."""

    # Taux de taxe valides au Maroc
    VALID_TAX_RATES = {7, 10, 14, 20}

    def __init__(self, session: Optional[Session] = None):
        """Initialise le moteur avec une session DB optionnelle.

        Args:
            session: Session SQLAlchemy. Si None, utilise SessionLocal().
        """
        self._session = session

    def _get_session(self) -> Session:
        """Retourne la session DB active ou en crée une nouvelle."""
        if self._session is not None:
            return self._session
        return SessionLocal()

    def _should_close_session(self) -> bool:
        """Indique si la session doit être fermée après utilisation."""
        return self._session is None

    def compute_features(self, invoice_data: dict) -> FeatureVector:
        """Calcule les 6 features à partir des données extraites.

        Args:
            invoice_data: Dictionnaire avec les champs extraits de la facture.
                Champs requis: amount, vendor, date, tax_rate.

        Returns:
            FeatureVector avec les 6 features calculées.

        Raises:
            ValueError: Si des champs requis sont manquants.
        """
        # Vérification des champs requis
        required_fields = ["amount", "vendor", "date", "tax_rate"]
        missing_fields = [
            field for field in required_fields if field not in invoice_data or invoice_data[field] is None
        ]

        if missing_fields:
            raise ValueError(
                f"Champs requis manquants: {', '.join(missing_fields)}"
            )

        amount = float(invoice_data["amount"])
        vendor = str(invoice_data["vendor"])
        date = str(invoice_data["date"])
        tax_rate = float(invoice_data["tax_rate"])

        # Calcul des 6 features
        amount_zscore = self._compute_amount_zscore(amount, vendor)
        tax_inconsistency = self._compute_tax_inconsistency(tax_rate)
        duplicate_flag = self._compute_duplicate_flag(amount, vendor, date)
        weekend_flag = self._compute_weekend_flag(date)
        round_amount_flag = self._compute_round_amount_flag(amount)
        vendor_deviation = self._compute_vendor_deviation(vendor)

        return FeatureVector(
            amount_zscore=amount_zscore,
            tax_inconsistency=tax_inconsistency,
            duplicate_flag=duplicate_flag,
            weekend_flag=weekend_flag,
            round_amount_flag=round_amount_flag,
            vendor_deviation=vendor_deviation,
        )

    def _compute_amount_zscore(self, amount: float, vendor: str) -> float:
        """Calcule le z-score du montant par rapport à l'historique du fournisseur.

        Z-score = (amount - mean) / std sur les 12 derniers mois.
        Retourne 0.0 si moins de 5 factures historiques.

        Args:
            amount: Montant HT de la facture en MAD.
            vendor: Nom du fournisseur.

        Returns:
            Z-score ou 0.0 si historique insuffisant.
        """
        session = self._get_session()
        try:
            # Fenêtre de 12 mois
            twelve_months_ago = (datetime.now() - timedelta(days=365)).strftime(
                "%Y-%m-%d"
            )

            # Récupérer les montants historiques du fournisseur
            historical_amounts = (
                session.query(Invoice.amount_ht)
                .filter(
                    Invoice.vendor_name == vendor,
                    Invoice.date >= twelve_months_ago,
                    Invoice.amount_ht.isnot(None),
                )
                .all()
            )

            amounts = [row[0] for row in historical_amounts]

            # Besoin d'au moins 5 factures historiques
            if len(amounts) < 5:
                return 0.0

            # Calcul du z-score
            mean = sum(amounts) / len(amounts)
            variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
            std = variance**0.5

            # Éviter la division par zéro
            if std == 0.0:
                return 0.0

            return (amount - mean) / std
        finally:
            if self._should_close_session():
                session.close()

    def _compute_tax_inconsistency(self, tax_rate: float) -> bool:
        """Vérifie si le taux de taxe est inconsistant.

        True si le taux n'est pas parmi les taux marocains valides {7, 10, 14, 20}.

        Args:
            tax_rate: Taux de taxe en pourcentage.

        Returns:
            True si taux invalide, False sinon.
        """
        return tax_rate not in self.VALID_TAX_RATES

    def _compute_duplicate_flag(
        self, amount: float, vendor: str, date: str
    ) -> bool:
        """Détecte les doublons: même fournisseur + même montant dans les 30 derniers jours.

        Args:
            amount: Montant HT de la facture.
            vendor: Nom du fournisseur.
            date: Date de la facture (format YYYY-MM-DD).

        Returns:
            True si un doublon existe, False sinon.
        """
        session = self._get_session()
        try:
            # Fenêtre de 30 jours avant la date de la facture
            invoice_date = datetime.strptime(date, "%Y-%m-%d")
            thirty_days_ago = (invoice_date - timedelta(days=30)).strftime(
                "%Y-%m-%d"
            )

            # Chercher un doublon dans les 30 derniers jours
            duplicate_count = (
                session.query(func.count(Invoice.id))
                .filter(
                    Invoice.vendor_name == vendor,
                    Invoice.amount_ht == amount,
                    Invoice.date >= thirty_days_ago,
                    Invoice.date < date,
                )
                .scalar()
            )

            return duplicate_count > 0
        finally:
            if self._should_close_session():
                session.close()

    def _compute_weekend_flag(self, date: str) -> bool:
        """Vérifie si la date tombe un samedi ou dimanche.

        Args:
            date: Date au format YYYY-MM-DD.

        Returns:
            True si samedi (5) ou dimanche (6), False sinon.
        """
        invoice_date = datetime.strptime(date, "%Y-%m-%d")
        return invoice_date.weekday() >= 5

    def _compute_round_amount_flag(self, amount: float) -> bool:
        """Vérifie si le montant est un chiffre rond suspect.

        True si le montant est un multiple de 1000 ET strictement supérieur à 10 000.

        Args:
            amount: Montant HT en MAD.

        Returns:
            True si montant rond et > 10 000, False sinon.
        """
        return amount % 1000 == 0 and amount > 10000

    def _compute_vendor_deviation(self, vendor: str) -> float:
        """Calcule le score de rareté du fournisseur.

        - 1.0 si le fournisseur n'a aucune facture antérieure.
        - Sinon: 1 - (vendor_count / max_count) sur les 12 derniers mois.
        - Toujours dans [0.0, 1.0].

        Args:
            vendor: Nom du fournisseur.

        Returns:
            Score de rareté entre 0.0 et 1.0.
        """
        session = self._get_session()
        try:
            # Fenêtre de 12 mois
            twelve_months_ago = (datetime.now() - timedelta(days=365)).strftime(
                "%Y-%m-%d"
            )

            # Compter les factures du fournisseur sur 12 mois
            vendor_count = (
                session.query(func.count(Invoice.id))
                .filter(
                    Invoice.vendor_name == vendor,
                    Invoice.date >= twelve_months_ago,
                )
                .scalar()
            )

            # Aucune facture antérieure → rareté maximale
            if vendor_count == 0:
                return 1.0

            # Trouver le max de factures parmi tous les fournisseurs sur 12 mois
            max_count_result = (
                session.query(func.count(Invoice.id))
                .filter(Invoice.date >= twelve_months_ago)
                .group_by(Invoice.vendor_name)
                .order_by(func.count(Invoice.id).desc())
                .first()
            )

            max_count = max_count_result[0] if max_count_result else 1

            # Éviter la division par zéro
            if max_count == 0:
                return 1.0

            # Score de rareté: plus le fournisseur est rare, plus le score est élevé
            deviation = 1.0 - (vendor_count / max_count)

            # Clamp dans [0.0, 1.0]
            return max(0.0, min(1.0, deviation))
        finally:
            if self._should_close_session():
                session.close()
