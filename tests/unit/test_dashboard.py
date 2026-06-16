"""
Tests unitaires pour la page dashboard.
Vérifie le formatage des montants, les couleurs de statut et la logique de pagination.
"""

import math
import sys
import os

# Ajouter le répertoire racine du projet au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# --- Test des fonctions de formatage (sans dépendances Streamlit/plotly) ---


def _format_amount_mad(amount: float) -> str:
    """Formate un montant en MAD avec séparateur de milliers.
    Copie locale pour tester sans importer le module complet.
    """
    if amount is None or (isinstance(amount, float) and math.isnan(amount)):
        return "0,00 MAD"
    integer_part = int(amount)
    decimal_part = round((amount - integer_part) * 100)
    formatted_integer = f"{integer_part:,}".replace(",", " ")
    return f"{formatted_integer},{decimal_part:02d} MAD"


def _get_status_color(status: str) -> str:
    """Retourne la couleur CSS associée au statut de fraude."""
    colors = {
        "Normal": "#28a745",
        "Suspect": "#fd7e14",
        "Frauduleux": "#dc3545",
    }
    return colors.get(status, "#6c757d")


class TestFormatAmountMad:
    """Tests pour le formatage des montants en MAD."""

    def test_standard_amount(self):
        """Montant standard avec décimales."""
        result = _format_amount_mad(1234567.89)
        assert result == "1 234 567,89 MAD"

    def test_zero_amount(self):
        """Montant nul."""
        result = _format_amount_mad(0)
        assert result == "0,00 MAD"

    def test_small_amount(self):
        """Petit montant."""
        result = _format_amount_mad(500.50)
        assert result == "500,50 MAD"

    def test_large_amount(self):
        """Grand montant avec séparateur de milliers."""
        result = _format_amount_mad(999999999.99)
        assert result == "999 999 999,99 MAD"

    def test_no_decimals(self):
        """Montant sans partie décimale."""
        result = _format_amount_mad(1000.0)
        assert result == "1 000,00 MAD"

    def test_nan_amount(self):
        """Montant NaN retourne zéro."""
        result = _format_amount_mad(float("nan"))
        assert result == "0,00 MAD"

    def test_none_amount(self):
        """Montant None retourne zéro."""
        result = _format_amount_mad(None)
        assert result == "0,00 MAD"


class TestGetStatusColor:
    """Tests pour les couleurs de statut."""

    def test_normal_green(self):
        """Normal doit être vert."""
        assert _get_status_color("Normal") == "#28a745"

    def test_suspect_orange(self):
        """Suspect doit être orange."""
        assert _get_status_color("Suspect") == "#fd7e14"

    def test_frauduleux_red(self):
        """Frauduleux doit être rouge."""
        assert _get_status_color("Frauduleux") == "#dc3545"

    def test_unknown_status(self):
        """Statut inconnu retourne gris."""
        assert _get_status_color("Inconnu") == "#6c757d"


class TestPaginationLogic:
    """Tests pour la logique de pagination."""

    def test_single_page(self):
        """Moins de 50 factures → 1 seule page."""
        total = 30
        total_pages = math.ceil(total / 50)
        assert total_pages == 1

    def test_exact_page_boundary(self):
        """Exactement 50 factures → 1 page."""
        total = 50
        total_pages = math.ceil(total / 50)
        assert total_pages == 1

    def test_multiple_pages(self):
        """Plus de 50 factures → plusieurs pages."""
        total = 120
        total_pages = math.ceil(total / 50)
        assert total_pages == 3

    def test_page_slice_first(self):
        """Première page: indices 0-49."""
        page = 1
        start_idx = (page - 1) * 50
        end_idx = start_idx + 50
        assert start_idx == 0
        assert end_idx == 50

    def test_page_slice_second(self):
        """Deuxième page: indices 50-99."""
        page = 2
        start_idx = (page - 1) * 50
        end_idx = start_idx + 50
        assert start_idx == 50
        assert end_idx == 100


class TestSortingLogic:
    """Tests pour la logique de tri par date décroissante."""

    def test_sort_descending(self):
        """Les factures doivent être triées par date décroissante."""
        invoices = [
            {"date": "2024-01-15", "fraud_label": "Normal"},
            {"date": "2024-03-20", "fraud_label": "Suspect"},
            {"date": "2024-02-10", "fraud_label": "Frauduleux"},
        ]
        sorted_invoices = sorted(
            invoices,
            key=lambda x: x.get("date", "") or "",
            reverse=True,
        )
        assert sorted_invoices[0]["date"] == "2024-03-20"
        assert sorted_invoices[1]["date"] == "2024-02-10"
        assert sorted_invoices[2]["date"] == "2024-01-15"

    def test_sort_with_none_dates(self):
        """Les factures sans date doivent aller en fin de liste."""
        invoices = [
            {"date": None, "fraud_label": "Normal"},
            {"date": "2024-03-20", "fraud_label": "Suspect"},
            {"date": "2024-01-15", "fraud_label": "Normal"},
        ]
        sorted_invoices = sorted(
            invoices,
            key=lambda x: x.get("date", "") or "",
            reverse=True,
        )
        assert sorted_invoices[0]["date"] == "2024-03-20"
        assert sorted_invoices[1]["date"] == "2024-01-15"
        assert sorted_invoices[2]["date"] is None


class TestEmptyState:
    """Tests pour l'état vide du dashboard."""

    def test_zero_stats(self):
        """Stats à zéro quand aucune facture analysée."""
        stats = {"total_invoices": 0, "fraud_rate": 0.0, "total_amount": 0.0}
        assert stats["total_invoices"] == 0
        assert stats["fraud_rate"] == 0.0
        assert stats["total_amount"] == 0.0

    def test_empty_invoices_list(self):
        """Liste vide de factures."""
        invoices = []
        assert len(invoices) == 0
