"""
Page Dashboard: KPIs, graphiques et tableau des factures analysées.
Affiche les statistiques de fraude, la distribution des statuts,
l'évolution temporelle du score de fraude et le tableau détaillé.
"""

import math
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# URL de base de l'API backend
API_BASE_URL = "http://127.0.0.1:8000"

# Nombre maximum de lignes par page dans le tableau
ROWS_PER_PAGE = 50


def _get_auth_headers() -> dict:
    """Retourne les headers d'authentification avec le token JWT."""
    token = st.session_state.get("token", "")
    return {"Authorization": f"Bearer {token}"}


def _format_amount_mad(amount: float) -> str:
    """Formate un montant en MAD avec séparateur de milliers.

    Exemple: 1234567.89 → "1 234 567,89 MAD"
    """
    if amount is None or math.isnan(amount):
        return "0,00 MAD"
    # Séparer partie entière et décimale
    integer_part = int(amount)
    decimal_part = round((amount - integer_part) * 100)
    # Formater avec espaces comme séparateur de milliers
    formatted_integer = f"{integer_part:,}".replace(",", " ")
    return f"{formatted_integer},{decimal_part:02d} MAD"


def _get_status_color(status: str) -> str:
    """Retourne la couleur CSS associée au statut de fraude."""
    colors = {
        "Normal": "#28a745",      # Vert
        "Suspect": "#fd7e14",     # Orange
        "Frauduleux": "#dc3545",  # Rouge
    }
    return colors.get(status, "#6c757d")


def _fetch_stats() -> dict:
    """Récupère les KPIs agrégés depuis l'API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/stats",
            headers=_get_auth_headers(),
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def _fetch_invoices() -> list:
    """Récupère la liste des factures analysées depuis l'API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/invoices",
            headers=_get_auth_headers(),
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        return []
    except requests.RequestException:
        return []


def _render_kpi_cards(stats: dict):
    """Affiche les cartes KPI: total factures, taux de fraude, montant total."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="📄 Total Factures",
            value=str(stats.get("total_invoices", 0)),
        )

    with col2:
        fraud_rate = stats.get("fraud_rate", 0.0)
        st.metric(
            label="⚠️ Taux de Fraude",
            value=f"{fraud_rate:.1f}%",
        )

    with col3:
        total_amount = stats.get("total_amount", 0.0)
        st.metric(
            label="💰 Montant Total",
            value=_format_amount_mad(total_amount),
        )


def _render_pie_chart(invoices: list):
    """Affiche le pie chart de la distribution Normal/Suspect/Frauduleux."""
    # Compter les statuts
    status_counts = Counter(inv.get("fraud_label", "Inconnu") for inv in invoices)

    # Préparer les données pour le graphique
    labels = []
    values = []
    colors = []

    # Ordre défini pour cohérence visuelle
    for status in ["Normal", "Suspect", "Frauduleux"]:
        count = status_counts.get(status, 0)
        if count > 0:
            labels.append(f"{status} ({count})")
            values.append(count)
            colors.append(_get_status_color(status))

    if not values:
        st.info("Aucune donnée disponible pour le graphique.")
        return

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo="label+value",
        textposition="outside",
        hole=0.3,
    )])

    fig.update_layout(
        title="Distribution des Statuts",
        showlegend=True,
        margin=dict(t=50, b=20, l=20, r=20),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_line_chart(invoices: list):
    """Affiche le line chart du score de fraude moyen par jour."""
    if not invoices:
        st.info("Aucune donnée disponible pour le graphique.")
        return

    # Construire un DataFrame avec les dates et scores
    data = []
    for inv in invoices:
        date_str = inv.get("date")
        score = inv.get("fraud_score")
        if date_str and score is not None:
            data.append({"date": date_str, "fraud_score": score})

    if not data:
        st.info("Aucune donnée disponible pour le graphique temporel.")
        return

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        st.info("Aucune date valide pour le graphique temporel.")
        return

    # Calculer le score moyen par jour
    daily_avg = df.groupby(df["date"].dt.date)["fraud_score"].mean().reset_index()
    daily_avg.columns = ["date", "score_moyen"]
    daily_avg = daily_avg.sort_values("date")

    fig = px.line(
        daily_avg,
        x="date",
        y="score_moyen",
        title="Score de Fraude Moyen par Jour",
        labels={"date": "Date", "score_moyen": "Score Fraude Moyen"},
        markers=True,
    )

    fig.update_yaxes(range=[0.0, 1.0], title="Score Fraude Moyen")
    fig.update_xaxes(title="Date")
    fig.update_layout(
        margin=dict(t=50, b=20, l=20, r=20),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_invoice_table(invoices: list):
    """Affiche le tableau des factures avec pagination et couleurs de statut."""
    if not invoices:
        st.info("Aucune facture analysée")
        return

    # Trier par date décroissante
    sorted_invoices = sorted(
        invoices,
        key=lambda x: x.get("date", "") or "",
        reverse=True,
    )

    # Pagination
    total_invoices = len(sorted_invoices)
    total_pages = math.ceil(total_invoices / ROWS_PER_PAGE)

    if total_pages > 1:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key="dashboard_page",
        )
    else:
        page = 1

    start_idx = (page - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE
    page_invoices = sorted_invoices[start_idx:end_idx]

    # Construire les données du tableau
    table_data = []
    for inv in page_invoices:
        table_data.append({
            "ID": inv.get("invoice_id") or str(inv.get("id", "")),
            "Fournisseur": inv.get("vendor_name") or "—",
            "Montant (MAD)": _format_amount_mad(inv.get("amount_ttc") or 0),
            "Date": inv.get("date") or "—",
            "Score fraude": f"{inv.get('fraud_score', 0):.2f}",
            "Statut": inv.get("fraud_label") or "—",
        })

    df = pd.DataFrame(table_data)

    # Afficher le tableau avec couleurs de statut via st.dataframe + styling
    def _color_status(val):
        """Applique la couleur de fond selon le statut."""
        color_map = {
            "Normal": "background-color: #d4edda; color: #155724;",
            "Suspect": "background-color: #fff3cd; color: #856404;",
            "Frauduleux": "background-color: #f8d7da; color: #721c24;",
        }
        return color_map.get(val, "")

    # Appliquer le style au DataFrame
    styled_df = df.style.applymap(
        _color_status,
        subset=["Statut"],
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=min(len(page_invoices) * 40 + 50, 600),
    )

    # Afficher les informations de pagination
    if total_pages > 1:
        st.caption(
            f"Page {page}/{total_pages} — "
            f"Factures {start_idx + 1} à {min(end_idx, total_invoices)} "
            f"sur {total_invoices}"
        )


def show_dashboard():
    """Fonction principale du dashboard. Appelée depuis app.py."""
    st.title("📊 Dashboard - Détection de Fraude")
    st.markdown("---")

    # Récupérer les données depuis l'API
    stats = _fetch_stats()
    invoices = _fetch_invoices()

    # Gestion de l'état vide
    if stats is None:
        st.error("Impossible de se connecter à l'API. Vérifiez que le serveur est en cours d'exécution.")
        return

    if stats.get("total_invoices", 0) == 0:
        # État vide: afficher KPIs à zéro et message
        _render_kpi_cards(stats)
        st.markdown("---")
        st.info("📭 Aucune facture analysée")
        return

    # Afficher les KPIs
    _render_kpi_cards(stats)
    st.markdown("---")

    # Graphiques côte à côte
    col_pie, col_line = st.columns(2)

    with col_pie:
        _render_pie_chart(invoices)

    with col_line:
        _render_line_chart(invoices)

    st.markdown("---")

    # Tableau des factures
    st.subheader("📋 Factures Analysées")
    _render_invoice_table(invoices)
