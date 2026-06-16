"""
Application Streamlit autonome - Détection de fraude sur factures.
Version déployable sur Streamlit Community Cloud (sans FastAPI).

Appelle directement les modules ML/OCR/DB au lieu de passer par HTTP.
"""

import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.database import Base, Invoice, SessionLocal, User, engine, init_db
from backend.auth import (
    authenticate,
    check_lockout,
    create_token,
    seed_demo_account,
    verify_token,
)
from ml.feature_engineering import FeatureEngine
from ml.fraud_detector import FraudDetector, COMPARISON_PATH, MODEL_PATH

# --- Résolution des chemins absolus ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ABS_MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_PATH)
ABS_COMPARISON_PATH = os.path.join(PROJECT_ROOT, COMPARISON_PATH)

# --- Configuration ---

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ACCEPTED_TYPES = ["pdf", "jpg", "jpeg", "png"]
ROWS_PER_PAGE = 50

# Configuration de la page
st.set_page_config(
    page_title="Détection de Fraude - Factures",
    page_icon="🔍",
    layout="wide",
)


# --- Initialisation DB (une seule fois) ---

@st.cache_resource
def init_application():
    """Initialise la base de données et le compte démo."""
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    init_db()
    db = SessionLocal()
    try:
        seed_demo_account(db)
    finally:
        db.close()
    return True


init_application()


# --- Fonctions utilitaires ---


def get_db_session():
    """Crée une nouvelle session DB."""
    return SessionLocal()


def is_authenticated() -> bool:
    """Vérifie si l'utilisateur est authentifié."""
    return bool(st.session_state.get("token", ""))


def do_logout():
    """Déconnecte l'utilisateur."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def _format_amount_mad(amount: float) -> str:
    """Formate un montant en MAD avec séparateur de milliers."""
    if amount is None or (isinstance(amount, float) and math.isnan(amount)):
        return "0,00 MAD"
    integer_part = int(amount)
    decimal_part = round((amount - integer_part) * 100)
    formatted_integer = f"{integer_part:,}".replace(",", " ")
    return f"{formatted_integer},{decimal_part:02d} MAD"


def _get_status_color(status: str) -> str:
    """Retourne la couleur CSS associée au statut."""
    colors = {
        "Normal": "#28a745",
        "Suspect": "#fd7e14",
        "Frauduleux": "#dc3545",
    }
    return colors.get(status, "#6c757d")


# --- Page: Connexion ---


def show_login():
    """Page de connexion avec authentification directe."""
    st.title("🔐 Détection de Fraude - Connexion")
    st.markdown("---")

    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")

        if submitted:
            if not username or not password:
                st.warning("Veuillez remplir tous les champs.")
            else:
                db = get_db_session()
                try:
                    token = authenticate(db, username, password)
                    if token:
                        st.session_state["token"] = token
                        st.session_state["username"] = username
                        st.rerun()
                    else:
                        st.error("Identifiants incorrects")
                finally:
                    db.close()

    st.markdown("---")
    st.caption("Compte démo : admin / admin123")


# --- Page: Tableau de bord ---


def show_dashboard():
    """Dashboard avec KPIs, graphiques et tableau des factures."""
    st.title("📊 Dashboard - Détection de Fraude")
    st.markdown("---")

    db = get_db_session()
    try:
        # Récupérer les stats
        invoices = db.query(Invoice).filter(Invoice.fraud_score.isnot(None)).all()
        total_invoices = len(invoices)

        if total_invoices == 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Total Factures", "0")
            with col2:
                st.metric("⚠️ Taux de Fraude", "0.0%")
            with col3:
                st.metric("💰 Montant Total", "0,00 MAD")
            st.markdown("---")
            st.info("📭 Aucune facture analysée")
            return

        # Calculer les KPIs
        fraud_count = sum(
            1 for inv in invoices
            if inv.fraud_label in ("Suspect", "Frauduleux")
        )
        fraud_rate = (fraud_count / total_invoices) * 100
        total_amount = sum(inv.amount_ttc or 0 for inv in invoices)

        # Afficher KPIs
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Total Factures", str(total_invoices))
        with col2:
            st.metric("⚠️ Taux de Fraude", f"{fraud_rate:.1f}%")
        with col3:
            st.metric("💰 Montant Total", _format_amount_mad(total_amount))

        st.markdown("---")

        # Graphiques
        col_pie, col_line = st.columns(2)

        with col_pie:
            _render_pie_chart(invoices)

        with col_line:
            _render_line_chart(invoices)

        st.markdown("---")

        # Tableau des factures
        st.subheader("📋 Factures Analysées")
        _render_invoice_table(invoices)

    finally:
        db.close()


def _render_pie_chart(invoices):
    """Graphique camembert de la distribution des statuts."""
    status_counts = Counter(inv.fraud_label for inv in invoices if inv.fraud_label)

    labels = []
    values = []
    colors = []

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


def _render_line_chart(invoices):
    """Graphique en ligne du score de fraude moyen par jour."""
    data = []
    for inv in invoices:
        if inv.date and inv.fraud_score is not None:
            data.append({"date": inv.date, "fraud_score": inv.fraud_score})

    if not data:
        st.info("Aucune donnée disponible pour le graphique temporel.")
        return

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        st.info("Aucune date valide pour le graphique.")
        return

    daily_avg = df.groupby(df["date"].dt.date)["fraud_score"].mean().reset_index()
    daily_avg.columns = ["date", "score_moyen"]
    daily_avg = daily_avg.sort_values("date")

    fig = px.line(
        daily_avg, x="date", y="score_moyen",
        title="Score de Fraude Moyen par Jour",
        labels={"date": "Date", "score_moyen": "Score Fraude Moyen"},
        markers=True,
    )
    fig.update_yaxes(range=[0.0, 1.0])
    fig.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=400)
    st.plotly_chart(fig, use_container_width=True)


def _render_invoice_table(invoices):
    """Tableau des factures avec pagination et couleurs."""
    sorted_invoices = sorted(
        invoices,
        key=lambda x: x.date or "",
        reverse=True,
    )

    total = len(sorted_invoices)
    total_pages = math.ceil(total / ROWS_PER_PAGE)

    if total_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    else:
        page = 1

    start_idx = (page - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE
    page_invoices = sorted_invoices[start_idx:end_idx]

    table_data = []
    for inv in page_invoices:
        table_data.append({
            "ID": inv.invoice_id or str(inv.id),
            "Fournisseur": inv.vendor_name or "—",
            "Montant (MAD)": _format_amount_mad(inv.amount_ttc or 0),
            "Date": inv.date or "—",
            "Score fraude": f"{inv.fraud_score:.2f}" if inv.fraud_score else "—",
            "Statut": inv.fraud_label or "—",
        })

    df = pd.DataFrame(table_data)

    def _color_status(val):
        color_map = {
            "Normal": "background-color: #d4edda; color: #155724;",
            "Suspect": "background-color: #fff3cd; color: #856404;",
            "Frauduleux": "background-color: #f8d7da; color: #721c24;",
        }
        return color_map.get(val, "")

    styled_df = df.style.map(_color_status, subset=["Statut"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    if total_pages > 1:
        st.caption(f"Page {page}/{total_pages} — {start_idx+1} à {min(end_idx, total)} sur {total}")


# --- Page: Upload & Analyse ---


def show_upload():
    """Page d'upload et d'analyse directe (sans passer par l'API)."""
    st.title("📄 Upload & Analyse de Facture")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Glissez-déposez une facture ou cliquez pour sélectionner",
        type=ACCEPTED_TYPES,
        help=f"Formats acceptés: PDF, JPG, PNG. Taille maximale: {MAX_FILE_SIZE_MB} Mo.",
    )

    if uploaded_file is not None:
        file_size_bytes = uploaded_file.size
        file_size_mb = file_size_bytes / (1024 * 1024)
        file_extension = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""

        if file_extension not in ACCEPTED_TYPES:
            st.error(f"❌ Format non supporté: .{file_extension}. Formats acceptés: PDF, JPG, PNG.")
            return

        if file_size_bytes > MAX_FILE_SIZE_BYTES:
            st.error(f"❌ Fichier trop volumineux: {file_size_mb:.1f} Mo. Maximum: {MAX_FILE_SIZE_MB} Mo.")
            return

        st.success(f"✅ Fichier: **{uploaded_file.name}** ({file_size_mb:.2f} Mo)")

        if st.button("🔍 Analyser", type="primary", use_container_width=True):
            _run_analysis_direct(uploaded_file, file_extension)


def _run_analysis_direct(uploaded_file, file_extension):
    """Exécute l'analyse directement via les modules Python."""
    progress_bar = st.progress(0)
    stage_text = st.empty()

    try:
        # Stage 1: Sauvegarde du fichier
        stage_text.markdown("⏳ **Stage 1/3**: Extraction OCR...")
        progress_bar.progress(0.25)

        # Sauvegarder temporairement le fichier
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"temp_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            uploaded_file.seek(0)
            f.write(uploaded_file.read())

        # Extraction OCR
        from ocr.pipeline import OCRPipeline
        ocr_pipeline = OCRPipeline()
        ocr_result = ocr_pipeline.extract(file_path)

        # Stage 2: Feature Engineering
        stage_text.markdown("⏳ **Stage 2/3**: Feature Engineering...")
        progress_bar.progress(0.5)

        invoice_data = {
            "amount": ocr_result.amount,
            "vendor": ocr_result.vendor_name,
            "date": ocr_result.date,
            "tax_rate": ocr_result.tax_rate,
        }

        db = get_db_session()
        try:
            feature_engine = FeatureEngine(session=db)
            feature_vector = feature_engine.compute_features(invoice_data)

            # Stage 3: Détection de fraude
            stage_text.markdown("⏳ **Stage 3/3**: Détection de fraude...")
            progress_bar.progress(0.75)

            detector = FraudDetector(model_path=ABS_MODEL_PATH)
            fraud_result = detector.predict(feature_vector)

            # Persistance en DB
            invoice_record = Invoice(
                invoice_id=ocr_result.invoice_id,
                vendor_name=ocr_result.vendor_name,
                amount_ht=ocr_result.amount,
                tax_rate=ocr_result.tax_rate,
                amount_ttc=ocr_result.total,
                date=ocr_result.date,
                file_path=file_path,
                file_type=file_extension,
                fraud_score=fraud_result.fraud_score,
                fraud_label=fraud_result.fraud_label,
                fraud_reason=fraud_result.fraud_reason,
                ocr_confidences=ocr_result.field_confidences,
                analyzed_at=datetime.utcnow(),
            )
            db.add(invoice_record)
            db.commit()
        finally:
            db.close()

        progress_bar.progress(1.0)
        stage_text.markdown("✅ **Analyse terminée avec succès !**")

        # Affichage des résultats
        _display_results(ocr_result, fraud_result)

    except Exception as e:
        progress_bar.empty()
        stage_text.empty()
        st.error(f"❌ Erreur dans le pipeline: {str(e)}")
        st.info("💡 Le fichier a été préservé. Vous pouvez réessayer.")


def _display_results(ocr_result, fraud_result):
    """Affiche les résultats d'analyse."""
    st.markdown("---")
    st.subheader("📋 Résultats de l'analyse")

    col_score, col_label = st.columns(2)

    with col_score:
        st.metric(
            label="Score de fraude",
            value=f"{fraud_result.fraud_score:.2f}",
            delta=fraud_result.fraud_label,
            delta_color="off",
        )

        if fraud_result.fraud_label == "Normal":
            st.markdown('<p style="color: green; font-size: 1.2em; font-weight: bold;">🟢 Normal</p>', unsafe_allow_html=True)
        elif fraud_result.fraud_label == "Suspect":
            st.markdown('<p style="color: orange; font-size: 1.2em; font-weight: bold;">🟠 Suspect</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: red; font-size: 1.2em; font-weight: bold;">🔴 Frauduleux</p>', unsafe_allow_html=True)

    with col_label:
        st.markdown("**Raison de la classification:**")
        st.info(fraud_result.fraud_reason or "Aucune raison disponible")

    st.markdown("---")
    st.subheader("📑 Données extraites")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Fournisseur:**")
        st.write(ocr_result.vendor_name or "Non extrait")
        st.markdown("**N° Facture:**")
        st.write(ocr_result.invoice_id or "Non extrait")
    with col2:
        st.markdown("**Date:**")
        st.write(ocr_result.date or "Non extrait")
        st.markdown("**Taux de taxe:**")
        st.write(f"{ocr_result.tax_rate}%" if ocr_result.tax_rate else "Non extrait")
    with col3:
        st.markdown("**Montant HT:**")
        st.write(f"{ocr_result.amount:,.2f} MAD" if ocr_result.amount else "Non extrait")
        st.markdown("**Montant TTC:**")
        st.write(f"{ocr_result.total:,.2f} MAD" if ocr_result.total else "Non extrait")


# --- Page: Rapport ML ---


def show_ml_report():
    """Rapport comparatif des modèles ML."""
    st.title("🤖 Rapport ML - Comparaison des Modèles")
    st.markdown("---")

    # Charger le rapport
    comparison_path = ABS_COMPARISON_PATH
    if not os.path.exists(comparison_path):
        st.warning("⚠️ Rapport non disponible. Exécutez `python ml/train.py` pour générer le rapport.")
        return

    try:
        with open(comparison_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, IOError):
        st.error("Erreur de lecture du fichier de rapport.")
        return

    models = report.get("models", [])
    if not models:
        st.warning("Aucun modèle dans le rapport.")
        return

    # Tableau comparatif
    st.subheader("📊 Métriques de Performance")

    metrics_data = []
    for model in models:
        metrics_data.append({
            "Modèle": model["name"],
            "F1 Score": model["f1_score"],
            "Precision": model["precision"],
            "Recall": model["recall"],
            "AUC-ROC": model["auc_roc"],
        })

    df = pd.DataFrame(metrics_data)

    # Mise en surbrillance des meilleures valeurs
    def highlight_max(s):
        is_max = s == s.max()
        return [
            "background-color: #d4edda; font-weight: bold" if v else ""
            for v in is_max
        ]

    metric_cols = ["F1 Score", "Precision", "Recall", "AUC-ROC"]
    styled_df = df.style.apply(highlight_max, subset=metric_cols)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Matrices de confusion
    st.subheader("🔢 Matrices de Confusion")

    cols = st.columns(len(models))
    for i, model in enumerate(models):
        with cols[i]:
            cm = model["confusion_matrix"]
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Normal", "Frauduleux"],
                y=["Normal", "Frauduleux"],
                text=[[str(v) for v in row] for row in cm],
                texttemplate="%{text}",
                colorscale="RdYlGn_r",
                showscale=False,
            ))
            fig.update_layout(
                title=model["name"],
                xaxis_title="Prédit",
                yaxis_title="Réel",
                height=300,
                margin=dict(t=40, b=40, l=40, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)


# --- Navigation principale ---


def main():
    """Point d'entrée principal."""
    if is_authenticated():
        with st.sidebar:
            st.markdown(f"👤 Connecté: **{st.session_state.get('username', 'Utilisateur')}**")
            st.markdown("---")
            page = st.radio(
                "Navigation",
                options=["Tableau de bord", "Upload & Analyse", "Rapport ML"],
                index=0,
            )
            st.markdown("---")
            if st.button("Déconnexion", use_container_width=True):
                do_logout()

        if page == "Tableau de bord":
            show_dashboard()
        elif page == "Upload & Analyse":
            show_upload()
        elif page == "Rapport ML":
            show_ml_report()
    else:
        show_login()


if __name__ == "__main__":
    main()
