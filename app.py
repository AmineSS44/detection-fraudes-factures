"""
Application Streamlit principale - Détection de fraude sur factures.
Dashboard multi-pages avec navigation latérale.

Pages:
- Connexion (login)
- Tableau de bord (dashboard avec KPIs)
- Upload & Analyse (upload de factures et analyse)
- Rapport ML (comparaison des modèles)
"""

import time

import requests
import streamlit as st

# URL de base de l'API backend
API_BASE_URL = "http://127.0.0.1:8000"

# Taille maximale de fichier (10 Mo)
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Formats de fichiers acceptés
ACCEPTED_TYPES = ["pdf", "jpg", "jpeg", "png"]

# Configuration de la page Streamlit (doit être le premier appel st.)
st.set_page_config(
    page_title="Détection de Fraude - Factures",
    page_icon="🔍",
    layout="wide",
)


# --- Fonctions utilitaires ---


def get_auth_headers() -> dict:
    """Retourne les headers d'authentification avec le token JWT."""
    token = st.session_state.get("token", "")
    return {"Authorization": f"Bearer {token}"}


def is_authenticated() -> bool:
    """Vérifie si l'utilisateur est authentifié (token présent en session)."""
    token = st.session_state.get("token", "")
    return bool(token)


def validate_token() -> bool:
    """Vérifie si le token JWT stocké en session est toujours valide.

    Effectue un appel à GET /api/stats pour vérifier que le token
    n'est pas expiré. Retourne True si valide, False sinon.
    """
    if not is_authenticated():
        return False

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/stats",
            headers=get_auth_headers(),
            timeout=5,
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # Si le serveur n'est pas joignable, on garde le token en session
        # mais on ne peut pas valider → on considère valide pour ne pas bloquer
        return True


def do_logout():
    """Déconnecte l'utilisateur en effaçant la session et relançant l'app."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# --- Page: Connexion ---


def show_login():
    """Affiche la page de connexion avec formulaire username/password.

    Appelle POST /api/login et stocke le JWT token en session.
    Affiche "Identifiants incorrects" si les identifiants sont invalides.
    Redirige automatiquement vers le dashboard si un token valide existe.
    """
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
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/login",
                        json={"username": username, "password": password},
                        timeout=10,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        # Stocker le token JWT et le username en session
                        st.session_state["token"] = data["token"]
                        st.session_state["username"] = username
                        # Rediriger vers le dashboard
                        st.rerun()
                    else:
                        # Afficher le message d'erreur spécifié dans les requirements
                        st.error("Identifiants incorrects")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Impossible de se connecter au serveur API. "
                        "Vérifiez que le backend est lancé sur http://127.0.0.1:8000"
                    )
                except requests.exceptions.RequestException:
                    st.error("Erreur de communication avec le serveur.")

    st.markdown("---")
    st.caption("Compte démo : admin / admin123")


# --- Page: Tableau de bord ---


def show_dashboard():
    """Affiche le dashboard avec KPIs, graphiques et tableau des factures.

    Placeholder - sera implémenté dans la tâche 11.2.
    """
    st.title("📊 Tableau de Bord")
    st.info("Le tableau de bord sera implémenté prochainement.")


# --- Page: Upload & Analyse ---


def show_upload():
    """Affiche la page d'upload et d'analyse de factures.

    Fonctionnalités:
    - Zone d'upload drag-and-drop (PDF, JPG, PNG, max 10 Mo)
    - Validation client-side du format et de la taille
    - Bouton "Analyser" pour déclencher le pipeline
    - Barre de progression avec les stages du pipeline
    - Affichage des résultats: données extraites, fraud_score, fraud_label, fraud_reason
    - Gestion des erreurs avec indication du stage en échec
    """
    st.title("📄 Upload & Analyse de Facture")
    st.markdown("---")

    # Zone d'upload de fichier (drag-and-drop natif Streamlit)
    uploaded_file = st.file_uploader(
        "Glissez-déposez une facture ou cliquez pour sélectionner",
        type=ACCEPTED_TYPES,
        help=f"Formats acceptés: PDF, JPG, PNG. Taille maximale: {MAX_FILE_SIZE_MB} Mo.",
    )

    # Validation et affichage des informations du fichier
    if uploaded_file is not None:
        file_size_bytes = uploaded_file.size
        file_size_mb = file_size_bytes / (1024 * 1024)
        file_extension = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""

        # Validation du format côté client
        if file_extension not in ACCEPTED_TYPES:
            st.error(
                f"❌ Format non supporté: .{file_extension}. "
                f"Formats acceptés: PDF, JPG, PNG."
            )
            return

        # Validation de la taille côté client
        if file_size_bytes > MAX_FILE_SIZE_BYTES:
            st.error(
                f"❌ Fichier trop volumineux: {file_size_mb:.1f} Mo. "
                f"Taille maximale autorisée: {MAX_FILE_SIZE_MB} Mo."
            )
            return

        # Affichage des informations du fichier valide
        st.success(
            f"✅ Fichier sélectionné: **{uploaded_file.name}** "
            f"({file_size_mb:.2f} Mo, format .{file_extension})"
        )

        # Bouton "Analyser" pour déclencher le pipeline
        if st.button("🔍 Analyser", type="primary", use_container_width=True):
            _run_analysis(uploaded_file)


def _run_analysis(uploaded_file):
    """Exécute l'analyse de la facture via l'API avec barre de progression.

    Affiche les 3 stages du pipeline avec progression visuelle,
    puis affiche les résultats ou une erreur en cas d'échec.

    Args:
        uploaded_file: Fichier uploadé via st.file_uploader.
    """
    # Définition des stages du pipeline pour la barre de progression
    stages = [
        "Extraction OCR",
        "Feature Engineering",
        "Détection de fraude",
    ]

    # Conteneurs pour la progression et les résultats
    progress_container = st.container()
    results_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        stage_text = st.empty()

        # Simulation de la progression (les stages sont exécutés côté API)
        for i, stage_name in enumerate(stages):
            stage_text.markdown(f"⏳ **Stage {i + 1}/{len(stages)}**: {stage_name}...")
            progress_bar.progress((i + 1) / (len(stages) + 1))

            if i < len(stages) - 1:
                time.sleep(0.3)

        # Appel API pour l'analyse complète
        stage_text.markdown("⏳ **Envoi au serveur et traitement en cours...**")

        try:
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            headers = get_auth_headers()

            response = requests.post(
                f"{API_BASE_URL}/api/upload",
                files=files,
                headers=headers,
                timeout=120,
            )

            progress_bar.progress(1.0)

            if response.status_code == 200:
                stage_text.markdown("✅ **Analyse terminée avec succès !**")
                data = response.json()
                _display_results(results_container, data)

            elif response.status_code == 401:
                stage_text.empty()
                progress_bar.empty()
                st.error("❌ Session expirée. Veuillez vous reconnecter.")

            elif response.status_code == 400:
                stage_text.empty()
                progress_bar.empty()
                error_detail = response.json().get("detail", "Erreur de validation")
                st.error(f"❌ Erreur de validation: {error_detail}")

            elif response.status_code == 422:
                stage_text.empty()
                progress_bar.empty()
                error_detail = response.json().get("detail", "Erreur dans le pipeline")
                _display_pipeline_error(error_detail)

            elif response.status_code == 500:
                stage_text.empty()
                progress_bar.empty()
                error_detail = response.json().get("detail", "Erreur interne du serveur")
                _display_pipeline_error(error_detail)

            else:
                stage_text.empty()
                progress_bar.empty()
                st.error(f"❌ Erreur inattendue (HTTP {response.status_code})")

        except requests.exceptions.ConnectionError:
            progress_bar.empty()
            stage_text.empty()
            st.error(
                "❌ Impossible de se connecter au serveur API. "
                "Vérifiez que le backend est en cours d'exécution."
            )
        except requests.exceptions.Timeout:
            progress_bar.empty()
            stage_text.empty()
            st.error("❌ Le serveur a mis trop de temps à répondre. Réessayez plus tard.")


def _display_pipeline_error(error_detail: str):
    """Affiche une erreur de pipeline avec le nom du stage en échec.

    Args:
        error_detail: Message d'erreur retourné par l'API.
    """
    if "Extraction OCR" in error_detail:
        failed_stage = "Extraction OCR"
    elif "Feature Engineering" in error_detail:
        failed_stage = "Feature Engineering"
    elif "Détection de fraude" in error_detail:
        failed_stage = "Détection de fraude"
    else:
        failed_stage = "Inconnu"

    st.error(
        f"❌ **Échec du pipeline au stage: {failed_stage}**\n\n"
        f"Détails: {error_detail}"
    )
    st.info("💡 Le fichier a été préservé. Vous pouvez réessayer l'analyse.")


def _display_results(container, data: dict):
    """Affiche les résultats de l'analyse de fraude.

    Args:
        container: Conteneur Streamlit pour l'affichage.
        data: Réponse JSON de l'API.
    """
    with container:
        st.markdown("---")
        st.subheader("📋 Résultats de l'analyse")

        invoice_data = data.get("invoice_data", {})
        fraud_score = data.get("fraud_score", 0.0)
        fraud_label = data.get("fraud_label", "Inconnu")
        fraud_reason = data.get("fraud_reason", "")

        # Score et label de fraude
        col_score, col_label = st.columns(2)

        with col_score:
            if fraud_label == "Normal":
                delta_color = "off"
            elif fraud_label == "Suspect":
                delta_color = "off"
            else:
                delta_color = "off"

            st.metric(
                label="Score de fraude",
                value=f"{fraud_score:.2f}",
                delta=fraud_label,
                delta_color=delta_color,
            )

            if fraud_label == "Normal":
                st.markdown(
                    '<p style="color: green; font-size: 1.2em; font-weight: bold;">'
                    f'🟢 {fraud_label}</p>',
                    unsafe_allow_html=True,
                )
            elif fraud_label == "Suspect":
                st.markdown(
                    '<p style="color: orange; font-size: 1.2em; font-weight: bold;">'
                    f'🟠 {fraud_label}</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p style="color: red; font-size: 1.2em; font-weight: bold;">'
                    f'🔴 {fraud_label}</p>',
                    unsafe_allow_html=True,
                )

        with col_label:
            st.markdown("**Raison de la classification:**")
            st.info(fraud_reason if fraud_reason else "Aucune raison disponible")

        # Données extraites de la facture
        st.markdown("---")
        st.subheader("📑 Données extraites de la facture")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Fournisseur:**")
            st.write(invoice_data.get("vendor_name") or "Non extrait")

            st.markdown("**N° Facture:**")
            st.write(invoice_data.get("invoice_id") or "Non extrait")

        with col2:
            st.markdown("**Date:**")
            st.write(invoice_data.get("date") or "Non extrait")

            st.markdown("**Taux de taxe:**")
            tax_rate = invoice_data.get("tax_rate")
            st.write(f"{tax_rate}%" if tax_rate is not None else "Non extrait")

        with col3:
            st.markdown("**Montant HT:**")
            amount_ht = invoice_data.get("amount_ht")
            st.write(f"{amount_ht:,.2f} MAD" if amount_ht is not None else "Non extrait")

            st.markdown("**Montant TTC:**")
            amount_ttc = invoice_data.get("amount_ttc")
            st.write(f"{amount_ttc:,.2f} MAD" if amount_ttc is not None else "Non extrait")


# --- Page: Rapport ML ---


def show_ml_report():
    """Affiche le rapport comparatif des performances des modèles ML.

    Placeholder - sera implémenté dans la tâche 11.4.
    """
    st.title("🤖 Rapport ML")
    st.info("Le rapport ML sera implémenté prochainement.")


# --- Navigation principale ---


def main():
    """Point d'entrée principal de l'application Streamlit.

    Gère la navigation entre les pages selon l'état d'authentification.
    - Si token valide en session → redirection automatique vers dashboard
    - Si pas de token → affichage de la page de connexion
    - Bouton "Déconnexion" dans la sidebar pour effacer la session
    """
    # Auto-redirect: si un token valide existe, afficher le dashboard
    if is_authenticated() and validate_token():
        # Sidebar avec navigation et bouton déconnexion
        with st.sidebar:
            st.markdown(
                f"👤 Connecté: **{st.session_state.get('username', 'Utilisateur')}**"
            )
            st.markdown("---")

            # Menu de navigation
            page = st.radio(
                "Navigation",
                options=["Tableau de bord", "Upload & Analyse", "Rapport ML"],
                index=0,
            )

            st.markdown("---")

            # Bouton de déconnexion
            if st.button("Déconnexion", use_container_width=True):
                do_logout()

        # Affichage de la page sélectionnée
        if page == "Tableau de bord":
            show_dashboard()
        elif page == "Upload & Analyse":
            show_upload()
        elif page == "Rapport ML":
            show_ml_report()
    else:
        # Token absent ou invalide → afficher la page de connexion
        if "token" in st.session_state and st.session_state["token"] is not None:
            # Token expiré ou invalide, on le nettoie
            st.session_state["token"] = None
        show_login()


if __name__ == "__main__":
    main()
