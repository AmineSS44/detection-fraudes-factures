"""
Point d'entrée principal de l'application.
Lance le serveur FastAPI (uvicorn) et l'application Streamlit en parallèle.

Usage:
    python main.py              # Lance les deux services
    python main.py --api-only   # Lance uniquement l'API FastAPI
    python main.py --ui-only    # Lance uniquement le dashboard Streamlit
"""

import argparse
import os
import subprocess
import sys
import threading
import time

import uvicorn

# Répertoire racine du projet
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Configuration par défaut des serveurs
API_HOST = "127.0.0.1"
API_PORT = 8000
STREAMLIT_PORT = 8501

# Chemin de l'application Streamlit
STREAMLIT_APP_PATH = os.path.join(PROJECT_ROOT, "app.py")


def init_database():
    """Initialise la base de données et crée le compte démo au premier lancement."""
    from backend.database import init_db, SessionLocal
    from backend.auth import seed_demo_account

    # Créer les tables si elles n'existent pas
    init_db()

    # Créer le compte démo (admin/admin123) si absent
    db = SessionLocal()
    try:
        seed_demo_account(db)
    finally:
        db.close()

    # Créer le répertoire uploads si nécessaire
    uploads_dir = os.path.join(PROJECT_ROOT, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    # Créer le répertoire data si nécessaire
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Créer le répertoire results si nécessaire
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Créer le répertoire models si nécessaire
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("[init] Base de données initialisée, compte démo créé (admin/admin123)")


def run_api_server(host: str = API_HOST, port: int = API_PORT):
    """Lance le serveur FastAPI avec uvicorn.

    Args:
        host: Adresse d'écoute.
        port: Port d'écoute.
    """
    print(f"[API] Démarrage du serveur FastAPI sur http://{host}:{port}")
    uvicorn.run(
        "backend.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


def run_streamlit(port: int = STREAMLIT_PORT):
    """Lance l'application Streamlit dans un sous-processus.

    Args:
        port: Port d'écoute pour Streamlit.
    """
    if not os.path.exists(STREAMLIT_APP_PATH):
        print(f"[UI] Fichier Streamlit introuvable: {STREAMLIT_APP_PATH}")
        print("[UI] Le dashboard Streamlit n'est pas encore implémenté.")
        return

    print(f"[UI] Démarrage du dashboard Streamlit sur http://127.0.0.1:{port}")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        STREAMLIT_APP_PATH,
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Attendre que le processus se termine ou qu'on l'arrête
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
    except FileNotFoundError:
        print("[UI] Streamlit n'est pas installé. Installez-le avec: pip install streamlit")


def main():
    """Point d'entrée principal: parse les arguments et lance les services."""
    parser = argparse.ArgumentParser(
        description="Lance l'application de détection de fraude sur factures"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Lance uniquement le serveur API FastAPI",
    )
    parser.add_argument(
        "--ui-only",
        action="store_true",
        help="Lance uniquement le dashboard Streamlit",
    )
    parser.add_argument(
        "--api-host",
        type=str,
        default=API_HOST,
        help=f"Adresse d'écoute de l'API (défaut: {API_HOST})",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=API_PORT,
        help=f"Port de l'API (défaut: {API_PORT})",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=STREAMLIT_PORT,
        help=f"Port du dashboard Streamlit (défaut: {STREAMLIT_PORT})",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Détection de Fraude sur Factures - Bureau Comptable")
    print("=" * 60)

    # Initialisation de la base de données et du compte démo
    init_database()

    if args.api_only:
        # Mode API uniquement
        run_api_server(host=args.api_host, port=args.api_port)

    elif args.ui_only:
        # Mode UI uniquement
        run_streamlit(port=args.ui_port)

    else:
        # Mode complet: lancer les deux services en parallèle
        print(f"\n[info] API:       http://{args.api_host}:{args.api_port}")
        print(f"[info] Dashboard: http://127.0.0.1:{args.ui_port}")
        print("[info] Ctrl+C pour arrêter les deux services\n")

        # Lancer Streamlit dans un thread séparé
        streamlit_thread = threading.Thread(
            target=run_streamlit,
            args=(args.ui_port,),
            daemon=True,
        )
        streamlit_thread.start()

        # Petit délai pour que Streamlit démarre avant FastAPI
        time.sleep(1)

        # Lancer FastAPI dans le thread principal (bloquant)
        try:
            run_api_server(host=args.api_host, port=args.api_port)
        except KeyboardInterrupt:
            print("\n[info] Arrêt des services...")


if __name__ == "__main__":
    main()
