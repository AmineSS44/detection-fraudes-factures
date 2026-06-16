"""
Routes FastAPI pour l'API backend de détection de fraude.
Expose les endpoints d'authentification, upload, consultation et statistiques.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, Header, UploadFile
from sqlalchemy.orm import Session

from backend.auth import authenticate, create_token, seed_demo_account, verify_token
from backend.database import Invoice, SessionLocal, get_db, init_db
from backend.schemas import (
    ErrorResponse,
    InvoiceResponse,
    LoginRequest,
    LoginResponse,
    ModelReportResponse,
    StatsResponse,
    UploadResponse,
)

# Formats de fichiers acceptés
ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}

# Taille maximale de fichier: 10 Mo
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB en octets

# Répertoire de stockage des fichiers uploadés
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")

# Chemin du rapport de comparaison des modèles
MODEL_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "model_comparison.json"
)

# Création de l'application FastAPI
app = FastAPI(
    title="Invoice Fraud Detection API",
    description="API de détection de fraude sur factures pour bureau comptable marocain",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    """Initialise la base de données et crée le compte démo au démarrage."""
    init_db()
    db = SessionLocal()
    try:
        seed_demo_account(db)
    finally:
        db.close()
    # Créer le répertoire uploads si nécessaire
    os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Dépendance JWT pour les endpoints protégés ---


def get_current_user(authorization: str = Header(None)) -> str:
    """Dépendance FastAPI pour valider le JWT dans le header Authorization.

    Vérifie la présence et la validité du token Bearer.
    Retourne le username extrait du token.

    Raises:
        HTTPException 401: Si le token est absent, invalide ou expiré.
    """
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="Token d'authentification manquant",
        )

    # Extraire le token du format "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Format d'autorisation invalide. Utilisez: Bearer <token>",
        )

    token = parts[1]

    try:
        payload = verify_token(token)
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=401,
                detail="Token invalide: claim 'sub' manquant",
            )
        return username
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Token invalide ou expiré",
        )


# --- Validation de fichier ---


def validate_file(file: UploadFile) -> str:
    """Valide le format et la taille du fichier uploadé.

    Args:
        file: Fichier uploadé via FastAPI.

    Returns:
        Extension du fichier (sans le point).

    Raises:
        HTTPException 400: Si le format ou la taille est invalide.
    """
    # Vérification du format via l'extension
    if file.filename is None:
        raise HTTPException(
            status_code=400,
            detail="Nom de fichier manquant",
        )

    extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté: .{extension}. Formats acceptés: PDF, JPG, PNG",
        )

    return extension


# --- Routes ---


@app.post("/api/login", response_model=LoginResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authentifie un utilisateur et retourne un JWT token.

    Accepte username/password, valide les identifiants via le module auth,
    et retourne un token JWT valide 24h en cas de succès.
    """
    token = authenticate(db, request.username, request.password)

    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Identifiants incorrects",
        )

    # Décoder le token pour récupérer l'expiration
    payload = verify_token(token)
    expires_at = datetime.utcfromtimestamp(payload["exp"]).isoformat()

    return LoginResponse(token=token, expires_at=expires_at)


@app.post("/api/upload", response_model=UploadResponse)
def upload_invoice(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload et analyse une facture (PDF/JPG/PNG ≤10MB).

    Déclenche le pipeline complet: validation → OCR → feature engineering → ML.
    Persiste les résultats en base de données.
    """
    # Validation du format
    extension = validate_file(file)

    # Lecture du contenu et vérification de la taille
    content = file.file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Fichier trop volumineux: {len(content) / (1024*1024):.1f} MB. Maximum: 10 MB",
        )

    # Sauvegarde du fichier sur disque
    file_id = str(uuid.uuid4())
    filename = f"{file_id}.{extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)

    # Variable pour tracer le stage en cours du pipeline
    current_stage = "Extraction OCR"

    try:
        # Étape 1: Extraction OCR
        from ocr.pipeline import OCRPipeline

        ocr_pipeline = OCRPipeline()
        ocr_result = ocr_pipeline.extract(file_path)

        # Préparer les données extraites pour le feature engineering
        invoice_data = {
            "amount": ocr_result.amount,
            "vendor": ocr_result.vendor_name,
            "date": ocr_result.date,
            "tax_rate": ocr_result.tax_rate,
        }

        # Étape 2: Feature Engineering
        current_stage = "Feature Engineering"
        from ml.feature_engineering import FeatureEngine

        feature_engine = FeatureEngine(session=db)
        feature_vector = feature_engine.compute_features(invoice_data)

        # Étape 3: Détection de fraude
        current_stage = "Détection de fraude"
        from ml.fraud_detector import FraudDetector

        detector = FraudDetector()
        fraud_result = detector.predict(feature_vector)

        # Étape 4: Persistance en base de données
        current_stage = "Persistance"
        invoice_record = Invoice(
            invoice_id=ocr_result.invoice_id,
            vendor_name=ocr_result.vendor_name,
            amount_ht=ocr_result.amount,
            tax_rate=ocr_result.tax_rate,
            amount_ttc=ocr_result.total,
            date=ocr_result.date,
            file_path=file_path,
            file_type=extension,
            fraud_score=fraud_result.fraud_score,
            fraud_label=fraud_result.fraud_label,
            fraud_reason=fraud_result.fraud_reason,
            ocr_confidences=ocr_result.field_confidences,
            feature_vector={
                "amount_zscore": feature_vector.amount_zscore,
                "tax_inconsistency": feature_vector.tax_inconsistency,
                "duplicate_flag": feature_vector.duplicate_flag,
                "weekend_flag": feature_vector.weekend_flag,
                "round_amount_flag": feature_vector.round_amount_flag,
                "vendor_deviation": feature_vector.vendor_deviation,
            },
            analyzed_at=datetime.utcnow(),
        )
        db.add(invoice_record)
        db.commit()
        db.refresh(invoice_record)

        # Construction de la réponse
        response_invoice_data = {
            "invoice_id": ocr_result.invoice_id,
            "vendor_name": ocr_result.vendor_name,
            "amount_ht": ocr_result.amount,
            "tax_rate": ocr_result.tax_rate,
            "amount_ttc": ocr_result.total,
            "date": ocr_result.date,
        }

        return UploadResponse(
            invoice_data=response_invoice_data,
            fraud_score=fraud_result.fraud_score,
            fraud_label=fraud_result.fraud_label,
            fraud_reason=fraud_result.fraud_reason,
        )

    except ValueError as e:
        # Erreur dans le pipeline (champs manquants, features invalides)
        raise HTTPException(
            status_code=422,
            detail=f"Échec au stage '{current_stage}': {str(e)}",
        )
    except (RuntimeError, FileNotFoundError) as e:
        # Erreur OCR ou fichier introuvable
        raise HTTPException(
            status_code=500,
            detail=f"Échec au stage '{current_stage}': {str(e)}",
        )
    except Exception as e:
        # Erreur inattendue dans le pipeline
        raise HTTPException(
            status_code=500,
            detail=f"Échec au stage '{current_stage}': {str(e)}",
        )


@app.get("/api/invoices", response_model=List[InvoiceResponse])
def get_invoices(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retourne toutes les factures analysées avec les résultats de fraude.

    Endpoint protégé par JWT. Retourne la liste complète des factures
    avec fraud_score, fraud_label, fraud_reason et montants en MAD.
    """
    invoices = (
        db.query(Invoice)
        .filter(Invoice.fraud_score.isnot(None))
        .order_by(Invoice.created_at.desc())
        .all()
    )

    return [
        InvoiceResponse(
            id=inv.id,
            invoice_id=inv.invoice_id,
            vendor_name=inv.vendor_name,
            amount_ht=inv.amount_ht,
            tax_rate=inv.tax_rate,
            amount_ttc=inv.amount_ttc,
            date=inv.date,
            fraud_score=inv.fraud_score,
            fraud_label=inv.fraud_label or "",
            fraud_reason=inv.fraud_reason or "",
        )
        for inv in invoices
    ]


@app.get("/api/stats", response_model=StatsResponse)
def get_stats(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retourne les KPIs agrégés: total factures, taux de fraude, montant total.

    - total_invoices: nombre total de factures analysées
    - fraud_rate: (Suspect + Frauduleux) / total * 100
    - total_amount: somme de tous les amount_ttc en MAD
    """
    # Compter toutes les factures analysées
    total_invoices = (
        db.query(Invoice)
        .filter(Invoice.fraud_score.isnot(None))
        .count()
    )

    if total_invoices == 0:
        return StatsResponse(
            total_invoices=0,
            fraud_rate=0.0,
            total_amount=0.0,
        )

    # Compter les factures suspectes et frauduleuses
    fraud_count = (
        db.query(Invoice)
        .filter(
            Invoice.fraud_score.isnot(None),
            Invoice.fraud_label.in_(["Suspect", "Frauduleux"]),
        )
        .count()
    )

    # Calculer le taux de fraude en pourcentage
    fraud_rate = (fraud_count / total_invoices) * 100

    # Somme des montants TTC
    from sqlalchemy import func

    total_amount_result = (
        db.query(func.sum(Invoice.amount_ttc))
        .filter(Invoice.fraud_score.isnot(None))
        .scalar()
    )
    total_amount = total_amount_result if total_amount_result is not None else 0.0

    return StatsResponse(
        total_invoices=total_invoices,
        fraud_rate=round(fraud_rate, 1),
        total_amount=round(total_amount, 2),
    )


@app.get("/api/models/report", response_model=ModelReportResponse)
def get_model_report(
    current_user: str = Depends(get_current_user),
):
    """Retourne le rapport comparatif des modèles ML.

    Lit le fichier results/model_comparison.json contenant les métriques
    (F1, Precision, Recall, AUC-ROC, confusion matrix) de chaque modèle.

    Retourne 404 si le fichier est absent ou illisible.
    """
    if not os.path.exists(MODEL_REPORT_PATH):
        raise HTTPException(
            status_code=404,
            detail="Rapport de comparaison des modèles non encore généré",
        )

    try:
        with open(MODEL_REPORT_PATH, "r", encoding="utf-8") as f:
            report_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        raise HTTPException(
            status_code=404,
            detail="Rapport de comparaison des modèles illisible ou corrompu",
        )

    return ModelReportResponse(**report_data)
