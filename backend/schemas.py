"""
Schémas Pydantic pour la validation des requêtes/réponses API.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """Requête de connexion utilisateur."""

    username: str = Field(max_length=50)
    password: str = Field(max_length=128)


class LoginResponse(BaseModel):
    """Réponse après authentification réussie."""

    token: str
    expires_at: str  # Format ISO 8601


class InvoiceResponse(BaseModel):
    """Réponse pour une facture analysée."""

    id: int
    invoice_id: Optional[str] = None
    vendor_name: Optional[str] = None
    amount_ht: Optional[float] = None
    tax_rate: Optional[float] = None
    amount_ttc: Optional[float] = None
    date: Optional[str] = None
    fraud_score: float
    fraud_label: str
    fraud_reason: str


class UploadResponse(BaseModel):
    """Réponse après upload et analyse d'une facture."""

    invoice_data: dict
    fraud_score: float
    fraud_label: str
    fraud_reason: str


class StatsResponse(BaseModel):
    """Réponse avec les KPIs agrégés."""

    total_invoices: int
    fraud_rate: float  # Pourcentage 0-100
    total_amount: float  # Montant total en MAD


class ModelMetrics(BaseModel):
    """Métriques de performance d'un modèle ML."""

    name: str
    f1_score: float
    precision: float
    recall: float
    auc_roc: float
    confusion_matrix: List[List[int]]  # [[TP, FP], [FN, TN]]


class ModelReportResponse(BaseModel):
    """Réponse avec le rapport comparatif des modèles ML."""

    models: List[ModelMetrics]


class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée."""

    error: str  # Code d'erreur court
    message: str  # Message lisible en français
    details: Optional[dict] = None  # Détails supplémentaires
    stage: Optional[str] = None  # Stage du pipeline en échec
