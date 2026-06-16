"""
Pipeline OCR pour l'extraction de données de factures.
Utilise YOLOv8 pour la détection de champs et Tesseract comme fallback.
Prétraitement d'image: niveaux de gris, débruitage, redressement.
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None


# Seuil de confiance minimum pour YOLOv8
YOLO_CONFIDENCE_THRESHOLD = 0.5

# Bornes de validation des champs
AMOUNT_MIN = 0.01
AMOUNT_MAX = 999_999_999.99
TAX_RATE_MIN = 0.0
TAX_RATE_MAX = 100.0
INVOICE_ID_MAX_LENGTH = 50
VENDOR_NAME_MAX_LENGTH = 200


@dataclass
class OCRResult:
    """Résultat d'extraction OCR avec les 6 champs de facture.

    Champs:
        invoice_id: Identifiant de facture (max 50 caractères)
        vendor_name: Nom du fournisseur (max 200 caractères)
        amount: Montant HT en MAD (0.01 - 999,999,999.99)
        date: Date au format ISO 8601 YYYY-MM-DD
        tax_rate: Taux de taxe en pourcentage (0 - 100)
        total: Montant TTC en MAD (0.01 - 999,999,999.99)
        field_confidences: Confiance par champ (0.0 - 1.0)
        warnings: Avertissements pour les champs manquants
    """

    invoice_id: Optional[str] = None
    vendor_name: Optional[str] = None
    amount: Optional[float] = None
    date: Optional[str] = None
    tax_rate: Optional[float] = None
    total: Optional[float] = None
    field_confidences: Dict[str, float] = field(default_factory=lambda: {
        "invoice_id": 0.0,
        "vendor_name": 0.0,
        "amount": 0.0,
        "date": 0.0,
        "tax_rate": 0.0,
        "total": 0.0,
    })
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Sérialise le résultat OCR en dictionnaire JSON-compatible."""
        return asdict(self)

    def to_json(self) -> str:
        """Sérialise le résultat OCR en chaîne JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "OCRResult":
        """Désérialise un dictionnaire en OCRResult."""
        return cls(
            invoice_id=data.get("invoice_id"),
            vendor_name=data.get("vendor_name"),
            amount=data.get("amount"),
            date=data.get("date"),
            tax_rate=data.get("tax_rate"),
            total=data.get("total"),
            field_confidences=data.get("field_confidences", {
                "invoice_id": 0.0,
                "vendor_name": 0.0,
                "amount": 0.0,
                "date": 0.0,
                "tax_rate": 0.0,
                "total": 0.0,
            }),
            warnings=data.get("warnings", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "OCRResult":
        """Désérialise une chaîne JSON en OCRResult."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class OCRPipeline:
    """Pipeline OCR complet: prétraitement → YOLOv8 → Tesseract fallback."""

    def __init__(self, yolo_model_path: Optional[str] = None):
        """Initialise le pipeline OCR.

        Args:
            yolo_model_path: Chemin vers le modèle YOLOv8. Si None, utilise
                le chemin par défaut models/yolo_invoice.pt.
        """
        self._yolo_model_path = yolo_model_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            "yolo_invoice.pt",
        )
        self._yolo_model = None

    def _load_yolo_model(self):
        """Charge le modèle YOLOv8 si disponible."""
        if YOLO is None:
            return None
        if not os.path.exists(self._yolo_model_path):
            return None
        try:
            self._yolo_model = YOLO(self._yolo_model_path)
            return self._yolo_model
        except Exception:
            return None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Prétraite l'image: conversion niveaux de gris, débruitage, redressement.

        Args:
            image: Image au format numpy BGR (OpenCV).

        Returns:
            Image prétraitée en niveaux de gris.
        """
        # Conversion en niveaux de gris
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Débruitage avec fastNlMeansDenoising
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Redressement (deskew) basé sur l'angle des contours
        deskewed = self._deskew(denoised)

        return deskewed

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Redresse l'image en détectant l'angle d'inclinaison.

        Args:
            image: Image en niveaux de gris.

        Returns:
            Image redressée.
        """
        # Détection des coordonnées non-nulles pour estimer l'angle
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 10:
            return image

        # Calcul de l'angle via minAreaRect
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Correction de l'angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Ne pas corriger si l'angle est trop faible
        if abs(angle) < 0.5:
            return image

        # Rotation pour redresser
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated

    def detect_fields_yolo(self, image: np.ndarray) -> Dict[str, Any]:
        """Détecte les champs de facture via YOLOv8.

        Retourne un dictionnaire avec les valeurs extraites et les confiances.
        Si le modèle YOLOv8 n'est pas disponible, retourne un dict vide.

        Args:
            image: Image prétraitée (niveaux de gris ou BGR).

        Returns:
            Dict avec clés: 'fields' (valeurs extraites), 'confidences' (scores par champ).
        """
        result = {"fields": {}, "confidences": {}}

        # Charger le modèle YOLOv8
        model = self._load_yolo_model()
        if model is None:
            return result

        try:
            # Exécuter la détection YOLOv8
            detections = model(image, verbose=False)

            if not detections or len(detections) == 0:
                return result

            # Mapper les classes détectées aux champs de facture
            # Les noms de classes dépendent du modèle entraîné
            field_mapping = {
                "invoice_id": "invoice_id",
                "vendor_name": "vendor_name",
                "amount": "amount",
                "date": "date",
                "tax_rate": "tax_rate",
                "total": "total",
            }

            for detection in detections[0].boxes:
                cls_id = int(detection.cls[0])
                confidence = float(detection.conf[0])
                class_name = model.names.get(cls_id, "")

                if class_name in field_mapping:
                    field_name = field_mapping[class_name]
                    result["confidences"][field_name] = confidence

                    # Extraire la région détectée pour OCR
                    if confidence >= YOLO_CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, detection.xyxy[0])
                        roi = image[y1:y2, x1:x2]
                        # Utiliser Tesseract pour lire le texte dans la ROI
                        if pytesseract is not None and roi.size > 0:
                            text = pytesseract.image_to_string(
                                roi, config="--psm 7"
                            ).strip()
                            result["fields"][field_name] = text

        except Exception:
            # En cas d'erreur YOLOv8, retourner un résultat vide
            pass

        return result

    def extract_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        """Extraction par Tesseract comme méthode de fallback.

        Effectue une extraction plein texte et tente de parser les 6 champs.

        Args:
            image: Image prétraitée en niveaux de gris.

        Returns:
            Dict avec clés: 'fields' (valeurs extraites), 'confidences' (scores par champ).
        """
        result = {"fields": {}, "confidences": {}}

        if pytesseract is None:
            return result

        try:
            # Extraction du texte complet avec données de confiance
            ocr_data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT
            )

            # Reconstruire le texte complet
            full_text = pytesseract.image_to_string(image)

            # Calculer la confiance moyenne globale
            confidences = [
                int(c) for c in ocr_data["conf"] if int(c) > 0
            ]
            avg_confidence = (
                sum(confidences) / len(confidences) / 100.0
                if confidences
                else 0.0
            )

            # Parser les champs depuis le texte
            parsed = self._parse_text_fields(full_text)

            for field_name, value in parsed.items():
                if value is not None:
                    result["fields"][field_name] = value
                    # Confiance basée sur la moyenne OCR Tesseract
                    result["confidences"][field_name] = min(avg_confidence, 0.9)

        except Exception:
            pass

        return result

    def _parse_text_fields(self, text: str) -> Dict[str, Optional[str]]:
        """Parse le texte OCR brut pour en extraire les champs de facture.

        Args:
            text: Texte brut extrait par Tesseract.

        Returns:
            Dict avec les champs parsés (ou None si non trouvé).
        """
        fields: Dict[str, Optional[str]] = {
            "invoice_id": None,
            "vendor_name": None,
            "amount": None,
            "date": None,
            "tax_rate": None,
            "total": None,
        }

        lines = text.strip().split("\n")
        full_text_lower = text.lower()

        # Recherche de l'identifiant de facture
        invoice_id_patterns = [
            r"(?:facture|invoice|fact|inv)[\s#.:]*([A-Za-z0-9\-/]+)",
            r"(?:n[°o]|num[ée]ro)[\s.:]*([A-Za-z0-9\-/]+)",
        ]
        for pattern in invoice_id_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["invoice_id"] = match.group(1)[:INVOICE_ID_MAX_LENGTH]
                break

        # Recherche du nom du fournisseur (première ligne significative)
        for line in lines[:5]:
            line_stripped = line.strip()
            if len(line_stripped) > 3 and not re.match(r"^\d", line_stripped):
                fields["vendor_name"] = line_stripped[:VENDOR_NAME_MAX_LENGTH]
                break

        # Recherche de la date (format ISO ou formats courants)
        date_patterns = [
            r"(\d{4}[-/]\d{2}[-/]\d{2})",
            r"(\d{2}[-/]\d{2}[-/]\d{4})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                fields["date"] = self._normalize_date(date_str)
                break

        # Recherche des montants
        amount_patterns = [
            r"(?:montant\s*(?:ht|hors\s*taxe))[\s:]*([0-9\s.,]+)",
            r"(?:sous[\s-]*total|subtotal)[\s:]*([0-9\s.,]+)",
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, full_text_lower)
            if match:
                fields["amount"] = match.group(1).strip()
                break

        # Recherche du total TTC
        total_patterns = [
            r"(?:total\s*(?:ttc|toutes\s*taxes))[\s:]*([0-9\s.,]+)",
            r"(?:net\s*[àa]\s*payer|total)[\s:]*([0-9\s.,]+)",
        ]
        for pattern in total_patterns:
            match = re.search(pattern, full_text_lower)
            if match:
                fields["total"] = match.group(1).strip()
                break

        # Recherche du taux de taxe
        tax_patterns = [
            r"(?:tva|taxe|tax)[\s:]*(\d+(?:[.,]\d+)?)\s*%?",
        ]
        for pattern in tax_patterns:
            match = re.search(pattern, full_text_lower)
            if match:
                fields["tax_rate"] = match.group(1).strip()
                break

        return fields

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalise une date au format ISO 8601 YYYY-MM-DD.

        Args:
            date_str: Date dans un format détecté.

        Returns:
            Date au format YYYY-MM-DD ou None si impossible à parser.
        """
        date_str = date_str.replace("/", "-")

        # Format YYYY-MM-DD
        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                return None

        # Format DD-MM-YYYY
        if re.match(r"\d{2}-\d{2}-\d{4}", date_str):
            try:
                parsed = datetime.strptime(date_str, "%d-%m-%Y")
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                return None

        return None

    def _convert_pdf_to_image(self, file_path: str) -> np.ndarray:
        """Convertit un fichier PDF en image numpy.

        Utilise pdf2image pour convertir la première page du PDF.

        Args:
            file_path: Chemin vers le fichier PDF.

        Returns:
            Image numpy BGR de la première page.

        Raises:
            RuntimeError: Si pdf2image n'est pas installé ou la conversion échoue.
        """
        if convert_from_path is None:
            raise RuntimeError(
                "pdf2image n'est pas installé. "
                "Installez-le avec: pip install pdf2image"
            )

        try:
            # Convertir la première page en image
            pages = convert_from_path(file_path, first_page=1, last_page=1, dpi=300)
            if not pages:
                raise RuntimeError("Aucune page extraite du PDF")

            # Convertir PIL Image en numpy array BGR
            pil_image = pages[0]
            image_rgb = np.array(pil_image)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            return image_bgr
        except Exception as e:
            raise RuntimeError(f"Échec de la conversion PDF: {str(e)}")

    def _validate_and_convert_fields(
        self, raw_fields: Dict[str, Any], confidences: Dict[str, float]
    ) -> OCRResult:
        """Valide et convertit les champs bruts en OCRResult typé.

        Applique les bornes de validation et génère les warnings.

        Args:
            raw_fields: Champs bruts extraits (valeurs textuelles).
            confidences: Confiances par champ (0.0-1.0).

        Returns:
            OCRResult validé avec warnings pour les champs manquants/invalides.
        """
        result = OCRResult()
        all_fields = ["invoice_id", "vendor_name", "amount", "date", "tax_rate", "total"]

        for field_name in all_fields:
            raw_value = raw_fields.get(field_name)
            confidence = confidences.get(field_name, 0.0)

            if raw_value is None or raw_value == "":
                # Champ manquant: null + confidence 0.0 + warning
                setattr(result, field_name, None)
                result.field_confidences[field_name] = 0.0
                result.warnings.append(
                    f"Champ '{field_name}' non extrait"
                )
                continue

            # Validation et conversion selon le type de champ
            validated = self._validate_field(field_name, raw_value)

            if validated is None:
                # Valeur invalide: null + confidence 0.0 + warning
                setattr(result, field_name, None)
                result.field_confidences[field_name] = 0.0
                result.warnings.append(
                    f"Champ '{field_name}' invalide: {raw_value}"
                )
            else:
                setattr(result, field_name, validated)
                # Clamp la confiance entre 0.0 et 1.0
                result.field_confidences[field_name] = max(0.0, min(1.0, confidence))

        return result

    def _validate_field(self, field_name: str, value: Any) -> Any:
        """Valide un champ individuel selon ses bornes.

        Args:
            field_name: Nom du champ à valider.
            value: Valeur brute à valider.

        Returns:
            Valeur validée et convertie, ou None si invalide.
        """
        try:
            if field_name == "invoice_id":
                # String, max 50 caractères
                str_val = str(value).strip()
                if not str_val:
                    return None
                return str_val[:INVOICE_ID_MAX_LENGTH]

            elif field_name == "vendor_name":
                # String, max 200 caractères
                str_val = str(value).strip()
                if not str_val:
                    return None
                return str_val[:VENDOR_NAME_MAX_LENGTH]

            elif field_name == "amount":
                # Float, 0.01 - 999,999,999.99
                float_val = self._parse_number(value)
                if float_val is None:
                    return None
                if float_val < AMOUNT_MIN or float_val > AMOUNT_MAX:
                    return None
                return round(float_val, 2)

            elif field_name == "date":
                # ISO 8601 YYYY-MM-DD
                str_val = str(value).strip()
                # Vérifier le format
                if re.match(r"^\d{4}-\d{2}-\d{2}$", str_val):
                    try:
                        datetime.strptime(str_val, "%Y-%m-%d")
                        return str_val
                    except ValueError:
                        return None
                # Tenter la normalisation
                normalized = self._normalize_date(str_val)
                return normalized

            elif field_name == "tax_rate":
                # Float, 0 - 100
                float_val = self._parse_number(value)
                if float_val is None:
                    return None
                if float_val < TAX_RATE_MIN or float_val > TAX_RATE_MAX:
                    return None
                return round(float_val, 2)

            elif field_name == "total":
                # Float, 0.01 - 999,999,999.99
                float_val = self._parse_number(value)
                if float_val is None:
                    return None
                if float_val < AMOUNT_MIN or float_val > AMOUNT_MAX:
                    return None
                return round(float_val, 2)

        except (ValueError, TypeError):
            return None

        return None

    def _parse_number(self, value: Any) -> Optional[float]:
        """Parse une valeur numérique depuis un texte OCR.

        Gère les formats: 1234.56, 1 234,56, 1,234.56, etc.

        Args:
            value: Valeur brute (string ou numérique).

        Returns:
            Float parsé ou None si impossible.
        """
        if isinstance(value, (int, float)):
            return float(value)

        str_val = str(value).strip()

        # Supprimer les espaces (séparateurs de milliers)
        str_val = str_val.replace(" ", "").replace("\u00a0", "")

        # Supprimer les symboles de devise courants
        str_val = re.sub(r"[MADdhsDHs€$]", "", str_val).strip()

        if not str_val:
            return None

        # Déterminer le séparateur décimal
        # Si le dernier séparateur est une virgule avec ≤2 chiffres après → virgule décimale
        if re.match(r"^[\d.]+,\d{1,2}$", str_val):
            # Format: 1.234,56 → remplacer . par rien, , par .
            str_val = str_val.replace(".", "").replace(",", ".")
        elif "," in str_val and "." not in str_val:
            # Format: 1234,56
            str_val = str_val.replace(",", ".")
        elif "," in str_val and "." in str_val:
            # Format: 1,234.56 (virgule = séparateur milliers)
            str_val = str_val.replace(",", "")

        try:
            return float(str_val)
        except ValueError:
            return None

    def extract(self, file_path: str) -> OCRResult:
        """Pipeline complet d'extraction OCR.

        Étapes:
        1. Charger l'image (ou convertir le PDF en image)
        2. Prétraiter l'image
        3. Tenter la détection YOLOv8
        4. Fallback Tesseract pour les champs avec confiance < 0.5
        5. Valider et structurer le résultat

        Args:
            file_path: Chemin vers le fichier (PDF, JPG, PNG, TIFF).

        Returns:
            OCRResult avec les champs extraits, confiances et warnings.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            RuntimeError: Si l'extraction échoue complètement.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")

        # Déterminer le type de fichier
        ext = os.path.splitext(file_path)[1].lower()

        # Charger l'image
        if ext == ".pdf":
            image = self._convert_pdf_to_image(file_path)
        elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
            image = cv2.imread(file_path)
            if image is None:
                raise RuntimeError(
                    f"Impossible de charger l'image: {file_path}"
                )
        else:
            raise RuntimeError(
                f"Format de fichier non supporté: {ext}. "
                "Formats supportés: PDF, JPG, PNG, TIFF."
            )

        # Prétraitement
        preprocessed = self.preprocess(image)

        # Étape 1: Détection YOLOv8
        yolo_result = self.detect_fields_yolo(preprocessed)

        # Étape 2: Fallback Tesseract pour champs manquants ou faible confiance
        tesseract_result = self.extract_tesseract(preprocessed)

        # Fusionner les résultats: YOLOv8 prioritaire si confiance >= 0.5
        merged_fields: Dict[str, Any] = {}
        merged_confidences: Dict[str, float] = {}

        all_fields = ["invoice_id", "vendor_name", "amount", "date", "tax_rate", "total"]

        for field_name in all_fields:
            yolo_conf = yolo_result["confidences"].get(field_name, 0.0)
            yolo_value = yolo_result["fields"].get(field_name)

            tess_conf = tesseract_result["confidences"].get(field_name, 0.0)
            tess_value = tesseract_result["fields"].get(field_name)

            # Utiliser YOLOv8 si confiance >= seuil
            if yolo_value is not None and yolo_conf >= YOLO_CONFIDENCE_THRESHOLD:
                merged_fields[field_name] = yolo_value
                merged_confidences[field_name] = yolo_conf
            # Sinon fallback vers Tesseract
            elif tess_value is not None:
                merged_fields[field_name] = tess_value
                merged_confidences[field_name] = tess_conf
            else:
                # Aucune extraction réussie
                merged_fields[field_name] = None
                merged_confidences[field_name] = 0.0

        # Validation et conversion des types
        result = self._validate_and_convert_fields(merged_fields, merged_confidences)

        return result
