"""
Détecteur de fraude par modèles ML.
Compare Isolation Forest, One-Class SVM et Random Forest.
Utilise Isolation Forest comme modèle principal pour la prédiction.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ml.feature_engineering import FeatureVector


# Noms des 6 features dans l'ordre
FEATURE_NAMES = [
    "amount_zscore",
    "tax_inconsistency",
    "duplicate_flag",
    "weekend_flag",
    "round_amount_flag",
    "vendor_deviation",
]

# Chemin par défaut du modèle sauvegardé
MODEL_PATH = "models/isolation_forest.pkl"

# Chemin par défaut des résultats de comparaison
COMPARISON_PATH = "results/model_comparison.json"


@dataclass
class FraudResult:
    """Résultat de l'analyse de fraude pour une facture."""

    fraud_score: float  # 0.0 - 1.0 (ou -1.0 en cas d'erreur)
    fraud_label: str  # "Normal" | "Suspect" | "Frauduleux" | "Erreur"
    fraud_reason: str  # Top 3 features contributives


class FraudDetector:
    """Détecteur de fraude utilisant Isolation Forest comme modèle principal."""

    def __init__(self, model_path: str = MODEL_PATH):
        """Initialise le détecteur avec le chemin du modèle.

        Args:
            model_path: Chemin vers le fichier du modèle Isolation Forest sauvegardé.
        """
        self._model_path = model_path
        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None

    def _load_model(self) -> bool:
        """Charge le modèle depuis le disque.

        Returns:
            True si le modèle a été chargé avec succès, False sinon.
        """
        try:
            if os.path.exists(self._model_path):
                data = joblib.load(self._model_path)
                self._model = data["model"]
                self._scaler = data["scaler"]
                return True
            return False
        except Exception:
            return False

    def train_models(self, dataset_path: str) -> dict:
        """Entraîne les 3 modèles et sauvegarde les métriques.

        Args:
            dataset_path: Chemin vers le fichier CSV ou SQLite du dataset.

        Returns:
            Dictionnaire avec les métriques de chaque modèle.
        """
        # Chargement des données
        if dataset_path.endswith(".csv"):
            df = pd.read_csv(dataset_path)
        else:
            # Chargement depuis SQLite
            import sqlite3

            conn = sqlite3.connect(dataset_path)
            df = pd.read_sql_query("SELECT * FROM invoices", conn)
            conn.close()

        # Préparation des features et labels
        X, y = self._prepare_data(df)

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Normalisation des features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        # Entraînement des 3 modèles
        # 1. Isolation Forest (modèle principal, non supervisé)
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.125,  # ~12.5% de fraudes attendues
            random_state=42,
        )
        iso_forest.fit(X_train_scaled)

        # 2. One-Class SVM (non supervisé)
        oc_svm = OneClassSVM(
            kernel="rbf",
            gamma="auto",
            nu=0.125,  # Paramètre similaire à contamination
        )
        oc_svm.fit(X_train_scaled)

        # 3. Random Forest (supervisé)
        rf_clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        )
        rf_clf.fit(X_train_scaled, y_train)

        # Sauvegarde du modèle principal (Isolation Forest)
        self._model = iso_forest
        self._save_model()

        # Évaluation des 3 modèles
        results = self.evaluate_models(
            X_test_scaled, y_test,
            models={
                "Isolation Forest": iso_forest,
                "One-Class SVM": oc_svm,
                "Random Forest": rf_clf,
            },
        )

        # Sauvegarde des résultats de comparaison
        self._save_comparison(results)

        return results

    def evaluate_models(
        self, X_test, y_test, models: Optional[dict] = None
    ) -> dict:
        """Compare F1, Precision, Recall, AUC-ROC + confusion matrices.

        Args:
            X_test: Données de test (déjà normalisées).
            y_test: Labels de test (0=Normal, 1=Frauduleux).
            models: Dictionnaire {nom: modèle}. Si None, utilise le modèle chargé.

        Returns:
            Dictionnaire avec les métriques pour chaque modèle.
        """
        if models is None:
            if self._model is None:
                self._load_model()
            models = {"Isolation Forest": self._model}

        results = {"models": []}

        for name, model in models.items():
            # Prédiction selon le type de modèle
            if isinstance(model, (IsolationForest, OneClassSVM)):
                # Modèles non supervisés: -1 = anomalie, 1 = normal
                raw_predictions = model.predict(X_test)
                # Conversion: -1 → 1 (fraude), 1 → 0 (normal)
                y_pred = np.where(raw_predictions == -1, 1, 0)

                # Score de décision pour AUC-ROC
                if isinstance(model, IsolationForest):
                    scores = model.decision_function(X_test)
                    # Inverser: scores plus négatifs = plus anomalique
                    y_scores = -scores
                else:
                    scores = model.decision_function(X_test)
                    y_scores = -scores
            else:
                # Random Forest (supervisé)
                y_pred = model.predict(X_test)
                y_scores = model.predict_proba(X_test)[:, 1]

            # Calcul des métriques
            f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
            precision = round(
                float(precision_score(y_test, y_pred, zero_division=0)), 4
            )
            recall = round(
                float(recall_score(y_test, y_pred, zero_division=0)), 4
            )

            # AUC-ROC (nécessite au moins 2 classes dans y_test)
            try:
                auc_roc = round(float(roc_auc_score(y_test, y_scores)), 4)
            except ValueError:
                auc_roc = 0.0

            # Matrice de confusion [[TP, FP], [FN, TN]]
            cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
            cm_list = cm.tolist()

            results["models"].append(
                {
                    "name": name,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "auc_roc": auc_roc,
                    "confusion_matrix": cm_list,
                }
            )

        return results

    def predict(self, feature_vector: FeatureVector) -> FraudResult:
        """Score une facture avec Isolation Forest (modèle principal).

        Args:
            feature_vector: Vecteur de 6 features calculées.

        Returns:
            FraudResult avec fraud_score, fraud_label et fraud_reason.

        Raises:
            ValueError: Si le vecteur de features est invalide.
        """
        # Validation du vecteur de features
        self._validate_feature_vector(feature_vector)

        # Chargement du modèle si nécessaire
        if self._model is None:
            if not self._load_model():
                # Modèle non disponible
                return FraudResult(
                    fraud_score=-1.0,
                    fraud_label="Erreur",
                    fraud_reason="Modèle non disponible",
                )

        try:
            # Conversion du FeatureVector en array numpy
            features_array = self._feature_vector_to_array(feature_vector)

            # Normalisation
            if self._scaler is not None:
                features_scaled = self._scaler.transform(
                    features_array.reshape(1, -1)
                )
            else:
                features_scaled = features_array.reshape(1, -1)

            # Score d'anomalie via Isolation Forest
            # decision_function: plus le score est négatif, plus c'est anomalique
            raw_score = self._model.decision_function(features_scaled)[0]

            # Conversion en score de fraude [0, 1]
            # Le score de décision est typiquement dans [-0.5, 0.5]
            # On normalise: score négatif → fraude (proche de 1)
            fraud_score = self._normalize_score(raw_score)

            # Attribution du label
            fraud_label = self._assign_label(fraud_score)

            # Génération de la raison
            # Utilisation des importances basées sur la profondeur moyenne
            importances = self._compute_feature_importances(features_scaled)
            fraud_reason = self._generate_reason(feature_vector, importances)

            return FraudResult(
                fraud_score=fraud_score,
                fraud_label=fraud_label,
                fraud_reason=fraud_reason,
            )

        except Exception:
            return FraudResult(
                fraud_score=-1.0,
                fraud_label="Erreur",
                fraud_reason="Modèle non disponible",
            )

    def _assign_label(self, score: float) -> str:
        """Assigne un label basé sur le score de fraude.

        Args:
            score: Score de fraude entre 0.0 et 1.0.

        Returns:
            "Normal" pour [0, 0.3[, "Suspect" pour [0.3, 0.7[, "Frauduleux" pour [0.7, 1.0].
        """
        if score < 0.3:
            return "Normal"
        elif score < 0.7:
            return "Suspect"
        else:
            return "Frauduleux"

    def _generate_reason(
        self, features: FeatureVector, importances: list
    ) -> str:
        """Génère la raison de fraude avec les top 3 features contributives.

        Args:
            features: Vecteur de features de la facture.
            importances: Liste des importances pour chaque feature.

        Returns:
            Texte listant les top 3 features contributives avec leurs valeurs.
        """
        # Associer chaque feature à son importance et sa valeur
        feature_values = self._feature_vector_to_dict(features)

        feature_importance_pairs = []
        for i, name in enumerate(FEATURE_NAMES):
            feature_importance_pairs.append(
                (name, importances[i], feature_values[name])
            )

        # Trier par importance décroissante
        feature_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        # Prendre les top 3
        top_3 = feature_importance_pairs[:3]

        # Formater la raison
        reasons = []
        for name, importance, value in top_3:
            reasons.append(f"{name}={value}")

        return "Top 3 features: " + ", ".join(reasons)

    def _validate_feature_vector(self, feature_vector: FeatureVector) -> None:
        """Valide que le vecteur de features contient les 6 features numériques.

        Args:
            feature_vector: Vecteur à valider.

        Raises:
            ValueError: Si des features sont manquantes ou non numériques.
        """
        missing = []
        invalid = []

        for name in FEATURE_NAMES:
            if not hasattr(feature_vector, name):
                missing.append(name)
            else:
                value = getattr(feature_vector, name)
                if value is None:
                    missing.append(name)
                elif not isinstance(value, (int, float, bool, np.integer, np.floating)):
                    invalid.append(name)

        if missing or invalid:
            error_parts = []
            if missing:
                error_parts.append(
                    f"Features manquantes: {', '.join(missing)}"
                )
            if invalid:
                error_parts.append(
                    f"Features non numériques: {', '.join(invalid)}"
                )
            raise ValueError(". ".join(error_parts))

    def _feature_vector_to_array(self, feature_vector: FeatureVector) -> np.ndarray:
        """Convertit un FeatureVector en array numpy.

        Args:
            feature_vector: Vecteur de features.

        Returns:
            Array numpy de shape (6,) avec les valeurs numériques.
        """
        values = []
        for name in FEATURE_NAMES:
            value = getattr(feature_vector, name)
            # Convertir les booléens en float
            values.append(float(value))
        return np.array(values)

    def _feature_vector_to_dict(self, feature_vector: FeatureVector) -> dict:
        """Convertit un FeatureVector en dictionnaire.

        Args:
            feature_vector: Vecteur de features.

        Returns:
            Dictionnaire {nom_feature: valeur}.
        """
        result = {}
        for name in FEATURE_NAMES:
            value = getattr(feature_vector, name)
            result[name] = value
        return result

    def _normalize_score(self, raw_score: float) -> float:
        """Normalise le score de décision Isolation Forest en [0, 1].

        Le score de décision est typiquement centré autour de 0.
        Valeurs négatives = anomalies, positives = normal.

        Args:
            raw_score: Score brut de decision_function.

        Returns:
            Score normalisé entre 0.0 et 1.0.
        """
        # Transformation sigmoïde inversée pour mapper en [0, 1]
        # Plus le score est négatif, plus la fraude est probable
        normalized = 1.0 / (1.0 + np.exp(5.0 * raw_score))
        # Clamp dans [0, 1]
        return float(max(0.0, min(1.0, normalized)))

    def _compute_feature_importances(self, features_scaled: np.ndarray) -> list:
        """Calcule les importances des features pour une prédiction donnée.

        Utilise la profondeur moyenne dans les arbres de l'Isolation Forest
        pour approximer l'importance de chaque feature.

        Args:
            features_scaled: Features normalisées (shape 1x6).

        Returns:
            Liste de 6 valeurs d'importance.
        """
        if not hasattr(self._model, "estimators_"):
            # Fallback: utiliser les valeurs absolues des features
            return [abs(float(x)) for x in features_scaled[0]]

        # Calculer l'importance basée sur la profondeur dans chaque arbre
        importances = np.zeros(len(FEATURE_NAMES))

        for tree in self._model.estimators_:
            # Pour chaque arbre, regarder les features utilisées aux noeuds
            tree_model = tree.tree_
            feature_indices = tree_model.feature

            # Compter combien de fois chaque feature est utilisée
            for feat_idx in feature_indices:
                if 0 <= feat_idx < len(FEATURE_NAMES):
                    importances[feat_idx] += 1

        # Normaliser
        total = importances.sum()
        if total > 0:
            importances = importances / total

        # Pondérer par les valeurs des features
        feature_abs = np.abs(features_scaled[0])
        weighted_importances = importances * feature_abs

        return weighted_importances.tolist()

    def _prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prépare les features et labels à partir du DataFrame.

        Args:
            df: DataFrame avec les colonnes de factures.

        Returns:
            Tuple (X, y) avec X: array de features, y: array de labels.
        """
        # Calculer les features à partir des colonnes disponibles
        feature_columns = []

        # amount_zscore: z-score du montant par fournisseur
        if "amount_ht" in df.columns and "vendor_name" in df.columns:
            # Calculer le z-score par fournisseur
            zscore_col = df.groupby("vendor_name")["amount_ht"].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
            )
            df["amount_zscore"] = zscore_col.fillna(0.0)
        else:
            df["amount_zscore"] = 0.0

        # tax_inconsistency: taux hors {7, 10, 14, 20}
        valid_rates = {7, 10, 14, 20}
        if "tax_rate" in df.columns:
            df["tax_inconsistency"] = df["tax_rate"].apply(
                lambda x: 1.0 if x not in valid_rates else 0.0
            )
        else:
            df["tax_inconsistency"] = 0.0

        # duplicate_flag: détection simplifiée de doublons
        if "vendor_name" in df.columns and "amount_ht" in df.columns:
            duplicates = df.duplicated(
                subset=["vendor_name", "amount_ht"], keep=False
            )
            df["duplicate_flag"] = duplicates.astype(float)
        else:
            df["duplicate_flag"] = 0.0

        # weekend_flag: date un samedi ou dimanche
        if "date" in df.columns:
            df["weekend_flag"] = pd.to_datetime(
                df["date"], errors="coerce"
            ).dt.dayofweek.apply(lambda x: 1.0 if x >= 5 else 0.0)
            df["weekend_flag"] = df["weekend_flag"].fillna(0.0)
        else:
            df["weekend_flag"] = 0.0

        # round_amount_flag: multiple de 1000 et > 10000
        if "amount_ht" in df.columns:
            df["round_amount_flag"] = df["amount_ht"].apply(
                lambda x: 1.0 if x % 1000 == 0 and x > 10000 else 0.0
            )
        else:
            df["round_amount_flag"] = 0.0

        # vendor_deviation: score de rareté
        if "vendor_name" in df.columns:
            vendor_counts = df["vendor_name"].value_counts()
            max_count = vendor_counts.max() if len(vendor_counts) > 0 else 1
            df["vendor_deviation"] = df["vendor_name"].apply(
                lambda v: 1.0 - (vendor_counts.get(v, 0) / max_count)
            )
        else:
            df["vendor_deviation"] = 1.0

        # Extraction des features
        X = df[FEATURE_NAMES].values

        # Labels: 0=Normal, 1=Frauduleux
        if "label" in df.columns:
            y = (df["label"] == "Frauduleux").astype(int).values
        else:
            # Pas de labels → tout est normal (pour non supervisé)
            y = np.zeros(len(df))

        return X, y

    def _save_model(self) -> None:
        """Sauvegarde le modèle Isolation Forest et le scaler sur le disque."""
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)

        data = {
            "model": self._model,
            "scaler": self._scaler,
        }
        joblib.dump(data, self._model_path)

    def _save_comparison(self, results: dict) -> None:
        """Sauvegarde les résultats de comparaison en JSON.

        Args:
            results: Dictionnaire avec les métriques des modèles.
        """
        os.makedirs(os.path.dirname(COMPARISON_PATH), exist_ok=True)

        with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
