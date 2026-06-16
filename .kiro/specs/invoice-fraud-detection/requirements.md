# Requirements Document

## Introduction

Système web complet de détection de fraude sur factures pour un bureau comptable marocain. L'utilisateur uploade une facture (PDF ou image), l'IA extrait les données via OCR, puis un modèle ML calcule un score de fraude. Le système permet de visualiser les résultats sur un dashboard interactif avec KPIs, graphiques et tableau comparatif des performances des modèles ML.

## Glossary

- **System**: L'application web complète de détection de fraude sur factures
- **Dashboard**: Interface principale Streamlit affichant les KPIs, graphiques et tableau des factures analysées
- **OCR_Pipeline**: Module d'extraction de données via prétraitement d'image et reconnaissance optique de caractères (YOLOv8 + Tesseract)
- **Fraud_Detector**: Module ML comparant 3 modèles (Isolation Forest, One-Class SVM, Random Forest) pour calculer un score de fraude
- **Feature_Engine**: Module de calcul des 6 features discriminantes pour la détection de fraude
- **Auth_Module**: Module d'authentification JWT gérant les sessions utilisateur
- **API_Backend**: Serveur FastAPI exposant les routes d'upload, d'analyse et de récupération des données
- **Dataset_Generator**: Script de génération de 200 factures synthétiques marocaines réalistes
- **Invoice**: Facture PDF ou image (JPG/PNG) uploadée pour analyse
- **Fraud_Score**: Valeur flottante entre 0 et 1 représentant la probabilité de fraude
- **Fraud_Label**: Classification textuelle parmi "Normal", "Suspect", "Frauduleux"
- **MAD**: Dirham marocain, monnaie utilisée pour tous les montants

## Requirements

### Requirement 1: Authentification utilisateur

**User Story:** As a comptable, I want to log in securely to the application, so that only authorized users can access fraud detection results.

#### Acceptance Criteria

1. WHEN a user submits valid credentials (username + password), THE Auth_Module SHALL issue a JWT token valid for 24 hours and redirect the user to the Dashboard
2. WHEN a user submits invalid credentials, THE Auth_Module SHALL display an error message "Identifiants incorrects" and remain on the login page
3. WHILE a valid JWT token exists in the session, THE System SHALL redirect the user directly to the Dashboard without requiring re-authentication
4. WHEN the JWT token expires after 24 hours, THE Auth_Module SHALL redirect the user to the login page
5. THE Auth_Module SHALL provide a demo account with username "admin" and password "admin123"
6. WHEN a user clicks the "Déconnexion" button, THE Auth_Module SHALL invalidate the current session token and redirect the user to the login page
7. WHEN a user submits invalid credentials 5 consecutive times, THE Auth_Module SHALL lock the account for 15 minutes and display a message indicating the lockout duration

### Requirement 2: Dashboard principal avec KPIs et graphiques

**User Story:** As a comptable, I want to view a summary dashboard of all analyzed invoices, so that I can quickly assess the fraud landscape.

#### Acceptance Criteria

1. THE Dashboard SHALL display a table listing all analyzed invoices with columns: ID, Fournisseur, Montant (MAD), Date, Score fraude (displayed as a value between 0.00 and 1.00 with two decimal places), Statut, sorted by Date descending by default, showing a maximum of 50 rows per page with pagination controls
2. THE Dashboard SHALL display each invoice status with color coding: "Normal" in green, "Suspect" in orange, "Frauduleux" in red
3. THE Dashboard SHALL display KPI cards showing: total number of analyzed invoices, fraud rate as a percentage (number of invoices with Suspect or Frauduleux status divided by total analyzed invoices, displayed with one decimal place), and total analyzed amount in MAD formatted with two decimal places and thousands separator
4. THE Dashboard SHALL display a pie chart showing the distribution of invoices across Normal, Suspect, and Frauduleux categories with each segment labeled with category name and count
5. THE Dashboard SHALL display a line chart showing the average fraud score per day on the Y-axis (scale 0.0 to 1.0) and date on the X-axis, plotting one data point per day where invoices were analyzed
6. WHEN new invoices are analyzed, THE Dashboard SHALL update KPIs, charts, and the invoice table to reflect the latest data upon page reload
7. IF no invoices have been analyzed, THEN THE Dashboard SHALL display the KPI cards with zero values and show an empty-state message in place of the table and charts indicating that no invoices have been analyzed yet

### Requirement 3: Upload et analyse de factures

**User Story:** As a comptable, I want to upload an invoice and trigger fraud analysis, so that I can determine if a specific invoice is fraudulent.

#### Acceptance Criteria

1. THE System SHALL provide a drag-and-drop zone accepting a single file in PDF, JPG, or PNG format with a maximum file size of 10 MB
2. WHEN a user clicks the "Analyser" button after uploading a file, THE System SHALL trigger the complete analysis pipeline (OCR extraction → feature engineering → fraud detection)
3. WHILE the analysis pipeline is running, THE System SHALL display a progress bar indicating the current stage of processing (OCR extraction, feature engineering, fraud detection)
4. WHEN the analysis completes, THE System SHALL display the extracted data fields (supplier name, invoice number, date, total amount in MAD, line items), the fraud_score (value between 0 and 1), the fraud_label, and the fraud_reason
5. IF an uploaded file is not in PDF, JPG, or PNG format, THEN THE System SHALL reject the file and display an error message indicating the supported formats (PDF, JPG, PNG)
6. IF an uploaded file exceeds 10 MB, THEN THE System SHALL reject the file and display an error message indicating the maximum allowed size
7. IF the analysis pipeline fails at any stage (OCR extraction, feature engineering, or fraud detection), THEN THE System SHALL display an error message indicating which stage failed and preserve the uploaded file for retry

### Requirement 4: Extraction OCR des données de facture

**User Story:** As a comptable, I want invoice data to be automatically extracted from uploaded documents, so that I do not need to manually enter invoice fields.

#### Acceptance Criteria

1. WHEN an image (JPEG, PNG, TIFF) or PDF document is received, THE OCR_Pipeline SHALL apply preprocessing steps: grayscale conversion, denoising, and deskewing
2. WHEN preprocessing completes, THE OCR_Pipeline SHALL attempt field detection using YOLOv8 as the primary method
3. IF YOLOv8 field detection returns no results, raises an error, or returns a confidence score below 0.5 for any detected field, THEN THE OCR_Pipeline SHALL fall back to direct Tesseract extraction for the affected fields
4. WHEN extraction completes, THE OCR_Pipeline SHALL output a structured JSON containing: invoice_id (string, max 50 characters), vendor_name (string, max 200 characters), amount (numeric, 0.01 to 999,999,999.99 in MAD), date (ISO 8601 format YYYY-MM-DD), tax_rate (numeric, 0 to 100 as percentage), and total (numeric, 0.01 to 999,999,999.99 in MAD)
5. IF the OCR_Pipeline cannot extract a required field, THEN THE OCR_Pipeline SHALL set the field value to null, include a per-field confidence score of 0.0, and include a warning entry in the output JSON identifying the missing field name
6. THE OCR_Pipeline SHALL produce idempotent structured output such that serializing the output JSON and deserializing it yields field values identical to the original extraction result for all six fields
7. WHEN extraction completes, THE OCR_Pipeline SHALL include a per-field confidence score between 0.0 and 1.0 for each of the six extracted fields in the output JSON

### Requirement 5: Feature Engineering pour la détection de fraude

**User Story:** As a data scientist, I want discriminating features to be computed from extracted invoice data, so that ML models can accurately detect anomalies.

#### Acceptance Criteria

1. WHEN extracted invoice data is received and the vendor has 5 or more historical invoices within the last 12 months, THE Feature_Engine SHALL compute amount_zscore as the z-score of the invoice amount relative to that vendor's historical mean and standard deviation over the same 12-month window
2. IF the vendor has fewer than 5 historical invoices within the last 12 months, THEN THE Feature_Engine SHALL set amount_zscore to 0.0
3. WHEN extracted invoice data is received, THE Feature_Engine SHALL compute tax_inconsistency as a boolean flag indicating the tax rate is not among 7%, 10%, 14%, or 20%
4. WHEN extracted invoice data is received, THE Feature_Engine SHALL compute duplicate_flag as a boolean indicating an exact-match amount and same vendor combination exists within the last 30 days
5. WHEN extracted invoice data is received, THE Feature_Engine SHALL compute weekend_flag as a boolean indicating the invoice date falls on a Saturday or Sunday
6. WHEN extracted invoice data is received, THE Feature_Engine SHALL compute round_amount_flag as a boolean indicating the amount is a multiple of 1000 MAD and strictly above 10,000 MAD
7. WHEN extracted invoice data is received, THE Feature_Engine SHALL compute vendor_deviation as a score between 0.0 and 1.0 where 1.0 indicates the vendor has no prior invoices in the historical database, and values are proportional to rarity defined as 1 minus the ratio of the vendor's invoice count to the maximum invoice count among all vendors over the last 12 months
8. THE Feature_Engine SHALL return a feature vector containing all 6 computed features for each invoice
9. IF extracted invoice data is missing one or more required fields (amount, vendor, date, or tax_rate), THEN THE Feature_Engine SHALL reject the input and return an error indication specifying which fields are missing

### Requirement 6: Détection de fraude par modèles ML

**User Story:** As a comptable, I want each invoice to receive a fraud score from the best available ML model, so that I can prioritize investigations on suspicious invoices.

#### Acceptance Criteria

1. THE Fraud_Detector SHALL train and compare three models (Isolation Forest, One-Class SVM, Random Forest with synthetic labels) using F1 Score, Precision, Recall, and AUC-ROC metrics, and persist the comparison results
2. WHEN a feature vector containing the 6 required features (amount_zscore, tax_inconsistency, duplicate_flag, weekend_flag, round_amount_flag, vendor_deviation) is provided, THE Fraud_Detector SHALL compute a fraud_score as a float between 0.0 and 1.0
3. WHEN a fraud_score is computed, THE Fraud_Detector SHALL assign a fraud_label: "Normal" for scores in [0.0, 0.3[, "Suspect" for scores in [0.3, 0.7[, "Frauduleux" for scores in [0.7, 1.0]
4. WHEN a fraud_label is assigned, THE Fraud_Detector SHALL generate a fraud_reason text listing the top 3 contributing features by importance along with their values
5. THE Fraud_Detector SHALL use Isolation Forest as the primary model for production scoring
6. WHEN the fraud_score is computed, THE Fraud_Detector SHALL persist the result (fraud_score, fraud_label, fraud_reason) in the SQLite database within the same transaction as the invoice record
7. IF the feature vector is missing one or more of the 6 required features or contains non-numeric values, THEN THE Fraud_Detector SHALL reject the input and return an error indication specifying which features are invalid or missing
8. IF the trained model file is unavailable or corrupted at scoring time, THEN THE Fraud_Detector SHALL return a fraud_score of -1.0 with fraud_label "Erreur" and a fraud_reason indicating model unavailability

### Requirement 7: Rapport comparatif des performances ML

**User Story:** As a data scientist, I want to compare the performance of all three ML models, so that I can validate the choice of the primary model and monitor model quality.

#### Acceptance Criteria

1. WHEN the evaluation is triggered on the 200-invoice synthetic dataset (175 normal, 25 fraudulent), THE Fraud_Detector SHALL compute F1 Score, Precision, Recall, and AUC-ROC for each of the three models (Isolation Forest, One-Class SVM, Random Forest), with metric values rounded to 4 decimal places
2. WHEN the evaluation is triggered, THE Fraud_Detector SHALL generate a 2×2 confusion matrix (classes: "normal", "fraudulent") for each of the three models, containing True Positives, True Negatives, False Positives, and False Negatives counts
3. WHEN all model metrics have been computed, THE System SHALL save the results to results/model_comparison.json containing for each model: model name, F1 Score, Precision, Recall, AUC-ROC, and the confusion matrix values
4. THE Dashboard SHALL provide a "Rapport ML" page displaying a comparison table of F1 Score, Precision, Recall, and AUC-ROC for all three models, with the highest value per metric visually highlighted
5. THE Dashboard SHALL display the confusion matrix visualization for each model on the "Rapport ML" page
6. IF results/model_comparison.json is missing or unreadable, THEN THE Dashboard SHALL display an error message indicating that model comparison data is unavailable and prompt the user to run the evaluation

### Requirement 8: Génération du dataset synthétique marocain

**User Story:** As a data scientist, I want a realistic synthetic dataset of Moroccan invoices, so that I can train and evaluate ML models without real sensitive data.

#### Acceptance Criteria

1. THE Dataset_Generator SHALL generate exactly 200 synthetic invoices: 175 labeled "Normal" and 25 labeled "Frauduleux"
2. THE Dataset_Generator SHALL assign vendor names from a predefined list of at least 5 distinct Moroccan vendors (e.g., Maroc Telecom, Atlas BTP, Souss Agro, Sahara Logistics, Fès Textile)
3. THE Dataset_Generator SHALL generate amounts in MAD ranging from 500.00 to 500,000.00 with 2 decimal places for normal invoices
4. THE Dataset_Generator SHALL apply Moroccan tax rates: 7%, 10%, 14%, or 20% for normal invoices
5. THE Dataset_Generator SHALL generate dates spanning 12 rolling months from the current date, with normal invoices dated on weekdays only (Monday to Friday)
6. THE Dataset_Generator SHALL introduce fraud patterns in the 25 fraudulent invoices, where each fraudulent invoice contains at least one of the following: duplicated vendor-amount-date combinations, tax rates not in {7%, 10%, 14%, 20%}, amounts outside the 500–500,000 MAD range, or dates falling on Saturday or Sunday
7. THE Dataset_Generator SHALL output the dataset as both a SQLite database (data/invoices.db) and a CSV file (data/invoices.csv) for model training
8. THE Dataset_Generator SHALL produce each invoice record with the following fields: invoice_id, vendor_name, amount_ht, tax_rate, amount_ttc, date, label, and fraud_type (empty string for normal invoices, description of the fraud pattern applied for fraudulent invoices)

### Requirement 9: Persistance et API Backend

**User Story:** As a developer, I want a REST API backend to handle data flow between the frontend, OCR, and ML modules, so that the architecture remains modular and maintainable.

#### Acceptance Criteria

1. THE API_Backend SHALL expose a POST /api/login endpoint accepting username (maximum 50 characters) and password (maximum 128 characters) and returning a JWT token upon successful authentication
2. IF the POST /api/login request contains invalid credentials, THEN THE API_Backend SHALL return an HTTP 401 status with an error message indicating authentication failure
3. THE API_Backend SHALL expose a POST /api/upload endpoint accepting a single file (PDF, JPG, or PNG, maximum 10 MB) and triggering the OCR extraction followed by ML fraud analysis pipeline
4. WHEN the POST /api/upload analysis pipeline completes, THE API_Backend SHALL return the extracted invoice data along with fraud_score (0 to 1), fraud_label, and fraud_reason for the uploaded document
5. IF the file submitted to POST /api/upload is not a supported format (PDF, JPG, PNG) or exceeds 10 MB, THEN THE API_Backend SHALL reject the request with an HTTP 400 status and an error message indicating the validation failure
6. THE API_Backend SHALL expose a GET /api/invoices endpoint returning all analyzed invoices with their fraud_score, fraud_label, fraud_reason, and extracted amounts in MAD
7. THE API_Backend SHALL expose a GET /api/stats endpoint returning aggregated KPIs: total number of invoices, fraud rate as a percentage (0 to 100), and total amount in MAD
8. THE API_Backend SHALL expose a GET /api/models/report endpoint returning the model comparison metrics from results/model_comparison.json
9. IF the model comparison file is unavailable when GET /api/models/report is called, THEN THE API_Backend SHALL return an HTTP 404 status with an error message indicating the report is not yet generated
10. WHILE no valid JWT token is present in the request header, THE API_Backend SHALL reject requests to protected endpoints with HTTP 401 status
11. THE API_Backend SHALL store all extracted invoice fields, fraud analysis results (fraud_score, fraud_label, fraud_reason), and upload metadata in a SQLite database using SQLAlchemy ORM
