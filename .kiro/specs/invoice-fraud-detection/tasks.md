# Implementation Plan: Invoice Fraud Detection

## Overview

Implémentation incrémentale du système de détection de fraude sur factures. Le plan suit l'ordre : structure projet → data layer → dataset synthétique → feature engineering → fraud detector → OCR pipeline → API backend → frontend Streamlit. Chaque étape est testable indépendamment et s'intègre progressivement.

## Tasks

- [x] 1. Set up project structure, dependencies, and database models
  - [x] 1.1 Create project directory structure and configuration files
    - Create directories: `backend/`, `ml/`, `ocr/`, `data/`, `results/`, `tests/unit/`, `tests/property/`, `tests/integration/`, `uploads/`
    - Create `requirements.txt` with all dependencies: fastapi, uvicorn, streamlit, sqlalchemy, pydantic, python-jose, passlib[bcrypt], python-multipart, ultralytics, pytesseract, opencv-python, numpy, pandas, scikit-learn, hypothesis, pytest, pytest-cov, httpx, factory_boy
    - Create `conftest.py` for pytest with shared fixtures (in-memory SQLite session, test client)
    - _Requirements: 9.11_

  - [x] 1.2 Implement SQLAlchemy ORM models and database setup
    - Create `backend/database.py` with engine, session factory, and Base
    - Implement `User` model (id, username, password_hash, failed_attempts, locked_until, created_at)
    - Implement `Invoice` model (all fields from design: invoice_id, vendor_name, amount_ht, tax_rate, amount_ttc, date, file_path, file_type, fraud_score, fraud_label, fraud_reason, label, fraud_type, ocr_confidences, feature_vector, analyzed_at, created_at, user_id FK)
    - Create `backend/schemas.py` with Pydantic models (LoginRequest, LoginResponse, InvoiceResponse, UploadResponse, StatsResponse, ModelMetrics, ModelReportResponse, ErrorResponse)
    - _Requirements: 9.11_

- [x] 2. Implement authentication module
  - [x] 2.1 Implement JWT auth module (`backend/auth.py`)
    - Implement `create_token(username)` → JWT token valid 24h with expiry claim
    - Implement `verify_token(token)` → decode and validate, raise on expired/invalid
    - Implement `authenticate(username, password)` → verify credentials, check lockout, return token or None
    - Implement `check_lockout(username)` → return True if account locked (5 failures → 15 min lock)
    - Implement failed attempt tracking and lockout logic
    - Create demo account: username="admin", password="admin123" (seeded on first run)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 2.2 Write property tests for JWT token round-trip
    - **Property 1: JWT token round-trip**
    - **Validates: Requirements 1.1, 1.4**

  - [x] 2.3 Write property tests for invalid credentials rejection
    - **Property 2: Invalid credentials rejection**
    - **Validates: Requirements 1.2**

  - [x] 2.4 Write property tests for account lockout threshold
    - **Property 3: Account lockout threshold**
    - **Validates: Requirements 1.7**

- [x] 3. Implement dataset generator
  - [x] 3.1 Implement synthetic dataset generation (`data/generate_dataset.py`)
    - Implement `DatasetGenerator` class with predefined VENDORS list (≥5 Moroccan vendors) and TAX_RATES [7, 10, 14, 20]
    - Implement `_generate_normal_invoice()`: weekday dates, valid tax rates, amounts 500-500,000 MAD
    - Implement `_generate_fraudulent_invoice()`: at least 1 fraud pattern (duplicate vendor-amount-date, invalid tax, out-of-range amount, weekend date)
    - Implement `generate()` → 200 invoices (175 Normal + 25 Frauduleux)
    - Implement `export_sqlite(df, path)` → `data/invoices.db`
    - Implement `export_csv(df, path)` → `data/invoices.csv`
    - Each record contains: invoice_id, vendor_name, amount_ht, tax_rate, amount_ttc, date, label, fraud_type
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

  - [x] 3.2 Write property tests for dataset size and distribution
    - **Property 22: Dataset size and distribution**
    - **Validates: Requirements 8.1**

  - [x] 3.3 Write property tests for normal invoice invariants
    - **Property 21: Normal invoice generation invariants**
    - **Validates: Requirements 8.2, 8.3, 8.4, 8.5, 8.8**

  - [x] 3.4 Write property tests for fraudulent invoice patterns
    - **Property 23: Fraudulent invoice pattern presence**
    - **Validates: Requirements 8.6**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement feature engineering module
  - [x] 5.1 Implement feature engine (`ml/feature_engineering.py`)
    - Implement `FeatureVector` dataclass with 6 features
    - Implement `_compute_amount_zscore(amount, vendor)`: z-score if ≥5 historical invoices in 12 months, else 0.0
    - Implement `_compute_tax_inconsistency(tax_rate)`: True if not in {7, 10, 14, 20}
    - Implement `_compute_duplicate_flag(amount, vendor, date)`: True if same vendor+amount in last 30 days
    - Implement `_compute_weekend_flag(date)`: True if Saturday or Sunday
    - Implement `_compute_round_amount_flag(amount)`: True if multiple of 1000 AND > 10,000
    - Implement `_compute_vendor_deviation(vendor)`: rarity score 0.0-1.0
    - Implement `compute_features(invoice_data)`: orchestrate all 6 features, reject if missing required fields
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [x] 5.2 Write property test for amount_zscore correctness
    - **Property 8: amount_zscore correctness**
    - **Validates: Requirements 5.1, 5.2**

  - [x] 5.3 Write property test for tax_inconsistency correctness
    - **Property 9: tax_inconsistency correctness**
    - **Validates: Requirements 5.3**

  - [x] 5.4 Write property test for duplicate_flag correctness
    - **Property 10: duplicate_flag correctness**
    - **Validates: Requirements 5.4**

  - [x] 5.5 Write property test for weekend_flag correctness
    - **Property 11: weekend_flag correctness**
    - **Validates: Requirements 5.5**

  - [x] 5.6 Write property test for round_amount_flag correctness
    - **Property 12: round_amount_flag correctness**
    - **Validates: Requirements 5.6**

  - [x] 5.7 Write property test for vendor_deviation correctness
    - **Property 13: vendor_deviation correctness**
    - **Validates: Requirements 5.7**

  - [x] 5.8 Write property test for missing field rejection
    - **Property 14: Feature vector missing field rejection**
    - **Validates: Requirements 5.9**

- [x] 6. Implement fraud detection module
  - [x] 6.1 Implement fraud detector (`ml/fraud_detector.py`)
    - Implement `FraudResult` dataclass (fraud_score, fraud_label, fraud_reason)
    - Implement `train_models(dataset_path)`: train Isolation Forest, One-Class SVM, Random Forest on synthetic data
    - Implement `evaluate_models(X_test, y_test)`: compute F1, Precision, Recall, AUC-ROC + confusion matrices for each model
    - Implement `predict(feature_vector)` using Isolation Forest as primary model → fraud_score [0,1]
    - Implement `_assign_label(score)`: [0, 0.3[ → "Normal", [0.3, 0.7[ → "Suspect", [0.7, 1.0] → "Frauduleux"
    - Implement `_generate_reason(features, importances)`: top 3 contributing features with values
    - Handle model unavailability: return fraud_score=-1.0, fraud_label="Erreur", fraud_reason indicating model unavailability
    - Save comparison results to `results/model_comparison.json`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [x] 6.2 Write property test for fraud score bounds and label assignment
    - **Property 15: Fraud score bounds and label assignment**
    - **Validates: Requirements 6.2, 6.3**

  - [x] 6.3 Write property test for fraud reason top 3 features
    - **Property 16: Fraud reason references top 3 features**
    - **Validates: Requirements 6.4**

  - [x] 6.4 Write property test for invalid feature vector rejection
    - **Property 17: Invalid feature vector rejection**
    - **Validates: Requirements 6.7**

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement OCR pipeline
  - [x] 8.1 Implement OCR pipeline (`ocr/pipeline.py`)
    - Implement `OCRResult` dataclass with 6 fields, field_confidences, and warnings
    - Implement `preprocess(image)`: grayscale conversion, denoising (cv2.fastNlMeansDenoising), deskewing
    - Implement `detect_fields_yolo(image)`: YOLOv8 field detection with confidence thresholds
    - Implement `extract_tesseract(image)`: Tesseract fallback extraction
    - Implement `extract(file_path)`: full pipeline with YOLOv8 → Tesseract fallback when confidence < 0.5
    - Handle PDF input (convert to image first using pdf2image or similar)
    - Set null value + confidence 0.0 + warning for fields that cannot be extracted
    - Validate field bounds (amount 0.01-999,999,999.99, tax_rate 0-100, etc.)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [x] 8.2 Write property test for OCR result serialization round-trip
    - **Property 4: OCR result serialization round-trip**
    - **Validates: Requirements 4.6**

  - [x] 8.3 Write property test for OCR output structure validity
    - **Property 5: OCR output structure validity**
    - **Validates: Requirements 4.4, 4.7**

  - [x] 8.4 Write property test for missing OCR field handling
    - **Property 6: Missing OCR field handling**
    - **Validates: Requirements 4.5**

  - [x] 8.5 Write property test for YOLOv8 fallback on low confidence
    - **Property 7: YOLOv8 fallback on low confidence**
    - **Validates: Requirements 4.3**

- [x] 9. Implement API backend
  - [x] 9.1 Implement FastAPI routes (`backend/api.py`)
    - Implement `POST /api/login`: accept username/password, validate credentials, return JWT or 401
    - Implement `POST /api/upload`: accept file (PDF/JPG/PNG ≤10MB), validate format+size, trigger pipeline (OCR → features → ML), persist results, return UploadResponse
    - Implement `GET /api/invoices`: return all analyzed invoices with fraud results (JWT protected)
    - Implement `GET /api/stats`: return aggregated KPIs (total_invoices, fraud_rate %, total_amount MAD) (JWT protected)
    - Implement `GET /api/models/report`: return model comparison from results/model_comparison.json (JWT protected), 404 if missing
    - Implement JWT dependency for protected endpoints (return 401 if missing/invalid/expired)
    - Implement file validation middleware (format check, size check → 400)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 9.11_

  - [x] 9.2 Write property test for file upload format validation
    - **Property 19: File upload format validation**
    - **Validates: Requirements 3.5, 9.5**

  - [x] 9.3 Write property test for protected endpoint auth enforcement
    - **Property 20: Protected endpoint authentication enforcement**
    - **Validates: Requirements 9.10**

  - [x] 9.4 Write property test for stats aggregation correctness
    - **Property 18: Stats aggregation correctness**
    - **Validates: Requirements 2.3, 9.7**

- [x] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement Streamlit frontend
  - [x] 11.1 Implement login page and session management (`app.py`)
    - Create login form with username/password fields
    - Call POST /api/login on submit, store JWT token in session state
    - Display "Identifiants incorrects" on failure
    - Implement logout button ("Déconnexion") to clear session and redirect
    - Auto-redirect to dashboard if valid token exists in session
    - _Requirements: 1.1, 1.2, 1.3, 1.6_

  - [x] 11.2 Implement dashboard page with KPIs and charts
    - Display KPI cards: total invoices, fraud rate (%), total amount (MAD with thousands separator)
    - Display pie chart (Normal/Suspect/Frauduleux distribution with labels and counts)
    - Display line chart (average fraud score per day, Y-axis 0.0-1.0)
    - Display invoice table with columns: ID, Fournisseur, Montant (MAD), Date, Score fraude (2 decimals), Statut
    - Color-code statut: green for Normal, orange for Suspect, red for Frauduleux
    - Sort by date descending, max 50 rows per page with pagination
    - Handle empty state: zero KPIs + message "Aucune facture analysée"
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [x] 11.3 Implement upload and analysis page
    - Create drag-and-drop file upload zone (PDF, JPG, PNG, max 10 MB)
    - Display file format/size validation error messages
    - Add "Analyser" button to trigger analysis
    - Display progress bar with stages: "Extraction OCR", "Feature Engineering", "Détection de fraude"
    - Display results: extracted data fields, fraud_score, fraud_label, fraud_reason
    - Handle pipeline failure: display error with failed stage name
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 11.4 Implement ML report page ("Rapport ML")
    - Display comparison table with F1 Score, Precision, Recall, AUC-ROC for 3 models
    - Highlight highest value per metric
    - Display confusion matrix visualization for each model
    - Handle missing report: display error message + prompt to run evaluation
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 12. Integration wiring and model training script
  - [x] 12.1 Create model training and evaluation script
    - Create `ml/train.py` that runs the full training pipeline: generate dataset → train 3 models → evaluate → save model_comparison.json
    - Persist trained Isolation Forest model to `models/isolation_forest.pkl`
    - Ensure the script is runnable standalone (`python ml/train.py`)
    - _Requirements: 6.1, 7.1, 7.2, 7.3_

  - [x] 12.2 Wire the full pipeline end-to-end
    - Ensure `POST /api/upload` correctly chains: file validation → OCR extraction → feature computation → fraud scoring → DB persistence → response
    - Verify database initialization creates tables and seeds demo user on first run
    - Create `main.py` entry point to launch both FastAPI (uvicorn) and Streamlit
    - _Requirements: 3.2, 9.3, 9.4, 9.11_

  - [x] 12.3 Write integration tests for the full pipeline
    - Test end-to-end upload → OCR → features → ML → DB → response
    - Test API authentication flow (login → use token → access protected routes)
    - Test model evaluation produces valid model_comparison.json
    - _Requirements: 3.2, 9.3, 9.4, 9.10_

- [x] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document (Hypothesis library, 100+ iterations each)
- Unit tests validate specific examples and edge cases
- Code in English, comments in French as per project conventions
- All monetary values in MAD (Dirham marocain)
- Every fraud analysis must return: fraud_score (0-1), fraud_label, fraud_reason

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["2.1", "3.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "2.4", "3.2", "3.3", "3.4"] },
    { "id": 4, "tasks": ["5.1"] },
    { "id": 5, "tasks": ["5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8"] },
    { "id": 6, "tasks": ["6.1"] },
    { "id": 7, "tasks": ["6.2", "6.3", "6.4", "8.1"] },
    { "id": 8, "tasks": ["8.2", "8.3", "8.4", "8.5", "9.1"] },
    { "id": 9, "tasks": ["9.2", "9.3", "9.4"] },
    { "id": 10, "tasks": ["11.1", "11.2", "11.3", "11.4", "12.1"] },
    { "id": 11, "tasks": ["12.2"] },
    { "id": 12, "tasks": ["12.3"] }
  ]
}
```
