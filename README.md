# AEGISCARE AI
### Integrated AI-Driven Healthcare Safety and ICU Risk Assessment System

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Academic%20Prototype-green.svg)]()

**AEGISCARE AI** is an integrated healthcare safety demonstration system designed to support patient monitoring, medication interaction checking, ICU risk assessment, and patient-state simulation.

The system combines rule-based safety checks, a trained machine learning model, synthetic patient data, and interactive visualisations within a unified Streamlit application.

It demonstrates how different computational approaches can be integrated into a healthcare-oriented software system.

---

## 1. Project Overview

Healthcare environments involve continuous patient monitoring, medication management, clinical data interpretation, and timely identification of potential risks.

AEGISCARE AI addresses these areas through four interconnected modules:

1. **Drug Interaction Checker:** Identifies potential medication interactions using a documented rule-based interaction database.
2. **Deterioration Prediction:** Demonstrates patient deterioration monitoring through simulated risk scoring and vital-sign analysis.
3. **ICU Risk Assessment:** Uses a trained Logistic Regression model to generate an ICU mortality risk score from clinical measurements.
4. **Patient Digital Twin:** Simulates patient-state changes and demonstrates how changes in clinical parameters can influence system responses.

The application also provides a central dashboard for viewing patient information, vital signs, trends, and safety-related information.

## 2. Key Features

| Module | Functionality |
|---|---|
| Patient Dashboard | Patient overview, vital signs, safety indicators and trends |
| Drug Interaction Checker | Medication interaction identification, severity information and source references |
| Deterioration Prediction | Simulated deterioration scoring and vital-sign monitoring |
| ICU Risk Assessment | Machine learning-based ICU mortality risk scoring |
| Patient Digital Twin | Interactive patient-state simulation and reassessment |
| Data Visualisation | Charts, clinical trends and model evaluation visualisations |

### 2.1 Drug Interaction Checker

The Drug Interaction Checker evaluates selected medications against a documented interaction database.

Key capabilities:
- Identifies known drug combinations.
- Displays interaction information and associated risks.
- Uses patient context where supported by the documented rules.
- Handles unknown medication names.
- Identifies duplicate medication entries.

**Implementation:** Rule-based interaction checking. This module does not use a trained machine learning model.

### 2.2 Deterioration Prediction

This module demonstrates patient deterioration monitoring using synthetic patient data and simulated scoring.

Key capabilities:
- Vital-sign monitoring.
- Deterioration score visualisation.
- Risk-contributor analysis.
- Simulated patient-state reassessment.

**Implementation:** Simulated scoring, not a trained predictive model.

### 2.3 ICU Risk Assessment

This module integrates a trained Logistic Regression model developed using the eICU Collaborative Research Database demo dataset.

The model uses aggregated clinical measurements and patient information to generate an uncalibrated ICU mortality risk score.

Key capabilities:
- Patient profile and ICU stay information.
- Vital-sign and laboratory input.
- Clinical input validation.
- Feature aggregation and preprocessing.
- Model-based risk scoring.
- Data completeness assessment.
- Model evaluation information.

The feature adapter converts human-readable clinical inputs into the 126 features expected by the trained model.

**Important:** The output is an uncalibrated model risk score, not a probability of death or a clinical diagnosis.

### 2.4 Patient Digital Twin

The Patient Digital Twin demonstrates how changes in patient parameters can influence the system's simulated response.

Key capabilities:
- Patient-state modification.
- Vital-sign changes.
- Medication changes.
- Reassessment of relevant system modules.
- Patient-state reset and timeline visualisation.

The Digital Twin is a demonstration module and does not independently use the trained ICU mortality model.

---

## 3. Machine Learning Methodology

The ICU Risk Assessment module uses supervised machine learning to estimate ICU mortality risk.

### 3.1 Dataset

The model was developed using the eICU Collaborative Research Database demo dataset.

| Dataset characteristic | Value |
|---|---:|
| Total ICU stays | 1,838 |
| Alive outcomes | 1,747 |
| Expired outcomes | 91 |
| Training samples | 1,470 |
| Testing samples | 368 |
| Model input features | 126 |

The dataset contains an imbalanced distribution of outcomes, with substantially fewer expired cases than alive cases.

The train-test split was performed by patient identifier to prevent the same patient from appearing in both training and testing sets.

### 3.2 Data Preprocessing

The preprocessing pipeline includes:

- Clinical data cleaning.
- Vital-sign aggregation.
- Laboratory measurement aggregation.
- Missing-value handling.
- Feature engineering.
- Categorical variable encoding.
- Constant-feature removal.
- Feature scaling.

The model uses measurements from the initial observation window rather than complete ICU stay information.

### 3.3 Models Evaluated

Three classification algorithms were evaluated:

| Model | Purpose |
|---|---|
| Logistic Regression | Linear classification baseline |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient-boosted decision trees |

Logistic Regression was selected using the predefined cross-validation model-selection procedure.

### 3.4 Model Performance

The following results were obtained on the held-out test dataset.

| Model | ROC-AUC | PR-AUC | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.813 | 0.306 | 0.389 | 0.350 |
| Random Forest | 0.855 | 0.352 | 0.444 | 0.327 |
| XGBoost | 0.845 | 0.476 | 0.444 | 0.381 |

**Evaluation interpretation:**

- ROC-AUC measures the model's ability to distinguish between the two outcome classes.
- PR-AUC evaluates the precision-recall trade-off and is particularly relevant for imbalanced datasets.
- Recall measures the proportion of expired cases correctly identified.
- F1-score combines precision and recall.

Logistic Regression was retained as the selected model based on the predefined cross-validation selection procedure, rather than selecting a model using the held-out test results.

---

## 4. System Architecture

AEGISCARE AI follows a modular application architecture.

### Application Layers

1. **Presentation Layer:** Streamlit interface, navigation, dashboards and interactive forms.
2. **Application Layer:** Patient management, medication interaction checking, deterioration simulation and Digital Twin operations.
3. **Machine Learning Layer:** Data preprocessing, feature adaptation, model inference and evaluation.
4. **Data Layer:** Synthetic patient information, medication interaction rules, model artefacts and supporting datasets.

### Data Flow

```text
             USER INPUT
                 |
                 v
        STREAMLIT APPLICATION
                 |
        +--------+--------+
        |        |        |
        v        v        v
     Patient   Drug     Patient
    Dashboard  Checker   Simulation
        |        |        |
        |        v        |
        |   Rule-Based    |
        |   Validation   |
        |                 |
        +--------+--------+
                 |
                 v
          ICU RISK MODULE
                 |
                 v
         FEATURE ADAPTER
                 |
                 v
       PREPROCESSING PIPELINE
                 |
                 v
       LOGISTIC REGRESSION
                 |
                 v
          RISK SCORE
                 |
                 v
       RESULT VISUALISATION
```

The ICU mortality model operates independently of the simulated deterioration scoring module.

---

## 5. Technology Stack

| Technology | Application |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| XGBoost | Model comparison |
| Matplotlib | Evaluation plots |
| Plotly | Interactive visualisations |
| Pytest | Automated testing |
| Git and GitHub | Version control and source management |

---

## 6. Project Structure

```text
ai_safety_system/
│
├── app.py
├── README.md
├── requirements.txt
├── pytest.ini
│
├── data/
│   ├── drug_interaction_rules.py
│   └── synthetic_patients.py
│
├── ml/
│   ├── data_audit.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── feature_adapter.py
│   │
│   ├── models/
│   │   ├── aegiscare_mortality_model.pkl
│   │   └── model_metadata.json
│   │
│   └── results/
│       ├── metrics.csv
│       ├── roc_curve.png
│       ├── pr_curve.png
│       └── confusion_matrix_best.png
│
├── tests/
│
├── ui/
│   └── styles.py
│
├── utils/
│
└── views/
    ├── dashboard.py
    ├── drug_interactions.py
    ├── deterioration.py
    ├── digital_twin.py
    └── icu_risk.py
```

**Dataset note:** The original eICU data and processed patient-level training CSV files are excluded from this repository. Access to the authorised dataset is required to reproduce model training.

---

## 7. Installation and Execution

### Prerequisites

- Python version compatible with the dependencies in `requirements.txt`.
- Git.
- A terminal or command prompt.

### Step 1: Clone the repository

```bash
git clone https://github.com/accidentaltacareer/ai_safety_system.git
```

### Step 2: Navigate into the project

```bash
cd ai_safety_system
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the environment

**Windows:**

```cmd
.venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Launch the application

```bash
streamlit run app.py
```

The application will open in your browser, normally at:

```text
http://localhost:8501
```

---

## 8. Testing

The project includes automated tests for application functionality, model inference and feature adaptation.

Run the application tests:

```bash
python -m pytest
```

Run the machine learning tests explicitly:

```bash
python -m pytest ml
```

The ML tests are invoked separately because the project's pytest configuration restricts default test discovery to the `tests/` directory.

---

## 9. Limitations

AEGISCARE AI is an academic proof-of-concept system. Its current limitations include:

- The system uses synthetic patient records for demonstration purposes.
- The deterioration prediction module uses simulated scoring rather than a trained predictive model.
- Drug interaction checking depends on the documented interaction rules and available medication information.
- The ICU mortality model was trained using the eICU demo dataset and has not undergone external clinical validation.
- The ICU mortality model's output is uncalibrated and must not be interpreted as a probability.
- The held-out test set contains only 18 expired cases, limiting the reliability of model comparisons.
- Model performance is weaker for ICU stays shorter than the initial 24-hour observation window.
- The system is not integrated with a live hospital information system or electronic health record platform.

---

## 10. Disclaimer

AEGISCARE AI is developed exclusively for academic, educational and demonstration purposes.

It is not a certified medical device, clinical decision-support system or substitute for professional medical judgement.

The system must not be used to diagnose patients, determine treatment plans, prescribe medication or make real-world clinical decisions.

All risk scores, simulated responses and medication interaction outputs should be interpreted within the limitations of the underlying demonstration data and implementation.

---

## 11. Project Information

**Project:** AEGISCARE AI – Integrated AI-Driven Healthcare Safety System

**Repository:** [ai_safety_system](https://github.com/accidentaltacareer/ai_safety_system)

**Development:** Academic engineering project

**Primary Domain:** Healthcare Technology, Artificial Intelligence and Machine Learning

**Purpose:** Demonstration of integrated healthcare safety modules using rule-based systems, machine learning and interactive patient simulation.

---

*Developed as an academic project exploring the integration of artificial intelligence, machine learning and software engineering in healthcare safety applications.*