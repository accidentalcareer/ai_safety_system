"""
Synthetic patient data for the demo.

EVERYTHING IN THIS FILE IS FICTIONAL.
The patients, medications, vital signs and lab results below were invented
for a college project. They do not describe any real person and must not be
used for clinical decisions.

The data is generated with a fixed random seed, so every run of the app
shows exactly the same numbers (useful for demos and tests).
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# All synthetic readings end at this fixed time so the data is reproducible.
REFERENCE_TIME = datetime(2026, 10, 10, 8, 0)

# 72 hours of history, one reading every 2 hours.
HOURS_OF_HISTORY = 72
READING_INTERVAL_HOURS = 2


# ---------------------------------------------------------------------------
# Typical adult reference ranges, used ONLY to shade charts and label values.
# Real reference ranges differ between labs and patient groups.
# ---------------------------------------------------------------------------
VITAL_SIGNS = {
    "heart_rate": {"short": "Heart rate", "label": "Heart rate", "unit": "bpm", "low": 60, "high": 100},
    "systolic_bp": {"short": "Systolic BP", "label": "Systolic BP", "unit": "mmHg", "low": 90, "high": 140},
    "diastolic_bp": {"short": "Diastolic BP", "label": "Diastolic BP", "unit": "mmHg", "low": 60, "high": 90},
    "spo2": {"short": "SpO₂", "label": "Oxygen saturation (SpO₂)", "unit": "%", "low": 95, "high": 100},
    "resp_rate": {"short": "Resp. rate", "label": "Respiratory rate", "unit": "/min", "low": 12, "high": 20},
    "temperature": {"short": "Temperature", "label": "Temperature", "unit": "°C", "low": 36.1, "high": 37.8},
}


# ---------------------------------------------------------------------------
# Patient profiles. Each "vitals_profile" describes where each vital sign
# starts, where it ends after 72 hours, and how noisy it is. The generator
# draws a smooth line between start and end and adds random noise.
# ---------------------------------------------------------------------------
PATIENTS = [
    {
        "patient_id": "SYN-001",
        "display_name": "Synthetic Patient A",
        "age": 68,
        "sex": "Male",
        "ward": "Cardiology – Bed 12",
        "admission_date": "2026-10-07",
        "attending": "Dr. Demo (fictional)",
        "allergies": ["Penicillin"],
        "diagnoses": ["Atrial fibrillation", "Type 2 diabetes mellitus", "Hypertension"],
        "scenario": "Stable cardiac patient on long-term anticoagulation.",
        "medications": [
            {"name": "Warfarin", "dose": "5 mg", "route": "Oral", "frequency": "Once daily", "indication": "Atrial fibrillation"},
            {"name": "Metformin", "dose": "500 mg", "route": "Oral", "frequency": "Twice daily", "indication": "Type 2 diabetes"},
            {"name": "Lisinopril", "dose": "10 mg", "route": "Oral", "frequency": "Once daily", "indication": "Hypertension"},
            {"name": "Atorvastatin", "dose": "20 mg", "route": "Oral", "frequency": "Once daily (night)", "indication": "Cholesterol"},
            {"name": "Aspirin", "dose": "75 mg", "route": "Oral", "frequency": "Once daily", "indication": "Cardiovascular prevention"},
        ],
        "vitals_profile": {
            "heart_rate": (88, 82, 4),
            "systolic_bp": (148, 136, 5),
            "diastolic_bp": (88, 82, 3),
            "spo2": (96, 97, 0.7),
            "resp_rate": (17, 16, 1),
            "temperature": (36.8, 36.7, 0.15),
        },
        "labs": [
            ("Hemoglobin", 13.2, "g/dL", 13.5, 17.5),
            ("White blood cells", 7.8, "×10⁹/L", 4.0, 11.0),
            ("Platelets", 215, "×10⁹/L", 150, 400),
            ("Sodium", 139, "mmol/L", 135, 145),
            ("Potassium", 4.9, "mmol/L", 3.5, 5.1),
            ("Creatinine", 1.3, "mg/dL", 0.7, 1.3),
            ("Glucose (fasting)", 142, "mg/dL", 70, 100),
            ("HbA1c", 7.4, "%", 4.0, 5.6),
            ("INR", 2.6, "ratio", 0.8, 1.2),
            ("Lactate", 1.1, "mmol/L", 0.5, 2.0),
        ],
    },
    {
        "patient_id": "SYN-002",
        "display_name": "Synthetic Patient B",
        "age": 54,
        "sex": "Female",
        "ward": "Respiratory – Bed 4",
        "admission_date": "2026-10-08",
        "attending": "Dr. Example (fictional)",
        "allergies": ["Sulfonamides"],
        "diagnoses": ["COPD exacerbation", "Community-acquired pneumonia"],
        "scenario": "Respiratory patient whose synthetic vitals drift over time.",
        "medications": [
            {"name": "Salbutamol", "dose": "2.5 mg", "route": "Nebulised", "frequency": "Every 4 hours", "indication": "Bronchospasm"},
            {"name": "Prednisolone", "dose": "30 mg", "route": "Oral", "frequency": "Once daily", "indication": "COPD exacerbation"},
            {"name": "Clarithromycin", "dose": "500 mg", "route": "Oral", "frequency": "Twice daily", "indication": "Pneumonia"},
            {"name": "Ceftriaxone", "dose": "1 g", "route": "IV", "frequency": "Once daily", "indication": "Pneumonia"},
            {"name": "Omeprazole", "dose": "20 mg", "route": "Oral", "frequency": "Once daily", "indication": "Gastric protection"},
        ],
        "vitals_profile": {
            "heart_rate": (92, 112, 4),
            "systolic_bp": (128, 108, 5),
            "diastolic_bp": (78, 66, 3),
            "spo2": (95, 90, 0.8),
            "resp_rate": (20, 26, 1.2),
            "temperature": (37.6, 38.4, 0.2),
        },
        "labs": [
            ("Hemoglobin", 12.4, "g/dL", 12.0, 15.5),
            ("White blood cells", 15.6, "×10⁹/L", 4.0, 11.0),
            ("Platelets", 310, "×10⁹/L", 150, 400),
            ("Sodium", 134, "mmol/L", 135, 145),
            ("Potassium", 3.6, "mmol/L", 3.5, 5.1),
            ("Creatinine", 1.0, "mg/dL", 0.6, 1.1),
            ("C-reactive protein", 96, "mg/L", 0, 10),
            ("Glucose (random)", 168, "mg/dL", 70, 140),
            ("Lactate", 2.4, "mmol/L", 0.5, 2.0),
            ("Procalcitonin", 0.9, "ng/mL", 0.0, 0.5),
        ],
    },
    {
        "patient_id": "SYN-003",
        "display_name": "Synthetic Patient C",
        "age": 37,
        "sex": "Female",
        "ward": "General Medicine – Bed 21",
        "admission_date": "2026-10-09",
        "attending": "Dr. Sample (fictional)",
        "allergies": ["None known"],
        "diagnoses": ["Migraine", "Depression", "Hypothyroidism"],
        "scenario": "Younger patient admitted for observation; vitals within typical ranges.",
        "medications": [
            {"name": "Sertraline", "dose": "50 mg", "route": "Oral", "frequency": "Once daily", "indication": "Depression"},
            {"name": "Sumatriptan", "dose": "50 mg", "route": "Oral", "frequency": "As needed", "indication": "Migraine"},
            {"name": "Levothyroxine", "dose": "75 mcg", "route": "Oral", "frequency": "Once daily (morning)", "indication": "Hypothyroidism"},
            {"name": "Ibuprofen", "dose": "400 mg", "route": "Oral", "frequency": "Every 8 hours as needed", "indication": "Pain"},
        ],
        "vitals_profile": {
            "heart_rate": (76, 72, 4),
            "systolic_bp": (118, 116, 4),
            "diastolic_bp": (76, 74, 3),
            "spo2": (98, 98, 0.6),
            "resp_rate": (14, 14, 1),
            "temperature": (36.7, 36.6, 0.15),
        },
        "labs": [
            ("Hemoglobin", 13.1, "g/dL", 12.0, 15.5),
            ("White blood cells", 6.2, "×10⁹/L", 4.0, 11.0),
            ("Platelets", 260, "×10⁹/L", 150, 400),
            ("Sodium", 140, "mmol/L", 135, 145),
            ("Potassium", 4.1, "mmol/L", 3.5, 5.1),
            ("Creatinine", 0.8, "mg/dL", 0.6, 1.1),
            ("TSH", 3.1, "mIU/L", 0.4, 4.0),
            ("Glucose (fasting)", 92, "mg/dL", 70, 100),
            ("ALT", 24, "U/L", 7, 56),
            ("Lactate", 0.9, "mmol/L", 0.5, 2.0),
        ],
    },
]

# Physical limits so random noise never produces impossible values.
_CLIP_LIMITS = {
    "heart_rate": (30, 200),
    "systolic_bp": (60, 220),
    "diastolic_bp": (30, 130),
    "spo2": (70, 100),
    "resp_rate": (6, 45),
    "temperature": (34.0, 41.5),
}

# One decimal place for temperature, whole numbers for everything else.
_DECIMALS = {"temperature": 1}


def list_patients():
    """Return a small table of all patients for the selector."""
    rows = [
        {
            "patient_id": p["patient_id"],
            "display_name": p["display_name"],
            "age": p["age"],
            "sex": p["sex"],
            "ward": p["ward"],
        }
        for p in PATIENTS
    ]
    return pd.DataFrame(rows)


def get_patient(patient_id):
    """Return the full profile dictionary for one patient."""
    for patient in PATIENTS:
        if patient["patient_id"] == patient_id:
            return patient
    raise KeyError(f"Unknown patient id: {patient_id}")


def get_medications(patient_id):
    """Return the current medication list as a DataFrame."""
    return medications_table(get_patient(patient_id)["medications"])


def medications_table(medications):
    """Turn a list of medication dicts into the display table (empty list allowed)."""
    meds = pd.DataFrame(medications, columns=["name", "dose", "route", "frequency", "indication"])
    meds.columns = ["Medication", "Dose", "Route", "Frequency", "Indication"]
    return meds


def generate_vitals(patient_id):
    """
    Create a time series of synthetic vital signs for one patient.

    Each vital sign follows a straight line from its start value to its end
    value, plus random noise. The seed is derived from the patient ID so the
    output is identical on every run.
    """
    patient = get_patient(patient_id)
    seed = int(patient_id.split("-")[1])
    rng = np.random.default_rng(seed)

    n_readings = HOURS_OF_HISTORY // READING_INTERVAL_HOURS + 1
    start_time = REFERENCE_TIME - timedelta(hours=HOURS_OF_HISTORY)
    timestamps = [start_time + timedelta(hours=i * READING_INTERVAL_HOURS) for i in range(n_readings)]

    data = {"timestamp": timestamps}
    for vital, (start, end, noise) in patient["vitals_profile"].items():
        trend = np.linspace(start, end, n_readings)
        values = trend + rng.normal(0, noise, n_readings)
        low, high = _CLIP_LIMITS[vital]
        values = np.clip(values, low, high)
        data[vital] = np.round(values, _DECIMALS.get(vital, 0))

    return pd.DataFrame(data)


def get_lab_results(patient_id):
    """Return lab results with a simple Low / Normal / High flag."""
    return lab_results_table(get_patient(patient_id)["labs"])


def lab_results_table(labs):
    """Build the lab table from a list of (test, value, unit, low, high) entries."""
    rows = []
    for test, value, unit, ref_low, ref_high in labs:
        rows.append(
            {
                "Test": test,
                "Value": value,
                "Unit": unit,
                "Reference range": f"{ref_low} – {ref_high}",
                "Status": classify_value(value, ref_low, ref_high),
            }
        )
    return pd.DataFrame(rows, columns=["Test", "Value", "Unit", "Reference range", "Status"])


def classify_value(value, low, high):
    """
    Compare a value against a reference range.

    This is a plain range check for display purposes. It is NOT a
    diagnosis or a clinical prediction.
    """
    if value < low:
        return "Low"
    if value > high:
        return "High"
    return "Normal"
