"""
Configuration for Multi-Disease Risk Prediction System
Defines features, encodings, model paths, and app settings.
"""

import os

# ============================================================
# App Settings
# ============================================================
APP_NAME = "VitalsAlign"
APP_VERSION = "1.0.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
DB_PATH = os.path.join(BASE_DIR, "risk_history.db")

# ============================================================
# Feature Definitions (16 Features)
# ============================================================
# Each feature has: name, type, encoding/range, display_name, category, description

FEATURE_DEFINITIONS = {
    # --- Demographic ---
    "age": {
        "type": "numerical",
        "range": (18, 90),
        "default": 35,
        "display_name": "Age",
        "category": "Demographic",
        "description": "Your current age in years",
        "unit": "years"
    },
    "gender": {
        "type": "categorical",
        "encoding": {"Male": 0, "Female": 1},
        "default": 0,
        "display_name": "Gender",
        "category": "Demographic",
        "description": "Biological gender",
        "unit": ""
    },

    # --- Body Measurements ---
    "bmi": {
        "type": "numerical",
        "range": (15.0, 50.0),
        "default": 24.0,
        "display_name": "BMI (Body Mass Index)",
        "category": "Body Measurements",
        "description": "Weight(kg) / Height(m)². Normal: 18.5–24.9",
        "unit": "kg/m²"
    },

    # --- Physical Activity ---
    "activity_level": {
        "type": "categorical",
        "encoding": {"Low": 1, "Moderate": 2, "High": 3},
        "default": 2,
        "display_name": "Physical Activity Level",
        "category": "Physical Activity",
        "description": "Your overall daily physical activity level",
        "unit": ""
    },
    "exercise_duration": {
        "type": "numerical",
        "range": (0, 120),
        "default": 30,
        "display_name": "Exercise Duration",
        "category": "Physical Activity",
        "description": "Average daily exercise time",
        "unit": "min/day"
    },

    # --- Diet & Nutrition ---
    "sugar_intake": {
        "type": "categorical",
        "encoding": {"Low": 0, "Moderate": 1, "High": 2},
        "default": 1,
        "display_name": "Sugar Intake",
        "category": "Diet & Nutrition",
        "description": "Daily sugar consumption level",
        "unit": ""
    },
    "salt_intake": {
        "type": "categorical",
        "encoding": {"Low": 0, "Moderate": 1, "High": 2},
        "default": 1,
        "display_name": "Salt Intake",
        "category": "Diet & Nutrition",
        "description": "Daily salt consumption level",
        "unit": ""
    },
    "fruit_veg_consumption": {
        "type": "numerical",
        "range": (0, 10),
        "default": 4,
        "display_name": "Fruit & Vegetable Servings",
        "category": "Diet & Nutrition",
        "description": "Combined daily servings of fruits and vegetables",
        "unit": "servings/day"
    },

    # --- Sleep ---
    "sleep_duration": {
        "type": "numerical",
        "range": (3, 12),
        "default": 7,
        "display_name": "Sleep Duration",
        "category": "Sleep",
        "description": "Average hours of sleep per night",
        "unit": "hours"
    },
    "sleep_quality": {
        "type": "categorical",
        "encoding": {"Poor": 0, "Average": 1, "Good": 2},
        "default": 1,
        "display_name": "Sleep Quality",
        "category": "Sleep",
        "description": "Self-assessed quality of sleep",
        "unit": ""
    },

    # --- Mental Health ---
    "stress_level": {
        "type": "categorical",
        "encoding": {"Low": 0, "Medium": 1, "High": 2},
        "default": 1,
        "display_name": "Stress Level",
        "category": "Mental Health",
        "description": "Your current stress level",
        "unit": ""
    },

    # --- Habits ---
    "smoking_status": {
        "type": "categorical",
        "encoding": {"Never": 0, "Former": 1, "Current": 2},
        "default": 0,
        "display_name": "Smoking Status",
        "category": "Habits",
        "description": "Your smoking history",
        "unit": ""
    },
    "alcohol_consumption": {
        "type": "categorical",
        "encoding": {"None": 0, "Moderate": 1, "Heavy": 2},
        "default": 0,
        "display_name": "Alcohol Consumption",
        "category": "Habits",
        "description": "Frequency and amount of alcohol consumed",
        "unit": ""
    },

    # --- Family History ---
    "family_history_diabetes": {
        "type": "categorical",
        "encoding": {"No": 0, "Yes": 1},
        "default": 0,
        "display_name": "Family History of Diabetes",
        "category": "Family History",
        "description": "Do any close relatives have diabetes?",
        "unit": ""
    },
    "family_history_hypertension": {
        "type": "categorical",
        "encoding": {"No": 0, "Yes": 1},
        "default": 0,
        "display_name": "Family History of Hypertension",
        "category": "Family History",
        "description": "Do any close relatives have hypertension?",
        "unit": ""
    },
    "family_history_heart_disease": {
        "type": "categorical",
        "encoding": {"No": 0, "Yes": 1},
        "default": 0,
        "display_name": "Family History of Heart Disease",
        "category": "Family History",
        "description": "Do any close relatives have heart disease?",
        "unit": ""
    },
}

# Ordered list of feature names (order matters for model input)
FEATURE_NAMES = [
    "age", "gender", "bmi",
    "activity_level", "exercise_duration",
    "sugar_intake", "salt_intake", "fruit_veg_consumption",
    "sleep_duration", "sleep_quality",
    "stress_level",
    "smoking_status", "alcohol_consumption",
    "family_history_diabetes", "family_history_hypertension",
    "family_history_heart_disease",
]

# Feature categories in display order (for multi-step form)
FEATURE_CATEGORIES = [
    "Demographic",
    "Body Measurements",
    "Physical Activity",
    "Diet & Nutrition",
    "Sleep",
    "Mental Health",
    "Habits",
    "Family History",
]

# ============================================================
# Target Diseases
# ============================================================
DISEASES = {
    "diabetes": {
        "display_name": "Diabetes",
        "icon": "🩸",
        "color": "#ff6b6b",
        "description": "Type 2 Diabetes Mellitus"
    },
    "hypertension": {
        "display_name": "Hypertension",
        "icon": "💓",
        "color": "#ffd93d",
        "description": "High Blood Pressure"
    },
    "heart_disease": {
        "display_name": "Heart Disease",
        "icon": "❤️",
        "color": "#00d4aa",
        "description": "Cardiovascular Disease"
    },
}

# ============================================================
# Model Configuration
# ============================================================
MODEL_CONFIGS = {
    "random_forest": {
        "display_name": "Random Forest",
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [5, 8, 12],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced"],
        }
    },
    "lightgbm": {
        "display_name": "LightGBM",
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 8],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
            "class_weight": ["balanced"],
        }
    },
    "xgboost": {
        "display_name": "XGBoost",
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 8],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
            "scale_pos_weight": [1],  # Will be overridden per disease
        }
    },
}

# ============================================================
# Risk Level Thresholds
# ============================================================
RISK_LEVELS = {
    "low": {"max_percentile": 33, "color": "#00d4aa", "label": "Low Risk"},
    "moderate": {"max_percentile": 66, "color": "#ffd93d", "label": "Moderate Risk"},
    "high": {"max_percentile": 100, "color": "#ff6b6b", "label": "High Risk"},
}

# ============================================================
# Dataset Configuration
# ============================================================
DATASET_SIZE = 5000
RANDOM_SEED = 42
TEST_SIZE = 0.2
