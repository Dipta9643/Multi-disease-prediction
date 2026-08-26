# %% [markdown]
# # 📊 Synthetic Dataset Generator
# ## Multi-Disease Risk Prediction System
# 
# Generates a realistic synthetic dataset with **16 features** and **3 target variables**
# (Diabetes, Hypertension, Heart Disease) using medically-informed correlations.

# %%
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_DEFINITIONS, FEATURE_NAMES, DATASET_SIZE, RANDOM_SEED, DATA_DIR

np.random.seed(RANDOM_SEED)

print(f"🔧 Generating synthetic dataset with {DATASET_SIZE} samples...")
print(f"📋 Features: {len(FEATURE_NAMES)}")
print(f"🎯 Targets: diabetes, hypertension, heart_disease")
print("=" * 60)

# %% [markdown]
# ## Step 1: Generate Feature Values
# Each feature is generated with realistic distributions based on population data.

# %%
n = DATASET_SIZE

# --- Demographic ---
age = np.random.normal(45, 15, n).clip(18, 90).astype(int)
gender = np.random.binomial(1, 0.48, n)  # ~48% female

# --- Body Measurements ---
# BMI correlates with age slightly
bmi_base = np.random.normal(26, 5, n)
bmi_age_effect = (age - 45) * 0.05
bmi = (bmi_base + bmi_age_effect).clip(15, 50).round(1)

# --- Physical Activity ---
# Younger people tend to be more active
activity_level = np.array([
    np.random.choice([1, 2, 3], p=[0.2, 0.4, 0.4]) if a < 30
    else np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3]) if a < 50
    else np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
    for a in age
])

# Exercise duration correlates with activity level
exercise_base = activity_level * 20 + np.random.normal(0, 15, n)
exercise_duration = exercise_base.clip(0, 120).astype(int)

# --- Diet & Nutrition ---
sugar_intake = np.random.choice([0, 1, 2], n, p=[0.3, 0.45, 0.25])
salt_intake = np.random.choice([0, 1, 2], n, p=[0.25, 0.50, 0.25])
fruit_veg = np.random.normal(4, 2, n).clip(0, 10).round(1)

# --- Sleep ---
sleep_duration = np.random.normal(7, 1.5, n).clip(3, 12).round(1)
sleep_quality = np.random.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])

# --- Mental Health ---
stress_level = np.random.choice([0, 1, 2], n, p=[0.3, 0.45, 0.25])

# --- Habits ---
smoking_status = np.random.choice([0, 1, 2], n, p=[0.55, 0.20, 0.25])
alcohol_consumption = np.random.choice([0, 1, 2], n, p=[0.40, 0.40, 0.20])

# --- Family History ---
family_history_diabetes = np.random.binomial(1, 0.25, n)
family_history_hypertension = np.random.binomial(1, 0.30, n)
family_history_heart_disease = np.random.binomial(1, 0.20, n)

print("✅ Feature values generated")

# %% [markdown]
# ## Step 2: Generate Target Variables
# 
# Target variables are generated using **weighted logistic functions** based on 
# established medical risk factors. This ensures realistic correlations between 
# features and disease outcomes.

# %%
def sigmoid(x):
    """Sigmoid function for probability mapping."""
    return 1 / (1 + np.exp(-x))

# --- Diabetes Risk Score ---
# Key factors: BMI, sugar intake, age, low activity, family history, sleep
diabetes_score = (
    0.08 * (bmi - 25) +             # High BMI increases risk
    0.5 * sugar_intake +              # High sugar increases risk
    0.03 * (age - 40) +              # Older age increases risk
    -0.4 * (activity_level - 1) +    # High activity reduces risk
    0.8 * family_history_diabetes +   # Family history increases risk
    -0.15 * (sleep_duration - 5) +   # Too little sleep increases risk
    0.3 * stress_level +             # Stress increases risk
    0.2 * smoking_status +           # Smoking increases risk
    -0.1 * fruit_veg +              # Fruits/vegetables reduce risk
    np.random.normal(0, 0.5, n)      # Random noise
)
diabetes_prob = sigmoid(diabetes_score - 1.0)
diabetes = (np.random.random(n) < diabetes_prob).astype(int)

# --- Hypertension Risk Score ---
# Key factors: salt intake, stress, BMI, alcohol, age, low activity
hypertension_score = (
    0.7 * salt_intake +              # High salt increases risk
    0.5 * stress_level +             # High stress increases risk
    0.06 * (bmi - 25) +             # High BMI increases risk
    0.4 * alcohol_consumption +      # Heavy alcohol increases risk
    0.03 * (age - 40) +             # Older age increases risk
    -0.3 * (activity_level - 1) +   # Activity reduces risk
    0.6 * family_history_hypertension +  # Family history
    -0.1 * (sleep_quality - 1) +    # Poor sleep increases risk
    0.3 * smoking_status +           # Smoking increases risk
    np.random.normal(0, 0.5, n)      # Random noise
)
hypertension_prob = sigmoid(hypertension_score - 1.5)
diabetes_hyp_interaction = diabetes * 0.3  # Diabetes increases hypertension risk
hypertension_prob = sigmoid(hypertension_score - 1.5 + diabetes_hyp_interaction)
hypertension = (np.random.random(n) < hypertension_prob).astype(int)

# --- Heart Disease Risk Score ---
# Key factors: age, cholesterol proxy (BMI+diet), smoking, BP proxy, BMI
heart_disease_score = (
    0.05 * (age - 40) +             # Older age increases risk
    0.06 * (bmi - 25) +             # High BMI increases risk
    0.5 * smoking_status +           # Smoking greatly increases risk
    0.4 * alcohol_consumption +      # Heavy alcohol increases risk
    0.4 * stress_level +            # Stress increases risk
    -0.35 * (activity_level - 1) +  # Activity reduces risk
    0.7 * family_history_heart_disease +  # Family history
    -0.15 * (exercise_duration / 30) +  # Exercise reduces risk
    -0.1 * fruit_veg +             # Diet reduces risk
    0.4 * hypertension +            # Hypertension increases heart risk
    0.3 * diabetes +                # Diabetes increases heart risk
    -0.15 * sleep_quality +         # Poor sleep increases risk
    np.random.normal(0, 0.5, n)     # Random noise
)
heart_disease_prob = sigmoid(heart_disease_score - 2.0)
heart_disease = (np.random.random(n) < heart_disease_prob).astype(int)

print("✅ Target variables generated")

# %% [markdown]
# ## Step 3: Create DataFrame and Save

# %%
# Build DataFrame
df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "activity_level": activity_level,
    "exercise_duration": exercise_duration,
    "sugar_intake": sugar_intake,
    "salt_intake": salt_intake,
    "fruit_veg_consumption": fruit_veg,
    "sleep_duration": sleep_duration,
    "sleep_quality": sleep_quality,
    "stress_level": stress_level,
    "smoking_status": smoking_status,
    "alcohol_consumption": alcohol_consumption,
    "family_history_diabetes": family_history_diabetes,
    "family_history_hypertension": family_history_hypertension,
    "family_history_heart_disease": family_history_heart_disease,
    "diabetes": diabetes,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
})

# Save
os.makedirs(DATA_DIR, exist_ok=True)
output_path = os.path.join(DATA_DIR, "dataset.csv")
df.to_csv(output_path, index=False)
print(f"\n💾 Dataset saved to: {output_path}")

# %% [markdown]
# ## Step 4: Dataset Summary Statistics

# %%
print("\n" + "=" * 60)
print("📊 DATASET SUMMARY")
print("=" * 60)
print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"📋 Features: {len(FEATURE_NAMES)}")
print(f"🎯 Targets: diabetes, hypertension, heart_disease")

print("\n--- Class Distribution ---")
for target in ["diabetes", "hypertension", "heart_disease"]:
    counts = df[target].value_counts()
    total = len(df)
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    pct = (pos / total) * 100
    print(f"  {target:25s}: {pos:5d} positive ({pct:5.1f}%) | {neg:5d} negative ({100-pct:5.1f}%)")

print("\n--- Feature Statistics ---")
print(df[FEATURE_NAMES].describe().round(2).to_string())

print("\n--- First 5 Rows ---")
print(df.head().to_string())

print("\n--- Missing Values ---")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✅ No missing values!")
else:
    print(missing[missing > 0])

print("\n✅ Dataset generation complete!")
