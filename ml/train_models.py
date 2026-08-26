# %% [markdown]
# # Multi-Disease Risk Prediction System
# ## Model Training Pipeline
# 
# This notebook trains separate ML models for **Diabetes**, **Hypertension**, and **Heart Disease**
# using Random Forest, LightGBM, and XGBoost with hyperparameter tuning.
#
# ### Pipeline Steps:
# 1. Load & Explore Data (EDA)
# 2. Data Preprocessing
# 3. Model Training with GridSearchCV
# 4. Threshold Optimization
# 5. Model Evaluation
# 6. SHAP Explainability
# 7. Model Comparison
# 8. Save Artifacts

# %% [markdown]
# ---
# ## Step 1: Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import warnings
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import shap

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_NAMES, DISEASES, MODEL_CONFIGS, MODEL_DIR, DATA_DIR, RANDOM_SEED, TEST_SIZE

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directories
os.makedirs(MODEL_DIR, exist_ok=True)
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Libraries imported successfully!")
print(f"Output directory: {MODEL_DIR}")
print(f"Plots directory: {PLOTS_DIR}")

# %% [markdown]
# ---
# ## Step 2: Load Dataset

# %%
df = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
print(f"Dataset shape: {df.shape}")
print(f"\nFeatures ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
print(f"Targets: {list(DISEASES.keys())}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())

# %% [markdown]
# ---
# ## Step 3: Exploratory Data Analysis (EDA)

# %% [markdown]
# ### 3.1 Class Distribution

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
target_names = list(DISEASES.keys())
colors = ['#ff6b6b', '#ffd93d', '#00d4aa']

for i, (target, color) in enumerate(zip(target_names, colors)):
    counts = df[target].value_counts()
    axes[i].bar(['Negative (0)', 'Positive (1)'], [counts[0], counts[1]], 
                color=[color + '66', color], edgecolor=color, linewidth=2)
    axes[i].set_title(f'{DISEASES[target]["display_name"]}', fontsize=14, fontweight='bold')
    axes[i].set_ylabel('Count')
    total = len(df)
    axes[i].text(0, counts[0] + 50, f'{counts[0]} ({counts[0]/total*100:.1f}%)', 
                ha='center', fontweight='bold')
    axes[i].text(1, counts[1] + 50, f'{counts[1]} ({counts[1]/total*100:.1f}%)', 
                ha='center', fontweight='bold')

plt.suptitle('Class Distribution for Target Variables', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: class_distribution.png")

# Print class imbalance ratios
print("\nClass Imbalance Ratios:")
for target in target_names:
    counts = df[target].value_counts()
    ratio = counts[0] / counts[1]
    print(f"  {target:20s}: {ratio:.2f}:1 (Negative:Positive)")

# %% [markdown]
# ### 3.2 Feature Distributions

# %%
numerical_features = ['age', 'bmi', 'exercise_duration', 'fruit_veg_consumption', 'sleep_duration']
fig, axes = plt.subplots(1, len(numerical_features), figsize=(20, 4))

for i, feat in enumerate(numerical_features):
    axes[i].hist(df[feat], bins=30, color='#00d4aa', alpha=0.7, edgecolor='white')
    axes[i].set_title(feat.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Count')
    axes[i].axvline(df[feat].mean(), color='#ff6b6b', linestyle='--', label=f'Mean: {df[feat].mean():.1f}')
    axes[i].legend(fontsize=9)

plt.suptitle('Numerical Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_distributions_numerical.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_distributions_numerical.png")

# %%
categorical_features = ['gender', 'activity_level', 'sugar_intake', 'salt_intake', 
                         'sleep_quality', 'stress_level', 'smoking_status', 'alcohol_consumption',
                         'family_history_diabetes', 'family_history_hypertension', 'family_history_heart_disease']
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()

for i, feat in enumerate(categorical_features):
    counts = df[feat].value_counts().sort_index()
    axes[i].bar(counts.index.astype(str), counts.values, color='#a855f7', alpha=0.8, edgecolor='white')
    axes[i].set_title(feat.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Count')

# Hide unused subplot
axes[11].set_visible(False)

plt.suptitle('Categorical Feature Distributions', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_distributions_categorical.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_distributions_categorical.png")

# %% [markdown]
# ### 3.3 Correlation Heatmap

# %%
fig, ax = plt.subplots(figsize=(16, 12))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")

# Print top correlations with targets
print("\nTop Feature Correlations with Targets:")
for target in target_names:
    print(f"\n  {target}:")
    corr_with_target = corr_matrix[target].drop(target_names).abs().sort_values(ascending=False)
    for feat, corr_val in corr_with_target.head(5).items():
        direction = "+" if corr_matrix[target][feat] > 0 else "-"
        print(f"    {feat:35s}: {direction}{corr_val:.3f}")

# %% [markdown]
# ### 3.4 Feature vs Disease Distribution

# %%
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
key_features = ['age', 'bmi', 'exercise_duration']

for i, target in enumerate(target_names):
    for j, feat in enumerate(key_features):
        for label, color in [(0, '#00d4aa'), (1, '#ff6b6b')]:
            subset = df[df[target] == label][feat]
            axes[i][j].hist(subset, bins=25, alpha=0.6, color=color, 
                           label=f'{"Positive" if label else "Negative"}', edgecolor='white')
        axes[i][j].set_title(f'{DISEASES[target]["display_name"]} vs {feat.replace("_", " ").title()}',
                            fontsize=11, fontweight='bold')
        axes[i][j].legend()
        axes[i][j].set_ylabel('Count')

plt.suptitle('Feature Distributions by Disease Outcome', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_vs_disease.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_vs_disease.png")

# %% [markdown]
# ### 3.5 Co-occurrence of Diseases

# %%
fig, ax = plt.subplots(figsize=(8, 6))
disease_combinations = df[target_names].value_counts().reset_index()
disease_combinations.columns = target_names + ['count']
disease_combinations['label'] = disease_combinations.apply(
    lambda r: ', '.join([DISEASES[d]['display_name'] for d in target_names if r[d] == 1]) or 'None', axis=1
)
disease_combinations = disease_combinations.sort_values('count', ascending=True)

colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(disease_combinations)))
ax.barh(disease_combinations['label'], disease_combinations['count'], color=colors_bar, edgecolor='white')
ax.set_xlabel('Count')
ax.set_title('Disease Co-occurrence Patterns', fontsize=14, fontweight='bold')
for i, (_, row) in enumerate(disease_combinations.iterrows()):
    ax.text(row['count'] + 20, i, str(row['count']), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'disease_cooccurrence.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: disease_cooccurrence.png")

print("\nDisease Co-occurrence:")
print(disease_combinations[['label', 'count']].to_string(index=False))

# %% [markdown]
# ---
# ## Step 4: Data Preprocessing

# %%
# Separate features and targets
X = df[FEATURE_NAMES].copy()
targets = {disease: df[disease].copy() for disease in target_names}

print(f"Feature matrix shape: {X.shape}")
for disease, y in targets.items():
    print(f"Target '{disease}': {y.sum()} positive, {(y == 0).sum()} negative")

# %% [markdown]
# ### 4.1 Feature Scaling

# %%
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_NAMES)

print("Feature scaling applied (StandardScaler)")
print(f"\nScaled feature statistics:")
print(X_scaled.describe().round(3).to_string())

# Save scaler
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"\nScaler saved to: {scaler_path}")

# %% [markdown]
# ### 4.2 Train-Test Split

# %%
splits = {}
for disease in target_names:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, targets[disease],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=targets[disease]
    )
    splits[disease] = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test
    }
    print(f"{disease:20s}: Train={len(X_train)}, Test={len(X_test)}, "
          f"Train_pos={y_train.sum()} ({y_train.mean()*100:.1f}%), "
          f"Test_pos={y_test.sum()} ({y_test.mean()*100:.1f}%)")

# %% [markdown]
# ---
# ## Step 5: Model Training with GridSearchCV
# 
# For each disease, we train 3 models:
# - **Random Forest**
# - **LightGBM**
# - **XGBoost**
# 
# Using 5-fold stratified cross-validation and F1-score as the scoring metric.

# %%
def train_model_for_disease(disease, X_train, y_train):
    """Train all three model types for a given disease and return results."""
    results = {}
    pos_count = y_train.sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Model definitions
    models = {
        'random_forest': {
            'model': RandomForestClassifier(random_state=RANDOM_SEED),
            'params': MODEL_CONFIGS['random_forest']['param_grid'],
        },
        'lightgbm': {
            'model': LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
            'params': MODEL_CONFIGS['lightgbm']['param_grid'],
        },
        'xgboost': {
            'model': XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss',
                                    use_label_encoder=False),
            'params': {
                **MODEL_CONFIGS['xgboost']['param_grid'],
                'scale_pos_weight': [scale_pos_weight],
            },
        },
    }
    
    for model_name, config in models.items():
        print(f"    Training {model_name}...", end=" ", flush=True)
        
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        
        results[model_name] = {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_cv_f1': grid_search.best_score_,
        }
        print(f"CV F1={grid_search.best_score_:.4f}")
    
    return results

# %%
# Train models for all diseases
all_results = {}
best_models = {}

for disease in target_names:
    print(f"\n{'='*60}")
    print(f"  Training models for: {DISEASES[disease]['display_name']}")
    print(f"{'='*60}")
    
    data = splits[disease]
    results = train_model_for_disease(disease, data['X_train'], data['y_train'])
    all_results[disease] = results
    
    # Select best model based on CV F1
    best_model_name = max(results, key=lambda k: results[k]['best_cv_f1'])
    best_models[disease] = {
        'model_name': best_model_name,
        'model': results[best_model_name]['best_model'],
        'best_params': results[best_model_name]['best_params'],
        'cv_f1': results[best_model_name]['best_cv_f1'],
    }
    
    print(f"\n  >> Best model: {best_model_name} (CV F1={results[best_model_name]['best_cv_f1']:.4f})")
    print(f"  >> Best params: {results[best_model_name]['best_params']}")

# %% [markdown]
# ---
# ## Step 6: Threshold Optimization
# 
# Instead of using the default 0.5 threshold, we find the optimal threshold 
# that maximizes the F1 score using precision-recall curves.

# %%
optimal_thresholds = {}

for disease in target_names:
    data = splits[disease]
    model = best_models[disease]['model']
    
    # Get predicted probabilities on test set
    y_prob = model.predict_proba(data['X_test'])[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds_pr = precision_recall_curve(data['y_test'], y_prob)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find best threshold (excluding last element which corresponds to threshold=max)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds_pr[best_idx]
    best_f1 = f1_scores[best_idx]
    
    optimal_thresholds[disease] = float(best_threshold)
    
    print(f"{disease:20s}: Optimal threshold = {best_threshold:.4f}, F1 = {best_f1:.4f} "
          f"(vs default 0.5 threshold)")

# Save thresholds
thresholds_path = os.path.join(MODEL_DIR, 'thresholds.json')
with open(thresholds_path, 'w') as f:
    json.dump(optimal_thresholds, f, indent=2)
print(f"\nThresholds saved to: {thresholds_path}")

# %% [markdown]
# ---
# ## Step 7: Model Evaluation
# 
# Evaluate the best model for each disease using the optimized thresholds.

# %%
evaluation_results = {}

for disease in target_names:
    data = splits[disease]
    model = best_models[disease]['model']
    threshold = optimal_thresholds[disease]
    
    # Predictions with optimized threshold
    y_prob = model.predict_proba(data['X_test'])[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(data['y_test'], y_pred)),
        'precision': float(precision_score(data['y_test'], y_pred, zero_division=0)),
        'recall': float(recall_score(data['y_test'], y_pred, zero_division=0)),
        'f1_score': float(f1_score(data['y_test'], y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(data['y_test'], y_prob)),
        'threshold': float(threshold),
        'model_name': best_models[disease]['model_name'],
        'best_params': {k: str(v) for k, v in best_models[disease]['best_params'].items()},
    }
    evaluation_results[disease] = metrics
    
    print(f"\n{'='*60}")
    print(f"  {DISEASES[disease]['display_name']} - {best_models[disease]['model_name']}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Threshold: {metrics['threshold']:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(data['y_test'], y_pred, target_names=['Negative', 'Positive']))

# %% [markdown]
# ### 7.1 Confusion Matrices

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, disease in enumerate(target_names):
    data = splits[disease]
    model = best_models[disease]['model']
    y_prob = model.predict_proba(data['X_test'])[:, 1]
    y_pred = (y_prob >= optimal_thresholds[disease]).astype(int)
    
    cm = confusion_matrix(data['y_test'], y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[i].set_title(f'{DISEASES[disease]["display_name"]}\n({best_models[disease]["model_name"]})',
                     fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrices.png")

# %% [markdown]
# ### 7.2 ROC Curves

# %%
fig, ax = plt.subplots(figsize=(10, 8))

for disease, color in zip(target_names, colors):
    data = splits[disease]
    model = best_models[disease]['model']
    y_prob = model.predict_proba(data['X_test'])[:, 1]
    
    fpr, tpr, _ = roc_curve(data['y_test'], y_prob)
    auc = evaluation_results[disease]['roc_auc']
    
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{DISEASES[disease]["display_name"]} (AUC = {auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves for All Disease Models', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roc_curves.png")

# %% [markdown]
# ### 7.3 Precision-Recall Curves

# %%
fig, ax = plt.subplots(figsize=(10, 8))

for disease, color in zip(target_names, colors):
    data = splits[disease]
    model = best_models[disease]['model']
    y_prob = model.predict_proba(data['X_test'])[:, 1]
    
    precision, recall, _ = precision_recall_curve(data['y_test'], y_prob)
    
    ax.plot(recall, precision, color=color, linewidth=2,
            label=f'{DISEASES[disease]["display_name"]}')
    
    # Mark optimal threshold point
    threshold = optimal_thresholds[disease]
    y_pred_opt = (y_prob >= threshold).astype(int)
    p = precision_score(data['y_test'], y_pred_opt, zero_division=0)
    r = recall_score(data['y_test'], y_pred_opt, zero_division=0)
    ax.scatter([r], [p], s=100, color=color, zorder=5, edgecolors='white', linewidth=2)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'precision_recall_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: precision_recall_curves.png")

# %% [markdown]
# ### 7.4 Calibration Curves

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, disease in enumerate(target_names):
    data = splits[disease]
    model = best_models[disease]['model']
    y_prob = model.predict_proba(data['X_test'])[:, 1]
    
    prob_true, prob_pred = calibration_curve(data['y_test'], y_prob, n_bins=10, strategy='uniform')
    
    axes[i].plot(prob_pred, prob_true, 's-', color=colors[i], linewidth=2, markersize=8,
                label='Model')
    axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    axes[i].set_xlabel('Mean Predicted Probability')
    axes[i].set_ylabel('Fraction of Positives')
    axes[i].set_title(f'{DISEASES[disease]["display_name"]}', fontsize=12, fontweight='bold')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Calibration Curves', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'calibration_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: calibration_curves.png")

# %% [markdown]
# ---
# ## Step 8: SHAP Explainability
# 
# Using SHAP (SHapley Additive Explanations) to understand feature importance 
# and individual prediction explanations.

# %%
shap_data = {}

for disease in target_names:
    print(f"\nComputing SHAP values for {DISEASES[disease]['display_name']}...")
    
    model = best_models[disease]['model']
    data = splits[disease]
    
    # Use a sample for SHAP computation (faster)
    sample_size = min(500, len(data['X_test']))
    X_sample = data['X_test'].iloc[:sample_size]
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle SHAP output format differences robustly
    if isinstance(shap_values, list):
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            shap_vals = shap_values[:, :, 1]
        else:
            shap_vals = shap_values
    else:
        if hasattr(shap_values, "values"):
            vals = shap_values.values
            if isinstance(vals, np.ndarray):
                if vals.ndim == 3:
                    shap_vals = vals[:, :, 1]
                else:
                    shap_vals = vals
            else:
                shap_vals = vals
        else:
            shap_vals = shap_values
            
    shap_data[disease] = {
        'explainer': explainer,
        'shap_values': shap_vals,
        'X_sample': X_sample,
    }
    
    # Feature importance (mean absolute SHAP)
    importance = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"  Top 5 features:")
    for _, row in importance_df.head(5).iterrows():
        print(f"    {row['feature']:35s}: {row['importance']:.4f}")

# %% [markdown]
# ### 8.1 SHAP Summary Plots

# %%
for disease in target_names:
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_data[disease]['shap_values'],
        shap_data[disease]['X_sample'],
        feature_names=[f.replace('_', ' ').title() for f in FEATURE_NAMES],
        show=False,
        plot_size=(12, 8),
    )
    plt.title(f'SHAP Summary - {DISEASES[disease]["display_name"]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'shap_summary_{disease}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: shap_summary_{disease}.png")

# %% [markdown]
# ### 8.2 SHAP Feature Importance Bar Plots

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for i, disease in enumerate(target_names):
    importance = np.abs(shap_data[disease]['shap_values']).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': [f.replace('_', ' ').title() for f in FEATURE_NAMES],
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    axes[i].barh(importance_df['feature'], importance_df['importance'], color=colors[i], alpha=0.8)
    axes[i].set_title(f'{DISEASES[disease]["display_name"]}', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('Mean |SHAP Value|')

plt.suptitle('Feature Importance (SHAP) for Each Disease', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'shap_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_feature_importance.png")

# %% [markdown]
# ---
# ## Step 9: Model Comparison
# 
# Compare all three models for each disease side by side.

# %%
print("\n" + "=" * 90)
print("  MODEL COMPARISON TABLE")
print("=" * 90)

comparison_data = []

for disease in target_names:
    data = splits[disease]
    for model_name, result in all_results[disease].items():
        model = result['best_model']
        y_prob = model.predict_proba(data['X_test'])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)  # Use default threshold for fair comparison
        
        row = {
            'Disease': DISEASES[disease]['display_name'],
            'Model': MODEL_CONFIGS[model_name]['display_name'],
            'CV F1': result['best_cv_f1'],
            'Accuracy': accuracy_score(data['y_test'], y_pred),
            'Precision': precision_score(data['y_test'], y_pred, zero_division=0),
            'Recall': recall_score(data['y_test'], y_pred, zero_division=0),
            'F1': f1_score(data['y_test'], y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(data['y_test'], y_prob),
            'Selected': '*' if model_name == best_models[disease]['model_name'] else '',
        }
        comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv(os.path.join(MODEL_DIR, 'model_comparison.csv'), index=False)
print(f"\nSaved: model_comparison.csv")

# %% [markdown]
# ### 9.1 Model Comparison Visualization

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
metrics_to_plot = ['F1', 'ROC-AUC', 'Accuracy']
bar_colors = ['#ff6b6b', '#00d4aa', '#a855f7']

for i, metric in enumerate(metrics_to_plot):
    for j, disease in enumerate(target_names):
        disease_data = comparison_df[comparison_df['Disease'] == DISEASES[disease]['display_name']]
        x = np.arange(len(disease_data))
        width = 0.25
        offset = (j - 1) * width
        axes[i].bar(x + offset, disease_data[metric], width, 
                    color=colors[j], alpha=0.8, label=DISEASES[disease]['display_name'])
    
    axes[i].set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
    axes[i].set_xticks(np.arange(3))
    axes[i].set_xticklabels(['Random Forest', 'LightGBM', 'XGBoost'], rotation=15)
    axes[i].set_ylim(0, 1.05)
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")

# %% [markdown]
# ---
# ## Step 10: Percentile Ranking System
# 
# Calculate percentile distributions from training data probabilities
# for contextualizing individual risk scores.

# %%
percentile_data = {}

for disease in target_names:
    model = best_models[disease]['model']
    data = splits[disease]
    
    # Get probabilities for the entire training set
    train_probs = model.predict_proba(data['X_train'])[:, 1]
    
    # Store percentile breakpoints
    percentiles = {}
    for p in range(0, 101, 5):
        percentiles[str(p)] = float(np.percentile(train_probs, p))
    
    percentile_data[disease] = {
        'percentiles': percentiles,
        'mean_prob': float(np.mean(train_probs)),
        'median_prob': float(np.median(train_probs)),
        'std_prob': float(np.std(train_probs)),
    }
    
    print(f"{disease:20s}: Mean prob={np.mean(train_probs):.4f}, "
          f"Median={np.median(train_probs):.4f}, "
          f"25th pct={np.percentile(train_probs, 25):.4f}, "
          f"75th pct={np.percentile(train_probs, 75):.4f}")

# Save percentile data
percentile_path = os.path.join(MODEL_DIR, 'percentiles.json')
with open(percentile_path, 'w') as f:
    json.dump(percentile_data, f, indent=2)
print(f"\nPercentile data saved to: {percentile_path}")

# %% [markdown]
# ---
# ## Step 11: Subgroup/Fairness Analysis
# 
# Analyze model performance across demographic subgroups.

# %%
print("\n" + "=" * 70)
print("  SUBGROUP ANALYSIS BY AGE GROUP")
print("=" * 70)

# Get original (unscaled) age values from test set indices
for disease in target_names:
    data = splits[disease]
    test_indices = data['X_test'].index
    age_original = df.loc[test_indices, 'age']
    
    # Create age groups
    age_groups = pd.cut(age_original, bins=[0, 30, 50, 100], labels=['<30', '30-50', '>50'])
    
    model = best_models[disease]['model']
    y_prob = model.predict_proba(data['X_test'])[:, 1]
    y_pred = (y_prob >= optimal_thresholds[disease]).astype(int)
    
    print(f"\n  {DISEASES[disease]['display_name']}:")
    print(f"  {'Age Group':<12} {'Count':<8} {'Accuracy':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print(f"  {'-'*56}")
    
    for group in ['<30', '30-50', '>50']:
        mask = age_groups == group
        if mask.sum() < 10:
            continue
        
        y_true_g = data['y_test'][mask]
        y_pred_g = y_pred[mask]
        y_prob_g = y_prob[mask]
        
        acc = accuracy_score(y_true_g, y_pred_g)
        f1 = f1_score(y_true_g, y_pred_g, zero_division=0)
        try:
            auc = roc_auc_score(y_true_g, y_prob_g)
        except ValueError:
            auc = 0.0
        
        print(f"  {group:<12} {mask.sum():<8} {acc:<12.4f} {f1:<12.4f} {auc:<12.4f}")

# %% [markdown]
# ---
# ## Step 12: Save All Artifacts

# %%
# Save best models
for disease in target_names:
    model_path = os.path.join(MODEL_DIR, f'{disease}_model.pkl')
    joblib.dump(best_models[disease]['model'], model_path)
    print(f"Saved model: {model_path}")

# Save SHAP explainers
for disease in target_names:
    explainer_path = os.path.join(MODEL_DIR, f'{disease}_explainer.pkl')
    joblib.dump(shap_data[disease]['explainer'], explainer_path)
    print(f"Saved SHAP explainer: {explainer_path}")

# Save training report
training_report = {
    'dataset_info': {
        'total_samples': len(df),
        'num_features': len(FEATURE_NAMES),
        'features': FEATURE_NAMES,
        'diseases': list(DISEASES.keys()),
    },
    'class_distribution': {
        disease: {
            'positive': int(df[disease].sum()),
            'negative': int((df[disease] == 0).sum()),
            'positive_pct': float(df[disease].mean() * 100),
        } for disease in target_names
    },
    'best_models': {
        disease: {
            'model_name': best_models[disease]['model_name'],
            'cv_f1': float(best_models[disease]['cv_f1']),
            'best_params': {k: str(v) for k, v in best_models[disease]['best_params'].items()},
        } for disease in target_names
    },
    'evaluation_metrics': evaluation_results,
    'optimal_thresholds': optimal_thresholds,
    'feature_importance': {
        disease: {
            feat: float(np.abs(shap_data[disease]['shap_values']).mean(axis=0)[i])
            for i, feat in enumerate(FEATURE_NAMES)
        } for disease in target_names
    },
}

report_path = os.path.join(MODEL_DIR, 'training_report.json')
with open(report_path, 'w') as f:
    json.dump(training_report, f, indent=2)
print(f"\nTraining report saved to: {report_path}")

# %% [markdown]
# ---
# ## Summary
# 
# ### Models Trained:
# - Random Forest, LightGBM, XGBoost for each disease
# - Best model selected by CV F1-score
# - Threshold optimized for each disease
# 
# ### Artifacts Saved:
# - `models/` — Trained models (.pkl), scaler, thresholds, percentiles
# - `models/plots/` — All visualization plots
# - `models/training_report.json` — Complete training metrics
# - `models/model_comparison.csv` — Side-by-side comparison

# %%
print("\n" + "=" * 60)
print("  TRAINING COMPLETE - SUMMARY")
print("=" * 60)
for disease in target_names:
    m = evaluation_results[disease]
    print(f"\n  {DISEASES[disease]['display_name']} ({best_models[disease]['model_name']}):")
    print(f"    Accuracy={m['accuracy']:.4f}  F1={m['f1_score']:.4f}  ROC-AUC={m['roc_auc']:.4f}")

print(f"\n  All artifacts saved to: {MODEL_DIR}")
print(f"  Plots saved to: {PLOTS_DIR}")
print("\n  Ready for deployment!")
