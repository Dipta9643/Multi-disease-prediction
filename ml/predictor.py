"""
Prediction Service for Multi-Disease Risk Prediction System.
Loads models, scaler, thresholds, percentiles and performs predictions and SHAP calculations.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_NAMES, MODEL_DIR, DISEASES, RISK_LEVELS

class DiseasePredictor:
    def __init__(self):
        self.models = {}
        self.explainers = {}
        self.scaler = None
        self.thresholds = {}
        self.percentiles = {}
        self.loaded = False
        
        self.load_artifacts()

    def load_artifacts(self):
        try:
            # Load scaler
            scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                print("Warning: Scaler pkl not found. Run train_models.py first.")
                return

            # Load thresholds
            thresholds_path = os.path.join(MODEL_DIR, "thresholds.json")
            if os.path.exists(thresholds_path):
                import json
                with open(thresholds_path, 'r') as f:
                    self.thresholds = json.load(f)
            else:
                self.thresholds = {disease: 0.5 for disease in DISEASES}

            # Load percentiles
            percentiles_path = os.path.join(MODEL_DIR, "percentiles.json")
            if os.path.exists(percentiles_path):
                import json
                with open(percentiles_path, 'r') as f:
                    self.percentiles = json.load(f)
            else:
                self.percentiles = {}

            # Load models and explainers
            for disease in DISEASES:
                model_path = os.path.join(MODEL_DIR, f"{disease}_model.pkl")
                explainer_path = os.path.join(MODEL_DIR, f"{disease}_explainer.pkl")
                
                if os.path.exists(model_path):
                    self.models[disease] = joblib.load(model_path)
                if os.path.exists(explainer_path):
                    self.explainers[disease] = joblib.load(explainer_path)
                
            self.loaded = True
            print("Successfully loaded all ML models, scale matrices, and explainers.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")

    def compute_percentile(self, disease, probability):
        """Map absolute probability to percentile based on training distribution."""
        if disease not in self.percentiles or 'percentiles' not in self.percentiles[disease]:
            return round(probability * 100, 1)
        
        pct_dict = self.percentiles[disease]['percentiles']
        # Convert keys to integers and values to floats
        points = sorted([(int(pct), float(val)) for pct, val in pct_dict.items()], key=lambda x: x[0])
        
        # Interpolate percentile
        pct_nums = [p[0] for p in points]
        val_nums = [p[1] for p in points]
        
        percentile = np.interp(probability, val_nums, pct_nums)
        return round(float(percentile), 1)

    def get_risk_level(self, percentile):
        """Determine risk level label and color based on percentile."""
        for level, config in RISK_LEVELS.items():
            if percentile <= config["max_percentile"]:
                return level, config["label"], config["color"]
        return "high", "High Risk", "#ff6b6b"

    def predict(self, raw_input_dict):
        """
        Generate predictions for all three diseases.
        raw_input_dict: dict of feature_name -> raw_value
        """
        if not self.loaded:
            self.load_artifacts()
            if not self.loaded:
                raise Exception("Models not trained or loaded. Please train first.")

        # Convert input dict to dataframe in correct order
        input_data = [raw_input_dict[feat] for feat in FEATURE_NAMES]
        df_input = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        # Scale inputs
        df_scaled = pd.DataFrame(self.scaler.transform(df_input), columns=FEATURE_NAMES)
        
        predictions = {}
        for disease in DISEASES:
            model = self.models[disease]
            threshold = self.thresholds.get(disease, 0.5)
            
            # Predict probability
            prob = float(model.predict_proba(df_scaled)[0, 1])
            
            # Calculate percentile
            percentile = self.compute_percentile(disease, prob)
            
            # Class decision based on optimized threshold
            binary_pred = 1 if prob >= threshold else 0
            
            # Risk level category
            level, label, color = self.get_risk_level(percentile)
            
            predictions[disease] = {
                "probability": round(prob, 4),
                "percentile": percentile,
                "binary_prediction": binary_pred,
                "risk_level": level,
                "risk_label": label,
                "color": color
            }
            
        return predictions

    def explain(self, raw_input_dict):
        """
        Calculate SHAP explanations for the current prediction.
        """
        if not self.loaded:
            self.load_artifacts()
            
        input_data = [raw_input_dict[feat] for feat in FEATURE_NAMES]
        df_input = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        df_scaled = pd.DataFrame(self.scaler.transform(df_input), columns=FEATURE_NAMES)
        
        explanations = {}
        for disease in DISEASES:
            explainer = self.explainers.get(disease)
            if explainer is None:
                # Recreate explainer if it wasn't saved/loaded
                model = self.models.get(disease)
                if model:
                    explainer = shap.TreeExplainer(model)
                    self.explainers[disease] = explainer
            
            if explainer:
                try:
                    shap_values = explainer.shap_values(df_scaled)
                    # Handle LightGBM/XGBoost/RF format differences
                    if isinstance(shap_values, list):
                        # Random Forest or multi-class classification
                        s_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                    elif len(shap_values.shape) == 3:
                        s_vals = shap_values[0, :, 1]
                    else:
                        s_vals = shap_values[0]
                        
                    # Map to feature names
                    feature_impact = []
                    for i, feat in enumerate(FEATURE_NAMES):
                        feature_impact.append({
                            "feature": feat,
                            "display_name": feat.replace("_", " ").title(),
                            "shap_value": float(s_vals[i]),
                            # Direction: positive means increases risk, negative means reduces risk
                            "direction": "increase" if s_vals[i] > 0 else "reduce",
                            "raw_value": raw_input_dict[feat]
                        })
                    
                    # Sort by absolute SHAP impact descending
                    feature_impact = sorted(feature_impact, key=lambda x: abs(x["shap_value"]), reverse=True)
                    explanations[disease] = feature_impact
                except Exception as e:
                    print(f"Error computing SHAP for {disease}: {e}")
                    explanations[disease] = []
            else:
                explanations[disease] = []
                
        return explanations
