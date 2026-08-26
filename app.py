"""
Flask Application for Multi-Disease Risk Prediction & Lifestyle Recommendation System.
Defines all web controller routes, form handlers, and REST API endpoints.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURE_NAMES, FEATURE_DEFINITIONS, DISEASES, APP_NAME
from ml.predictor import DiseasePredictor
from ml.recommender import RecommendationEngine
import database

app = Flask(__name__)
app.secret_key = "multi_disease_predictor_secret_key_mca"

# Initialize predictor and recommender
predictor = DiseasePredictor()
recommender = RecommendationEngine()

# Initialize DB on start
database.init_db()

@app.context_processor
def inject_global_vars():
    """Inject variables automatically into all templates."""
    return {
        "diseases": DISEASES,
        "app_name": APP_NAME
    }

@app.route("/")
def home():
    return render_template("index.html", active_page="home")

@app.route("/predict", methods=["GET"])
def predict():
    return render_template("predict.html", active_page="predict")

@app.route("/results", methods=["POST"])
def results():
    try:
        # Collect raw features from form post parameters
        raw_input = {}
        for feat in FEATURE_NAMES:
            feat_def = FEATURE_DEFINITIONS[feat]
            
            # Fetch from form
            val_str = request.form.get(feat)
            if val_str is None:
                # Use default value if missing
                val = feat_def["default"]
            else:
                # Cast based on feature type
                if feat_def["type"] == "numerical":
                    val = float(val_str)
                    if val.is_integer():
                        val = int(val)
                else:
                    val = int(val_str)
                    
            raw_input[feat] = val

        # Run predictions
        predictions = predictor.predict(raw_input)
        
        # Run SHAP explanations
        shap_exps = predictor.explain(raw_input)
        
        # Run recommendations
        recommendations = recommender.generate_recommendations(raw_input, predictions)
        
        # Save to SQLite database
        record_id = database.save_prediction(raw_input, predictions)
        
        return render_template(
            "results.html",
            active_page="predict",
            predictions=predictions,
            shap_explanations=shap_exps,
            recommendations=recommendations,
            user_input=raw_input,
            record_id=record_id
        )
    except Exception as e:
        flash(f"An error occurred during prediction: {str(e)}", "danger")
        return redirect(url_for("predict"))

@app.route("/results/<int:record_id>")
def view_results(record_id):
    try:
        # Fetch record from DB
        history = database.get_history()
        record = next((r for r in history if r["id"] == record_id), None)
        
        if not record:
            flash("Prediction record not found.", "warning")
            return redirect(url_for("history"))
            
        # Reconstruct raw_input_dict from DB record values
        gender_reverse = {"Male": 0, "Female": 1}
        activity_reverse = {"Low": 1, "Moderate": 2, "High": 3}
        sugar_salt_reverse = {"Low": 0, "Moderate": 1, "High": 2}
        sleep_qual_reverse = {"Poor": 0, "Average": 1, "Good": 2}
        stress_reverse = {"Low": 0, "Medium": 1, "High": 2}
        smoking_reverse = {"Never": 0, "Former": 1, "Current": 2}
        alcohol_reverse = {"None": 0, "Moderate": 1, "Heavy": 2}
        yes_no_reverse = {"No": 0, "Yes": 1}

        raw_input = {
            "age": int(record["age"]),
            "gender": gender_reverse.get(record["gender"], 0),
            "bmi": float(record["bmi"]),
            "activity_level": activity_reverse.get(record["activity_level"], 2),
            "exercise_duration": int(record["exercise_duration"]),
            "sugar_intake": sugar_salt_reverse.get(record["sugar_intake"], 1),
            "salt_intake": sugar_salt_reverse.get(record["salt_intake"], 1),
            "fruit_veg_consumption": float(record["fruit_veg_consumption"]),
            "sleep_duration": float(record["sleep_duration"]),
            "sleep_quality": sleep_qual_reverse.get(record["sleep_quality"], 1),
            "stress_level": stress_reverse.get(record["stress_level"], 1),
            "smoking_status": smoking_reverse.get(record["smoking_status"], 0),
            "alcohol_consumption": alcohol_reverse.get(record["alcohol_consumption"], 0),
            "family_history_diabetes": yes_no_reverse.get(record["family_history_diabetes"], 0),
            "family_history_hypertension": yes_no_reverse.get(record["family_history_hypertension"], 0),
            "family_history_heart_disease": yes_no_reverse.get(record["family_history_heart_disease"], 0),
        }

        # Re-predict and explain to display results dashboard
        predictions = predictor.predict(raw_input)
        shap_exps = predictor.explain(raw_input)
        recommendations = recommender.generate_recommendations(raw_input, predictions)
        
        return render_template(
            "results.html",
            active_page="history",
            predictions=predictions,
            shap_explanations=shap_exps,
            recommendations=recommendations,
            user_input=raw_input,
            record_id=record_id
        )
    except Exception as e:
        flash(f"Error loading record: {str(e)}", "danger")
        return redirect(url_for("history"))

@app.route("/simulator")
def simulator():
    # Pre-fill simulator with either latest prediction or defaults
    history = database.get_history()
    
    if history:
        # Load from latest record
        latest = history[0]
        gender_reverse = {"Male": 0, "Female": 1}
        activity_reverse = {"Low": 1, "Moderate": 2, "High": 3}
        sugar_salt_reverse = {"Low": 0, "Moderate": 1, "High": 2}
        sleep_qual_reverse = {"Poor": 0, "Average": 1, "Good": 2}
        stress_reverse = {"Low": 0, "Medium": 1, "High": 2}
        smoking_reverse = {"Never": 0, "Former": 1, "Current": 2}
        alcohol_reverse = {"None": 0, "Moderate": 1, "Heavy": 2}
        yes_no_reverse = {"No": 0, "Yes": 1}

        original_input = {
            "age": int(latest["age"]),
            "gender": gender_reverse.get(latest["gender"], 0),
            "bmi": float(latest["bmi"]),
            "activity_level": activity_reverse.get(latest["activity_level"], 2),
            "exercise_duration": int(latest["exercise_duration"]),
            "sugar_intake": sugar_salt_reverse.get(latest["sugar_intake"], 1),
            "salt_intake": sugar_salt_reverse.get(latest["salt_intake"], 1),
            "fruit_veg_consumption": float(latest["fruit_veg_consumption"]),
            "sleep_duration": float(latest["sleep_duration"]),
            "sleep_quality": sleep_qual_reverse.get(latest["sleep_quality"], 1),
            "stress_level": stress_reverse.get(latest["stress_level"], 1),
            "smoking_status": smoking_reverse.get(latest["smoking_status"], 0),
            "alcohol_consumption": alcohol_reverse.get(latest["alcohol_consumption"], 0),
            "family_history_diabetes": yes_no_reverse.get(latest["family_history_diabetes"], 0),
            "family_history_hypertension": yes_no_reverse.get(latest["family_history_hypertension"], 0),
            "family_history_heart_disease": yes_no_reverse.get(latest["family_history_heart_disease"], 0),
        }
    else:
        # Pre-fill defaults
        original_input = {feat: FEATURE_DEFINITIONS[feat]["default"] for feat in FEATURE_NAMES}

    original_predictions = predictor.predict(original_input)
    
    return render_template(
        "simulator.html",
        active_page="simulator",
        original_input=original_input,
        original_predictions=original_predictions
    )

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        data = request.json
        if not data:
            return jsonify({"success": false, "error": "No input data provided"}), 400
            
        predictions = predictor.predict(data)
        return jsonify({"success": True, "predictions": predictions})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/history")
def history():
    records = database.get_history()
    return render_template("history.html", active_page="history", history=records)

@app.route("/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id):
    database.delete_prediction(record_id)
    flash("Record deleted.", "info")
    return redirect(url_for("history"))

@app.route("/clear-history", methods=["POST"])
def clear_history_data():
    database.clear_history()
    flash("All prediction history cleared.", "success")
    return redirect(url_for("history"))

@app.route("/results/<int:record_id>/print")
def pdf_report(record_id):
    # Simply render a printer-friendly layout and trigger printing automatically!
    try:
        history = database.get_history()
        record = next((r for r in history if r["id"] == record_id), None)
        
        if not record:
            return "Record not found", 404
            
        return render_template("print_report.html", record=record)
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, debug=True)
