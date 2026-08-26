"""
Database helper for Multi-Disease Risk Prediction System.
Handles SQLite connection to store prediction history.
"""

import sqlite3
import os
import json
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DB_PATH

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create predictions table if not exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            bmi REAL,
            activity_level TEXT,
            exercise_duration INTEGER,
            sugar_intake TEXT,
            salt_intake TEXT,
            fruit_veg_consumption REAL,
            sleep_duration REAL,
            sleep_quality TEXT,
            stress_level TEXT,
            smoking_status TEXT,
            alcohol_consumption TEXT,
            family_history_diabetes TEXT,
            family_history_hypertension TEXT,
            family_history_heart_disease TEXT,
            
            diabetes_prob REAL,
            diabetes_risk TEXT,
            
            hypertension_prob REAL,
            hypertension_risk TEXT,
            
            heart_disease_prob REAL,
            heart_disease_risk TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def save_prediction(user_input, predictions):
    """
    Saves user input and predictions to database.
    user_input: raw inputs (dict)
    predictions: outputs (dict)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Map encoded features back to display values for readable history
    # Helper to clean up display
    gender_map = {0: "Male", 1: "Female"}
    activity_map = {1: "Low", 2: "Moderate", 3: "High"}
    sugar_salt_map = {0: "Low", 1: "Moderate", 2: "High"}
    sleep_qual_map = {0: "Poor", 1: "Average", 2: "Good"}
    stress_map = {0: "Low", 1: "Medium", 2: "High"}
    smoking_map = {0: "Never", 1: "Former", 2: "Current"}
    alcohol_map = {0: "None", 1: "Moderate", 2: "Heavy"}
    yes_no_map = {0: "No", 1: "Yes"}
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO risk_history (
            timestamp, age, gender, bmi, activity_level, exercise_duration,
            sugar_intake, salt_intake, fruit_veg_consumption, sleep_duration,
            sleep_quality, stress_level, smoking_status, alcohol_consumption,
            family_history_diabetes, family_history_hypertension, family_history_heart_disease,
            
            diabetes_prob, diabetes_risk,
            hypertension_prob, hypertension_risk,
            heart_disease_prob, heart_disease_risk
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        int(user_input.get("age", 35)),
        gender_map.get(int(user_input.get("gender", 0)), "Male"),
        float(user_input.get("bmi", 24.0)),
        activity_map.get(int(user_input.get("activity_level", 2)), "Moderate"),
        int(user_input.get("exercise_duration", 30)),
        sugar_salt_map.get(int(user_input.get("sugar_intake", 1)), "Moderate"),
        sugar_salt_map.get(int(user_input.get("salt_intake", 1)), "Moderate"),
        float(user_input.get("fruit_veg_consumption", 4.0)),
        float(user_input.get("sleep_duration", 7.0)),
        sleep_qual_map.get(int(user_input.get("sleep_quality", 1)), "Average"),
        stress_map.get(int(user_input.get("stress_level", 1)), "Medium"),
        smoking_map.get(int(user_input.get("smoking_status", 0)), "Never"),
        alcohol_map.get(int(user_input.get("alcohol_consumption", 0)), "None"),
        yes_no_map.get(int(user_input.get("family_history_diabetes", 0)), "No"),
        yes_no_map.get(int(user_input.get("family_history_hypertension", 0)), "No"),
        yes_no_map.get(int(user_input.get("family_history_heart_disease", 0)), "No"),
        
        float(predictions["diabetes"]["probability"]),
        predictions["diabetes"]["risk_label"],
        
        float(predictions["hypertension"]["probability"]),
        predictions["hypertension"]["risk_label"],
        
        float(predictions["heart_disease"]["probability"]),
        predictions["heart_disease"]["risk_label"]
    ))
    
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id

def get_history():
    """Retrieve history in descending chronological order."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM risk_history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_prediction(record_id):
    """Delete a specific prediction record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM risk_history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

def clear_history():
    """Delete all prediction records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM risk_history")
    conn.commit()
    conn.close()
