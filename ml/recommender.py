"""
Lifestyle Recommendation Engine for Multi-Disease Risk Prediction.
Generates personalized lifestyle recommendations based on prediction results and user features.
"""

class RecommendationEngine:
    def __init__(self):
        pass

    def generate_recommendations(self, user_input, predictions):
        """
        Generates structured recommendations.
        user_input: dict of raw feature values
        predictions: dict of disease outputs
        """
        recs = []

        # Extract features
        age = user_input.get("age", 35)
        bmi = user_input.get("bmi", 24.0)
        activity_level = user_input.get("activity_level", 2)
        exercise_duration = user_input.get("exercise_duration", 30)
        sugar_intake = user_input.get("sugar_intake", 1)
        salt_intake = user_input.get("salt_intake", 1)
        fruit_veg = user_input.get("fruit_veg_consumption", 4)
        sleep_dur = user_input.get("sleep_duration", 7)
        sleep_qual = user_input.get("sleep_quality", 1)
        stress = user_input.get("stress_level", 1)
        smoking = user_input.get("smoking_status", 0)
        alcohol = user_input.get("alcohol_consumption", 0)
        
        # Extract disease risk levels
        diab_risk = predictions.get("diabetes", {}).get("risk_level", "low")
        hyp_risk = predictions.get("hypertension", {}).get("risk_level", "low")
        heart_risk = predictions.get("heart_disease", {}).get("risk_level", "low")

        # ============================================================
        # 1. DIET RECOMMENDATIONS
        # ============================================================
        if diab_risk in ["high", "moderate"] or sugar_intake == 2:
            priority = "CRITICAL" if (diab_risk == "high" or sugar_intake == 2) else "HIGH"
            recs.append({
                "category": "Diet & Nutrition",
                "title": "Reduce Refined Sugar & Processed Carbs",
                "text": "Limit sweets, sugary drinks, white bread, and white rice. Swap them with whole grains (oats, brown rice, quinoa) to avoid rapid blood sugar spikes.",
                "priority": priority,
                "icon": "🍎"
            })

        if hyp_risk in ["high", "moderate"] or salt_intake == 2:
            priority = "CRITICAL" if (hyp_risk == "high" or salt_intake == 2) else "HIGH"
            recs.append({
                "category": "Diet & Nutrition",
                "title": "Adopt Low-Sodium Diet (DASH Diet)",
                "text": "Reduce daily salt intake to under 1,500 mg. Avoid processed meats, canned soups, and salty snacks. Eat more potassium-rich foods (bananas, spinach, sweet potatoes) to lower blood pressure.",
                "priority": priority,
                "icon": "🧂"
            })

        if bmi >= 25.0:
            priority = "CRITICAL" if bmi >= 30.0 else "HIGH"
            recs.append({
                "category": "Diet & Nutrition",
                "title": "Caloric Control & Weight Management",
                "text": f"Your BMI is {bmi} (Overweight/Obese). Aim for a modest 5-10% body weight reduction by creating a daily caloric deficit of 300-500 kcal, focus on portion control.",
                "priority": priority,
                "icon": "⚖️"
            })

        if fruit_veg < 5:
            recs.append({
                "category": "Diet & Nutrition",
                "title": "Increase Fruits & Vegetables",
                "text": f"You currently consume {fruit_veg} servings per day. Aim for at least 5 servings daily. These provide vital antioxidants, fiber, and minerals that protect blood vessels and improve insulin sensitivity.",
                "priority": "MODERATE",
                "icon": "🥦"
            })

        # ============================================================
        # 2. PHYSICAL ACTIVITY RECOMMENDATIONS
        # ============================================================
        is_inactive = (activity_level == 1 or exercise_duration < 20)
        has_high_risk = (diab_risk == "high" or hyp_risk == "high" or heart_risk == "high")
        
        if is_inactive:
            priority = "CRITICAL" if has_high_risk else "HIGH"
            recs.append({
                "category": "Physical Activity",
                "title": "Start Regular Aerobic Exercise",
                "text": "Engage in at least 150 minutes of moderate-intensity exercise (brisk walking, cycling, swimming) per week. Start with 20 minutes a day and build up gradually.",
                "priority": priority,
                "icon": "🏃‍♂️"
            })
        elif exercise_duration < 45 and has_high_risk:
            recs.append({
                "category": "Physical Activity",
                "title": "Boost Cardio Exercise Duration",
                "text": "Increase your daily exercise duration to 45-60 minutes. Aerobic exercise strengthens the heart muscle, increases insulin sensitivity, and lowers arterial stiffness.",
                "priority": "MODERATE",
                "icon": "🚴‍♀️"
            })

        if activity_level == 1:
            recs.append({
                "category": "Physical Activity",
                "title": "Break Sedentary Habits",
                "text": "Avoid sitting for long hours. Take a 5-minute standing or walking break for every 45 minutes of sitting. Set reminders to move.",
                "priority": "MODERATE",
                "icon": "🚶‍♂️"
            })

        # ============================================================
        # 3. LIFESTYLE HABITS RECOMMENDATIONS
        # ============================================================
        if smoking == 2:  # Current smoker
            recs.append({
                "category": "Habits & Substance Use",
                "title": "Quit Smoking Immediately",
                "text": "Smoking multiplies heart disease risk by damaging blood vessels and raising heart rate. Consider nicotine replacement therapy or consulting a specialist.",
                "priority": "CRITICAL",
                "icon": "🚬"
            })
        elif smoking == 1:  # Former smoker
            recs.append({
                "category": "Habits & Substance Use",
                "title": "Maintain Smoke-Free Status",
                "text": "Great job on quitting! Avoid secondhand smoke, which can still cause significant arterial damage and increase cardiovascular disease risks.",
                "priority": "MODERATE",
                "icon": "🚭"
            })

        if alcohol == 2:  # Heavy drinker
            priority = "CRITICAL" if heart_risk == "high" or hyp_risk == "high" else "HIGH"
            recs.append({
                "category": "Habits & Substance Use",
                "title": "Reduce Alcohol Consumption",
                "text": "Limit alcohol intake to a maximum of 1 drink per day for women or 2 for men. Heavy drinking raises blood pressure, increases triglycerides, and damages heart tissue.",
                "priority": priority,
                "icon": "🍺"
            })

        # ============================================================
        # 4. SLEEP & STRESS RECOMMENDATIONS
        # ============================================================
        if sleep_dur < 6 or sleep_qual == 0:
            priority = "HIGH" if (stress == 2 or heart_risk == "high") else "MODERATE"
            recs.append({
                "category": "Sleep & Mental Health",
                "title": "Improve Sleep Hygiene",
                "text": f"Your current sleep is {sleep_dur} hours. Target 7-8 hours of restful sleep. Keep a regular sleep schedule, dim lights 1 hour before bed, and avoid screen usage in bed.",
                "priority": priority,
                "icon": "😴"
            })

        if stress == 2:
            recs.append({
                "category": "Sleep & Mental Health",
                "title": "Incorporate Stress-Relief Activities",
                "text": "Practice mindfulness, deep breathing exercises, yoga, or meditation for 10-15 minutes daily. High stress releases cortisol and adrenaline, elevating blood pressure and blood sugar.",
                "priority": "HIGH",
                "icon": "🧘‍♀️"
            })

        # ============================================================
        # 5. CLINICAL MONITORING RECOMMENDATIONS
        # ============================================================
        if heart_risk == "high" or hyp_risk == "high" or diab_risk == "high":
            recs.append({
                "category": "Medical & Monitoring",
                "title": "Consult a Primary Care Physician",
                "text": "Given your high risk status, schedule a professional check-up for blood pressure monitoring, HbA1c screening, and a lipid profile test to check cholesterol levels.",
                "priority": "CRITICAL",
                "icon": "🩺"
            })
        elif heart_risk == "moderate" or hyp_risk == "moderate" or diab_risk == "moderate":
            recs.append({
                "category": "Medical & Monitoring",
                "title": "Schedule Annual Health Screenings",
                "text": "Monitor your vital numbers yearly. Keep track of your blood pressure, fasting glucose, and cholesterol to identify changes early.",
                "priority": "MODERATE",
                "icon": "📋"
            })
            
        if hyp_risk in ["high", "moderate"]:
            recs.append({
                "category": "Medical & Monitoring",
                "title": "Regular Blood Pressure Tracking",
                "text": "Check your blood pressure twice a week. Keep a log of your systolic/diastolic readings and share them with your doctor.",
                "priority": "HIGH",
                "icon": "💓"
            })

        # Sort recommendations: Critical first, then High, Moderate
        priority_map = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}
        recs = sorted(recs, key=lambda x: priority_map.get(x["priority"], 3))

        return recs
