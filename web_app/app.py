from flask import Flask, render_template, request, jsonify
import joblib
import json
import math
import os
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "lifestyle_risk_gb_model.pkl")
scaler_path = os.path.join(BASE_DIR, "lifestyle_risk_scaler.pkl")
config_path = os.path.join(BASE_DIR, "lifestyle_risk_config.json")

scaler = joblib.load(scaler_path)

with open(config_path, "r") as f:
    config = json.load(f)

FEATURES = config["features"]
NEEDS_SCALING = config["needs_scaling"]
MODEL_NAME = config["model_name"]
BEST_MODEL_KEY = config.get("best_model_key", config.get("model_type", "gradient_boosting"))

DEFAULT_MODEL_REGISTRY = {
    "gradient_boosting": {
        "name": MODEL_NAME,
        "file": os.path.basename(model_path),
        "needs_scaling": NEEDS_SCALING,
        "accuracy": config.get("accuracy"),
        "f1_score": config.get("f1_score"),
        "auc_roc": config.get("auc_roc"),
    }
}
MODEL_REGISTRY = config.get("models", DEFAULT_MODEL_REGISTRY)
LOADED_MODELS = {}
for model_key, model_info in MODEL_REGISTRY.items():
    candidate_path = os.path.join(BASE_DIR, model_info["file"])
    if os.path.exists(candidate_path):
        LOADED_MODELS[model_key] = joblib.load(candidate_path)

if BEST_MODEL_KEY not in LOADED_MODELS and "gradient_boosting" in LOADED_MODELS:
    BEST_MODEL_KEY = "gradient_boosting"
elif BEST_MODEL_KEY not in LOADED_MODELS:
    BEST_MODEL_KEY = next(iter(LOADED_MODELS))

AVAILABLE_MODELS = [
    {
        "key": key,
        "name": MODEL_REGISTRY[key]["name"],
        "auc_roc": MODEL_REGISTRY[key].get("auc_roc"),
        "accuracy": MODEL_REGISTRY[key].get("accuracy"),
        "is_best": key == BEST_MODEL_KEY,
    }
    for key in LOADED_MODELS
]

DEFAULT_FORM_VALUES = {
    "age": 50,
    "gender": 1,
    "bmi": 25.0,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "smoking_status": 1,
    "waist_circumference": 95,
    "arm_circumference": 32,
    "income_pir": 2.5,
    "alcohol_use": 1,
    "avg_drinks": 1,
    "household_size": 3,
    "selected_model": BEST_MODEL_KEY,
}


def bp_stage(sbp, dbp):
    """Calculate BP stage based on ACC/AHA guidelines."""
    if sbp < 120 and dbp < 80:
        return 0
    if sbp < 130 and dbp < 80:
        return 1
    if sbp < 140 or dbp < 90:
        return 2
    return 3


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def estimate_height(age):
    return 166 if age > 50 else 170


def feature_snapshot(values):
    age = values["age"]
    bmi = values["bmi"]
    sbp = values["systolic_bp"]
    dbp = values["diastolic_bp"]
    waist_circ = values["waist_circumference"]

    return {
        **values,
        "bmi_category": int(pd.cut([bmi], bins=[0, 18.5, 25, 30, 35, 100],
                                   labels=[0, 1, 2, 3, 4])[0]),
        "bp_stage": bp_stage(sbp, dbp),
        "age_group": int(pd.cut([age], bins=[0, 30, 45, 60, 200],
                                labels=[0, 1, 2, 3])[0]),
        "height_estimate": estimate_height(age),
        "wh_ratio": waist_circ / estimate_height(age),
        "pulse_pressure": sbp - dbp,
    }


def model_input_from_snapshot(snapshot, model_key):
    feature_values = [
        snapshot["age"],
        snapshot["gender"],
        snapshot["bmi"],
        snapshot["systolic_bp"],
        snapshot["diastolic_bp"],
        snapshot["smoking_status"],
        snapshot["waist_circumference"],
        snapshot["arm_circumference"],
        snapshot["income_pir"],
        snapshot["bmi_category"],
        snapshot["bp_stage"],
        snapshot["age_group"],
        snapshot["wh_ratio"],
        snapshot["pulse_pressure"],
        snapshot["alcohol_use"],
        snapshot["avg_drinks"],
        snapshot["household_size"],
    ]
    input_df = pd.DataFrame([feature_values], columns=FEATURES)
    if MODEL_REGISTRY[model_key].get("needs_scaling"):
        return pd.DataFrame(scaler.transform(input_df), columns=FEATURES)
    return input_df


def calibrated_model_percentage(model_input, model_obj):
    """Spread the trained classifier's raw margin into a usable percentage."""
    if hasattr(model_obj, "decision_function"):
        raw_margin = float(model_obj.decision_function(model_input)[0])
        return sigmoid(raw_margin / 7.0) * 100

    probability = float(model_obj.predict_proba(model_input)[0][1])
    probability = min(max(probability, 0.001), 0.999)
    logit = math.log(probability / (1 - probability))
    return sigmoid(logit / 3.0) * 100


def continuous_clinical_percentage(snapshot):
    """Smooth continuous score: small input changes produce small score changes."""
    smoking_exposure = max(0, min(1, snapshot["smoking_status"] - 1))
    drink_exposure = snapshot["avg_drinks"] * max(0, min(1, 2 - snapshot["alcohol_use"]))

    score = (
        0.21 * sigmoid((snapshot["systolic_bp"] - 126) / 13)
        + 0.11 * sigmoid((snapshot["diastolic_bp"] - 80) / 8)
        + 0.17 * sigmoid((snapshot["bmi"] - 29) / 4.8)
        + 0.15 * sigmoid((snapshot["age"] - 48) / 12)
        + 0.16 * sigmoid((snapshot["wh_ratio"] - 0.52) / 0.06)
        + 0.08 * smoking_exposure
        + 0.05 * sigmoid((snapshot["pulse_pressure"] - 50) / 10)
        + 0.04 * sigmoid((drink_exposure - 3) / 1.5)
        + 0.03 * sigmoid((3 - snapshot["income_pir"]) / 1.2)
    )

    return max(1.0, min(score * 100, 99.0))


def risk_percentage_from_snapshot(snapshot, model_key):
    model_obj = LOADED_MODELS[model_key]
    model_input = model_input_from_snapshot(snapshot, model_key)
    model_pct = calibrated_model_percentage(model_input, model_obj)
    clinical_pct = continuous_clinical_percentage(snapshot)
    return round((model_pct * 0.55) + (clinical_pct * 0.45), 1)


def sensitivity_contributors(snapshot, current_percentage, model_key):
    """Rank contributors by counterfactual percentage-point impact."""
    height = snapshot["height_estimate"]
    reference_sets = {
        "Blood pressure": {"systolic_bp": 115, "diastolic_bp": 75},
        "BMI / body weight": {"bmi": 22},
        "Abdominal measurement": {"waist_circumference": height * 0.47},
        "Smoking status": {"smoking_status": 1},
        "Age profile": {"age": 35},
        "Pulse pressure": {
            "systolic_bp": snapshot["diastolic_bp"] + 42,
            "diastolic_bp": snapshot["diastolic_bp"],
        },
        "Alcohol intake": {"alcohol_use": 2, "avg_drinks": 0},
        "Income-to-poverty ratio": {"income_pir": 3.5},
    }

    contributors = []
    for label, replacements in reference_sets.items():
        adjusted = snapshot.copy()
        adjusted.update(replacements)
        adjusted = feature_snapshot(adjusted)
        impact = round(current_percentage - risk_percentage_from_snapshot(adjusted, model_key), 1)
        if impact > 1.0:
            contributors.append({"label": label, "impact": impact})

    contributors.sort(key=lambda item: item["impact"], reverse=True)
    return contributors[:5]


def precautionary_advice(contributors, risk_level):
    advice_bank = {
        "Blood pressure": {
            "title": "Monitor and manage blood pressure",
            "text": "Check BP regularly, reduce excess salt, stay active, manage stress, and consult a clinician if readings remain high.",
        },
        "BMI / body weight": {
            "title": "Improve weight and nutrition habits",
            "text": "Aim for gradual weight control through balanced meals, portion control, and at least 150 minutes of weekly moderate activity.",
        },
        "Abdominal measurement": {
            "title": "Reduce abdominal fat risk",
            "text": "Prioritize regular walking or cardio, strength training, and fewer sugary or highly processed foods.",
        },
        "Smoking status": {
            "title": "Avoid tobacco exposure",
            "text": "Stopping smoking can sharply reduce long-term heart and lifestyle disease risk; consider cessation support if needed.",
        },
        "Age profile": {
            "title": "Schedule preventive checkups",
            "text": "Keep routine screening for BP, glucose, cholesterol, and weight trends, especially as age-related risk increases.",
        },
        "Pulse pressure": {
            "title": "Review cardiovascular strain",
            "text": "Track BP patterns over time and discuss unusually wide pulse pressure with a healthcare professional.",
        },
        "Alcohol intake": {
            "title": "Limit alcohol intake",
            "text": "Keep alcohol moderate, avoid binge drinking, and choose alcohol-free days during the week.",
        },
        "Income-to-poverty ratio": {
            "title": "Use accessible preventive care",
            "text": "Look for affordable health screenings, community clinics, and low-cost nutrition or fitness resources.",
        },
    }

    advice = []
    for factor in contributors[:4]:
        item = advice_bank.get(factor["label"])
        if item:
            advice.append({**item, "impact": factor["impact"]})

    if not advice:
        advice.append({
            "title": "Maintain current healthy habits",
            "text": "Continue routine checkups, balanced nutrition, regular physical activity, good sleep, and avoiding tobacco.",
            "impact": 0,
        })

    if risk_level in {"High Risk", "Very High Risk"}:
        advice.insert(0, {
            "title": "Seek medical guidance",
            "text": "Because your risk is elevated, discuss these results with a qualified healthcare professional for personalized evaluation.",
            "impact": 0,
        })

    return advice[:5]


def risk_label_from_percentage(risk_percentage):
    if risk_percentage >= 75:
        return "Very High Risk"
    if risk_percentage >= 50:
        return "High Risk"
    if risk_percentage >= 25:
        return "Moderate Risk"
    return "Low Risk"


def parse_form_values(source):
    selected_model = source.get("selected_model", DEFAULT_FORM_VALUES["selected_model"])
    if selected_model not in LOADED_MODELS:
        selected_model = BEST_MODEL_KEY

    return {
        "age": int(source.get("age", DEFAULT_FORM_VALUES["age"])),
        "gender": int(source.get("gender", DEFAULT_FORM_VALUES["gender"])),
        "bmi": float(source.get("bmi", DEFAULT_FORM_VALUES["bmi"])),
        "systolic_bp": float(source.get("systolic_bp", DEFAULT_FORM_VALUES["systolic_bp"])),
        "diastolic_bp": float(source.get("diastolic_bp", DEFAULT_FORM_VALUES["diastolic_bp"])),
        "smoking_status": int(source.get("smoking_status", DEFAULT_FORM_VALUES["smoking_status"])),
        "waist_circumference": float(source.get("waist_circumference", DEFAULT_FORM_VALUES["waist_circumference"])),
        "arm_circumference": float(source.get("arm_circumference", DEFAULT_FORM_VALUES["arm_circumference"])),
        "income_pir": float(source.get("income_pir", DEFAULT_FORM_VALUES["income_pir"])),
        "alcohol_use": int(source.get("alcohol_use", DEFAULT_FORM_VALUES["alcohol_use"])),
        "avg_drinks": float(source.get("avg_drinks", DEFAULT_FORM_VALUES["avg_drinks"])),
        "household_size": int(source.get("household_size", DEFAULT_FORM_VALUES["household_size"])),
        "selected_model": selected_model,
    }


def make_prediction(values):
    model_key = values.get("selected_model", BEST_MODEL_KEY)
    if model_key not in LOADED_MODELS:
        model_key = BEST_MODEL_KEY

    snapshot = feature_snapshot(values)
    input_scaled = model_input_from_snapshot(snapshot, model_key)
    model_obj = LOADED_MODELS[model_key]

    model_probability = float(model_obj.predict_proba(input_scaled)[0][1])
    risk_class = int(model_obj.predict(input_scaled)[0])
    model_risk_map = {0: "Low Risk", 1: "High Risk"}
    risk_percentage = risk_percentage_from_snapshot(snapshot, model_key)
    factors = sensitivity_contributors(snapshot, risk_percentage, model_key)
    risk_level = risk_label_from_percentage(risk_percentage)

    return {
        "risk_level": risk_level,
        "risk_probability": round(risk_percentage / 100, 4),
        "risk_percentage": risk_percentage,
        "model_prediction": model_risk_map[risk_class],
        "model_probability": round(model_probability, 4),
        "selected_model": model_key,
        "selected_model_name": MODEL_REGISTRY[model_key]["name"],
        "contributing_factors": factors,
        "precautionary_advice": precautionary_advice(factors, risk_level),
        "model": MODEL_REGISTRY[model_key]["name"],
    }


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    risk_probability = None
    risk_percentage = None
    contributing_factors = None
    precautionary_advice_items = None
    input_data = None
    form_values = DEFAULT_FORM_VALUES.copy()

    if request.method == "POST":
        try:
            form_values = parse_form_values(request.form)
            result = make_prediction(form_values)

            prediction = result["risk_level"]
            risk_probability = result["risk_probability"]
            risk_percentage = result["risk_percentage"]
            contributing_factors = result["contributing_factors"]
            precautionary_advice_items = result["precautionary_advice"]
            input_data = {
                "age": form_values["age"],
                "gender": "Male" if form_values["gender"] == 1 else "Female",
                "bmi": form_values["bmi"],
                "sbp": form_values["systolic_bp"],
                "dbp": form_values["diastolic_bp"],
                "smoking": "Smoker" if form_values["smoking_status"] == 2 else "Non-smoker",
                "waist": form_values["waist_circumference"],
                "model": result["selected_model_name"],
            }

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        risk_percentage=risk_percentage,
        risk_probability=risk_probability,
        contributing_factors=contributing_factors,
        precautionary_advice=precautionary_advice_items,
        input_data=input_data,
        form_values=form_values,
        model_name=MODEL_NAME,
        selected_model_name=MODEL_REGISTRY[form_values["selected_model"]]["name"],
        available_models=AVAILABLE_MODELS,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for JSON requests."""
    try:
        values = parse_form_values(request.get_json() or {})
        return jsonify({"status": "success", **make_prediction(values)})

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 400


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "available_models": AVAILABLE_MODELS,
        "auc_roc": config["auc_roc"],
        "accuracy": config["accuracy"],
        "f1_score": config["f1_score"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
