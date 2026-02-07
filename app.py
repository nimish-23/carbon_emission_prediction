from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd


# Load existing models
driver_models = joblib.load("models/driver_models.pkl")
co2_model = joblib.load("models/co2_model.pkl")

# NEW: Load SHAP explainer
shap_explainer = joblib.load("models/shap_explainer.pkl")
training_stats = joblib.load("models/training_stats.pkl")

FEATURES = [
    "energy_per_capita",
    "fossil_share_energy",
    "energy_per_gdp"
]

POLICY_DOMAIN_MAP = {
    "energy_per_capita": {
        "theme": "Energy Demand Reduction",
        "description": "High total energy consumption across households, transport, and industry",
        "policy_areas": [
            "Public transport expansion",
            "Energy-efficient buildings",
            "Urban planning and densification",
            "Appliance efficiency standards",
            "Behavioral energy conservation"
        ]
    },
    "fossil_share_energy": {
        "theme": "Energy Supply Decarbonization",
        "description": "High dependence on fossil fuels in the energy mix",
        "policy_areas": [
            "Renewable energy scale-up",
            "Coal phase-down",
            "Grid modernization",
            "Energy storage deployment",
            "Carbon pricing mechanisms"
        ]
    },
    "energy_per_gdp": {
        "theme": "Economic Energy Efficiency",
        "description": "Low energy efficiency of economic output",
        "policy_areas": [
            "Industrial efficiency programs",
            "Technology modernization",
            "Electrification of industry",
            "Process optimization"
        ]
    }
}


print("="*70)
print("🚀 MODELS LOADED SUCCESSFULLY")
print("="*70)
print(f"✓ Driver models: {len(driver_models)} features")
print(f"✓ CO2 model loaded")
print(f"✓ SHAP explainer loaded (baseline: {training_stats['baseline']:.4f})")
print("="*70)


app = Flask(__name__, static_folder="src", static_url_path="")
CORS(app)

def predict_drivers_for_year(year: int) -> dict:
    """Predict energy drivers for a given year using trained trend models"""
    year_df = pd.DataFrame({"year": [year]})

    drivers_pred = {}
    for driver, model in driver_models.items():
        drivers_pred[driver] = float(model.predict(year_df)[0])

    return drivers_pred


def predict_co2_from_drivers(drivers_pred: dict) -> float:
    """Predict CO2 per capita from projected drivers"""
    X = pd.DataFrame([drivers_pred])[FEATURES]
    co2_pred = float(co2_model.predict(X)[0])

    return co2_pred


def explain_prediction_with_shap(drivers_pred: dict) -> dict:
    """
    Explain a CO2 prediction using SHAP values
    
    Args:
        drivers_pred: Dictionary of predicted driver values
        
    Returns:
        Dictionary containing:
            - prediction: CO2 prediction
            - baseline: Average prediction (expected value)
            - contributions: SHAP values for each feature
            - percentages: Percentage contribution of each feature
    """
    # Convert drivers to DataFrame (same format as training)
    X = pd.DataFrame([drivers_pred])[FEATURES]
    
    # Get prediction
    prediction = float(co2_model.predict(X)[0])
    
    # Compute SHAP values
    shap_values = shap_explainer.shap_values(X)
    
    # Extract SHAP values (it's a 2D array, we want the first row)
    shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    
    # Create contributions dictionary
    contributions = {
        feature: float(shap_vals[i])
        for i, feature in enumerate(FEATURES)
    }
    
    # Calculate percentages (absolute contribution)
    total_abs_contribution = sum(abs(v) for v in contributions.values())
    
    percentages = {
        feature: (abs(contrib) / total_abs_contribution * 100) if total_abs_contribution > 0 else 0
        for feature, contrib in contributions.items()
    }
    
    return {
        'prediction': prediction,
        'baseline': training_stats['baseline'],
        'contributions': contributions,
        'percentages': percentages
    }


def build_responsibility_profile(explanation: dict, threshold: float = 5.0):
    """
    Build a structured responsibility profile from SHAP output
    """
    profile = []

    for feature, pct in explanation["percentages"].items():
        profile.append({
            "factor": feature,
            "impact_percent": pct,
            "impact_value": explanation["contributions"][feature],
            "policy_relevant": pct >= threshold,
            "policy_context": POLICY_DOMAIN_MAP.get(feature, {})
        })

    # Sort by importance
    profile.sort(key=lambda x: x["impact_percent"], reverse=True)
    return profile

def generate_policy_insights(responsibility_profile: list) -> list:
    """
    Generate policy insights from responsibility profile.
    This is where GenAI will later be plugged in.
    """
    insights = []

    for item in responsibility_profile:
        if not item["policy_relevant"]:
            continue

        context = item["policy_context"]

        insights.append({
            "factor": item["factor"],
            "theme": context.get("theme"),
            "why_it_matters": context.get("description"),
            "policy_focus": context.get("policy_areas"),
            "model_signal": f"Accounts for {item['impact_percent']:.1f}% of the predicted emissions impact"
        })

    return insights



@app.route("/")
def index():
    """Serve the frontend HTML page"""
    return send_from_directory("src", "index.html")



@app.route("/predict/explain", methods=["POST"])
def predict_explain():
    """
    Prediction with SHAP explanation
    
    Returns prediction + explanation of which features contributed
    """
    data = request.get_json()

    # Validate input
    if not data or "year" not in data:
        return jsonify({"error": "Missing 'year' in request"}), 400

    year = data["year"]

    if not isinstance(year, int):
        return jsonify({"error": "'year' must be an integer"}), 400

    if year < 1965 or year > 2100:
        return jsonify({"error": "Year out of supported range"}), 400

    try:
        # 1. Predict drivers
        drivers_pred = predict_drivers_for_year(year)

        # 2. Get prediction + SHAP explanation
        explanation = explain_prediction_with_shap(drivers_pred)

        # 3. Response
        return jsonify({
            "year": year,
            "predicted_co2_per_capita": round(explanation['prediction'], 3),
            "baseline": round(explanation['baseline'], 3),
            "projected_drivers": {
                k: round(v, 3) for k, v in drivers_pred.items()
            },
            "explanation": {
                "contributions": {
                    k: round(v, 4) for k, v in explanation['contributions'].items()
                },
                "percentages": {
                    k: round(v, 1) for k, v in explanation['percentages'].items()
                },
                "interpretation": generate_interpretation(explanation)
            }
        })
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# NEW HELPER FUNCTION
def generate_interpretation(explanation: dict) -> str:
    """
    Generate a human-readable interpretation of SHAP values
    
    Args:
        explanation: Dictionary with contributions and percentages
        
    Returns:
        String with interpretation
    """
    contributions = explanation['contributions']
    percentages = explanation['percentages']
    
    # Sort by absolute contribution
    sorted_features = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # Get top 2 features
    top_features = sorted_features[:2]
    
    interpretations = []
    
    for feature, contrib in top_features:
        pct = percentages[feature]
        direction = "increases" if contrib > 0 else "decreases"
        
        # Make feature name readable
        readable_name = feature.replace('_', ' ').title()
        
        interpretations.append(
            f"{readable_name} {direction} emissions ({pct:.1f}% impact)"
        )
    
    return "; ".join(interpretations)

@app.route("/predict/explain-policy", methods=["POST"])
def predict_explain_policy():
    """
    Prediction + SHAP explanation + Policy insights
    """
    data = request.get_json()

    if not data or "year" not in data:
        return jsonify({"error": "Missing 'year' in request"}), 400

    year = data["year"]

    if not isinstance(year, int):
        return jsonify({"error": "'year' must be an integer"}), 400

    try:
        drivers_pred = predict_drivers_for_year(year)
        explanation = explain_prediction_with_shap(drivers_pred)

        responsibility_profile = build_responsibility_profile(explanation)
        policy_insights = generate_policy_insights(responsibility_profile)

        return jsonify({
            "year": year,
            "predicted_co2_per_capita": round(explanation["prediction"], 3),
            "baseline": round(explanation["baseline"], 3),
            "responsibility_profile": responsibility_profile,
            "policy_insights": policy_insights,
            "note": (
                "Policy insights are generated by interpreting model explanations. "
                "They are indicative, not prescriptive."
            )
        })

    except Exception as e:
        return jsonify({"error": f"Policy explanation failed: {str(e)}"}), 500



if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌍 India CO₂ Emissions Prediction API")
    print("="*70)
    print("\nAvailable endpoints:")
    print("  • POST /predict/explain - CO₂ prediction with SHAP explanation")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True)