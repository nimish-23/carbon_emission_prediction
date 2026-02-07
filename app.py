from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# ============================================================================
# LOAD MODELS
# ============================================================================

# Load existing models
driver_models = joblib.load("models/driver_models.pkl")
co2_model = joblib.load("models/co2_model.pkl")

# NEW: Load SHAP explainer
shap_explainer = joblib.load("models/shap_explainer.pkl")
training_stats = joblib.load("models/training_stats.pkl")

FEATURES = [
    "energy_per_capita",
    "fossil_energy_per_capita",
    "renewables_share_energy",
    "energy_per_gdp"
]

print("="*70)
print("🚀 MODELS LOADED SUCCESSFULLY")
print("="*70)
print(f"✓ Driver models: {len(driver_models)} features")
print(f"✓ CO2 model loaded")
print(f"✓ SHAP explainer loaded (baseline: {training_stats['baseline']:.4f})")
print("="*70)

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__, static_folder="src", static_url_path="")
CORS(app)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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


# NEW FUNCTION
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

# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def index():
    """Serve the frontend HTML page"""
    return send_from_directory("src", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Basic prediction endpoint (unchanged)"""
    data = request.get_json()

    # Validate input
    if not data or "year" not in data:
        return jsonify({"error": "Missing 'year' in request"}), 400

    year = data["year"]

    if not isinstance(year, int):
        return jsonify({"error": "'year' must be an integer"}), 400

    if year < 1965 or year > 2100:
        return jsonify({"error": "Year out of supported range"}), 400

    # 1. Predict drivers
    drivers_pred = predict_drivers_for_year(year)

    # 2. Predict CO2
    co2_pred = predict_co2_from_drivers(drivers_pred)

    # 3. Response
    return jsonify({
        "year": year,
        "predicted_co2_per_capita": round(co2_pred, 3),
        "projected_drivers": {
            k: round(v, 3) for k, v in drivers_pred.items()
        }
    })


# NEW ENDPOINT
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


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌍 India CO₂ Emissions Prediction API")
    print("="*70)
    print("\nAvailable endpoints:")
    print("  • POST /predict         - Basic prediction")
    print("  • POST /predict/explain - Prediction with SHAP explanation")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True)