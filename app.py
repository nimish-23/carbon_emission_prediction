from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import shap

# --------------------
# Load models at startup
# --------------------
driver_models = joblib.load("models/driver_models.pkl")
co2_model = joblib.load("models/co2_model.pkl")

FEATURES = [
    "energy_per_capita",
    "fossil_energy_per_capita",
    "renewables_share_energy",
    "energy_per_gdp"
]

# Create SHAP explainer (background = training-style data)
# Use small dummy background (acceptable for linear models)
background = pd.DataFrame(
    [{f: 0 for f in FEATURES}]
)
explainer = shap.Explainer(co2_model, background)

# --------------------
# Flask app
# --------------------
app = Flask(__name__, static_folder='src', static_url_path='')
CORS(app)  # Enable CORS for all routes

# --------------------
# Helper functions
# --------------------
def predict_drivers_for_year(year):
    year_df = pd.DataFrame({"year": [year]})
    return {
        driver: float(model.predict(year_df)[0])
        for driver, model in driver_models.items()
    }

def predict_co2_from_drivers(drivers_pred):
    X = pd.DataFrame([drivers_pred])
    co2_pred = float(co2_model.predict(X)[0])
    return co2_pred, X

def compute_shap_contributions(X):
    shap_values = explainer(X)
    contributions = {
        FEATURES[i]: float(shap_values.values[0][i])
        for i in range(len(FEATURES))
    }
    base_value = float(shap_values.base_values[0])
    return contributions, base_value

# --------------------
# Routes
# --------------------
@app.route("/")
def index():
    """Serve the frontend HTML page"""
    return send_from_directory('src', 'index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "year" not in data:
        return jsonify({"error": "Missing 'year' in request"}), 400

    year = data["year"]

    if not isinstance(year, int) or year < 1965 or year > 2100:
        return jsonify({"error": "Year out of supported range"}), 400

    # 1. Predict drivers
    drivers_pred = predict_drivers_for_year(year)

    # 2. Predict CO2
    co2_pred, X = predict_co2_from_drivers(drivers_pred)

    # 3. Explain prediction
    contributions, base_value = compute_shap_contributions(X)

    return jsonify({
        "year": year,
        "predicted_co2_per_capita": round(co2_pred, 3),
        "baseline_co2": round(base_value, 3),
        "projected_drivers": {k: round(v, 3) for k, v in drivers_pred.items()},
        "factor_contributions": {k: round(v, 3) for k, v in contributions.items()}
    })

# --------------------
# Run app
# --------------------
if __name__ == "__main__":
    app.run(debug=True)
