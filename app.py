from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd

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

# --------------------
# Flask app
# --------------------
app = Flask(__name__, static_folder="src", static_url_path="")
CORS(app)

# --------------------
# Helper functions
# --------------------
def predict_drivers_for_year(year: int) -> dict:
    """
    Predict energy drivers for a given year using trained trend models
    """
    year_df = pd.DataFrame({"year": [year]})

    drivers_pred = {}
    for driver, model in driver_models.items():
        drivers_pred[driver] = float(model.predict(year_df)[0])

    return drivers_pred


def predict_co2_from_drivers(drivers_pred: dict) -> float:
    """
    Predict CO2 per capita from projected drivers
    """
    X = pd.DataFrame([drivers_pred])[FEATURES]
    co2_pred = float(co2_model.predict(X)[0])
    return co2_pred


# --------------------
# Routes
# --------------------
@app.route("/")
def index():
    """Serve the frontend HTML page"""
    return send_from_directory("src", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
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


# --------------------
# Run app
# --------------------
if __name__ == "__main__":
    app.run(debug=True)
