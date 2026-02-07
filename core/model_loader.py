# Model loading module
import joblib

# Feature list used across the application
FEATURES = [
    "energy_per_capita",
    "fossil_share_energy",
    "energy_per_gdp"
]

# Load existing models
driver_models = joblib.load("models/driver_models.pkl")
co2_model = joblib.load("models/co2_model.pkl")

# Load SHAP explainer
shap_explainer = joblib.load("models/shap_explainer.pkl")
training_stats = joblib.load("models/training_stats.pkl")

# Print confirmation
print("="*70)
print("🚀 MODELS LOADED SUCCESSFULLY")
print("="*70)
print(f"✓ Driver models: {len(driver_models)} features")
print(f"✓ CO2 model loaded")
print(f"✓ SHAP explainer loaded (baseline: {training_stats['baseline']:.4f})")
print("="*70)
