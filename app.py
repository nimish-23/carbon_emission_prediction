from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import atexit
import time
import sys

# Import from core modules
from core.model_loader import driver_models, co2_model, shap_explainer, training_stats
from core.prediction import predict_drivers_for_year
from core.shap_explainer import explain_prediction_with_shap
from core.interpretation import generate_interpretation

# Import GenAI for policy recommendations
from genai.summarizer import PolicySummarizer
from genai.policy_context import enrich_responsibility_profile

# Import from utils
from utils.validators import validate_year_input


app = Flask(__name__, static_folder="src", static_url_path="")
CORS(app)

# Ollama server management
ollama_process = None

def is_ollama_running():
    """Check if Ollama server is already running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_ollama_server():
    """Start Ollama server if not already running"""
    global ollama_process
    
    if is_ollama_running():
        print("✓ Ollama server is already running")
        return True
    
    try:
        print("🚀 Starting Ollama server...")
        # Start Ollama server in the background
        ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        
        # Wait for server to be ready (max 10 seconds)
        for i in range(20):
            time.sleep(0.5)
            if is_ollama_running():
                print("✓ Ollama server started successfully")
                return True
        
        print("⚠ Ollama server started but not responding yet")
        return False
        
    except FileNotFoundError:
        print("❌ Ollama not found. Please install it from https://ollama.com")
        return False
    except Exception as e:
        print(f"❌ Failed to start Ollama server: {e}")
        return False

def stop_ollama_server():
    """Stop Ollama server if it was started by this app"""
    global ollama_process
    if ollama_process:
        print("\n🛑 Stopping Ollama server...")
        try:
            ollama_process.terminate()
            ollama_process.wait(timeout=5)
            print("✓ Ollama server stopped")
        except:
            ollama_process.kill()
            print("✓ Ollama server force stopped")

# Register cleanup handler
atexit.register(stop_ollama_server)

# Start Ollama server on app initialization
start_ollama_server()

# Initialize GenAI policy generator
policy_generator = PolicySummarizer(model_name="llama3.2", use_llm=True)


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
    is_valid, error_msg, year = validate_year_input(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

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


@app.route("/predict/explain-policy", methods=["POST"])
def predict_explain_policy():
    """
    Prediction + SHAP explanation + GenAI-powered Policy insights
    """
    data = request.get_json()

    # Validate input
    is_valid, error_msg, year = validate_year_input(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    try:
        # 1. Predict drivers and get SHAP explanation
        drivers_pred = predict_drivers_for_year(year)
        explanation = explain_prediction_with_shap(drivers_pred)

        # 2. Build responsibility profile (simplified from SHAP)
        responsibility_profile = []
        for feature, contribution in explanation['contributions'].items():
            responsibility_profile.append({
                "factor": feature,
                "impact_value": round(contribution, 4),
                "impact_percent": round(explanation['percentages'][feature], 1)
            })
        
        # Sort by absolute impact
        responsibility_profile.sort(key=lambda x: abs(x['impact_value']), reverse=True)
        
        # Enrich with policy context (adds 'policy_relevant' and 'policy_context' fields)
        responsibility_profile = enrich_responsibility_profile(responsibility_profile)
        
        # 3. Generate GenAI policy recommendations directly
        try:
            policy_insights = policy_generator.generate_policy_recommendations(
                explanation=explanation,
                responsibility_profile=responsibility_profile,
                year=year
            )
            genai_enabled = any(ins.get("source") == "genai" for ins in policy_insights)
        except Exception as genai_error:
            print(f"⚠ GenAI generation failed: {genai_error}")
            # Fallback: simple message
            policy_insights = [{
                "theme": "GenAI Unavailable",
                "why_it_matters": "Policy recommendations require Ollama to be running. Start it with 'ollama serve'.",
                "policy_focus": [],
                "source": "fallback"
            }]
            genai_enabled = False

        return jsonify({
            "year": year,
            "predicted_co2_per_capita": round(explanation["prediction"], 3),
            "baseline": round(explanation["baseline"], 3),
            "responsibility_profile": responsibility_profile,
            "policy_insights": policy_insights,
            "genai_enabled": genai_enabled,
            "note": (
                "Policy insights are generated using GenAI analysis of model explanations. "
                "They are indicative and context-aware for India's climate policy landscape."
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
    print("  • POST /predict/explain-policy - Prediction with policy insights")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True)