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


app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# ─── Ollama server management ─────────────────────────────────
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
    global ollama_process
    if is_ollama_running():
        print("✓ Ollama server is already running")
        return True
    try:
        print("🚀 Starting Ollama server...")
        ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
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

atexit.register(stop_ollama_server)
start_ollama_server()

policy_generator = PolicySummarizer(model_name="llama3.2", use_llm=True)


# ─── Helper ───────────────────────────────────────────────────
def _build_responsibility_profile(explanation):
    """Build + enrich responsibility profile from SHAP explanation."""
    profile = []
    for feature, contribution in explanation['contributions'].items():
        profile.append({
            "factor": feature,
            "impact_value": round(contribution, 4),
            "impact_percent": round(explanation['percentages'][feature], 1)
        })
    profile.sort(key=lambda x: abs(x['impact_value']), reverse=True)
    return enrich_responsibility_profile(profile)


# ─── Routes ───────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend HTML page"""
    return send_from_directory("frontend", "index.html")


@app.route("/predict/explain", methods=["POST"])
def predict_explain():
    """Prediction with SHAP explanation."""
    data = request.get_json()
    is_valid, error_msg, year = validate_year_input(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    try:
        drivers_pred = predict_drivers_for_year(year)
        explanation  = explain_prediction_with_shap(drivers_pred)

        return jsonify({
            "year": year,
            "predicted_co2_per_capita": round(explanation['prediction'], 3),
            "baseline": round(explanation['baseline'], 3),
            "projected_drivers": {k: round(v, 3) for k, v in drivers_pred.items()},
            "explanation": {
                "contributions": {k: round(v, 4) for k, v in explanation['contributions'].items()},
                "percentages":   {k: round(v, 1) for k, v in explanation['percentages'].items()},
                "interpretation": generate_interpretation(explanation)
            }
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict/explain-policy", methods=["POST"])
def predict_explain_policy():
    """Prediction + SHAP explanation + GenAI-powered policy insights."""
    data = request.get_json()
    is_valid, error_msg, year = validate_year_input(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    try:
        drivers_pred           = predict_drivers_for_year(year)
        explanation            = explain_prediction_with_shap(drivers_pred)
        responsibility_profile = _build_responsibility_profile(explanation)

        try:
            policy_insights = policy_generator.generate_policy_recommendations(
                explanation=explanation,
                responsibility_profile=responsibility_profile,
                year=year
            )
            genai_enabled = any(ins.get("source") == "genai" for ins in policy_insights)
        except Exception as genai_error:
            print(f"⚠ GenAI generation failed: {genai_error}")
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
            "projected_drivers": {k: round(v, 3) for k, v in drivers_pred.items()},
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


@app.route("/predict/compare", methods=["POST"])
def predict_compare():
    """
    Multi-year comparison endpoint.

    Request body:
        { "years": [2025, 2030, 2040, 2050] }

    Returns predictions + SHAP for each year, plus a trend summary.
    Max 10 years per request.
    """
    data = request.get_json()

    if not data or "years" not in data:
        return jsonify({"error": "Missing 'years' list in request"}), 400

    years = data["years"]
    if not isinstance(years, list) or len(years) == 0:
        return jsonify({"error": "'years' must be a non-empty list"}), 400
    if len(years) > 10:
        return jsonify({"error": "Maximum 10 years per comparison request"}), 400

    results = []
    for year in years:
        if not isinstance(year, int) or year < 1965 or year > 2100:
            return jsonify({"error": f"Invalid year: {year}. Must be int between 1965-2100"}), 400
        try:
            drivers_pred = predict_drivers_for_year(year)
            explanation  = explain_prediction_with_shap(drivers_pred)
            results.append({
                "year": year,
                "predicted_co2_per_capita": round(explanation['prediction'], 3),
                "baseline": round(explanation['baseline'], 3),
                "delta_from_baseline": round(explanation['prediction'] - explanation['baseline'], 3),
                "projected_drivers": {k: round(v, 3) for k, v in drivers_pred.items()},
                "top_driver": max(
                    explanation['contributions'].items(),
                    key=lambda x: abs(x[1])
                )[0]
            })
        except Exception as e:
            results.append({"year": year, "error": str(e)})

    # Trend summary
    valid = [r for r in results if "error" not in r]
    trend_summary = {}
    if len(valid) >= 2:
        first, last = valid[0], valid[-1]
        trend_summary = {
            "from_year": first["year"],
            "to_year":   last["year"],
            "co2_change": round(last["predicted_co2_per_capita"] - first["predicted_co2_per_capita"], 3),
            "pct_change": round(
                (last["predicted_co2_per_capita"] - first["predicted_co2_per_capita"])
                / first["predicted_co2_per_capita"] * 100, 1
            ) if first["predicted_co2_per_capita"] != 0 else 0,
            "direction": "increasing" if last["predicted_co2_per_capita"] > first["predicted_co2_per_capita"] else "decreasing"
        }

    return jsonify({
        "comparison": results,
        "trend_summary": trend_summary,
        "count": len(results)
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint — also returns model metadata."""
    return jsonify({
        "status": "ok",
        "ollama_running": is_ollama_running(),
        "baseline": round(training_stats.get('baseline', 0), 4),
        "endpoints": [
            "POST /predict/explain",
            "POST /predict/explain-policy",
            "POST /predict/compare",
            "GET  /health"
        ]
    })


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🌍 India CO₂ Emissions Prediction API  —  CarbonLens")
    print("=" * 70)
    print("\nEndpoints:")
    print("  • POST /predict/explain        — CO₂ prediction + SHAP")
    print("  • POST /predict/explain-policy — + GenAI policy insights")
    print("  • POST /predict/compare        — Multi-year comparison")
    print("  • GET  /health                 — Health check")
    print("\n  Frontend: http://localhost:5000/")
    print("=" * 70 + "\n")
    app.run(debug=True, port=5000)
