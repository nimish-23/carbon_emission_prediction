# 🌍 India CO₂ Emissions Prediction

> **A machine learning-powered web application for forecasting India's per capita CO₂ emissions with explainable AI insights**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-purple.svg)](https://shap.readthedocs.io/)

---

## 🎯 Project Overview

This end-to-end machine learning project predicts India's future CO₂ emissions per capita using historical energy consumption trends from 1965-2022. Built with **explainable AI** capabilities, the system provides transparent insights into which energy factors drive emission predictions.

### 🔑 Key Highlights

- **Two-Stage Forecasting Pipeline**: Energy drivers → CO₂ predictions
- **Explainable AI**: SHAP-powered feature importance analysis
- **Policy Insights**: Automated policy recommendations based on model explanations
- **Modular Architecture**: Clean separation of ML, policy, and API logic
- **Modern Minimalist UI**: Clean, responsive interface with focus-driven design
- **Production-Ready API**: RESTful Flask backend with comprehensive error handling
- **Smart Data Strategy**: Recent-window forecasting for renewables to capture structural break (2015+)

---

## ✨ Features

### 🤖 Machine Learning

- **Multi-Model Architecture**: Separate trend models for 4 energy drivers
- **SHAP Explanations**: Understand feature contributions to each prediction
- **Adaptive Modeling**: Recent-window strategy for renewables growth trend

### 🎨 User Interface

- **Minimalist Design**: No-scroll layout with clean, modern aesthetics
- **Dynamic UX**: Input form transforms into results-only view
- **Instant Predictions**: Enter year → get prediction + explanation
- **Visual Explanations**: Feature contribution bars with percentage breakdowns

### 🔌 API

- `/predict/explain` - CO₂ predictions with SHAP feature explanations
- `/predict/explain-policy` - Predictions with policy insights and recommendations
- Comprehensive error handling and validation

---

## 🏗️ Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────┐
│                    User Input (Year)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Energy Driver Models (3 Features)            │
│  • Energy per capita  • Fossil share  • Energy per GDP  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              CO₂ Prediction Model                        │
│           (Trained on historical data)                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         SHAP Explainer (Feature Importance)              │
│    Baseline + Contribution Analysis + Percentages        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Policy Insight Generation (Optional)            │
│  Responsibility Profiling + Policy Recommendations       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│    CO₂ Prediction + Explanation + Policy Insights       │
└─────────────────────────────────────────────────────────┘
```

### Modular Code Architecture

The codebase is organized into focused packages for maintainability:

- **`core/`** - ML model loading, predictions, and SHAP explanations
- **`policy/`** - Policy domain mapping and insight generation
- **`utils/`** - Input validation and helper functions
- **`genai/`** - Future AI-powered enhancements (stubs)
- **`app.py`** - Thin Flask API layer (routes only)

---

## 📂 Project Structure

```
carbon_emission_prediction/
├── app.py                         - Flask API server (thin entrypoint)
├── requirements.txt               - Python dependencies
├── core/                          - ML prediction & SHAP logic
│   ├── model_loader.py           - Model loading & constants
│   ├── prediction.py             - Driver & CO₂ predictions
│   ├── shap_explainer.py         - SHAP explanation generation
│   └── interpretation.py         - Human-readable interpretations
├── policy/                        - Policy analysis modules
│   ├── policy_map.py             - Policy domain mappings
│   ├── responsibility.py         - Responsibility profiling
│   └── policy_engine.py          - Policy insight generation
├── genai/                         - Future GenAI integration (stubs)
│   ├── prompts.py                - Prompt templates (placeholder)
│   ├── ollama_client.py          - Ollama API wrapper (stub)
│   └── summarizer.py             - Policy summarizer (stub)
├── utils/                         - Utilities & helpers
│   └── validators.py             - Input validation
├── data/
│   ├── owid-co2-data.csv         - Historical CO₂ emissions
│   └── owid-energy-data.csv      - Energy consumption data
├── models/
│   ├── driver_models.pkl         - 4 energy trend models
│   ├── co2_model.pkl             - CO₂ regression model
│   ├── shap_explainer.pkl        - SHAP explainer object
│   └── training_stats.pkl        - Model metadata & baseline
├── notebooks/
│   ├── carbon_emission.ipynb     - Initial EDA & modeling
│   └── carbon_emission_2.ipynb   - Advanced model development
└── src/                           - Frontend application
    ├── index.html                 - Minimalist UI
    ├── script.js                  - Client logic + explanations
    └── style.css                  - Modern responsive design
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Modern web browser

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/nimish-23/carbon_emission_prediction.git
   cd carbon_emission_prediction
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv env

   # Windows
   env\Scripts\activate

   # macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask server**

   ```bash
   python app.py
   ```

   The server will start at `http://localhost:5000`

2. **Open the web interface**

   Navigate to `http://localhost:5000` in your browser

3. **Make predictions**
   - Enter a year (1965-2100)
   - Click **🔍 Predict & Explain**
   - View prediction + feature explanations

---

## 🔌 API Documentation

### Prediction with Explanation

**Endpoint:** `POST /predict/explain`

**Request:**

```bash
curl -X POST http://localhost:5000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{"year": 2030}'
```

**Response:**

```json
{
  "year": 2030,
  "predicted_co2_per_capita": 2.456,
  "baseline": 1.823,
  "projected_drivers": {
    "energy_per_capita": 23.789,
    "fossil_share_energy": 78.234,
    "energy_per_gdp": 1.234
  },
  "explanation": {
    "contributions": {
      "energy_per_capita": 0.3456,
      "fossil_share_energy": 0.2123,
      "energy_per_gdp": 0.0234
    },
    "percentages": {
      "energy_per_capita": 52.3,
      "fossil_share_energy": 32.1,
      "energy_per_gdp": 15.6
    },
    "interpretation": "Energy Per Capita increases emissions (52.3% impact); Fossil Share Energy increases emissions (32.1% impact)"
  }
}
```

### Prediction with Policy Insights

**Endpoint:** `POST /predict/explain-policy`

**Request:**

```bash
curl -X POST http://localhost:5000/predict/explain-policy \
  -H "Content-Type: application/json" \
  -d '{"year": 2030}'
```

**Response:**

```json
{
  "year": 2030,
  "predicted_co2_per_capita": 2.456,
  "baseline": 1.823,
  "responsibility_profile": [
    {
      "factor": "energy_per_capita",
      "impact_percent": 52.3,
      "impact_value": 0.3456,
      "policy_relevant": true,
      "policy_context": {
        "theme": "Energy Demand Reduction",
        "description": "High total energy consumption across households, transport, and industry",
        "policy_areas": [
          "Public transport expansion",
          "Energy-efficient buildings",
          "Urban planning and densification",
          "Appliance efficiency standards",
          "Behavioral energy conservation"
        ]
      }
    }
  ],
  "policy_insights": [
    {
      "factor": "energy_per_capita",
      "theme": "Energy Demand Reduction",
      "why_it_matters": "High total energy consumption across households, transport, and industry",
      "policy_focus": [
        "Public transport expansion",
        "Energy-efficient buildings",
        "Urban planning and densification",
        "Appliance efficiency standards",
        "Behavioral energy conservation"
      ],
      "model_signal": "Accounts for 52.3% of the predicted emissions impact"
    }
  ],
  "note": "Policy insights are generated by interpreting model explanations. They are indicative, not prescriptive."
}
```

### Error Responses

- **400 Bad Request**: Missing or invalid `year` parameter
- **500 Internal Server Error**: Prediction failure

---

## 🧮 Technical Approach

### 1. Energy Driver Models

- **Algorithm**: Linear Regression with time-based features
- **Training Data**: 1965-2022 (India-specific)
- **Features Predicted**:
  - `energy_per_capita` (kWh)
  - `fossil_energy_per_capita` (kWh)
  - `renewables_share_energy` (%)
  - `energy_per_gdp` (kWh per $ of GDP)

**Special Strategy**: Renewables model uses **recent-window forecasting** (2015-2022 only) to capture the structural break where renewables shifted from declining to growing.

### 2. CO₂ Prediction Model

- **Algorithm**: Regression model (trained on historical CO₂ vs energy drivers)
- **Input**: 4 projected energy driver values
- **Output**: CO₂ emissions per capita (tons)
- **Training Period**: 1965-2022

### 3. SHAP Explainability

- **Framework**: SHAP (SHapley Additive exPlanations)
- **Purpose**: Decompose each prediction into feature contributions
- **Outputs**:
  - Baseline (expected value over training data)
  - Feature contributions (SHAP values)
  - Percentage importance
  - Human-readable interpretation

---

## 📊 Data Source

All data sourced from **Our World in Data**:

- [CO₂ and Greenhouse Gas Emissions](https://github.com/owid/co2-data)
- [Energy Dataset](https://github.com/owid/energy-data)

**Coverage**: 1965-2022, India-specific metrics

---

## 🎨 Design Philosophy

The UI follows **minimalist modern design principles**:

- **No-scroll layout**: Everything fits in viewport
- **Monochrome palette**: Clean blacks, whites, grays
- **Focus-driven UX**: Input form disappears → results appear
- **System fonts**: Native rendering for crisp typography
- **Subtle interactions**: Minimal animations, maximum clarity

---

## 🛠️ Tech Stack

| Layer                 | Technologies                    |
| --------------------- | ------------------------------- |
| **Backend**           | Flask 2.0+, Flask-CORS          |
| **ML/Data**           | scikit-learn, pandas, numpy     |
| **Explainability**    | SHAP                            |
| **Visualization**     | matplotlib, seaborn (notebooks) |
| **Model Persistence** | joblib                          |
| **Frontend**          | Vanilla HTML5/CSS3/JavaScript   |
| **Data Source**       | Our World in Data (CSV)         |

---

## 📈 Key Insights from Analysis

1. **Structural Break (2015)**: Renewables share changed from declining trend to growth trajectory
2. **Fossil Dominance**: Fossil energy per capita remains the strongest contributor to emissions
3. **Efficiency Gains**: Improving energy intensity (energy/GDP) helps moderate emissions
4. **Forecasting Confidence**:
   - **2023-2050**: Reasonable projections
   - **2051-2100**: Use with caution (extrapolation limits)

---

## 💡 Future Enhancements

- [ ] **GenAI Integration**: Implement LLM-powered policy summarization using Ollama (stubs already in place)
- [ ] Add confidence intervals for predictions
- [ ] Multi-scenario forecasting (optimistic/pessimistic paths)
- [ ] Expand to multiple countries
- [ ] Historical data visualization on frontend
- [ ] Model retraining pipeline with new data
- [ ] Docker containerization for deployment
- [ ] Unit tests for core modules

### 🤖 GenAI Integration Ready

The project includes stub files for future generative AI integration:

- `genai/prompts.py` - Prompt template definitions
- `genai/ollama_client.py` - Ollama API wrapper
- `genai/summarizer.py` - AI-powered policy summarization

These stubs provide clear extension points for adding LLM-based policy recommendations.

---

## 📝 Development

### Project Organization

The modular structure makes development more organized:

**ML/Prediction Logic** (`core/`):

- Modify prediction algorithms in `core/prediction.py`
- Update SHAP explanations in `core/shap_explainer.py`
- Change model loading in `core/model_loader.py`

**Policy Analysis** (`policy/`):

- Add new policy domains to `policy/policy_map.py`
- Update profiling logic in `policy/responsibility.py`
- Enhance policy insights in `policy/policy_engine.py`

**API Layer** (`app.py`):

- Add new endpoints to `app.py`
- Routes are thin and delegate to core/policy modules

**Future AI Integration** (`genai/`):

- Implement Ollama client in `genai/ollama_client.py`
- Design prompts in `genai/prompts.py`
- Build summarizers in `genai/summarizer.py`

### Explore the Notebooks

1. Navigate to `notebooks/`
2. Run `carbon_emission_2.ipynb` for complete analysis
3. See data preprocessing, EDA, model training, and SHAP integration

### Modify Models

Models are persisted as `.pkl` files in `models/`:

- Update training code in notebooks
- Save new models with joblib
- Restart `app.py` to load updated models

---

## 🤝 Contributing

Contributions welcome! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Areas for improvement:

- Model optimization
- UI/UX enhancements
- Additional features
- Documentation improvements

---

## 📄 License

This project is open source and available for educational and research purposes.

---

## 🙏 Acknowledgments

- **Data**: [Our World in Data](https://ourworldindata.org/) for comprehensive CO₂ and energy datasets
- **SHAP**: For making ML models interpretable
- **Open Source Community**: For the amazing tools that made this possible

---

## ⚠️ Disclaimer

This is a forecasting tool built for **educational and research purposes**. Predictions should not be used for official policy decisions without validation by climate science and energy domain experts.

---

**Built with ❤️ for transparency in climate predictions**
