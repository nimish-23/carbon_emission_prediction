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

- `/predict` - Basic CO₂ predictions
- `/predict/explain` - Predictions with SHAP feature explanations
- Comprehensive error handling and validation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Input (Year)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Energy Driver Models (4 Models)              │
│  • Energy per capita        • Renewables share          │
│  • Fossil energy per capita • Energy intensity (GDP)    │
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
│        CO₂ Prediction + Explanation (JSON)               │
└─────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
carbon_emission_prediction/
├── app.py                          # Flask API server (259 lines)
├── requirements.txt                # Python dependencies
├── data/
│   ├── owid-co2-data.csv          # Historical CO₂ emissions
│   └── owid-energy-data.csv       # Energy consumption data
├── models/
│   ├── driver_models.pkl          # 4 energy trend models
│   ├── co2_model.pkl              # CO₂ regression model
│   ├── shap_explainer.pkl         # SHAP explainer object
│   └── training_stats.pkl         # Model metadata & baseline
├── notebooks/
│   ├── carbon_emission.ipynb      # Initial EDA & modeling
│   └── carbon_emission_2.ipynb    # Advanced model development
└── src/                            # Frontend application
    ├── index.html                 # Minimalist UI
    ├── script.js                  # Client logic + explanations
    └── style.css                  # Modern responsive design
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

### Standard Prediction

**Endpoint:** `POST /predict`

**Request:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"year": 2030}'
```

**Response:**

```json
{
  "year": 2030,
  "predicted_co2_per_capita": 2.456,
  "projected_drivers": {
    "energy_per_capita": 23.789,
    "fossil_energy_per_capita": 18.234,
    "renewables_share_energy": 15.678,
    "energy_per_gdp": 1.234
  }
}
```

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
    "fossil_energy_per_capita": 18.234,
    "renewables_share_energy": 15.678,
    "energy_per_gdp": 1.234
  },
  "explanation": {
    "contributions": {
      "fossil_energy_per_capita": 0.3456,
      "energy_per_capita": 0.2123,
      "renewables_share_energy": -0.0987,
      "energy_per_gdp": 0.0234
    },
    "percentages": {
      "fossil_energy_per_capita": 52.3,
      "energy_per_capita": 32.1,
      "renewables_share_energy": 14.9,
      "energy_per_gdp": 0.7
    },
    "interpretation": "Fossil Energy Per Capita increases emissions (52.3% impact); Energy Per Capita increases emissions (32.1% impact)"
  }
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

- [ ] Add confidence intervals for predictions
- [ ] Multi-scenario forecasting (optimistic/pessimistic paths)
- [ ] Expand to multiple countries
- [ ] Historical data visualization on frontend
- [ ] Model retraining pipeline with new data
- [ ] Docker containerization for deployment

---

## 📝 Development

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
