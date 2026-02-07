# Prediction module
import pandas as pd
from .model_loader import driver_models, co2_model, FEATURES


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
