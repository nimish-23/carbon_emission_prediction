# SHAP explainer module

import pandas as pd
from .model_loader import co2_model, shap_explainer, training_stats, FEATURES


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
