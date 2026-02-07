# Interpretation module


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
