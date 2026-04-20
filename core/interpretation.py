# Interpretation module — richer, human-readable SHAP summaries

FEATURE_DESCRIPTIONS = {
    "energy_per_capita": {
        "label": "Energy per Capita",
        "increase": "rising household and industrial energy demand",
        "decrease": "improved energy efficiency gains",
    },
    "fossil_share_energy": {
        "label": "Fossil Share of Energy",
        "increase": "continued reliance on coal, oil, and natural gas",
        "decrease": "growing shift toward renewable energy sources",
    },
    "energy_per_gdp": {
        "label": "Economic Energy Intensity",
        "increase": "energy-intensive economic activity",
        "decrease": "a more efficient, cleaner economic structure",
    },
    "renewables_share_energy": {
        "label": "Renewables Share",
        "increase": "stronger uptake of solar and wind power",
        "decrease": "slower renewable energy adoption",
    },
}


def generate_interpretation(explanation: dict) -> str:
    """
    Generate a rich human-readable interpretation of SHAP values.

    Args:
        explanation: Dictionary with contributions and percentages

    Returns:
        Descriptive string summarising top feature impacts
    """
    contributions = explanation['contributions']
    percentages   = explanation['percentages']
    prediction    = explanation.get('prediction', None)
    baseline      = explanation.get('baseline', None)

    # Sort by absolute contribution, descending
    sorted_features = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    parts = []
    for feature, contrib in sorted_features[:3]:
        pct   = percentages.get(feature, 0)
        meta  = FEATURE_DESCRIPTIONS.get(feature, {
            "label":    feature.replace('_', ' ').title(),
            "increase": "increasing pressure on emissions",
            "decrease": "reducing emission pressure",
        })
        direction_key = "increase" if contrib > 0 else "decrease"
        impact_word   = "pushes emissions higher" if contrib > 0 else "pulls emissions lower"

        parts.append(
            f"{meta['label']} ({pct:.1f}%) — {meta[direction_key]}, "
            f"which {impact_word} by {abs(contrib):.4f} t"
        )

    summary = "; ".join(parts)

    # Add overall delta note if prediction + baseline available
    if prediction is not None and baseline is not None:
        delta = prediction - baseline
        direction = "above" if delta >= 0 else "below"
        summary += (
            f". Overall, the model predicts {abs(delta):.3f} t/capita "
            f"{direction} the historical baseline of {baseline:.3f} t."
        )

    return summary
