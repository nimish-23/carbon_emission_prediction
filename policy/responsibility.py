"""
Responsibility profiling - builds structured responsibility profiles from SHAP output
"""
from .policy_map import POLICY_DOMAIN_MAP


def build_responsibility_profile(explanation: dict, threshold: float = 5.0):
    """
    Build a structured responsibility profile from SHAP output
    """
    profile = []

    for feature, pct in explanation["percentages"].items():
        profile.append({
            "factor": feature,
            "impact_percent": pct,
            "impact_value": explanation["contributions"][feature],
            "policy_relevant": pct >= threshold,
            "policy_context": POLICY_DOMAIN_MAP.get(feature, {})
        })

    # Sort by importance
    profile.sort(key=lambda x: x["impact_percent"], reverse=True)
    return profile
