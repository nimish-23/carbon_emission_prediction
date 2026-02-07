"""
Policy engine - generates policy insights from responsibility profiles
"""


def generate_policy_insights(responsibility_profile: list) -> list:
    """
    Generate policy insights from responsibility profile.
    This is where GenAI will later be plugged in.
    """
    insights = []

    for item in responsibility_profile:
        if not item["policy_relevant"]:
            continue

        context = item["policy_context"]

        insights.append({
            "factor": item["factor"],
            "theme": context.get("theme"),
            "why_it_matters": context.get("description"),
            "policy_focus": context.get("policy_areas"),
            "model_signal": f"Accounts for {item['impact_percent']:.1f}% of the predicted emissions impact"
        })

    return insights
