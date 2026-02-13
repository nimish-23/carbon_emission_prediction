"""
Policy context mapping for emission drivers

Maps model features to policy-relevant themes and actionable areas
"""

# Policy context for each emission driver feature
POLICY_CONTEXT_MAP = {
    "energy_per_capita": {
        "theme": "Energy Consumption & Efficiency",
        "description": "Per capita energy use directly drives emissions. Reducing energy intensity while maintaining economic growth is critical.",
        "policy_areas": [
            "Demand-side energy efficiency programs",
            "Building energy codes and retrofits",
            "Industrial energy audits and standards",
            "Public awareness campaigns on energy conservation"
        ],
        "policy_relevant": True
    },
    "fossil_share_energy": {
        "theme": "Energy Transition & Decarbonization",
        "description": "High fossil fuel dependency locks in emissions. Transitioning to renewables is the most direct decarbonization pathway.",
        "policy_areas": [
            "Accelerate renewable energy capacity additions",
            "Coal phase-down roadmap with just transition support",
            "Grid modernization for renewable integration",
            "Carbon pricing mechanisms"
        ],
        "policy_relevant": True
    },
    "energy_per_gdp": {
        "theme": "Economic Energy Intensity",
        "description": "Energy intensity reflects how efficiently the economy converts energy to output. Lower intensity means cleaner growth.",
        "policy_areas": [
            "Structural economic transformation toward services",
            "Technology transfer and innovation incentives",
            "Energy productivity targets by sector",
            "Industrial cluster modernization programs"
        ],
        "policy_relevant": True
    },
    "renewables_share_energy": {
        "theme": "Renewable Energy Deployment",
        "description": "Renewable energy share directly displaces fossil fuels. India has strong solar and wind potential.",
        "policy_areas": [
            "Fast-track renewable project approvals",
            "Renewable Purchase Obligations (RPO) enforcement",
            "Energy storage and grid balancing infrastructure",
            "Green hydrogen and emerging technologies"
        ],
        "policy_relevant": True
    }
}


def get_policy_context(feature_name: str) -> dict:
    """
    Get policy context for a given feature
    
    Args:
        feature_name: Name of the model feature (e.g., 'energy_per_capita')
        
    Returns:
        Dict with 'theme', 'description', 'policy_areas', 'policy_relevant'
        If feature not found, returns a default context
    """
    if feature_name in POLICY_CONTEXT_MAP:
        return POLICY_CONTEXT_MAP[feature_name].copy()
    
    # Default context for unknown features
    return {
        "theme": f"Policy Area: {feature_name.replace('_', ' ').title()}",
        "description": f"This factor impacts emissions and should be considered in policy planning.",
        "policy_areas": [
            f"Monitor and analyze {feature_name.replace('_', ' ')}",
            "Develop targeted interventions"
        ],
        "policy_relevant": True  # Assume all features are policy-relevant by default
    }


def enrich_responsibility_profile(responsibility_profile: list) -> list:
    """
    Add policy context to each item in the responsibility profile
    
    Args:
        responsibility_profile: List of dicts with 'factor', 'impact_value', 'impact_percent'
        
    Returns:
        Same list with added 'policy_relevant' and 'policy_context' fields
    """
    enriched = []
    for item in responsibility_profile:
        factor = item['factor']
        context = get_policy_context(factor)
        
        enriched_item = item.copy()
        enriched_item['policy_relevant'] = context['policy_relevant']
        enriched_item['policy_context'] = {
            'theme': context['theme'],
            'description': context['description'],
            'policy_areas': context['policy_areas']
        }
        enriched.append(enriched_item)
    
    return enriched
