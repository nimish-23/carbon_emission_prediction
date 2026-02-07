"""
Policy domain mapping - maps features to policy themes and areas
"""

POLICY_DOMAIN_MAP = {
    "energy_per_capita": {
        "theme": "Energy Demand Reduction",
        "description": "High total energy consumption across households, transport, and industry",
        "policy_areas": [
            "Public transport expansion",
            "Energy-efficient buildings",
            "Urban planning and densification",
            "Appliance efficiency standards",
            "Behavioral energy conservation"
        ]
    },
    "fossil_share_energy": {
        "theme": "Energy Supply Decarbonization",
        "description": "High dependence on fossil fuels in the energy mix",
        "policy_areas": [
            "Renewable energy scale-up",
            "Coal phase-down",
            "Grid modernization",
            "Energy storage deployment",
            "Carbon pricing mechanisms"
        ]
    },
    "energy_per_gdp": {
        "theme": "Economic Energy Efficiency",
        "description": "Low energy efficiency of economic output",
        "policy_areas": [
            "Industrial efficiency programs",
            "Technology modernization",
            "Electrification of industry",
            "Process optimization"
        ]
    }
}
