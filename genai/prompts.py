"""
Prompt templates for GenAI policy summarization

This module contains carefully engineered prompts for generating
India-specific climate policy insights from model predictions and explanations.

IMPROVEMENTS (2026-02-13):
- Added few-shot examples for better LLM guidance
- Stricter JSON output requirements (no markdown, no extra text)
- Enhanced India-specific context and requirements
- Clearer constraints on response length and structure
- Quantitative requirement to reference SHAP percentages
"""

# System role for the LLM
POLICY_ANALYST_ROLE = """You are an expert climate policy analyst specializing in India's energy transition and carbon emission reduction strategies. 
Your role is to translate machine learning model insights into actionable policy recommendations that consider India's unique context as a developing nation balancing economic growth with climate commitments."""


POLICY_INSIGHT_PROMPT = """You are an expert climate policy analyst specializing in India's energy transition and carbon emission strategies.

CONTEXT - INDIA CO₂ PREDICTION FOR YEAR {year}:
- Predicted CO₂ per capita: {prediction:.3f} tonnes
- Baseline (historical average): {baseline:.3f} tonnes
- Change from baseline: {change:+.3f} tonnes ({change_percent:+.1f}%)

MODEL INSIGHTS - TOP EMISSION DRIVERS (SHAP analysis):
{driver_summary}

POLICY DOMAINS RELEVANT TO THESE DRIVERS:
{policy_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK: Generate 2-3 actionable policy recommendations for India based on the model insights above.

EXAMPLE OUTPUT FORMAT (for reference only, adapt to actual data above):
{{
  "recommendations": [
    {{
      "policy_area": "Renewable Energy Transition",
      "rationale": "Fossil fuel consumption accounts for 52% of the predicted emission increase. Accelerating renewable energy adoption could directly address the majority of emission growth while supporting India's energy security goals.",
      "actions": [
        "Fast-track solar and wind capacity additions to reach 500 GW by 2030",
        "Implement coal phase-down timeline with just transition support for affected regions",
        "Expand renewable energy purchase obligations for industrial consumers"
      ]
    }},
    {{
      "policy_area": "Industrial Energy Efficiency",
      "rationale": "Energy intensity contributes 28% to emission changes. Improving industrial efficiency offers co-benefits of cost savings and competitiveness.",
      "actions": [
        "Mandate energy audits for high-consumption industries",
        "Provide financial incentives for upgrading to energy-efficient machinery"
      ]
    }}
  ]
}}

REQUIREMENTS:
✓ Return ONLY valid JSON (no markdown code blocks, no extra text)
✓ Exactly 2-3 recommendations
✓ Each rationale must reference the quantitative SHAP contribution percentages shown above
✓ Actions must be specific to India's context (development stage, existing policies, institutional capacity)
✓ Actions should be implementable by national or state governments
✓ Focus ONLY on the highest-impact factors (sort by absolute percentage)
✓ Each recommendation: 2-3 concrete actions (not generic advice)

YOUR RESPONSE (valid JSON only, starting with {{ and ending with }}):
"""


SUMMARY_PROMPT = """You are summarizing climate model insights for policymakers.

RESPONSIBILITY PROFILE:
{responsibility_profile}

TASK: Write a 2-3 sentence executive summary.

EXAMPLE (adapt to actual data above):
"Fossil fuel dependency and energy consumption per capita are the primary emission drivers, together accounting for 78% of predicted changes. Targeted policies in renewable energy transition and demand-side efficiency could address the majority of projected emission growth. Industrial and transport sector reforms should be prioritized given their high model-predicted impact."

REQUIREMENTS:
- Exactly 2-3 sentences
- Identify the TOP 2 factors by name
- State their combined impact percentage
- End with one actionable policy insight
- Professional tone for policy brief
- No bullet points or lists

YOUR SUMMARY:
"""


# NOTE: This prompt is currently unused but kept for potential future feature
# to generate individual factor descriptions
FACTOR_DESCRIPTION_PROMPT = """Explain in ONE sentence what the factor "{factor}" means in the context of carbon emissions.

Factor: {factor}
Current contribution to emissions: {contribution:.3f} tonnes CO₂ per capita
Percentage of total impact: {percentage:.1f}%

Your explanation should be:
- Non-technical (understandable to policymakers)
- Specific to India where relevant
- Action-oriented (hint at what could be done)

Format: Just return the single sentence, nothing else."""


# Template for when LLM is unavailable
FALLBACK_RECOMMENDATION_TEMPLATE = {
    "policy_area": "{theme}",
    "rationale": "{description}. The model shows this factor accounts for {impact_percent:.1f}% of predicted emission changes.",
    "actions": [
        "Implement policies targeting {theme_lower}",
        "Consider focused interventions in this area given its high impact"
    ]
}


def format_driver_summary(explanation: dict, top_n: int = 3) -> str:
    """
    Format SHAP explanation into a readable driver summary
    
    Args:
        explanation: Dict with 'contributions' and 'percentages'
        top_n: Number of top drivers to include
        
    Returns:
        Formatted string describing top emission drivers
    """
    contributions = explanation['contributions']
    percentages = explanation['percentages']
    
    # Get top N by absolute percentage
    sorted_factors = sorted(
        percentages.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )[:top_n]
    
    lines = []
    for factor, pct in sorted_factors:
        contrib = contributions[factor]
        direction = "increases" if contrib > 0 else "decreases"
        lines.append(
            f"- {factor}: {direction} emissions by {abs(contrib):.4f} tonnes "
            f"({abs(pct):.1f}% of total impact)"
        )
    
    return "\n".join(lines)


def format_policy_context(responsibility_profile: list) -> str:
    """
    Extract and format policy context from responsibility profile
    
    Args:
        responsibility_profile: List of responsibility items with policy_context
        
    Returns:
        Formatted string with policy themes and areas
    """
    contexts = []
    for item in responsibility_profile:
        if item['policy_relevant'] and item['policy_context']:
            ctx = item['policy_context']
            theme = ctx.get('theme', 'Unknown')
            desc = ctx.get('description', '')
            areas = ctx.get('policy_areas', [])
            
            contexts.append(
                f"**{theme}**: {desc}\n"
                f"  Relevant areas: {', '.join(areas[:3])}"
            )
    
    return "\n\n".join(contexts) if contexts else "No specific policy context available."
