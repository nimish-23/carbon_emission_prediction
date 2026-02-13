"""
GenAI-powered policy summarizer

This module uses LLMs to generate sophisticated, context-aware policy insights
from SHAP model explanations and responsibility profiles.
"""
import json
import traceback
from typing import Dict, List, Optional
from .ollama_client import OllamaClient
from .prompts import (
    POLICY_ANALYST_ROLE,
    POLICY_INSIGHT_PROMPT,
    SUMMARY_PROMPT,
    FALLBACK_RECOMMENDATION_TEMPLATE,
    format_driver_summary,
    format_policy_context
)


class PolicySummarizer:
    """
    Uses GenAI to generate enhanced policy insights from model predictions
    
    This class integrates with Ollama to produce natural language policy
    recommendations that are:
    - Context-aware (India-specific)
    - Grounded in model explanations (SHAP values)
    - Actionable for policymakers
    """
    
    def __init__(self, model_name: str = "llama3.2", use_llm: bool = True):
        """
        Initialize policy summarizer
        
        Args:
            model_name: Ollama model to use (e.g., 'llama3.2', 'mistral', 'phi')
            use_llm: Whether to use LLM or fallback to rule-based approach
        """
        self.client = OllamaClient(model_name=model_name)
        self.use_llm = use_llm  # Don't check availability at init, check at runtime
    
    def summarize_responsibility_profile(self, responsibility_profile: List[dict]) -> str:
        """
        Generate natural language summary of responsibility profile
        
        Args:
            responsibility_profile: List of factor contributions with policy context
            
        Returns:
            Natural language summary of top emission drivers
            
        Example:
            >>> summarizer = PolicySummarizer()
            >>> profile = [...] # from build_responsibility_profile()
            >>> summary = summarizer.summarize_responsibility_profile(profile)
            >>> print(summary)
            "Energy consumption per capita and fossil fuel dependency are the 
            dominant drivers, accounting for 85% of predicted emission changes."
        """
        # Check LLM availability at runtime
        if not self.use_llm or not self.client.is_available() or not responsibility_profile:
            # Fallback: simple text summary
            top_factors = [
                item for item in responsibility_profile 
                if item['policy_relevant']
            ][:2]
            
            if not top_factors:
                return "No significant policy-relevant drivers identified."
            
            total_pct = sum(item['impact_percent'] for item in top_factors)
            factor_names = [item['factor'].replace('_', ' ') for item in top_factors]
            
            return (
                f"{' and '.join(factor_names)} are the dominant drivers, "
                f"accounting for {total_pct:.0f}% of predicted emission changes."
            )
        
        # LLM-based summary
        profile_text = "\n".join([
            f"- {item['factor']}: {item['impact_percent']:.1f}% impact"
            for item in responsibility_profile[:5]
        ])
        
        prompt = SUMMARY_PROMPT.format(responsibility_profile=profile_text)
        
        try:
            summary = self.client.generate(
                prompt,
                temperature=0.5,
                max_tokens=150
            )
            return summary if not summary.startswith("[LLM Error") else self._fallback_summary(responsibility_profile)
        except Exception as e:
            print(f"⚠ LLM summarization failed: {e}")
            return self._fallback_summary(responsibility_profile)
    
    def _fallback_summary(self, responsibility_profile: List[dict]) -> str:
        """Fallback summary when LLM fails"""
        top_factors = [
            item for item in responsibility_profile 
            if item['policy_relevant']
        ][:2]
        
        if not top_factors:
            return "No significant policy-relevant drivers identified."
        
        factor_names = [item['factor'].replace('_', ' ') for item in top_factors]
        total_pct = sum(item['impact_percent'] for item in top_factors)
        
        return (
            f"{' and '.join(factor_names)} are the top emission drivers, "
            f"collectively representing {total_pct:.0f}% of the predicted impact."
        )
    
    def generate_policy_recommendations(
        self, 
        explanation: Dict, 
        responsibility_profile: List[dict],
        year: int
    ) -> List[dict]:
        """
        Generate detailed policy recommendations using GenAI
        
        Args:
            explanation: Dict with 'prediction', 'baseline', 'contributions', 'percentages'
            responsibility_profile: List of policy-relevant factors
            year: Prediction year
            
        Returns:
            List of policy recommendation dicts with:
            - policy_area: Name of policy domain
            - rationale: Why this matters
            - actions: List of specific actions
            
        Example:
            >>> summarizer = PolicySummarizer()
            >>> recommendations = summarizer.generate_policy_recommendations(
            ...     explanation, profile, 2030
            ... )
            >>> for rec in recommendations:
            ...     print(rec['policy_area'], ':', rec['rationale'])
        """
        # Check LLM availability at runtime
        if not self.use_llm or not self.client.is_available():
            print("⚠ LLM not available, using fallback recommendations")
            return self._generate_fallback_recommendations(responsibility_profile)
        
        # Prepare prompt data
        prediction = explanation['prediction']
        baseline = explanation['baseline']
        change = prediction - baseline
        change_percent = (change / baseline * 100) if baseline != 0 else 0
        
        driver_summary = format_driver_summary(explanation, top_n=3)
        policy_context = format_policy_context(responsibility_profile)
        
        prompt = POLICY_INSIGHT_PROMPT.format(
            year=year,
            prediction=prediction,
            baseline=baseline,
            change=change,
            change_percent=change_percent,
            driver_summary=driver_summary,
            policy_context=policy_context
        )
        
        try:
            print("🤖 Generating policy recommendations with LLM...")
            response = self.client.generate(
                prompt,
                temperature=0.6,
                max_tokens=800
            )
            
            if response.startswith("[LLM Error"):
                print(f"⚠ LLM returned error: {response}")
                return self._generate_fallback_recommendations(responsibility_profile)
            
            # Parse JSON response
            recommendations = self._parse_llm_recommendations(response)
            
            if not recommendations:
                print("⚠ Failed to parse LLM response, using fallback")
                return self._generate_fallback_recommendations(responsibility_profile)
            
            # Add source field to mark as LLM-generated
            for rec in recommendations:
                rec['source'] = 'genai'
            
            print(f"✅ Generated {len(recommendations)} LLM-powered recommendations")
            return recommendations
            
        except Exception as e:
            print(f"⚠ Policy recommendation generation failed: {e}")
            traceback.print_exc()
            return self._generate_fallback_recommendations(responsibility_profile)
    
    def _parse_llm_recommendations(self, llm_response: str) -> Optional[List[dict]]:
        """
        Parse LLM JSON response into structured recommendations
        
        Args:
            llm_response: Raw LLM response text
            
        Returns:
            List of recommendation dicts or None if parsing fails
        """
        try:
            # Try to extract JSON from response
            # LLMs sometimes add explanation text before/after JSON or use markdown code blocks
            
            # Remove markdown code blocks if present
            response = llm_response.strip()
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                response = response[start:end].strip()
            
            # Extract JSON object
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                print(f"⚠ No JSON object found in LLM response")
                print(f"Response preview: {llm_response[:200]}...")
                return None
            
            json_str = response[start_idx:end_idx]
            
            # Try to parse
            data = json.loads(json_str)
            
            recommendations = data.get('recommendations', [])
            
            if not recommendations:
                print("⚠ No 'recommendations' key found in JSON")
                return None
            
            # Validate structure
            valid_recommendations = []
            for rec in recommendations:
                # Be flexible - accept recommendations with at least policy_area and rationale
                if 'policy_area' in rec and 'rationale' in rec:
                    # Ensure actions is a list
                    if 'actions' not in rec:
                        rec['actions'] = []
                    elif isinstance(rec['actions'], str):
                        rec['actions'] = [rec['actions']]
                    valid_recommendations.append(rec)
                else:
                    print(f"⚠ Skipping invalid recommendation: {rec}")
            
            if not valid_recommendations:
                return None
            
            return valid_recommendations
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"⚠ Failed to parse LLM response as JSON: {e}")
            return None
    
    def _generate_fallback_recommendations(self, responsibility_profile: List[dict]) -> List[dict]:
        """
        Generate rule-based recommendations when LLM is unavailable
        
        Args:
            responsibility_profile: List of policy-relevant factors
            
        Returns:
            List of basic policy recommendations
        """
        recommendations = []
        
        # Get top policy-relevant factors
        relevant_factors = [
            item for item in responsibility_profile 
            if item['policy_relevant'] and item['policy_context']
        ][:3]
        
        for item in relevant_factors:
            context = item['policy_context']
            theme = context.get('theme', 'Unknown Policy Area')
            description = context.get('description', 'This factor impacts emissions')
            policy_areas = context.get('policy_areas', [])
            
            recommendations.append({
                'policy_area': theme,
                'rationale': (
                    f"{description}. Model analysis shows this factor accounts for "
                    f"{item['impact_percent']:.1f}% of predicted emission changes, "
                    f"making it a high-priority area for policy intervention."
                ),
                'actions': policy_areas[:3] if policy_areas else [
                    f"Implement targeted policies for {theme.lower()}",
                    "Monitor and evaluate impact regularly"
                ],
                'source': 'fallback'
            })
        
        return recommendations
