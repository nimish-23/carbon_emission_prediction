"""
Quick test script to diagnose GenAI integration issues
"""
import sys
from genai.ollama_client import OllamaClient
from genai.summarizer import PolicySummarizer

def test_ollama_connection():
    """Test basic Ollama connectivity"""
    print("=" * 60)
    print("Testing Ollama Connection")
    print("=" * 60)
    
    client = OllamaClient(model_name="llama3.2")
    
    # Check server availability
    if client.is_available():
        print("✓ Ollama server is reachable")
    else:
        print("❌ Ollama server is NOT reachable")
        return False
    
    # Test simple generation
    print("\nTesting text generation...")
    try:
        response = client.generate("Say 'Hello, I am working!' in one sentence.", temperature=0.7)
        if response and not response.startswith("[LLM Error"):
            print(f"✓ Generation successful: {response[:100]}")
            return True
        else:
            print(f"❌ Generation failed: {response}")
            return False
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def test_policy_summarizer():
    """Test PolicySummarizer initialization"""
    print("\n" + "=" * 60)
    print("Testing PolicySummarizer")
    print("=" * 60)
    
    try:
        summarizer = PolicySummarizer(model_name="llama3.2", use_llm=True)
        print(f"✓ PolicySummarizer initialized")
        print(f"  - use_llm: {summarizer.use_llm}")
        
        # Test simple recommendation generation with mock data
        mock_profile = [
            {
                'factor': 'energy_per_capita',
                'impact_value': 0.5,
                'impact_percent': 60.0,
                'policy_relevant': True,
                'policy_context': {
                    'theme': 'Energy Transition',
                    'description': 'Energy consumption drives emissions',
                    'policy_areas': ['Renewable energy', 'Energy efficiency']
                }
            }
        ]
        
        mock_explanation = {
            'prediction': 2.0,
            'baseline': 1.5,
            'contributions': {'energy_per_capita': 0.5},
            'percentages': {'energy_per_capita': 60.0}
        }
        
        print("\nTesting policy recommendation generation...")
        recommendations = summarizer.generate_policy_recommendations(
            mock_explanation,
            mock_profile,
            2030
        )
        
        print(f"✓ Generated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  Recommendation {i}:")
            print(f"    - Policy Area: {rec.get('policy_area', 'N/A')}")
            print(f"    - Source: {rec.get('source', 'genai')}")
            if 'why_it_matters' in rec:
                print(f"    - Why: {rec['why_it_matters'][:80]}...")
            if 'rationale' in rec:
                print(f"    - Rationale: {rec['rationale'][:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ PolicySummarizer error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🔍 GenAI Diagnostic Test\n")
    
    # Run tests
    ollama_ok = test_ollama_connection()
    policy_ok = test_policy_summarizer()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Ollama Connection: {'✓ PASS' if ollama_ok else '❌ FAIL'}")
    print(f"Policy Summarizer: {'✓ PASS' if policy_ok else '❌ FAIL'}")
    
    if ollama_ok and policy_ok:
        print("\n✅ All tests passed! GenAI should be working.")
    else:
        print("\n⚠ Some tests failed. Check errors above.")
    
    sys.exit(0 if (ollama_ok and policy_ok) else 1)
