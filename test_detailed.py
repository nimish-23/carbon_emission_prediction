"""
Detailed test to see full API response structure
"""
import requests
import json

response = requests.post(
    "http://localhost:5000/predict/explain-policy",
    json={"year": 2030},
    timeout=30
)

data = response.json()

# Save to file for inspection
with open("api_response.json", "w") as f:
    json.dump(data, f, indent=2)

print("Full response saved to api_response.json")
print(f"\ngenai_enabled: {data.get('genai_enabled')}")
print(f"Number of insights: {len(data.get('policy_insights', []))}")

# Check each insight's source
for i, insight in enumerate(data.get('policy_insights', []), 1):
    print(f"\n  Insight {i}:")
    print(f"    Source: {insight.get('source', 'NO SOURCE FIELD')}")
    print(f"    Theme/Area: {insight.get('theme', insight.get('policy_area', 'N/A'))[:50]}")
