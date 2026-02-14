async function predict() {
  const yearInput = document.getElementById("yearInput");
  const resultDiv = document.getElementById("result");
  const errorDiv = document.getElementById("error");

  resultDiv.classList.add("hidden");
  errorDiv.classList.add("hidden");

  const year = yearInput.value;

  if (!year) {
    errorDiv.textContent = "Please enter a year.";
    errorDiv.classList.remove("hidden");
    return;
  }

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ year: parseInt(year) }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Something went wrong");
    }

    // Update UI
    document.getElementById("resultYear").textContent = data.year;
    document.getElementById("co2Value").textContent =
      data.predicted_co2_per_capita.toFixed(3);

    resultDiv.classList.remove("hidden");
  } catch (err) {
    errorDiv.textContent = err.message;
    errorDiv.classList.remove("hidden");
  }
}

async function predictWithExplanation() {
  const yearInput = document.getElementById("yearInput");
  const resultDiv = document.getElementById("result");
  const errorDiv = document.getElementById("error");
  const inputBox = document.querySelector(".input-box");
  const btnExplain = document.querySelector(".btn-explain");

  // Hide previous results and errors
  resultDiv.classList.add("hidden");
  errorDiv.classList.add("hidden");

  const year = yearInput.value;

  // Validate year input
  if (!year) {
    errorDiv.textContent = "Please enter a year.";
    errorDiv.classList.remove("hidden");
    return;
  }

  const yearNum = parseInt(year);
  
  // Validate year range
  if (yearNum < 1965 || yearNum > 2100) {
    errorDiv.textContent = "Please enter a year between 1965 and 2100.";
    errorDiv.classList.remove("hidden");
    return;
  }

  // Show loading state
  btnExplain.textContent = "⏳ Generating predictions & policy insights...";
  btnExplain.disabled = true;

  try {
    // Call the policy endpoint to get both ML analysis AND policy recommendations
    const response = await fetch(
      "http://127.0.0.1:5000/predict/explain-policy",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ year: parseInt(year) }),
      },
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Something went wrong");
    }

    // Hide input and button sections
    inputBox.style.display = "none";
    btnExplain.style.display = "none";

    // Show basic prediction using the correct element IDs
    document.getElementById("resultYear").textContent = data.year;
    document.getElementById("co2Value").textContent =
      data.predicted_co2_per_capita.toFixed(3);

        resultDiv.classList.remove("hidden");
        
        // Create grid container for side-by-side layout (if not exists)
        let insightsGrid = document.getElementById('insights-grid');
        if (!insightsGrid) {
            insightsGrid = document.createElement('div');
            insightsGrid.id = 'insights-grid';
            insightsGrid.className = 'insights-grid';
            resultDiv.parentNode.insertBefore(insightsGrid, resultDiv.nextSibling);
        }
        
        // Show SHAP explanation
        displayExplanation(data.responsibility_profile, data.baseline);
        
        // Show policy recommendations
        displayPolicyRecommendations(data.policy_insights, data.genai_enabled);
        
        // Move both sections into grid
        const explanationSection = document.getElementById('explanation-section');
        const policySection = document.getElementById('policy-section');
        if (explanationSection && policySection) {
            insightsGrid.appendChild(explanationSection);
            insightsGrid.appendChild(policySection);
        }
  } catch (error) {
    console.error("Error:", error);
    btnExplain.textContent = "🔍 Predict & Explain";
    btnExplain.disabled = false;
    errorDiv.textContent = error.message;
    errorDiv.classList.remove("hidden");
  }
}

function displayExplanation(responsibilityProfile, baseline) {
    const section = document.getElementById('explanation-section');
    const content = document.getElementById('explanation-content');
    
    // Show the section
    section.style.display = 'block';
    
    // Build HTML for explanation
    let html = `
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
            <p><strong>Baseline (Average):</strong> ${baseline.toFixed(3)} tons/capita</p>
            
            <h4 style="margin-top: 20px; margin-bottom: 15px;">🧠 Model-Based Feature Contributions:</h4>
    `;
    
    // Display responsibility profile items
    responsibilityProfile.forEach(item => {
        const factor = item.factor;
        const impact = item.impact_value;
        const pct = item.impact_percent;
        const color = impact > 0 ? '#e74c3c' : '#27ae60';
        const arrow = impact > 0 ? '↑' : '↓';
        const barWidth = Math.abs(pct);
        
        html += `
            <div style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>${factor.replace(/_/g, ' ')}</span>
                    <span style="color: ${color}; font-weight: bold;">
                        ${arrow} ${impact > 0 ? '+' : ''}${impact.toFixed(4)} (${pct.toFixed(1)}%)
                    </span>
                </div>
                <div style="background: #e0e0e0; height: 8px; border-radius: 4px;">
                    <div style="
                        background: ${color};
                        width: ${barWidth}%;
                        height: 100%;
                        border-radius: 4px;
                        transition: width 0.5s ease;
                    "></div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    content.innerHTML = html;
}

function displayPolicyRecommendations(policyInsights, genaiEnabled) {
    // Check if policy section already exists, if not create it
    let policySection = document.getElementById('policy-section');
    
    if (!policySection) {
        // Create policy section after explanation section
        const explanationSection = document.getElementById('explanation-section');
        policySection = document.createElement('div');
        policySection.id = 'policy-section';
        policySection.className = 'policy-section';
        explanationSection.parentNode.insertBefore(policySection, explanationSection.nextSibling);
    }
    
    // Show the section
    policySection.style.display = 'block';
    
    // Build HTML for policy recommendations
    let html = `
        <h3>🏛️ Policy Recommendations</h3>
        ${genaiEnabled ? 
            '<p class="genai-badge">✨ Generated by AI • Based on model insights</p>' : 
            '<p class="rule-based-badge">📋 Rule-based recommendations</p>'
        }
    `;
    
    if (!policyInsights || policyInsights.length === 0) {
        html += '<p style="color: #666;">No policy recommendations available.</p>';
    } else {
        html += '<div class="policy-cards">';
        
        policyInsights.forEach((insight, index) => {
            const policyArea = insight.theme || insight.policy_area || insight.factor;
            const rationale = insight.why_it_matters || insight.rationale || 'N/A';
            const actions = insight.policy_focus || insight.actions || [];
            
            html += `
                <div class="policy-card">
                    <div class="policy-header">
                        <span class="policy-number">${index + 1}</span>
                        <h4 class="policy-title">${policyArea}</h4>
                    </div>
                    <p class="policy-rationale">${rationale}</p>
                    ${actions.length > 0 ? `
                        <div class="policy-actions">
                            <strong>Recommended Actions:</strong>
                            <ul>
                                ${actions.slice(0, 3).map(action => 
                                    `<li>${action}</li>`
                                ).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            `;
        });
        
        html += '</div>';
    }
    
    policySection.innerHTML = html;
}

// Add Enter key support for the year input
document.addEventListener('DOMContentLoaded', function() {
    const yearInput = document.getElementById('yearInput');
    if (yearInput) {
        yearInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                predictWithExplanation();
            }
        });
    }
});
