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
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ year: parseInt(year) })
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
    
    try {
        const response = await fetch('http://127.0.0.1:5000/predict/explain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ year: parseInt(year) })
        });
        
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
        
        // Show explanation
        displayExplanation(data.explanation, data.baseline);
        
    } catch (error) {
        console.error('Error:', error);
        errorDiv.textContent = error.message;
        errorDiv.classList.remove("hidden");
    }
}

function displayExplanation(explanation, baseline) {
    const section = document.getElementById('explanation-section');
    const content = document.getElementById('explanation-content');
    
    // Show the section
    section.style.display = 'block';
    
    // Build HTML for explanation
    let html = `
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px;">
            <p><strong>Baseline (Average):</strong> ${baseline.toFixed(3)} tons/capita</p>
            <p style="color: #666; font-size: 14px;">
                ${explanation.interpretation}
            </p>
            
            <h4 style="margin-top: 20px;">Feature Contributions:</h4>
    `;
    
    // Sort contributions by absolute value
    const sorted = Object.entries(explanation.contributions)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    
    sorted.forEach(([feature, contrib]) => {
        const pct = explanation.percentages[feature];
        const color = contrib > 0 ? '#e74c3c' : '#27ae60';
        const arrow = contrib > 0 ? '↑' : '↓';
        const barWidth = pct;
        
        html += `
            <div style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>${feature.replace(/_/g, ' ')}</span>
                    <span style="color: ${color}; font-weight: bold;">
                        ${arrow} ${contrib > 0 ? '+' : ''}${contrib.toFixed(4)} (${pct.toFixed(1)}%)
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