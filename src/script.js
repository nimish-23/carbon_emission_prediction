async function predict() {
    const year = document.getElementById("yearInput").value;
    const resultDiv = document.getElementById("result");
    const errorDiv = document.getElementById("error");

    resultDiv.classList.add("hidden");
    errorDiv.classList.add("hidden");

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
        document.getElementById("co2Value").textContent = data.predicted_co2_per_capita;

        const contributionsList = document.getElementById("contributions");
        contributionsList.innerHTML = "";

        for (const [factor, value] of Object.entries(data.factor_contributions)) {
            const li = document.createElement("li");
            li.textContent = `${factor}: ${value}`;
            contributionsList.appendChild(li);
        }

        resultDiv.classList.remove("hidden");

    } catch (err) {
        errorDiv.textContent = err.message;
        errorDiv.classList.remove("hidden");
    }
}
