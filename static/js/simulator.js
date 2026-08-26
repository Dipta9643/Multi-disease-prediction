/**
 * JavaScript controlling the Simulator inputs and real-time Chart.js updates.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Initialise slider label mappings
    initRangeSliderLabel("age", "age-val", "");
    initRangeSliderLabel("bmi", "bmi-val", "");
    initRangeSliderLabel("exercise_duration", "exercise_duration-val", " min");
    initRangeSliderLabel("fruit_veg_consumption", "fruit_veg_consumption-val", " servings");
    initRangeSliderLabel("sleep_duration", "sleep_duration-val", " hrs");

    const charts = {};
    const diseaseIds = ["diabetes", "hypertension", "heart_disease"];
    const riskClasses = ["low", "moderate", "high"];

    // Initialize Chart.js Gauges
    diseaseIds.forEach(diseaseId => {
        const origProb = originalPredictions[diseaseId] ? originalPredictions[diseaseId].probability : 0.0;
        const origColor = originalPredictions[diseaseId] ? originalPredictions[diseaseId].color : "#00d4aa";
        
        const canvas = document.getElementById(`gauge-${diseaseId}`);
        if (canvas) {
            const ctx = canvas.getContext('2d');
            charts[diseaseId] = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: [origProb * 100, 100 - (origProb * 100)],
                        backgroundColor: [origColor, 'rgba(255, 255, 255, 0.05)'],
                        borderWidth: 0,
                        borderRadius: 4
                    }]
                },
                options: {
                    rotation: 270,
                    circumference: 180,
                    cutout: '80%',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        tooltip: { enabled: false },
                        legend: { display: false }
                    }
                }
            });
        }
        
        // Initial text display
        updateDisplayElements(diseaseId, origProb, originalPredictions[diseaseId]);
    });

    // Debounce Helper
    function debounce(func, wait) {
        let timeout;
        return function(...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    // Perform API call to simulate
    const runSimulation = debounce(() => {
        const formData = new FormData(document.getElementById("sim-form"));
        const data = {};
        formData.forEach((value, key) => {
            // Convert to floats or ints
            if (key === "bmi" || key === "fruit_veg_consumption" || key === "sleep_duration") {
                data[key] = parseFloat(value);
            } else {
                data[key] = parseInt(value);
            }
        });

        // Call Flask API endpoint
        fetch(simulateApiUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        })
        .then(res => res.json())
        .then(response => {
            if (response.success) {
                const results = response.predictions;
                diseaseIds.forEach(diseaseId => {
                    const result = results[diseaseId];
                    const prob = result.probability;
                    const chart = charts[diseaseId];
                    
                    // Update Chart.js dataset
                    if (chart) {
                        chart.data.datasets[0].data = [prob * 100, 100 - (prob * 100)];
                        chart.data.datasets[0].backgroundColor[0] = result.color;
                        chart.update();
                    }
                    
                    // Update display labels
                    updateDisplayElements(diseaseId, prob, result);
                });
            }
        })
        .catch(err => console.error("Simulation error: ", err));
    }, 250);

    function updateDisplayElements(diseaseId, prob, result) {
        const valLabel = document.getElementById(`val-${diseaseId}`);
        const tagLabel = document.getElementById(`tag-${diseaseId}`);
        const diffLabel = document.getElementById(`diff-${diseaseId}`);
        const cardElement = document.getElementById(`sim-card-${diseaseId}`);

        if (valLabel) valLabel.textContent = `${(prob * 100).toFixed(1)}%`;
        
        if (tagLabel) {
            tagLabel.textContent = result.risk_label;
            // Clear old risk level classes and apply new one
            riskClasses.forEach(c => tagLabel.classList.remove(c));
            tagLabel.classList.add(result.risk_level);
        }

        if (cardElement) {
            // Apply border highlight color matching risk status
            cardElement.style.borderTop = `3px solid ${result.color}`;
        }

        if (diffLabel && originalPredictions[diseaseId]) {
            const origProb = originalPredictions[diseaseId].probability;
            const diff = prob - origProb;
            const diffPct = Math.abs(diff * 100).toFixed(1);
            
            // Clear classes
            diffLabel.classList.remove("up", "down", "neutral");
            
            if (diff > 0.005) { // Threshold for significant change
                diffLabel.classList.add("up");
                diffLabel.innerHTML = `<span>▲</span> <span>+${diffPct}% (Increased)</span>`;
            } else if (diff < -0.005) {
                diffLabel.classList.add("down");
                diffLabel.innerHTML = `<span>▼</span> <span>-${diffPct}% (Decreased)</span>`;
            } else {
                diffLabel.classList.add("neutral");
                diffLabel.innerHTML = `<span>▬</span> <span>No change</span>`;
            }
        }
    }

    // Bind event listeners to form controls
    const formInputs = document.querySelectorAll("#sim-form input, #sim-form select");
    formInputs.forEach(input => {
        input.addEventListener("input", runSimulation);
        input.addEventListener("change", runSimulation);
    });

    // Reset handler
    const resetBtn = document.getElementById("btn-reset-sim");
    if (resetBtn) {
        resetBtn.addEventListener("click", () => {
            const form = document.getElementById("sim-form");
            
            // Map inputs back to originalInput values
            Object.keys(originalInput).forEach(key => {
                const element = document.getElementById(key);
                if (element) {
                    element.value = originalInput[key];
                    // Manually trigger range label change updates
                    const event = new Event("input");
                    element.dispatchEvent(event);
                }
            });
            
            // Re-run simulation calculations
            runSimulation();
        });
    }
});
