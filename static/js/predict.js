/**
 * JavaScript controlling the Multi-Step prediction form navigation.
 */

document.addEventListener("DOMContentLoaded", () => {
    let currentStep = 1;
    const totalSteps = 8;
    
    // Elements
    const prevBtn = document.getElementById("btn-prev");
    const nextBtn = document.getElementById("btn-next");
    const submitBtn = document.getElementById("btn-submit");
    const progressBarFill = document.getElementById("progress-bar-fill");
    
    // Range sliders linking
    initRangeSliderLabel("bmi", "bmi-val", "");
    initRangeSliderLabel("exercise_duration", "exercise_duration-val", " min");
    initRangeSliderLabel("fruit_veg_consumption", "fruit_veg_consumption-val", " servings");
    initRangeSliderLabel("sleep_duration", "sleep_duration-val", " hrs");

    // BMI Calculator functionality
    const bmiCalcToggle = document.getElementById("bmi-calc-toggle");
    const bmiCalcContainer = document.getElementById("bmi-calc-container");
    const heightInput = document.getElementById("bmi-height");
    const weightInput = document.getElementById("bmi-weight");
    const calculateBmiBtn = document.getElementById("btn-calculate-bmi");
    const bmiCalcResult = document.getElementById("bmi-calc-result");
    const bmiRangeInput = document.getElementById("bmi");
    const bmiValDisplay = document.getElementById("bmi-val");

    if (bmiCalcToggle && bmiCalcContainer) {
        bmiCalcToggle.addEventListener("click", () => {
            const isHidden = bmiCalcContainer.style.display === "none";
            bmiCalcContainer.style.display = isHidden ? "block" : "none";
            bmiCalcToggle.innerHTML = isHidden 
                ? "📐 Hide BMI Calculator" 
                : "📐 Calculate my BMI (I don't know my BMI)";
        });
    }

    function computeBmi() {
        const height = parseFloat(heightInput.value);
        const weight = parseFloat(weightInput.value);
        
        if (height > 0 && weight > 0) {
            const heightInMeters = height / 100;
            const bmi = weight / (heightInMeters * heightInMeters);
            return parseFloat(bmi.toFixed(1));
        }
        return null;
    }

    function updateBmiResult() {
        const bmi = computeBmi();
        if (bmi) {
            let category = "Normal";
            if (bmi < 18.5) category = "Underweight";
            else if (bmi >= 25 && bmi < 30) category = "Overweight";
            else if (bmi >= 30) category = "Obese";
            
            bmiCalcResult.textContent = `Result: ${bmi} (${category})`;
        } else {
            bmiCalcResult.textContent = "";
        }
    }

    if (heightInput && weightInput) {
        heightInput.addEventListener("input", updateBmiResult);
        weightInput.addEventListener("input", updateBmiResult);
    }

    if (calculateBmiBtn && bmiRangeInput && bmiValDisplay) {
        calculateBmiBtn.addEventListener("click", () => {
            const bmi = computeBmi();
            if (bmi) {
                if (bmi >= 15.0 && bmi <= 50.0) {
                    bmiRangeInput.value = bmi;
                    bmiValDisplay.textContent = bmi;
                    alert(`BMI of ${bmi} applied to the form!`);
                } else {
                    alert(`Calculated BMI is ${bmi}, which is outside the range (15.0 - 50.0). We will clip it.`);
                    const clippedBmi = Math.max(15.0, Math.min(50.0, bmi));
                    bmiRangeInput.value = clippedBmi;
                    bmiValDisplay.textContent = clippedBmi;
                }
            } else {
                alert("Please enter both height (cm) and weight (kg) values.");
            }
        });
    }

    function updateFormProgress() {
        // Show/hide correct step div
        document.querySelectorAll(".form-step").forEach(step => {
            step.classList.remove("active");
        });
        document.querySelector(`.form-step[data-step="${currentStep}"]`).classList.add("active");

        // Update progress bar width
        const progressPercent = (currentStep / totalSteps) * 100;
        progressBarFill.style.width = `${progressPercent}%`;

        // Update step labels active state
        document.querySelectorAll(".step-label").forEach(label => {
            const stepNum = parseInt(label.getAttribute("data-step"));
            if (stepNum <= currentStep) {
                label.classList.add("active");
            } else {
                label.classList.remove("active");
            }
        });

        // Toggle buttons visibility
        if (currentStep === 1) {
            prevBtn.style.display = "none";
        } else {
            prevBtn.style.display = "inline-flex";
        }

        if (currentStep === totalSteps) {
            nextBtn.style.display = "none";
            submitBtn.style.display = "inline-flex";
        } else {
            nextBtn.style.display = "inline-flex";
            submitBtn.style.display = "none";
        }
    }

    function validateCurrentStep() {
        const stepContainer = document.querySelector(`.form-step[data-step="${currentStep}"]`);
        
        // Age validation in step 1
        if (currentStep === 1) {
            const ageInput = document.getElementById("age");
            if (ageInput) {
                const val = parseInt(ageInput.value);
                if (isNaN(val) || val < 18 || val > 90) {
                    alert("Please enter a valid age between 18 and 90.");
                    ageInput.focus();
                    return false;
                }
            }
        }
        
        return true;
    }

    nextBtn.addEventListener("click", () => {
        if (validateCurrentStep()) {
            if (currentStep < totalSteps) {
                currentStep++;
                updateFormProgress();
            }
        }
    });

    prevBtn.addEventListener("click", () => {
        if (currentStep > 1) {
            currentStep--;
            updateFormProgress();
        }
    });

    // Handle form submit overlay
    const form = document.getElementById("predict-form");
    if (form) {
        form.addEventListener("submit", (e) => {
            if (!validateCurrentStep()) {
                e.preventDefault();
            }
        });
    }

    // Init
    updateFormProgress();
});
