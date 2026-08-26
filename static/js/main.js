/**
 * Main JS Helper Utilities for Multi-Risk AI Predictor.
 */

// Show the global loading spinner
function showLoading(text = "Processing health data...") {
    const overlay = document.getElementById("loading-overlay");
    const label = document.getElementById("loading-text");
    if (overlay) {
        if (label) label.textContent = text;
        overlay.classList.add("visible");
    }
}

// Hide the global loading spinner
function hideLoading() {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) {
        overlay.classList.remove("visible");
    }
}

// Format range slider label outputs
function initRangeSliderLabel(sliderId, labelId, suffix = "") {
    const slider = document.getElementById(sliderId);
    const label = document.getElementById(labelId);
    
    if (slider && label) {
        const updateLabel = () => {
            label.textContent = slider.value + suffix;
        };
        slider.addEventListener("input", updateLabel);
        updateLabel(); // Run initially
    }
}

// Trigger standard alert notifications if any
document.addEventListener("DOMContentLoaded", () => {
    // Hide loader if page is fully loaded
    hideLoading();
});
