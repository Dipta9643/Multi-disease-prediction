/**
 * JavaScript for rendering Gauges and charts on the Results dashboard.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Generate gauge charts using Chart.js donut configuration
    Object.keys(predData).forEach(diseaseId => {
        const prob = predData[diseaseId];
        const color = predColors[diseaseId];
        const canvas = document.getElementById(`gauge-${diseaseId}`);
        
        if (canvas) {
            const ctx = canvas.getContext('2d');
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: [prob * 100, 100 - (prob * 100)],
                        backgroundColor: [color, 'rgba(255, 255, 255, 0.05)'],
                        borderWidth: 0,
                        borderRadius: 4
                    }]
                },
                options: {
                    rotation: 270, // 270 degrees is top, but for gauge starting from left we use 270
                    circumference: 180, // half circle
                    cutout: '83%',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        tooltip: { enabled: false },
                        legend: { display: false }
                    }
                }
            });
        }
    });
});
