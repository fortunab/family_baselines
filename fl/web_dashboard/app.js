document.addEventListener('DOMContentLoaded', () => {
    initFig3Chart();
    initFig4Chart();
    initFig5Chart();
});

function initFig3Chart() {
    const ctx = document.getElementById('fig3Chart').getContext('2d');
    const models = ['ResNet', 'AlexNet', 'ZFNet', 'Bionnica', 'BFNet'];

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: models,
            datasets: [
                {
                    label: 'SubstraFL (on-premises)',
                    data: [88.2, 85.4, 84.5, 93.7, 92.5],
                    borderColor: '#f59e0b',
                    borderDash: [5, 5],
                    backgroundColor: '#f59e0b',
                    tension: 0.2
                },
                {
                    label: 'SubstraFL (hosted)',
                    data: [86.5, 84.1, 83.2, 92.0, 90.2],
                    borderColor: '#f59e0b',
                    backgroundColor: '#f59e0b',
                    tension: 0.2
                },
                {
                    label: 'Fed-BioMed (on-premises)',
                    data: [87.5, 84.8, 84.0, 92.4, 91.4],
                    borderColor: '#10b981',
                    borderDash: [5, 5],
                    backgroundColor: '#10b981',
                    tension: 0.2
                },
                {
                    label: 'Fed-BioMed (hosted)',
                    data: [86.8, 84.2, 83.5, 91.1, 90.4],
                    borderColor: '#10b981',
                    backgroundColor: '#10b981',
                    tension: 0.2
                },
                {
                    label: 'Classical env (hosted)',
                    data: [80.2, 79.1, 78.8, 87.2, 84.9],
                    borderColor: '#ec4899',
                    backgroundColor: '#ec4899',
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } }
            },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { min: 75, max: 95, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}

function initFig4Chart() {
    const ctx = document.getElementById('fig4Chart').getContext('2d');
    const models = ['ResNet-50', 'AlexNet', 'ZFNet', 'Bionnica', 'BFNet'];

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: [87.1, 85.4, 84.2, 92.0, 90.2],
                    backgroundColor: 'rgba(6, 182, 212, 0.7)',
                    borderColor: '#06b6d4',
                    borderWidth: 1
                },
                {
                    label: 'Sensitivity (%)',
                    data: [85.6, 83.0, 82.1, 90.5, 88.0],
                    backgroundColor: 'rgba(236, 72, 153, 0.7)',
                    borderColor: '#ec4899',
                    borderWidth: 1
                },
                {
                    label: 'Specificity (%)',
                    data: [86.3, 84.1, 83.2, 91.3, 89.0],
                    backgroundColor: 'rgba(245, 158, 11, 0.7)',
                    borderColor: '#f59e0b',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } }
            },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { min: 75, max: 95, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}

function initFig5Chart() {
    const ctx = document.getElementById('fig5Chart').getContext('2d');
    const models = ['ResNet', 'AlexNet', 'ZFNet', 'Bionnica', 'BFNet'];

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: models,
            datasets: [
                {
                    label: 'MetisFL (on-premises)',
                    data: [86.2, 83.1, 81.5, 90.5, 92.0],
                    borderColor: '#ec4899',
                    borderDash: [5, 5],
                    backgroundColor: '#ec4899',
                    tension: 0.2
                },
                {
                    label: 'MetisFL (hosted)',
                    data: [85.5, 82.4, 80.2, 89.6, 90.7],
                    borderColor: '#8b5cf6',
                    backgroundColor: '#8b5cf6',
                    tension: 0.2
                },
                {
                    label: 'Classical (on-premises)',
                    data: [80.2, 77.1, 76.5, 85.2, 89.1],
                    borderColor: '#64748b',
                    borderDash: [5, 5],
                    backgroundColor: '#64748b',
                    tension: 0.2
                },
                {
                    label: 'Classical (hosted)',
                    data: [78.9, 76.2, 75.1, 83.5, 88.5],
                    borderColor: '#64748b',
                    backgroundColor: '#64748b',
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } }
            },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { min: 70, max: 95, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}
