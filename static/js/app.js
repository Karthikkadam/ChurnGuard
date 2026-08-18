/**
 * app.js — ChurnGuard Frontend Logic
 *
 * Handles:
 *  - Form submission → /predict API
 *  - Metric loading → /metrics API
 *  - Animated gauge, bars, and rings
 *  - Navigation & section toggling
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    loadMetrics();
    initPredictionForm();
    initExplanationToggle();
    injectGaugeGradient();
});


/* ==========================================
   SVG Gradient Injection
   ========================================== */
function injectGaugeGradient() {
    // Add gradient definition for gauge circles
    const svg = document.querySelector('.result-gauge .gauge-svg') ||
                document.querySelector('.placeholder-ring .gauge-svg');
    if (svg) {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        defs.innerHTML = `
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#6366f1"/>
                <stop offset="50%" stop-color="#a855f7"/>
                <stop offset="100%" stop-color="#ec4899"/>
            </linearGradient>
        `;
        svg.insertBefore(defs, svg.firstChild);
    }

    // Also add to result gauge
    const resultSvg = document.querySelector('#result-content .gauge-svg');
    if (resultSvg && resultSvg !== svg) {
        const defs2 = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        defs2.innerHTML = `
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#6366f1"/>
                <stop offset="50%" stop-color="#a855f7"/>
                <stop offset="100%" stop-color="#ec4899"/>
            </linearGradient>
        `;
        resultSvg.insertBefore(defs2, resultSvg.firstChild);
    }
}


/* ==========================================
   Navigation
   ========================================== */
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.dataset.section;
            switchSection(section);
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

function switchSection(sectionName) {
    const sections = {
        prediction: document.getElementById('prediction'),
        performance: document.getElementById('performance'),
        info: document.getElementById('info'),
    };

    Object.values(sections).forEach(s => {
        if (s) s.classList.remove('hidden');
    });

    if (sectionName === 'prediction') {
        document.getElementById('info')?.classList.add('hidden');
    } else if (sectionName === 'performance') {
        document.getElementById('info')?.classList.add('hidden');
    } else if (sectionName === 'info') {
        // Show info, hide others optionally
        document.getElementById('info')?.classList.remove('hidden');
    }

    // Smooth scroll
    const target = document.getElementById(sectionName);
    if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}


/* ==========================================
   Prediction Form
   ========================================== */
function initPredictionForm() {
    const form = document.getElementById('predict-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await makePrediction();
    });
}

async function makePrediction() {
    const btn = document.getElementById('predict-btn');
    const placeholder = document.getElementById('result-placeholder');
    const resultContent = document.getElementById('result-content');
    const loading = document.getElementById('result-loading');

    // Show loading
    placeholder.classList.add('hidden');
    resultContent.classList.add('hidden');
    loading.classList.remove('hidden');
    btn.disabled = true;
    btn.style.opacity = '0.7';

    // Collect form data
    const data = {
        CreditScore: parseInt(document.getElementById('credit-score').value),
        Age: parseInt(document.getElementById('age').value),
        Geography: document.getElementById('geography').value,
        Gender: document.getElementById('gender').value,
        Tenure: parseInt(document.getElementById('tenure').value),
        Balance: parseFloat(document.getElementById('balance').value),
        NumOfProducts: parseInt(document.getElementById('products').value),
        HasCrCard: document.getElementById('credit-card').checked ? 1 : 0,
        IsActiveMember: document.getElementById('active-member').checked ? 1 : 0,
        EstimatedSalary: parseFloat(document.getElementById('salary').value),
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        const result = await response.json();

        if (result.success) {
            displayResult(result);
        } else {
            alert('Prediction failed: ' + (result.error || 'Unknown error'));
            placeholder.classList.remove('hidden');
        }
    } catch (err) {
        alert('Connection error: ' + err.message);
        placeholder.classList.remove('hidden');
    } finally {
        loading.classList.add('hidden');
        btn.disabled = false;
        btn.style.opacity = '1';
    }
}


/* ==========================================
   Display Prediction Result
   ========================================== */
function displayResult(result) {
    const resultContent = document.getElementById('result-content');
    resultContent.classList.remove('hidden');

    // Animate gauge
    const prob = result.probability;
    const circumference = 2 * Math.PI * 70; // r=70
    const offset = (prob / 100) * circumference;
    const gaugeCircle = document.getElementById('gauge-circle');
    const gaugeValue = document.getElementById('gauge-value');

    // Set stroke color based on risk
    const riskColors = {
        LOW: '#10b981',
        MEDIUM: '#f59e0b',
        HIGH: '#f97316',
        CRITICAL: '#ef4444',
    };

    // Animate gauge value counting up
    animateCounter(gaugeValue, 0, prob, 1200);

    setTimeout(() => {
        gaugeCircle.style.strokeDasharray = `${offset}, ${circumference}`;
        gaugeCircle.style.stroke = riskColors[result.risk] || '#6366f1';
    }, 100);

    // Risk badge
    const badge = document.getElementById('risk-badge');
    const riskText = document.getElementById('risk-text');
    badge.className = 'risk-badge ' + result.risk.toLowerCase();
    riskText.textContent = result.risk + ' RISK';

    // Explanation text
    const expDiv = document.getElementById('result-explanation');
    if (prob >= 50) {
        expDiv.innerHTML = '<p>Customer is <strong>likely to churn</strong>. Immediate retention measures recommended.</p>';
    } else if (prob >= 30) {
        expDiv.innerHTML = '<p>Customer shows <strong>moderate churn risk</strong>. Monitor engagement closely.</p>';
    } else {
        expDiv.innerHTML = '<p>Customer is <strong>unlikely to churn</strong>. Low risk profile detected.</p>';
    }

    // Explanation details
    const list = document.getElementById('explanation-list');
    list.innerHTML = '';
    result.explanations.forEach(exp => {
        const li = document.createElement('li');
        li.textContent = exp;
        list.appendChild(li);
    });

    // Hide explanation details by default
    document.getElementById('explanation-details').classList.add('hidden');
}


/* ==========================================
   Explanation Toggle
   ========================================== */
function initExplanationToggle() {
    const btn = document.getElementById('explanation-toggle');
    const details = document.getElementById('explanation-details');

    btn.addEventListener('click', () => {
        details.classList.toggle('hidden');
        btn.textContent = details.classList.contains('hidden')
            ? '🔍 View Explanation'
            : '✕ Hide Explanation';
    });
}


/* ==========================================
   Load Metrics
   ========================================== */
async function loadMetrics() {
    try {
        const response = await fetch('/metrics');
        const data = await response.json();
        renderMetricCards(data);
        renderModelComparison(data.classifier_scores);
        renderFeatureImportance(data.feature_importances);
    } catch (err) {
        console.error('Failed to load metrics:', err);
    }
}

function renderMetricCards(data) {
    const metrics = {
        accuracy: data.accuracy,
        precision: data.precision,
        recall: data.recall,
        f1: data.f1_score,
    };

    const circumference = 2 * Math.PI * 34; // r=34

    Object.entries(metrics).forEach(([key, value]) => {
        const valueEl = document.querySelector(`[data-metric="${key}"]`);
        if (valueEl) {
            animateCounter(valueEl, 0, value, 1500);
        }

        const ringClass = key === 'f1' ? 'f1-ring' : key + '-ring';
        const ring = document.querySelector('.' + ringClass);
        if (ring) {
            setTimeout(() => {
                const offset = (value / 100) * circumference;
                ring.style.strokeDasharray = `${offset}, ${circumference}`;
            }, 300);
        }
    });
}

function renderModelComparison(scores) {
    const container = document.getElementById('model-comparison-chart');
    if (!container) return;

    // Sort descending
    const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    const maxScore = Math.max(...sorted.map(s => s[1]));
    const bestName = sorted[0][0];

    const gradientClasses = [
        'gradient-1', 'gradient-2', 'gradient-3', 'gradient-4',
        'gradient-5', 'gradient-6', 'gradient-7', 'gradient-8', 'gradient-9'
    ];

    container.innerHTML = '';
    sorted.forEach(([name, score], i) => {
        const item = document.createElement('div');
        item.className = 'bar-item' + (name === bestName ? ' best' : '');

        const widthPct = (score / maxScore) * 100;
        const gradientClass = gradientClasses[i % gradientClasses.length];

        item.innerHTML = `
            <span class="bar-label" title="${name}">${name}</span>
            <div class="bar-track">
                <div class="bar-fill ${gradientClass}" style="width: 0%;">
                    <span>${score}%</span>
                </div>
            </div>
        `;
        container.appendChild(item);

        // Animate bar
        setTimeout(() => {
            item.querySelector('.bar-fill').style.width = widthPct + '%';
        }, 200 + i * 100);
    });
}

function renderFeatureImportance(importances) {
    const container = document.getElementById('feature-importance-chart');
    if (!container) return;

    const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]);
    const maxImp = sorted[0][1];

    const gradientClasses = [
        'gradient-1', 'gradient-6', 'gradient-3', 'gradient-4',
        'gradient-5', 'gradient-2', 'gradient-7', 'gradient-8', 'gradient-9', 'gradient-1'
    ];

    // Feature label mapping for friendlier names
    const labelMap = {
        'CreditScore': 'Credit Score',
        'Geography': 'Geography',
        'Gender': 'Gender',
        'Age': 'Age',
        'Tenure': 'Tenure',
        'Balance': 'Balance',
        'NumOfProducts': 'Products',
        'HasCrCard': 'Credit Card',
        'IsActiveMember': 'Active Member',
        'EstimatedSalary': 'Salary',
    };

    container.innerHTML = '';
    sorted.forEach(([name, value], i) => {
        const item = document.createElement('div');
        item.className = 'bar-item';

        const widthPct = (value / maxImp) * 100;
        const displayPct = (value * 100).toFixed(1);
        const gradientClass = gradientClasses[i % gradientClasses.length];
        const displayName = labelMap[name] || name;

        item.innerHTML = `
            <span class="bar-label" title="${displayName}">${displayName}</span>
            <div class="bar-track">
                <div class="bar-fill ${gradientClass}" style="width: 0%;">
                    <span>${displayPct}%</span>
                </div>
            </div>
        `;
        container.appendChild(item);

        // Animate bar
        setTimeout(() => {
            item.querySelector('.bar-fill').style.width = widthPct + '%';
        }, 400 + i * 100);
    });
}


/* ==========================================
   Counter Animation
   ========================================== */
function animateCounter(element, start, end, duration) {
    const startTime = performance.now();
    const isPercentage = true;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * eased;

        element.textContent = Math.round(current * 10) / 10 + '%';

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = end + '%';
        }
    }

    requestAnimationFrame(update);
}
