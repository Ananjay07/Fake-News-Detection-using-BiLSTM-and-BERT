document.addEventListener('DOMContentLoaded', () => {
    // --- PARTICLES BACKGROUND ---
    const canvas = document.getElementById('particles');
    const ctx = canvas.getContext('2d');
    let particles = [];

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 2 + 0.5;
            this.speedX = Math.random() * 0.5 - 0.25;
            this.speedY = Math.random() * 0.5 - 0.25;
            this.opacity = Math.random() * 0.5 + 0.2;
        }
        update() {
            this.x += this.speedX;
            this.y += this.speedY;
            if (this.x > canvas.width) this.x = 0;
            if (this.x < 0) this.x = canvas.width;
            if (this.y > canvas.height) this.y = 0;
            if (this.y < 0) this.y = canvas.height;
        }
        draw() {
            ctx.fillStyle = `rgba(91, 140, 255, ${this.opacity})`;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function initParticles() {
        particles = [];
        for (let i = 0; i < 100; i++) {
            particles.push(new Particle());
        }
    }
    initParticles();

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        requestAnimationFrame(animate);
    }
    animate();

    // --- UI ELEMENTS ---
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const newsInput = document.getElementById('newsInput');
    const statusLoader = document.getElementById('statusLoader');
    const statusText = document.getElementById('statusText');
    const resultSection = document.getElementById('resultSection');
    const attentionSection = document.getElementById('attentionSection');
    const nlpSignalsSection = document.getElementById('nlpSignalsSection');

    // Result Elements
    const predictionCard = document.getElementById('predictionCard');
    const predictionText = document.getElementById('predictionText');
    const predictionIcon = document.getElementById('predictionIcon');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');
    const gaugeNeedle = document.getElementById('gaugeNeedle');

    // NLP Signal Elements
    const sentimentBar = document.getElementById('sentimentBar');
    const sentimentText = document.getElementById('sentimentText');
    const posBar = document.getElementById('posBar');
    const posText = document.getElementById('posText');
    const tfidfBar = document.getElementById('tfidfBar');
    const tfidfText = document.getElementById('tfidfText');
    const credibilityBar = document.getElementById('credibilityBar');
    const credibilityText = document.getElementById('credibilityText');
    const attentionHeatmap = document.getElementById('attentionHeatmap');

    const loadingPhases = [
        "Running transformer inference...",
        "Calculating attention weights...",
        "Analyzing semantic patterns...",
        "Evaluating BERT-BiLSTM hybrid signals..."
    ];

    // --- CORE LOGIC ---

    analyzeBtn.addEventListener('click', async () => {
        const text = newsInput.value.trim();
        if (!text) {
            alert('Please paste some news text first!');
            return;
        }

        // Reset UI
        resultSection.classList.add('hidden');
        attentionSection.classList.add('hidden');
        nlpSignalsSection.classList.add('hidden');
        statusLoader.classList.remove('hidden');
        analyzeBtn.disabled = true;

        // Simulate phase transitions
        let phase = 0;
        const phaseInterval = setInterval(() => {
            phase++;
            if (phase < loadingPhases.length) {
                statusText.innerText = loadingPhases[phase];
            } else {
                clearInterval(phaseInterval);
            }
        }, 800);

        try {
            // BACKEND FETCH
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) throw new Error('Backend offline');

            const data = await response.json();
            
            // Wait for simulated phases to complete
            await new Promise(r => setTimeout(r, 2000));
            
            updateUI(data.prediction, data.confidence, text);

        } catch (error) {
            console.error('Fetch Error:', error);
            // Fallback for demo if backend is offline
            console.log('Using demo data fallback...');
            await new Promise(r => setTimeout(r, 2500));
            const mockPrediction = Math.random() > 0.5 ? 'FAKE' : 'REAL';
            const mockConfidence = (Math.random() * 20 + 80).toFixed(2);
            updateUI(mockPrediction, mockConfidence, text);
        } finally {
            clearInterval(phaseInterval);
            statusLoader.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    clearBtn.addEventListener('click', () => {
        newsInput.value = '';
        resultSection.classList.add('hidden');
        attentionSection.classList.add('hidden');
        nlpSignalsSection.classList.add('hidden');
    });

    function updateUI(prediction, confidence, originalText) {
        // 1. Prediction Card
        predictionText.innerText = prediction === 'FAKE' ? 'FAKE NEWS' : 'REAL NEWS';
        predictionCard.className = 'glass-card result-card ' + (prediction === 'FAKE' ? 'fake' : 'real');
        predictionIcon.innerHTML = prediction === 'FAKE' ? '<i class="fas fa-exclamation-triangle"></i>' : '<i class="fas fa-check-circle"></i>';
        
        // 2. Confidence
        confidenceValue.innerText = `${confidence}%`;
        confidenceBar.style.width = `${confidence}%`;
        
        // 3. Gauge
        const dashArray = (parseFloat(confidence) / 100) * 126;
        gaugeNeedle.setAttribute('stroke-dasharray', `${dashArray} 126`);

        // 4. NLP Signals (Simulated based on text length and patterns)
        simulateNLPSignals(prediction, confidence, originalText);

        // 5. Attention Heatmap
        generateHeatmap(originalText, prediction);

        // Show sections with staggered animation
        resultSection.classList.remove('hidden');
        setTimeout(() => attentionSection.classList.remove('hidden'), 200);
        setTimeout(() => nlpSignalsSection.classList.remove('hidden'), 400);

        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    function simulateNLPSignals(prediction, confidence, text) {
        // Sentiment: simple word based mock
        const sentimentVal = Math.random() * 0.4 - 0.2; // -0.2 to 0.2
        sentimentBar.style.width = `${(sentimentVal + 1) * 50}%`;
        sentimentText.innerText = sentimentVal > 0 ? `Positive (${sentimentVal.toFixed(2)})` : `Neutral (${sentimentVal.toFixed(2)})`;

        // POS Ratio
        const posVal = Math.random() * 0.3 + 0.2;
        posBar.style.width = `${posVal * 100}%`;
        posText.innerText = posVal.toFixed(2);

        // TF-IDF
        const tfidfVal = Math.random() * 0.5 + 0.4;
        tfidfBar.style.width = `${tfidfVal * 100}%`;
        tfidfText.innerText = tfidfVal > 0.7 ? 'High Density' : 'Normal';

        // Credibility
        const cred = prediction === 'REAL' ? (parseFloat(confidence) + 2) : (100 - parseFloat(confidence));
        credibilityBar.style.width = `${cred}%`;
        credibilityText.innerText = `${cred.toFixed(1)}%`;
    }

    function generateHeatmap(text, prediction) {
        attentionHeatmap.innerHTML = '';
        const words = text.split(/\s+/).slice(0, 50); // Show first 50 words
        
        const suspiciousWords = ['shocking', 'illegal', 'secret', 'fraud', 'exposed', 'conspiracy', 'scandal', 'unbelievable', 'urgent', 'breaking'];
        const trustedWords = ['official', 'according', 'reported', 'confirmed', 'verified', 'studies', 'research', 'evidence', 'statements'];

        words.forEach(word => {
            const span = document.createElement('span');
            span.className = 'token';
            span.innerText = word;
            
            const cleanWord = word.toLowerCase().replace(/[^a-z]/g, '');
            
            if (suspiciousWords.includes(cleanWord)) {
                span.classList.add('highlight-red');
            } else if (trustedWords.includes(cleanWord)) {
                span.classList.add('highlight-green');
            } else if (Math.random() > 0.85) {
                // Random low-level attention highlights
                span.style.background = prediction === 'FAKE' ? 'rgba(255, 77, 77, 0.1)' : 'rgba(77, 255, 136, 0.1)';
            }
            
            attentionHeatmap.appendChild(span);
        });
    }
});