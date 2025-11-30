// Nail Disease Screening Application - Frontend JavaScript

const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const detectionVis = document.getElementById('detectionVis');
const detectionImage = document.getElementById('detectionImage');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const clearResultsBtn = document.getElementById('clearResultsBtn');

let selectedFile = null;

// Upload area click handler
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// File input change handler
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
    } else {
        showError('Please drop a valid image file.');
    }
});

// Handle file selection
function handleFileSelect(file) {
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size exceeds 16MB limit. Please choose a smaller image.');
        return;
    }

    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        analyzeBtn.disabled = false;
        hideError();
        hideResults();
    };
    reader.readAsDataURL(file);
}

// Analyze button handler
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }

    // Show loading, hide results
    showLoading();
    hideResults();
    hideError();

    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'An error occurred during analysis');
        }

        // Display results
        displayResults(data);
    } catch (error) {
        showError(error.message || 'Failed to analyze image. Please try again.');
    } finally {
        hideLoading();
    }
});

// Display results
function displayResults(data) {
    resultsSection.style.display = 'block';

    // Show detection visualization if available
    if (data.detection_visualization) {
        detectionImage.src = 'data:image/jpeg;base64,' + data.detection_visualization;
        detectionVis.style.display = 'block';
    } else {
        detectionVis.style.display = 'none';
    }

    // Clear previous results
    resultsContainer.innerHTML = '';

    // Check if any credible diseases were detected
    if (data.num_credible_detections === 0) {
        // No credible detections found, but still show result with Grad-CAM if available
        if (data.results && data.results.length > 0) {
            const result = data.results[0];
            // If Grad-CAM is available, show the result card with visualization
            if (result.gradcam_heatmap) {
                const resultCard = createResultCard(result, 0);
                if (resultCard) {
                    resultsContainer.appendChild(resultCard);
                }
            } else {
                // No Grad-CAM, show simple no disease message
                const noDiseaseCard = document.createElement('div');
                noDiseaseCard.className = 'result-card';
                noDiseaseCard.style.textAlign = 'center';
                noDiseaseCard.style.padding = '40px';
                noDiseaseCard.innerHTML = `
                    <div style="font-size: 3em; margin-bottom: 20px;">✓</div>
                    <h3 style="color: #28a745; margin-bottom: 15px;">No Disease Identified</h3>
                    <p style="color: #666; font-size: 1.1em; line-height: 1.6;">
                        The analysis did not detect any nail conditions with sufficient confidence (>50%).<br>
                        This is a positive result indicating no significant abnormalities were found.
                    </p>
                    <p style="color: #999; font-size: 0.9em; margin-top: 20px;">
                        <strong>Note:</strong> ${data.num_nails_detected} nail(s) were analyzed. The result shown corresponds to the nail with the highest disease probability, but it did not exceed the 50% confidence threshold.
                    </p>
                `;
                resultsContainer.appendChild(noDiseaseCard);
            }
        }
    } else {
        // Display the best result (nail with highest disease probability)
        if (data.results && data.results.length > 0) {
            const result = data.results[0]; // Only one result now (the best one)
            
            // Debug: log the result to see if gradcam_heatmap is present
            console.log('Result data:', result);
            console.log('Grad-CAM heatmap present:', !!result.gradcam_heatmap);
            
            // Show result if it has a credible detection
            if (result.disease !== null && !result.no_disease_detected) {
                const resultCard = createResultCard(result, 0);
                if (resultCard) {
                    // Add a note about multiple nails if applicable
                    if (data.num_nails_detected > 1) {
                        const infoNote = document.createElement('div');
                        infoNote.style.cssText = 'margin-bottom: 20px; padding: 15px; background-color: #f8f9fa; border-left: 4px solid #007bff; border-radius: 4px;';
                        infoNote.innerHTML = `
                            <p style="margin: 0; color: #666; font-size: 0.95em;">
                                <strong>ℹ️ Analysis Summary:</strong> ${data.num_nails_detected} nail(s) were detected. 
                                The result below shows the nail with the highest disease probability.
                            </p>
                        `;
                        resultsContainer.appendChild(infoNote);
                    }
                    resultsContainer.appendChild(resultCard);
                }
            } else {
                // Even if no credible detection, show result with Grad-CAM if available
                // This helps users see what the model is looking at
                if (result.gradcam_heatmap) {
                    console.log('Showing result card even without credible detection (has Grad-CAM)');
                    const resultCard = createResultCard(result, 0);
                    if (resultCard) {
                        resultsContainer.appendChild(resultCard);
                    }
                }
            }
        }
    }
}

// Create result card
function createResultCard(result, index) {
    const card = document.createElement('div');
    card.className = 'result-card';

    // Allow cards even without credible detections if Grad-CAM is available
    // This helps users understand what the model is looking at
    if ((result.disease === null || result.no_disease_detected) && !result.gradcam_heatmap) {
        return null;
    }

    const diseaseName = result.disease ? formatDiseaseName(result.disease) : 'No Disease Detected';
    const probability = (result.probability * 100).toFixed(1);

    card.innerHTML = `
        <h3>
            <span class="nail-badge">Best Result (Nail ${result.nail_index})</span>
            ${diseaseName}
        </h3>
        
        <div class="prediction">
            <div class="prediction-label">
                <span>Predicted Condition:</span>
                <span class="prediction-probability">${probability}%</span>
            </div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${probability}%"></div>
            </div>
        </div>

        ${result.gradcam_heatmap ? `
            <div class="gradcam-visualization" style="margin: 20px 0;">
                <h4 style="margin-bottom: 10px; color: #333;">Grad-CAM Visualization</h4>
                <p style="font-size: 0.9em; color: #666; margin-bottom: 10px;">
                    This heatmap shows which regions of the nail the model focused on when making its prediction. 
                    Red/yellow areas indicate higher attention.
                </p>
                <div style="border: 2px solid #ddd; border-radius: 8px; overflow: hidden; background: #f8f9fa; text-align: center; padding: 10px;">
                    <img src="data:image/jpeg;base64,${result.gradcam_heatmap}" 
                         alt="Grad-CAM Visualization" 
                         class="gradcam-image"
                         style="max-width: 150px; max-height: 150px; width: auto; height: auto; display: block; margin: 0 auto;">
                </div>
            </div>
        ` : ''}

        <div class="all-probabilities">
            <h4>All Condition Probabilities:</h4>
            ${Object.entries(result.all_probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([disease, prob]) => `
                    <div class="probability-item">
                        <span class="probability-item-label">${formatDiseaseName(disease)}</span>
                        <span class="probability-item-value">${(prob * 100).toFixed(1)}%</span>
                    </div>
                `).join('')}
        </div>

        <div style="margin-top: 15px; font-size: 0.85em; color: #999;">
            Detection Confidence: ${(result.detection_confidence * 100).toFixed(1)}%
        </div>
    `;

    return card;
}

// Format disease name for display
function formatDiseaseName(disease) {
    if (!disease) return 'Unknown';
    return disease
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

// Show/hide functions
function showLoading() {
    loadingSection.style.display = 'block';
    analyzeBtn.disabled = true;
}

function hideLoading() {
    loadingSection.style.display = 'none';
    analyzeBtn.disabled = false;
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
}

function hideError() {
    errorSection.style.display = 'none';
}

// Clear results button handler
clearResultsBtn.addEventListener('click', () => {
    clearResults();
});

// Clear all results and reset UI
function clearResults() {
    // Hide results section
    hideResults();
    
    // Clear results container
    resultsContainer.innerHTML = '';
    
    // Clear detection visualization
    detectionImage.src = '';
    detectionVis.style.display = 'none';
    
    // Clear preview
    previewImage.src = '';
    previewSection.style.display = 'none';
    
    // Clear file input
    imageInput.value = '';
    selectedFile = null;
    
    // Reset analyze button
    analyzeBtn.disabled = true;
    
    // Hide any errors
    hideError();
}

