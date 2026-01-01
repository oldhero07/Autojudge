// JavaScript for Flask ML Web App client interaction

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('problem-form');
    const resultsDiv = document.getElementById('results');
    const predictionContent = document.getElementById('prediction-content');
    const errorMessage = document.getElementById('error-message');
    const submitBtn = form.querySelector('.submit-btn');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Clear previous results and errors
        hideResults();
        hideError();
        
        // Get form data
        const formData = new FormData(form);
        const data = {
            description: formData.get('description').trim(),
            input_desc: formData.get('input_desc').trim(),
            output_desc: formData.get('output_desc').trim()
        };
        
        // Validate form data
        if (!data.description || !data.input_desc || !data.output_desc) {
            showError('Please fill in all fields.');
            return;
        }
        
        // Disable submit button and show loading state
        submitBtn.disabled = true;
        submitBtn.textContent = 'Processing...';
        
        try {
            // Make AJAX request to prediction endpoint
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                showResults(result);
            } else {
                showError(result.message || result.error || 'An error occurred while processing your request.');
            }
            
        } catch (error) {
            console.error('Error:', error);
            showError('Network error. Please check your connection and try again.');
        } finally {
            // Re-enable submit button
            submitBtn.disabled = false;
            submitBtn.textContent = 'Classify Problem';
        }
    });
    
    function showResults(result) {
        // Create difficulty class badge with color coding
        const classColor = getClassColor(result.class);
        
        predictionContent.innerHTML = `
            <div class="prediction-item">
                <span class="prediction-label">Problem Class:</span>
                <span class="prediction-value class-badge class-${result.class.toLowerCase()}" style="background-color: ${classColor}">
                    ${result.class}
                </span>
            </div>
            <div class="prediction-item">
                <span class="prediction-label">Difficulty Score:</span>
                <span class="prediction-value score-value">
                    ${result.score}/100
                </span>
            </div>
            <div class="prediction-explanation">
                <small>
                    <strong>Explanation:</strong> This problem has been classified as <strong>${result.class}</strong> 
                    difficulty with a score of <strong>${result.score}</strong> out of 100 based on the problem description, 
                    input format, and output format.
                </small>
            </div>
        `;
        resultsDiv.style.display = 'block';
        
        // Scroll to results
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    function getClassColor(problemClass) {
        switch(problemClass.toLowerCase()) {
            case 'easy': return '#27ae60';
            case 'medium': return '#f39c12';
            case 'hard': return '#e74c3c';
            default: return '#3498db';
        }
    }
    
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        
        // Scroll to error message
        errorMessage.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    function hideResults() {
        resultsDiv.style.display = 'none';
    }
    
    function hideError() {
        errorMessage.style.display = 'none';
    }
});