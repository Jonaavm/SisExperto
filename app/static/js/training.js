// Training page JavaScript - Sistema Experto

document.addEventListener('DOMContentLoaded', function() {
    console.log('Training page loaded');
    
    // Elements
    const csvFile = document.getElementById('csvFile');
    const uploadArea = document.getElementById('uploadArea');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadInfo = document.getElementById('uploadInfo');
    const configCard = document.getElementById('configCard');
    const progressCard = document.getElementById('progressCard');
    const targetCol = document.getElementById('targetCol');
    const algorithm = document.getElementById('algorithm');
    const validation = document.getElementById('validation');
    const trainBtn = document.getElementById('trainBtn');
    const trainInfo = document.getElementById('trainInfo');
    
    let currentDataset = null;
    
    // File upload handling
    function setupFileUpload() {
        // Click to select file
        uploadArea.addEventListener('click', () => {
            csvFile.click();
        });
        
        // Drag and drop
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
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                csvFile.files = files;
                handleFileSelect();
            }
        });
        
        csvFile.addEventListener('change', handleFileSelect);
        uploadBtn.addEventListener('click', uploadFile);
    }
    
    function handleFileSelect() {
        const file = csvFile.files[0];
        if (file) {
            console.log('File selected:', file.name, 'Size:', file.size);
            
            // Validate file
            const errors = window.fileHelper.validateFile(file, ['.csv'], 10 * 1024 * 1024);
            if (errors.length > 0) {
                showMessage(errors.join(', '), 'error');
                return;
            }
            
            uploadBtn.disabled = false;
            uploadArea.innerHTML = `
                <div class="upload-content">
                    <i class="upload-icon">📄</i>
                    <p><strong>${file.name}</strong></p>
                    <p>Tamaño: ${window.fileHelper.formatFileSize(file.size)}</p>
                    <p class="text-success">✓ Archivo válido - Listo para subir</p>
                </div>
            `;
        } else {
            uploadBtn.disabled = true;
            resetUploadArea();
        }
    }
    
    function resetUploadArea() {
        uploadArea.innerHTML = `
            <div class="upload-content">
                <i class="upload-icon">📤</i>
                <p>Arrastra tu archivo CSV aquí o haz clic para seleccionar</p>
            </div>
        `;
    }
    
    async function uploadFile() {
        const file = csvFile.files[0];
        if (!file) {
            showMessage('Por favor selecciona un archivo CSV', 'error');
            return;
        }
        
        console.log('Starting file upload...');
        window.showLoading(uploadBtn);
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            console.log('Sending file to server...');
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            console.log('Response status:', response.status);
            const data = await response.json();
            console.log('Response data:', data);
            
            if (!response.ok) {
                throw new Error(data.error || 'Error al subir archivo');
            }
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Success
            currentDataset = data;
            displayUploadSuccess(data);
            setupModelConfiguration(data);
            
        } catch (error) {
            console.error('Upload error:', error);
            showMessage('Error al subir archivo: ' + error.message, 'error');
            uploadInfo.innerHTML = `
                <div class="alert alert-error">
                    <strong>Error:</strong> ${error.message}
                </div>
            `;
        } finally {
            window.hideLoading(uploadBtn);
        }
    }
    
    function displayUploadSuccess(data) {
        uploadInfo.innerHTML = `
            <div class="alert alert-success">
                <h4>✅ Dataset cargado exitosamente</h4>
                <p><strong>Filas:</strong> ${data.shape[0].toLocaleString()}</p>
                <p><strong>Columnas:</strong> ${data.shape[1]}</p>
                <p><strong>Características:</strong> ${data.columns.join(', ')}</p>
            </div>
        `;
        
        showMessage('Dataset cargado correctamente', 'success');
    }
    
    function setupModelConfiguration(data) {
        // Populate target column select
        targetCol.innerHTML = '<option value="">Selecciona la variable objetivo...</option>';
        data.columns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            targetCol.appendChild(option);
        });
        
        // Enable configuration card
        configCard.style.display = 'block';
        trainBtn.disabled = true;
        
        // Enable train button when target is selected
        targetCol.addEventListener('change', function() {
            trainBtn.disabled = !this.value;
        });
    }
    
    function setupAlgorithmOptions() {
        algorithm.addEventListener('change', function() {
            const knnParams = document.getElementById('knnParams');
            const id3Params = document.getElementById('id3Params');
            
            if (this.value === 'knn') {
                knnParams.style.display = 'block';
                id3Params.style.display = 'none';
            } else {
                knnParams.style.display = 'none';
                id3Params.style.display = 'block';
            }
        });
    }
    
    function setupValidationOptions() {
        validation.addEventListener('change', function() {
            const kfoldOptions = document.getElementById('kfoldOptions');
            const holdoutOptions = document.getElementById('holdoutOptions');
            
            if (this.value === 'holdout') {
                kfoldOptions.style.display = 'none';
                holdoutOptions.style.display = 'block';
            } else {
                kfoldOptions.style.display = 'block';
                holdoutOptions.style.display = 'none';
            }
        });
    }
    
    async function trainModel() {
        if (!currentDataset) {
            showMessage('Por favor sube un dataset primero', 'error');
            return;
        }
        
        if (!targetCol.value) {
            showMessage('Por favor selecciona la columna objetivo', 'error');
            return;
        }
        
        const payload = {
            algorithm: algorithm.value,
            target: targetCol.value,
            validation: validation.value,
            kFolds: parseInt(document.getElementById('kFolds').value) || 5,
            testSize: parseFloat(document.getElementById('testSize').value) || 0.2,
            knnK: parseInt(document.getElementById('knnK').value) || 3,
            maxDepth: parseInt(document.getElementById('maxDepth').value) || 10
        };
        
        console.log('Training with payload:', payload);
        
        try {
            // Show progress
            progressCard.style.display = 'block';
            updateProgress(0, 'Iniciando entrenamiento...');
            
            window.showLoading(trainBtn);
            
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Error en el entrenamiento');
            }
            
            updateProgress(100, 'Entrenamiento completado');
            displayTrainingResults(data);
            
        } catch (error) {
            console.error('Training error:', error);
            showMessage('Error en el entrenamiento: ' + error.message, 'error');
            updateProgress(0, 'Error en el entrenamiento');
        } finally {
            window.hideLoading(trainBtn);
        }
    }
    
    function updateProgress(percentage, message) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }
        
        if (progressText) {
            progressText.textContent = message;
        }
        
        // Add to training log
        const logEntry = `${new Date().toLocaleTimeString()}: ${message}`;
        trainInfo.innerHTML += `<div>${logEntry}</div>`;
        trainInfo.scrollTop = trainInfo.scrollHeight;
    }
    
    function displayTrainingResults(results) {
        trainInfo.innerHTML += `
            <div class="training-complete">
                <h4>🎉 Entrenamiento Completado</h4>
                <p><strong>Algoritmo:</strong> ${results.algorithm}</p>
                <p><strong>Precisión:</strong> ${(results.accuracy * 100).toFixed(2)}%</p>
                <p><strong>Tiempo:</strong> ${results.training_time || 'N/A'}</p>
                <p><strong>Modelo ID:</strong> ${results.id || results.model_id}</p>
                <div class="result-actions">
                    <button onclick="viewResults('${results.id || results.model_id}')" class="btn btn-primary">
                        Ver Resultados Completos
                    </button>
                    <a href="/classification" class="btn btn-secondary">Clasificar Ejemplos</a>
                </div>
            </div>
        `;
        
        // Save the model ID for later use
        window.localStorage.setItem('lastTrainedModel', results.id || results.model_id);
        
        showMessage('Modelo entrenado exitosamente. Puedes ver los resultados completos haciendo clic en el botón.', 'success');
    }
    
    // Function to redirect to results page with model selected
    window.viewResults = function(modelId) {
        // Save the model ID and redirect
        window.localStorage.setItem('selectedModelId', modelId);
        window.location.href = '/results';
    };
    
    function showMessage(message, type = 'info') {
        if (window.showMessage) {
            window.showMessage(message, type);
        } else {
            alert(message);
        }
    }
    
    // Initialize everything
    function init() {
        console.log('Initializing training page...');
        setupFileUpload();
        setupAlgorithmOptions();
        setupValidationOptions();
        
        // Train button event
        trainBtn.addEventListener('click', trainModel);
        
        console.log('Training page initialized');
    }
    
    init();
});
