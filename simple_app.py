"""
Simple Sistema Experto - Versión de prueba
Aplicación Flask simple para probar las interfaces sin dependencias complejas
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import json
from datetime import datetime

# Configure Flask to look for templates in the app directory
app = Flask(__name__, 
            template_folder='app/templates',
            static_folder='app/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory and initialize models file
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models.json if it doesn't exist
models_file = os.path.join(app.config['UPLOAD_FOLDER'], 'models.json')
if not os.path.exists(models_file):
    with open(models_file, 'w') as f:
        json.dump([], f)

# Routes for pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/classification')
def classification():
    return render_template('classification.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/contacto')
def contacto():
    return render_template('contacto.html')

# API Routes
@app.route('/api/routes', methods=['GET'])
def list_routes():
    """List all available routes for debugging"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule)
        })
    return jsonify({'routes': routes})

@app.route('/api/test', methods=['GET'])
def api_test():
    """Test endpoint to verify API is working"""
    return jsonify({'status': 'OK', 'message': 'API is working!', 'timestamp': datetime.now().isoformat()})

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Upload CSV dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400
        
        # Save file
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')
        file.save(csv_path)
        
        # Read and analyze the dataset
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return jsonify({'error': f'Invalid CSV format: {str(e)}'}), 400
        
        # Save as pickle for faster loading
        pkl_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.pkl')
        df.to_pickle(pkl_path)
        
        # Return dataset info
        return jsonify({
            'message': 'File uploaded successfully',
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'sample_data': df.head(5).to_dict(orient='records')
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train model (simplified version)"""
    try:
        data = request.get_json()
        
        # Load dataset
        pkl_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.pkl')
        if not os.path.exists(pkl_path):
            return jsonify({'error': 'No dataset uploaded'}), 400
        
        df = pd.read_pickle(pkl_path)
        
        # Simulate training process
        import time
        time.sleep(2)  # Simulate training time
        
        # Create mock results
        mock_results = {
            'id': f"model_{int(datetime.now().timestamp())}",
            'name': f"{data.get('algorithm', 'id3').upper()}_model_{int(datetime.now().timestamp())}",
            'algorithm': data.get('algorithm', 'id3'),
            'target': data.get('target'),
            'accuracy': 0.85 + (hash(str(data)) % 100) / 1000,  # Mock accuracy
            'training_time': '2.3 seconds',
            'model_id': f"model_{int(datetime.now().timestamp())}",
            'created_at': datetime.now().isoformat()
        }
        
        # Save model info
        models_file = os.path.join(app.config['UPLOAD_FOLDER'], 'models.json')
        models = []
        if os.path.exists(models_file):
            with open(models_file, 'r') as f:
                models = json.load(f)
        
        models.append(mock_results)
        
        with open(models_file, 'w') as f:
            json.dump(models, f, indent=2)
        
        return jsonify(mock_results)
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of trained models"""
    print("🔍 GET /api/models endpoint called")
    try:
        models_file = os.path.join(app.config['UPLOAD_FOLDER'], 'models.json')
        print(f"📂 Looking for models file: {models_file}")
        print(f"📂 File exists: {os.path.exists(models_file)}")
        
        if not os.path.exists(models_file):
            print("❌ No models file found, returning empty list")
            return jsonify({'models': []})
        
        with open(models_file, 'r') as f:
            models = json.load(f)
        
        print(f"✅ Found {len(models)} models")
        return jsonify({'models': models})
        
    except Exception as e:
        print(f"❌ Error in get_models: {e}")
        return jsonify({'error': f'Failed to load models: {str(e)}'}), 500

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model_info(model_id):
    """Get detailed model information"""
    try:
        models_file = os.path.join(app.config['UPLOAD_FOLDER'], 'models.json')
        if not os.path.exists(models_file):
            return jsonify({'error': 'No models found'}), 404
        
        with open(models_file, 'r') as f:
            models = json.load(f)
        
        model = next((m for m in models if m['id'] == model_id or m['model_id'] == model_id), None)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Load dataset to get real features
        pkl_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.pkl')
        features = ['feature1', 'feature2', 'feature3', 'feature4']
        classes = ['class_a', 'class_b', 'class_c']
        
        if os.path.exists(pkl_path):
            try:
                df = pd.read_pickle(pkl_path)
                target_col = model.get('target')
                if target_col and target_col in df.columns:
                    features = [col for col in df.columns if col != target_col]
                    classes = df[target_col].unique().tolist() if target_col else classes
            except Exception as e:
                print(f"Error loading dataset for features: {e}")
        
        # Add mock detailed info
        model_details = {
            **model,
            'features': features,
            'classes': classes,
            'validation_method': 'k-fold'
        }
        
        return jsonify(model_details)
        
    except Exception as e:
        return jsonify({'error': f'Failed to load model info: {str(e)}'}), 500

@app.route('/api/models/<model_id>/results', methods=['GET'])
def get_model_results(model_id):
    """Get model results and metrics"""
    try:
        # Load model info
        models_file = os.path.join(app.config['UPLOAD_FOLDER'], 'models.json')
        if not os.path.exists(models_file):
            return jsonify({'error': 'No models found'}), 404
        
        with open(models_file, 'r') as f:
            models = json.load(f)
        
        model = next((m for m in models if m['id'] == model_id or m['model_id'] == model_id), None)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Load dataset to get real class information
        classes = ['class_a', 'class_b', 'class_c']
        pkl_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.pkl')
        
        if os.path.exists(pkl_path):
            try:
                df = pd.read_pickle(pkl_path)
                target_col = model.get('target')
                if target_col and target_col in df.columns:
                    classes = df[target_col].unique().tolist()
            except Exception as e:
                print(f"Error loading dataset for classes: {e}")
        
        # Generate realistic confusion matrix based on number of classes
        num_classes = len(classes)
        import random
        random.seed(42)  # For consistent results
        
        confusion_matrix = []
        for i in range(num_classes):
            row = []
            for j in range(num_classes):
                if i == j:  # Diagonal (correct predictions)
                    value = random.randint(80, 120)
                else:  # Off-diagonal (errors)
                    value = random.randint(5, 25)
                row.append(value)
            confusion_matrix.append(row)
        
        # Mock results for demonstration
        mock_results = {
            'algorithm': model.get('algorithm', 'id3'),
            'metrics': {
                'accuracy': model.get('accuracy', 0.87),
                'precision': model.get('accuracy', 0.87) - 0.02,
                'recall': model.get('accuracy', 0.87) - 0.04,
                'f1_score': model.get('accuracy', 0.87) - 0.03,
                'support': sum([sum(row) for row in confusion_matrix]),
                'classification_report': {}
            },
            'confusion_matrix': confusion_matrix,
            'class_names': classes,
            'feature_importance': {
                f'feature_{i+1}': random.uniform(0.1, 0.4) for i in range(4)
            }
        }
        
        # Generate classification report
        for i, class_name in enumerate(classes):
            mock_results['metrics']['classification_report'][class_name] = {
                'precision': random.uniform(0.75, 0.95),
                'recall': random.uniform(0.70, 0.90),
                'f1-score': random.uniform(0.72, 0.92),
                'support': sum(confusion_matrix[i])
            }
        
        # Add macro and weighted averages
        mock_results['metrics']['classification_report']['macro avg'] = {
            'precision': mock_results['metrics']['precision'],
            'recall': mock_results['metrics']['recall'],
            'f1-score': mock_results['metrics']['f1_score'],
            'support': mock_results['metrics']['support']
        }
        
        mock_results['metrics']['classification_report']['weighted avg'] = {
            'precision': mock_results['metrics']['precision'] + 0.01,
            'recall': mock_results['metrics']['accuracy'],
            'f1-score': mock_results['metrics']['f1_score'] + 0.01,
            'support': mock_results['metrics']['support']
        }
        
        # Add tree structure for ID3
        if model.get('algorithm') == 'id3':
            mock_results['tree_structure'] = {
                'feature': classes[0] if classes else 'feature1',
                'condition': '> 0.5',
                'samples': mock_results['metrics']['support'],
                'children': [
                    {
                        'feature': classes[1] if len(classes) > 1 else 'feature2',
                        'condition': '<= 0.3',
                        'samples': int(mock_results['metrics']['support'] * 0.6),
                        'children': [
                            {'class': classes[0], 'samples': int(mock_results['metrics']['support'] * 0.35)},
                            {'class': classes[1] if len(classes) > 1 else 'class_b', 'samples': int(mock_results['metrics']['support'] * 0.25)}
                        ]
                    },
                    {
                        'class': classes[-1],
                        'samples': int(mock_results['metrics']['support'] * 0.4)
                    }
                ]
            }
        
        return jsonify(mock_results)
        
    except Exception as e:
        print(f"Error in get_model_results: {e}")
        return jsonify({'error': f'Failed to load results: {str(e)}'}), 500

@app.route('/api/classify', methods=['POST'])
def classify_example():
    """Classify a new example"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        input_data = data.get('input_data')
        
        # Mock classification
        import random
        classes = ['class_a', 'class_b', 'class_c']
        prediction = random.choice(classes)
        confidence = random.uniform(0.7, 0.95)
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'processing_time': '0.05 seconds',
            'explanation': f'Classified as {prediction} based on input features'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Sistema Experto...")
    print("📂 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("🌐 Server starting at http://localhost:5000")
    print("🔧 Debug mode: ON")
    print("📊 Available endpoints:")
    print("   - GET  /              (Home page)")
    print("   - GET  /training      (Training interface)")
    print("   - GET  /classification (Classification interface)")
    print("   - GET  /results       (Results interface)")
    print("   - POST /api/upload    (Upload CSV)")
    print("   - POST /api/train     (Train model)")
    print("   - GET  /api/models    (List models)")
    print("   - GET  /api/models/<id> (Model info)")
    print("   - GET  /api/models/<id>/results (Model results)")
    print("   - POST /api/classify  (Classify example)")
    app.run(debug=True, port=5000)
