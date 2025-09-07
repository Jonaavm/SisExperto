"""
Test app - Sistema Experto
Aplicación Flask simple para probar las nuevas interfaces
"""

from flask import Flask, render_template
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/training')
def training():
    """Training interface"""
    return render_template('training.html')

@app.route('/classification')
def classification():
    """Classification interface"""
    return render_template('classification.html')

@app.route('/results')
def results():
    """Results interface"""
    return render_template('results.html')

@app.route('/contacto')
def contacto():
    """Contact page"""
    return render_template('contacto.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
