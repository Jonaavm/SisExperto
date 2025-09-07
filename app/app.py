"""
Sistema Experto (Flask) - Refactorizado
Aplicación Flask para entrenar modelos de machine learning (ID3 y K-NN)
y realizar clasificaciones sobre datasets CSV.
"""

from flask import Flask
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from routes.main_routes import create_main_routes
from routes.data_routes import create_data_routes
from routes.model_routes import create_model_routes


def create_app():
    app = Flask(__name__)
    
    # Load configuration
    app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
    app.config['MODELS_FOLDER'] = Config.MODELS_FOLDER
    app.config['DEBUG'] = Config.DEBUG
    
    # Initialize required directories
    Config.init_folders()
    
    # Register blueprints
    app.register_blueprint(create_main_routes())
    app.register_blueprint(create_data_routes(Config.UPLOAD_FOLDER))
    app.register_blueprint(create_model_routes(Config.UPLOAD_FOLDER, Config.MODELS_FOLDER))
    
    return app


# Create app instance
app = create_app()


if __name__ == '__main__':
    app.run(debug=Config.DEBUG, port=Config.PORT)
