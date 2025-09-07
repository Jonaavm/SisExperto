"""
Configuration utilities
"""
import os


class Config:
    """Application configuration"""
    UPLOAD_FOLDER = 'uploads'
    MODELS_FOLDER = 'models'
    DEBUG = True
    PORT = 5000
    
    @classmethod
    def init_folders(cls):
        """Initialize required folders"""
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.MODELS_FOLDER, exist_ok=True)
