"""
Main application routes
"""
from flask import Blueprint, render_template

def create_main_routes():
    """Create main routes blueprint"""
    bp = Blueprint('main', __name__)

    @bp.route('/')
    def index():
        """Main page"""
        return render_template('index.html')

    @bp.route('/training')
    def training():
        """Training interface"""
        return render_template('training.html')

    @bp.route('/classification')
    def classification():
        """Classification interface"""
        return render_template('classification.html')

    @bp.route('/results')
    def results():
        """Results interface"""
        return render_template('results.html')

    @bp.route('/contacto')
    def contacto():
        """Contact page"""
        return render_template('contacto.html')

    return bp
