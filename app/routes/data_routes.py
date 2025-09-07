"""
API routes for data operations
"""
from flask import Blueprint, request, jsonify
from services.data_service import DataService

def create_data_routes(upload_folder):
    """Create data routes blueprint"""
    bp = Blueprint('data', __name__)
    data_service = DataService(upload_folder)

    @bp.route('/api/upload', methods=['POST'])
    def upload_csv():
        """Upload CSV dataset"""
        try:
            file = request.files.get('file')
            result = data_service.save_uploaded_file(file)
            return jsonify(result)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500

    @bp.route('/api/preview', methods=['GET'])
    def preview_dataset():
        """Get dataset preview"""
        try:
            preview = data_service.get_dataset_preview()
            return jsonify(preview)
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': f'Preview failed: {str(e)}'}), 500

    return bp
