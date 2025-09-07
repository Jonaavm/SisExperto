"""
Data management service
"""
import pandas as pd
import os


class DataService:
    """Service for handling data upload and management"""
    
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)
    
    def save_uploaded_file(self, file):
        """Save uploaded CSV file and return dataset info"""
        if not file:
            raise ValueError('No file uploaded')
        
        csv_path = os.path.join(self.upload_folder, 'dataset.csv')
        pkl_path = os.path.join(self.upload_folder, 'dataset.pkl')
        
        file.save(csv_path)
        df = pd.read_csv(csv_path)
        df.to_pickle(pkl_path)
        
        return {
            'message': 'File uploaded successfully',
            'shape': df.shape,
            'columns': df.columns.tolist()
        }
    
    def get_dataset(self):
        """Load and return the current dataset"""
        pkl_path = os.path.join(self.upload_folder, 'dataset.pkl')
        if not os.path.exists(pkl_path):
            raise FileNotFoundError('No dataset uploaded')
        return pd.read_pickle(pkl_path)
    
    def get_dataset_preview(self, n_rows=10):
        """Get a preview of the dataset"""
        df = self.get_dataset()
        return df.head(n_rows).to_dict(orient='records')
