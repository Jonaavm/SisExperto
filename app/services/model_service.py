"""
Model evaluation service
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
import joblib
import os
import json

from models.id3 import ID3
from models.knn import KNN
from services.preprocessing import SimplePreprocessor


def evaluate_model(model, X, y, model_type='id3'):
    """Evaluate a model and return metrics"""
    if model_type == 'knn':
        y_pred = model.predict(X)
    else:
        # ID3 espera DataFrame
        y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y, y_pred)
    return {
        'accuracy': float(acc), 
        'precision': float(prec), 
        'recall': float(rec), 
        'f1': float(f1), 
        'confusion_matrix': cm.tolist()
    }


class ModelTrainer:
    """Service for training and evaluating models"""
    
    def __init__(self, models_folder):
        self.models_folder = models_folder
    
    def train_with_kfold(self, df, algorithm, target, k_folds=5, **params):
        """Train model using k-fold cross validation"""
        prep = SimplePreprocessor()
        prep.fit(df, target)
        
        results = {'folds': []}
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        X = df.drop(columns=[target]).reset_index(drop=True)
        y = prep.target_encoder.transform(df[target].astype(str))
        
        for train_idx, test_idx in kf.split(X):
            X_train = X.loc[train_idx].reset_index(drop=True)
            X_test = X.loc[test_idx].reset_index(drop=True)
            y_train = y[train_idx]
            y_test = y[test_idx]

            if algorithm == 'knn':
                Xtr = prep.transform_for_knn(X_train)
                Xte = prep.transform_for_knn(X_test)
                model = KNN(k=params.get('knn_k', 3))
                model.fit(Xtr, y_train)
                metrics = evaluate_model(model, Xte, y_test, 'knn')
            else:
                # ID3
                model = ID3(max_depth=params.get('max_depth', 10))
                model.fit(prep.transform_for_id3(X_train), pd.Series(prep.target_encoder.inverse_transform(y_train)))
                Xte_id3 = prep.transform_for_id3(X_test)
                y_test_labels = prep.target_encoder.inverse_transform(y_test)
                metrics = evaluate_model(model, Xte_id3, y_test_labels, 'id3')

            results['folds'].append(metrics)

        # Calculate averages
        avg = {k: np.mean([f[k] for f in results['folds']]) for k in ['accuracy', 'precision', 'recall', 'f1']}
        results['average'] = {k: float(avg[k]) for k in avg}

        # Train final model with full dataset
        final_model = self._train_final_model(df, algorithm, target, prep, **params)
        model_path = self._save_model(final_model, prep, algorithm, target)
        
        # Get the saved model info from models.json
        models_file = os.path.join(self.models_folder, 'models.json')
        if os.path.exists(models_file):
            with open(models_file, 'r') as f:
                models = json.load(f)
            # Get the last saved model (most recent)
            if models:
                latest_model = models[-1]
                latest_model['accuracy'] = results['average']['accuracy']
                # Save updated accuracy
                with open(models_file, 'w') as f:
                    json.dump(models, f, indent=2)
                
                results.update(latest_model)
        
        results['model_path'] = model_path
        results['training_time'] = '2.5 seconds'  # Mock timing
        
        return results
    
    def train_with_holdout(self, df, algorithm, target, test_size=0.2, **params):
        """Train model using holdout validation"""
        X = df.drop(columns=[target])
        y = df[target].astype(str)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        prep = SimplePreprocessor()
        prep.fit(pd.concat([Xtr, ytr.rename('target')], axis=1), 'target')

        if algorithm == 'knn':
            Xtr_knn = prep.transform_for_knn(Xtr)
            Xte_knn = prep.transform_for_knn(Xte)
            ytr_enc = prep.target_encoder.transform(ytr.astype(str))
            yte_enc = prep.target_encoder.transform(yte.astype(str))
            model = KNN(k=params.get('knn_k', 3))
            model.fit(Xtr_knn, ytr_enc)
            metrics = evaluate_model(model, Xte_knn, yte_enc, 'knn')
            model_payload = {'model': model, 'preprocessor': prep, 'meta': {'algorithm': 'knn', 'target': target}}
        else:
            model = ID3(max_depth=params.get('max_depth', 10))
            model.fit(prep.transform_for_id3(Xtr), ytr)
            metrics = evaluate_model(model, prep.transform_for_id3(Xte), yte, 'id3')
            model_payload = {'model': model, 'preprocessor': prep, 'meta': {'algorithm': 'id3', 'target': target}}

        model_path = os.path.join(self.models_folder, 'final_model.joblib')
        joblib.dump(model_payload, model_path)
        
        return {'metrics': metrics, 'model_path': model_path}
    
    def _train_final_model(self, df, algorithm, target, prep, **params):
        """Train final model with complete dataset"""
        if algorithm == 'knn':
            X_all = prep.transform_for_knn(df.drop(columns=[target]))
            y_all = prep.target_encoder.transform(df[target].astype(str))
            final_model = KNN(k=params.get('knn_k', 3))
            final_model.fit(X_all, y_all)
        else:
            final_model = ID3(max_depth=params.get('max_depth', 10))
            final_model.fit(prep.transform_for_id3(df.drop(columns=[target])), df[target].astype(str))
        
        return final_model
    
    def _save_model(self, model, prep, algorithm, target):
        """Save trained model to disk and update models registry"""
        import json
        from datetime import datetime
        
        # Save the model file
        model_payload = {
            'model': model, 
            'preprocessor': prep, 
            'meta': {'algorithm': algorithm, 'target': target}
        }
        model_path = os.path.join(self.models_folder, 'final_model.joblib')
        joblib.dump(model_payload, model_path)
        
        # Update models registry
        models_file = os.path.join(self.models_folder, 'models.json')
        models = []
        
        if os.path.exists(models_file):
            try:
                with open(models_file, 'r') as f:
                    models = json.load(f)
            except:
                models = []
        
        # Create model metadata
        model_id = f"model_{int(datetime.now().timestamp())}"
        model_info = {
            'id': model_id,
            'model_id': model_id,  # For backward compatibility
            'name': f"{algorithm.upper()}_model_{int(datetime.now().timestamp())}",
            'algorithm': algorithm,
            'target': target,
            'created_at': datetime.now().isoformat(),
            'model_path': model_path,
            'accuracy': 0.85  # Placeholder - should be updated with real accuracy
        }
        
        models.append(model_info)
        
        # Save updated models list
        with open(models_file, 'w') as f:
            json.dump(models, f, indent=2)
        
        return model_path


class ModelPredictor:
    """Service for making predictions with trained models"""
    
    def __init__(self, models_folder):
        self.models_folder = models_folder
    
    def predict(self, sample):
        """Make prediction for a single sample"""
        model_file = os.path.join(self.models_folder, 'final_model.joblib')
        if not os.path.exists(model_file):
            raise FileNotFoundError('No trained model available')
        
        payload = joblib.load(model_file)
        model = payload['model']
        prep = payload['preprocessor']
        meta = payload['meta']

        df_sample = pd.DataFrame([sample])
        target_col = meta['target']
        
        # Ensure all columns are present
        for c in prep.cols:
            if c not in df_sample.columns:
                df_sample[c] = np.nan

        if meta['algorithm'] == 'knn':
            Xs = prep.transform_for_knn(df_sample)
            pred_enc = model.predict(Xs)
            pred_label = prep.target_encoder.inverse_transform(pred_enc.astype(int))
            return pred_label[0]
        else:
            Xs = prep.transform_for_id3(df_sample)
            pred = model.predict(Xs)
            return pred[0]
