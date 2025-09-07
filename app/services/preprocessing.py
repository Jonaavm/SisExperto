"""
Data preprocessing service
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class SimplePreprocessor:
    """Simple data preprocessor for handling categorical and numerical data"""
    
    def __init__(self):
        self.encoders = {}  # col -> LabelEncoder (solo para categóricas)
        self.cols = []

    def fit(self, df: pd.DataFrame, target_col: str):
        """Fit the preprocessor on training data"""
        self.cols = [c for c in df.columns if c != target_col]
        for c in self.cols:
            if not pd.api.types.is_numeric_dtype(df[c]):
                le = LabelEncoder()
                df[c] = df[c].astype(str)
                le.fit(df[c])
                self.encoders[c] = le
        # target
        self.target_encoder = LabelEncoder()
        self.target_encoder.fit(df[target_col].astype(str))

    def transform(self, df: pd.DataFrame):
        """Transform data using fitted encoders"""
        out = df.copy()
        for c, le in self.encoders.items():
            out[c] = out[c].astype(str)
            # si encuentra un valor nuevo que no está en clases_ lanzará error -> manejar
            out[c] = out[c].map(lambda x: x if x in le.classes_ else None)
            # para evitar None, usamos -1 y reindexaremos
            # ahora transform solo para los que están
            known_mask = out[c].notnull()
            temp = out.loc[known_mask, c]
            out.loc[known_mask, c] = le.transform(temp)
            # para desconocidos:
            out.loc[~known_mask, c] = -1
        return out

    def transform_for_knn(self, df: pd.DataFrame):
        """Transform data for KNN (requires numerical arrays)"""
        t = self.transform(df)
        return t[self.cols].astype(float).values

    def transform_for_id3(self, df: pd.DataFrame):
        """Transform data for ID3 (keeps original types)"""
        return df[self.cols].copy()
