"""
ID3 Decision Tree Algorithm Implementation
"""
import pandas as pd
import numpy as np
import math


def entropy(y):
    """Calculate entropy of a target variable"""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum([p * math.log2(p) for p in probs if p > 0])


def info_gain_continuous(X_column, y):
    """Calculate information gain for continuous variables"""
    # Para variables continuas probamos umbrales en midpoints
    sorted_idx = np.argsort(X_column)
    Xs = X_column[sorted_idx]
    ys = y[sorted_idx]
    candidates = []
    for i in range(1, len(Xs)):
        if ys[i] != ys[i-1]:
            candidates.append((Xs[i] + Xs[i-1]) / 2.0)
    best_gain = -1
    best_thresh = None
    base_entropy = entropy(y)
    for t in candidates:
        left = ys[Xs <= t]
        right = ys[Xs > t]
        if len(left) == 0 or len(right) == 0:
            continue
        gain = base_entropy - (len(left)/len(ys))*entropy(left) - (len(right)/len(ys))*entropy(right)
        if gain > best_gain:
            best_gain = gain
            best_thresh = t
    return best_gain, best_thresh


def info_gain_categorical(X_column, y):
    """Calculate information gain for categorical variables"""
    base_entropy = entropy(y)
    vals, counts = np.unique(X_column, return_counts=True)
    weighted = 0
    for v, c in zip(vals, counts):
        weighted += (c/len(X_column))*entropy(y[X_column == v])
    return base_entropy - weighted


class ID3Node:
    """Node class for ID3 decision tree"""
    def __init__(self, *, feature=None, threshold=None, is_leaf=False, prediction=None, children=None):
        self.feature = feature
        self.threshold = threshold  # para continuas
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.children = children or {}  # para categóricas: valor -> nodo


class ID3:
    """ID3 Decision Tree Classifier"""
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_types = {}  # 'continuous' o 'categorical'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the decision tree"""
        # detectar tipos
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.feature_types[col] = 'continuous'
            else:
                self.feature_types[col] = 'categorical'
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        # condiciones de parada
        if len(np.unique(y)) == 1:
            return ID3Node(is_leaf=True, prediction=y.iloc[0])
        if depth >= self.max_depth or len(y) < self.min_samples_split or X.shape[1] == 0:
            # leaf: la clase mayoritaria
            vals, counts = np.unique(y, return_counts=True)
            return ID3Node(is_leaf=True, prediction=vals[np.argmax(counts)])

        # buscar mejor caracteristica
        best_gain = -1
        best_feat = None
        best_thresh = None
        for col in X.columns:
            if self.feature_types[col] == 'continuous':
                gain, thresh = info_gain_continuous(X[col].values.astype(float), y.values)
            else:
                gain = info_gain_categorical(X[col].values, y.values)
                thresh = None
            if gain is None:
                continue
            if gain > best_gain:
                best_gain = gain
                best_feat = col
                best_thresh = thresh

        if best_gain <= 0 or best_feat is None:
            vals, counts = np.unique(y, return_counts=True)
            return ID3Node(is_leaf=True, prediction=vals[np.argmax(counts)])

        node = ID3Node(feature=best_feat, threshold=best_thresh)
        if self.feature_types[best_feat] == 'continuous':
            left_mask = X[best_feat].astype(float) <= best_thresh
            right_mask = X[best_feat].astype(float) > best_thresh
            node.children['<='] = self._build_tree(X[left_mask].drop(columns=[best_feat]), y[left_mask], depth+1)
            node.children['>'] = self._build_tree(X[right_mask].drop(columns=[best_feat]), y[right_mask], depth+1)
        else:
            for v in np.unique(X[best_feat].astype(str)):
                mask = X[best_feat].astype(str) == v
                node.children[v] = self._build_tree(X[mask].drop(columns=[best_feat]), y[mask], depth+1)
        return node

    def _predict_row(self, row, node: ID3Node):
        """Predict a single row"""
        if node.is_leaf:
            return node.prediction
        val = row.get(node.feature)
        if node.threshold is not None:
            # continuo
            try:
                v = float(val)
            except:
                # si no convertible, asignar a una rama (>)
                v = float('-inf')
            if v <= node.threshold:
                return self._predict_row(row, node.children['<='])
            else:
                return self._predict_row(row, node.children['>'])
        else:
            # categórico
            key = str(val)
            if key in node.children:
                return self._predict_row(row, node.children[key])
            else:
                # si no existe la rama, devolver la rama mayoritaria (fallback)
                # escoger cualquiera: primer hijo
                child = list(node.children.values())[0]
                return self._predict_row(row, child)

    def predict(self, X: pd.DataFrame):
        """Predict multiple rows"""
        results = []
        for _, row in X.iterrows():
            results.append(self._predict_row(row, self.root))
        return np.array(results)
