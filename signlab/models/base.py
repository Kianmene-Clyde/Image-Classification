"""Contrat commun et opérations numériques utilisées par les estimateurs."""

import numpy as np


def matrix(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
        raise ValueError("X doit être une matrice non vide (observations, variables).")
    if not np.isfinite(X).all():
        raise ValueError("X contient des valeurs non finies.")
    return X


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def squared_distances(X, Y):
    return np.maximum((X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2 * X @ Y.T, 0)


def cross_entropy(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(y)), y]))


class Classifier:
    """fit(X, y), predict(X), decision_function(X), score(X, y)."""

    def _prepare(self, X, y):
        X = matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y doit être un vecteur de labels, un par observation.")
        if y.dtype.kind in "fc" and not np.isfinite(y).all():
            raise ValueError("Labels non finis.")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Au moins deux classes sont nécessaires.")
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(self.classes_)
        return X, encoded

    def _check(self, X):
        if not hasattr(self, "classes_"):
            raise ValueError("Modèle non entraîné. Appelez fit avant predict.")
        X = matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Attendu {self.n_features_in_} variables, reçu {X.shape[1]}.")
        return X

    def predict(self, X):
        X = self._check(X)
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

