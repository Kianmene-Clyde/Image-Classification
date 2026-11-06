"""Référence naïve, méthodes de voisinage et Bayes naïf."""

import numpy as np
from .base import Classifier, softmax, squared_distances


class Majority(Classifier):
    def fit(self, X, y):
        X, y = self._prepare(X, y)
        self.priors_ = np.bincount(y, minlength=self.n_classes_) / len(y)
        return self

    def decision_function(self, X):
        return np.tile(self.priors_, (len(self._check(X)), 1))


class NearestCentroid(Classifier):
    def fit(self, X, y):
        X, y = self._prepare(X, y)
        self.centers_ = np.array([X[y == c].mean(0) for c in range(self.n_classes_)])
        return self

    def decision_function(self, X):
        return -squared_distances(self._check(X), self.centers_)


class KNN(Classifier):
    def __init__(self, n_neighbors=5, distance_weighted=True):
        self.n_neighbors = n_neighbors
        self.distance_weighted = distance_weighted

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors doit être positif.")
        self.X_, self.y_ = X.copy(), y.copy()
        return self

    def decision_function(self, X):
        X = self._check(X)
        k = min(self.n_neighbors, len(self.X_))
        result = np.zeros((len(X), self.n_classes_))
        for start in range(0, len(X), 128):
            distances = squared_distances(X[start:start + 128], self.X_)
            for offset, row in enumerate(distances):
                indices = np.argpartition(row, k - 1)[:k]
                exact = row[indices] < 1e-20
                if exact.any():
                    indices = indices[exact]
                    weights = np.ones(len(indices))
                else:
                    weights = 1 / np.sqrt(row[indices]) if self.distance_weighted else np.ones(k)
                result[start + offset] = np.bincount(self.y_[indices], weights=weights, minlength=self.n_classes_)
        return result


class NaiveBayes(Classifier):
    def __init__(self, kind="gaussian", alpha=1.0, variance_floor=1e-8):
        self.kind, self.alpha, self.variance_floor = kind, alpha, variance_floor

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        if self.kind not in {"gaussian", "bernoulli", "multinomial", "complement"} or self.alpha <= 0:
            raise ValueError("Type Bayes ou lissage invalide.")
        counts = np.bincount(y, minlength=self.n_classes_)
        self.log_prior_ = np.log(counts / len(y))
        if self.kind == "gaussian":
            self.means_ = np.array([X[y == c].mean(0) for c in range(self.n_classes_)])
            self.vars_ = np.array([X[y == c].var(0) for c in range(self.n_classes_)])
            self.vars_ += max(self.variance_floor, float(X.var(0).max()) * 1e-9)
        else:
            if X.min() < 0:
                raise ValueError("Les variantes discrètes de Bayes exigent des variables positives.")
            values = (X > 0.5).astype(float) if self.kind == "bernoulli" else X
            sums = np.array([values[y == c].sum(0) for c in range(self.n_classes_)])
            if self.kind == "bernoulli":
                p = (sums + self.alpha) / (counts[:, None] + 2 * self.alpha)
                self.log_p_, self.log_not_p_ = np.log(p), np.log1p(-p)
            else:
                if self.kind == "complement":
                    sums = sums.sum(0, keepdims=True) - sums
                sums += self.alpha
                self.log_p_ = np.log(sums / sums.sum(1, keepdims=True))
        return self

    def decision_function(self, X):
        X = self._check(X)
        if self.kind == "gaussian":
            return np.column_stack([
                -0.5 * (np.log(2 * np.pi * self.vars_[c]) + (X - self.means_[c]) ** 2 / self.vars_[c]).sum(1)
                + self.log_prior_[c] for c in range(self.n_classes_)])
        if X.min() < 0:
            raise ValueError("Variables négatives incompatibles avec Bayes discret.")
        if self.kind == "bernoulli":
            X = (X > 0.5).astype(float)
            return X @ self.log_p_.T + (1 - X) @ self.log_not_p_.T + self.log_prior_
        if self.kind == "complement":
            return -X @ self.log_p_.T
        return X @ self.log_p_.T + self.log_prior_

    def predict_proba(self, X):
        if self.kind == "complement":
            raise ValueError("ComplementNB fournit des scores, pas des probabilités calibrées.")
        return softmax(self.decision_function(X))
