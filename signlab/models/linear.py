"""Modèles linéaires personnels : softmax, perceptron, ridge, SVM et LDA/QDA."""

import numpy as np
from .base import Classifier, cross_entropy, softmax


class Logistic(Classifier):
    def __init__(self, learning_rate=0.1, epochs=200, l2=0.001, random_state=42):
        self.learning_rate, self.epochs, self.l2, self.random_state = learning_rate, epochs, l2, random_state

    def loss_gradient(self, X, y, W, b):
        logits = X @ W + b
        residual = softmax(logits)
        residual[np.arange(len(y)), y] -= 1
        return (cross_entropy(logits, y) + self.l2 * float((W * W).sum()) / 2,
                X.T @ residual / len(X) + self.l2 * W, residual.mean(0))

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        if self.learning_rate <= 0 or self.epochs < 1 or self.l2 < 0:
            raise ValueError("Hyperparamètres invalides.")
        self.W_ = np.zeros((X.shape[1], self.n_classes_))
        self.b_ = np.zeros(self.n_classes_)
        self.history_ = []
        for _ in range(self.epochs):
            loss, dw, db = self.loss_gradient(X, y, self.W_, self.b_)
            self.W_ -= self.learning_rate * dw
            self.b_ -= self.learning_rate * db
            self.history_.append(float(loss))
        return self

    def decision_function(self, X):
        return self._check(X) @ self.W_ + self.b_

    def predict_proba(self, X):
        return softmax(self.decision_function(X))


class Perceptron(Classifier):
    def __init__(self, learning_rate=0.05, epochs=40, random_state=42):
        self.learning_rate, self.epochs, self.random_state = learning_rate, epochs, random_state

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        self.W_ = np.zeros((X.shape[1], self.n_classes_))
        self.b_ = np.zeros(self.n_classes_)
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.epochs):
            errors = 0
            for i in rng.permutation(len(X)):
                pred = int(np.argmax(X[i] @ self.W_ + self.b_))
                if pred != y[i]:
                    delta = self.learning_rate * X[i]
                    self.W_[:, y[i]] += delta
                    self.W_[:, pred] -= delta
                    self.b_[y[i]] += self.learning_rate
                    self.b_[pred] -= self.learning_rate
                    errors += 1
            if not errors:
                break
        return self

    def decision_function(self, X):
        return self._check(X) @ self.W_ + self.b_


class RidgeClassifier(Classifier):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        if self.alpha <= 0:
            raise ValueError("alpha doit être positif.")
        self.mean_ = X.mean(0)
        Z = X - self.mean_
        targets = np.eye(self.n_classes_)[y]
        self.b_ = targets.mean(0)
        targets -= self.b_
        if X.shape[1] <= len(X):
            self.W_ = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(X.shape[1]), Z.T @ targets)
        else:
            self.W_ = Z.T @ np.linalg.solve(Z @ Z.T + self.alpha * np.eye(len(X)), targets)
        return self

    def decision_function(self, X):
        return (self._check(X) - self.mean_) @ self.W_ + self.b_


class LinearSVM(Classifier):
    """SVM multiclasse de Crammer-Singer : marge hinge et régularisation L2."""
    def __init__(self, learning_rate=0.05, epochs=200, l2=0.01, random_state=42):
        self.learning_rate, self.epochs, self.l2, self.random_state = learning_rate, epochs, l2, random_state

    def loss_gradient(self, X, y, W, b):
        scores = X @ W + b
        rows = np.arange(len(X))
        margins = scores - scores[rows, y, None] + 1
        margins[rows, y] = -np.inf
        rivals = margins.argmax(axis=1)
        loss = np.maximum(0, margins[rows, rivals])
        grad = np.zeros_like(scores)
        active = rows[loss > 0]
        grad[active, rivals[active]] = 1
        grad[active, y[active]] = -1
        return (float(loss.mean() + self.l2 * (W * W).sum() / 2),
                X.T @ grad / len(X) + self.l2 * W, grad.mean(0))

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        self.W_ = np.zeros((X.shape[1], self.n_classes_))
        self.b_ = np.zeros(self.n_classes_)
        self.history_ = []
        for epoch in range(self.epochs):
            loss, dw, db = self.loss_gradient(X, y, self.W_, self.b_)
            rate = self.learning_rate / (1 + 0.01 * epoch)
            self.W_ -= rate * dw
            self.b_ -= rate * db
            self.history_.append(loss)
        return self

    def decision_function(self, X):
        return self._check(X) @ self.W_ + self.b_


class Discriminant(Classifier):
    def __init__(self, quadratic=False, shrinkage=0.2):
        self.quadratic, self.shrinkage = quadratic, shrinkage

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        if not 0 < self.shrinkage <= 1:
            raise ValueError("shrinkage doit être dans ]0, 1].")
        d = X.shape[1]
        if d > 1024:
            raise ValueError("LDA/QDA : utilisez PCA ou au plus 1024 variables.")
        self.means_ = np.array([X[y == c].mean(0) for c in range(self.n_classes_)])
        self.log_prior_ = np.log(np.bincount(y) / len(y))
        centered = X - self.means_[y]
        shared = centered.T @ centered / len(X)
        self.inverse_, self.logdet_ = [], []
        for c in range(self.n_classes_):
            z = centered[y == c]
            cov = z.T @ z / max(1, len(z)) if self.quadratic else shared
            scale = max(float(np.trace(cov) / d), 1e-8)
            cov = (1 - self.shrinkage) * cov + self.shrinkage * scale * np.eye(d)
            self.inverse_.append(np.linalg.inv(cov))
            self.logdet_.append(float(np.linalg.slogdet(cov)[1]))
        return self

    def decision_function(self, X):
        X = self._check(X)
        return np.column_stack([
            -0.5 * np.sum(((X - self.means_[c]) @ self.inverse_[c]) * (X - self.means_[c]), axis=1)
            - 0.5 * self.logdet_[c] + self.log_prior_[c] for c in range(self.n_classes_)])

    def predict_proba(self, X):
        return softmax(self.decision_function(X))
