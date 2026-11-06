"""MLP, RBF et SVM à noyau écrits avec NumPy."""

import numpy as np
from .base import Classifier, cross_entropy, matrix, softmax, squared_distances


def kmeans(X, n_clusters, random_state=42, max_iter=80):
    """Initialisation k-means++, puis itérations de Lloyd ; clusters vides réparés."""
    X = matrix(X)
    k = min(int(n_clusters), len(X))
    if k < 1:
        raise ValueError("Nombre de centres invalide.")
    rng = np.random.default_rng(random_state)
    centers = [X[rng.integers(len(X))].copy()]
    nearest = squared_distances(X, np.array(centers))[:, 0]
    for _ in range(1, k):
        index = rng.choice(len(X), p=nearest / nearest.sum()) if nearest.sum() > 1e-14 else rng.integers(len(X))
        centers.append(X[index].copy())
        nearest = np.minimum(nearest, squared_distances(X, np.array([centers[-1]]))[:, 0])
    centers = np.array(centers)
    for _ in range(max_iter):
        distances = squared_distances(X, centers)
        labels = distances.argmin(1)
        updated = np.array([X[labels == c].mean(0) if np.any(labels == c)
                            else X[np.argmax(distances.min(1))] for c in range(k)])
        if np.allclose(updated, centers, atol=1e-7, rtol=0):
            centers = updated
            break
        centers = updated
    return centers


class MLP(Classifier):
    def __init__(self, hidden_units=32, learning_rate=0.03, epochs=160, batch_size=64,
                 l2=0.0001, random_state=42):
        self.hidden_units, self.learning_rate, self.epochs = hidden_units, learning_rate, epochs
        self.batch_size, self.l2, self.random_state = batch_size, l2, random_state

    def loss_gradients(self, X, y, W1, b1, W2, b2):
        h = np.tanh(X @ W1 + b1)
        logits = h @ W2 + b2
        dz = softmax(logits)
        dz[np.arange(len(y)), y] -= 1
        dz /= len(X)
        dh = (dz @ W2.T) * (1 - h * h)
        loss = cross_entropy(logits, y) + self.l2 * ((W1 * W1).sum() + (W2 * W2).sum()) / 2
        return float(loss), (X.T @ dh + self.l2 * W1, dh.sum(0), h.T @ dz + self.l2 * W2, dz.sum(0))

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        if min(self.hidden_units, self.epochs, self.batch_size) < 1 or self.learning_rate <= 0 or self.l2 < 0:
            raise ValueError("Hyperparamètres MLP invalides.")
        rng = np.random.default_rng(self.random_state)
        self.W1_ = rng.normal(0, np.sqrt(2 / (X.shape[1] + self.hidden_units)), (X.shape[1], self.hidden_units))
        self.W2_ = rng.normal(0, np.sqrt(2 / (self.hidden_units + self.n_classes_)), (self.hidden_units, self.n_classes_))
        self.b1_, self.b2_ = np.zeros(self.hidden_units), np.zeros(self.n_classes_)
        self.history_ = []
        params = (self.W1_, self.b1_, self.W2_, self.b2_)
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                indices = order[start:start + self.batch_size]
                _, gradients = self.loss_gradients(X[indices], y[indices], *params)
                for param, grad in zip(params, gradients):
                    param -= self.learning_rate * grad
            self.history_.append(self.loss_gradients(X, y, *params)[0])
        return self

    def decision_function(self, X):
        return np.tanh(self._check(X) @ self.W1_ + self.b1_) @ self.W2_ + self.b2_

    def predict_proba(self, X):
        return softmax(self.decision_function(X))


class RBFNetwork(Classifier):
    def __init__(self, n_centers=24, gamma=None, learning_rate=0.3, epochs=160,
                 l2=0.001, random_state=42):
        self.n_centers, self.gamma, self.learning_rate = n_centers, gamma, learning_rate
        self.epochs, self.l2, self.random_state = epochs, l2, random_state

    def _basis(self, X):
        return np.exp(-self.gamma_ * squared_distances(X, self.centers_))

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        self.centers_ = kmeans(X, self.n_centers, self.random_state)
        d = squared_distances(self.centers_, self.centers_)
        positives = d[d > 1e-12]
        self.gamma_ = float(self.gamma if self.gamma is not None else 1 / max(float(np.median(positives)) if len(positives) else 1, 1e-8))
        if self.gamma_ <= 0:
            raise ValueError("gamma doit être positif.")
        Z = self._basis(X)
        self.W_, self.b_ = np.zeros((Z.shape[1], self.n_classes_)), np.zeros(self.n_classes_)
        self.history_ = []
        for _ in range(self.epochs):
            logits = Z @ self.W_ + self.b_
            residual = softmax(logits)
            residual[np.arange(len(y)), y] -= 1
            self.W_ -= self.learning_rate * (Z.T @ residual / len(y) + self.l2 * self.W_)
            self.b_ -= self.learning_rate * residual.mean(0)
            self.history_.append(cross_entropy(logits, y) + float(self.l2 * (self.W_ * self.W_).sum() / 2))
        return self

    def decision_function(self, X):
        return self._basis(self._check(X)) @ self.W_ + self.b_

    def predict_proba(self, X):
        return softmax(self.decision_function(X))


class KernelSVM(Classifier):
    """Dual C-SVM binaire résolu par SMO simplifié, combinaison one-versus-rest."""
    def __init__(self, kernel="rbf", C=1.0, gamma=None, degree=3, coef0=1.0,
                 max_iter=100, tolerance=0.001, random_state=42, max_samples=2500):
        self.kernel, self.C, self.gamma = kernel, C, gamma
        self.degree, self.coef0, self.max_iter = degree, coef0, max_iter
        self.tolerance, self.random_state, self.max_samples = tolerance, random_state, max_samples

    def _kernel(self, X, Y):
        if self.kernel == "rbf":
            return np.exp(-self.gamma_ * squared_distances(X, Y))
        return (self.gamma_ * X @ Y.T + self.coef0) ** self.degree

    def _binary(self, K, y, rng):
        n = len(y)
        alpha, f, b = np.zeros(n), np.zeros(n), 0.0
        passes = 0
        sweeps = 0
        for sweeps in range(1, self.max_iter + 1):
            changed = 0
            for i in rng.permutation(n):
                ei = f[i] - y[i]
                if not ((y[i] * ei < -self.tolerance and alpha[i] < self.C - 1e-10)
                        or (y[i] * ei > self.tolerance and alpha[i] > 1e-10)):
                    continue
                errors = f - y
                # Essayer plusieurs paires évite un blocage sur des bornes L == H.
                order = np.argsort(-np.abs(ei - errors))
                order = np.concatenate((order[:min(12, n)], rng.permutation(n)[:min(6, n)]))
                for j in order:
                    if i == j:
                        continue
                    ai, aj = alpha[i], alpha[j]
                    if y[i] != y[j]:
                        low, high = max(0, aj - ai), min(self.C, self.C + aj - ai)
                    else:
                        low, high = max(0, ai + aj - self.C), min(self.C, ai + aj)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if high - low < 1e-12 or eta >= -1e-12:
                        continue
                    ej = errors[j]
                    new_aj = float(np.clip(aj - y[j] * (ei - ej) / eta, low, high))
                    if abs(new_aj - aj) < 1e-7:
                        continue
                    new_ai = ai + y[i] * y[j] * (aj - new_aj)
                    b1 = b - ei - y[i] * (new_ai - ai) * K[i, i] - y[j] * (new_aj - aj) * K[i, j]
                    b2 = b - ej - y[i] * (new_ai - ai) * K[i, j] - y[j] * (new_aj - aj) * K[j, j]
                    new_b = b1 if 1e-10 < new_ai < self.C - 1e-10 else b2 if 1e-10 < new_aj < self.C - 1e-10 else (b1 + b2) / 2
                    f += y[i] * (new_ai - ai) * K[:, i] + y[j] * (new_aj - aj) * K[:, j] + new_b - b
                    alpha[i], alpha[j], b = new_ai, new_aj, new_b
                    changed += 1
                    break
            passes = passes + 1 if changed == 0 else 0
            if passes >= 5:
                break
        margins = y * f
        violation = np.maximum(np.where(alpha < self.C - 1e-7, 1 - margins, 0),
                               np.where(alpha > 1e-7, margins - 1, 0))
        return alpha, b, {"sweeps": sweeps, "max_kkt_violation": float(max(0, violation.max())),
                          "dual_equality_error": float(abs(alpha @ y))}

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        if len(X) > self.max_samples:
            raise ValueError(f"SVM à noyau limité à {self.max_samples} images : matrice quadratique. Réduisez train_limit pour tous les modèles.")
        if self.C <= 0 or self.kernel not in {"rbf", "poly"} or self.degree < 1 or self.coef0 < 0:
            raise ValueError("Hyperparamètres du noyau invalides.")
        self.gamma_ = float(self.gamma if self.gamma is not None else 1 / X.shape[1])
        if self.gamma_ <= 0:
            raise ValueError("gamma doit être positif.")
        K = self._kernel(X, X)
        if not np.isfinite(K).all():
            raise ValueError("Noyau non fini.")
        rng = np.random.default_rng(self.random_state)
        self.support_, self.dual_, self.bias_, self.diagnostics_ = [], [], [], []
        for c in range(self.n_classes_):
            target = np.where(y == c, 1.0, -1.0)
            alpha, b, diagnostics = self._binary(K, target, rng)
            mask = alpha > 1e-8
            self.support_.append(X[mask].copy())
            self.dual_.append(alpha[mask] * target[mask])
            self.bias_.append(float(b))
            self.diagnostics_.append(diagnostics)
        return self

    def decision_function(self, X):
        X = self._check(X)
        result = np.zeros((len(X), self.n_classes_))
        for start in range(0, len(X), 128):
            for c in range(self.n_classes_):
                result[start:start + 128, c] = self._kernel(X[start:start + 128], self.support_[c]) @ self.dual_[c] + self.bias_[c]
        return result
