"""CART à seuils candidats, bagging, forêts, AdaBoost SAMME et gradient boosting."""

import numpy as np
from .base import Classifier, softmax, cross_entropy


def grow_tree(X, target, rng, depth, min_leaf, max_features, random_thresholds=False,
              weights=None, regression=False, n_classes=2):
    n, d = X.shape
    weights = np.ones(n) if weights is None else weights
    mass = weights.sum()
    if regression:
        value = np.average(target, weights=weights)
        impurity = np.sum(weights * (target - value) ** 2)
    else:
        counts = np.bincount(target, weights=weights, minlength=n_classes)
        value = counts / mass
        impurity = mass * (1 - np.sum(value ** 2))
    leaf = {"value": float(value) if regression else value.tolist()}
    if depth <= 0 or n < 2 * min_leaf or impurity <= 1e-12:
        return leaf
    count = d if max_features is None else max(1, min(d, int(max_features)))
    features = rng.choice(d, count, replace=False)
    best, best_split = impurity, None
    for feature in features:
        column = X[:, feature]
        unique = np.unique(column)
        if len(unique) < 2:
            continue
        if random_thresholds:
            thresholds = [rng.uniform(unique[0], unique[-1])]
        elif len(unique) <= 17:
            thresholds = (unique[1:] + unique[:-1]) / 2
        else:
            thresholds = np.unique(np.quantile(column, np.linspace(0, 1, 17)[1:-1]))
        for threshold in thresholds:
            left = column <= threshold
            if left.sum() < min_leaf or (~left).sum() < min_leaf:
                continue
            score = 0.0
            for mask in (left, ~left):
                w, t = weights[mask], target[mask]
                m = w.sum()
                if m <= 0:
                    score = np.inf
                    break
                if regression:
                    score += np.sum(w * (t - np.average(t, weights=w)) ** 2)
                else:
                    counts = np.bincount(t, weights=w, minlength=n_classes)
                    score += m - (counts @ counts) / m
            if score < best - 1e-12:
                best, best_split = score, (int(feature), float(threshold), left)
    if best_split is None:
        return leaf
    feature, threshold, left = best_split
    kwargs = dict(rng=rng, depth=depth - 1, min_leaf=min_leaf, max_features=max_features,
                  random_thresholds=random_thresholds, regression=regression, n_classes=n_classes)
    return {"feature": feature, "threshold": threshold,
            "left": grow_tree(X[left], target[left], weights=weights[left], **kwargs),
            "right": grow_tree(X[~left], target[~left], weights=weights[~left], **kwargs)}


def tree_predict(tree, X):
    values = []
    for row in X:
        node = tree
        while "feature" in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        values.append(node["value"])
    return np.asarray(values)


class DecisionTree(Classifier):
    def __init__(self, max_depth=6, min_leaf=2, max_features=None, random_state=42):
        self.max_depth, self.min_leaf, self.max_features, self.random_state = max_depth, min_leaf, max_features, random_state

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        self.tree_ = grow_tree(X, y, np.random.default_rng(self.random_state), self.max_depth,
                               self.min_leaf, self.max_features, n_classes=self.n_classes_)
        return self

    def decision_function(self, X):
        return tree_predict(self.tree_, self._check(X))

    def predict_proba(self, X):
        return self.decision_function(X)


class Forest(Classifier):
    def __init__(self, n_estimators=15, max_depth=6, min_leaf=2, kind="random_forest", random_state=42):
        self.n_estimators, self.max_depth, self.min_leaf = n_estimators, max_depth, min_leaf
        self.kind, self.random_state = kind, random_state

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        if self.kind not in {"random_forest", "bagging", "extra_trees"} or self.n_estimators < 1:
            raise ValueError("Configuration de forêt invalide.")
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            indices = np.arange(len(X)) if self.kind == "extra_trees" else rng.integers(len(X), size=len(X))
            max_features = None if self.kind == "bagging" else max(1, int(np.sqrt(X.shape[1])))
            self.trees_.append(grow_tree(X[indices], y[indices], rng, self.max_depth, self.min_leaf,
                                         max_features, self.kind == "extra_trees", n_classes=self.n_classes_))
        return self

    def decision_function(self, X):
        X = self._check(X)
        return np.mean([tree_predict(tree, X) for tree in self.trees_], axis=0)

    def predict_proba(self, X):
        return self.decision_function(X)


class AdaBoost(Classifier):
    """SAMME discret multiclasse, arbres faibles de profondeur 1 par défaut."""
    def __init__(self, n_estimators=30, learning_rate=0.7, max_depth=1, random_state=42):
        self.n_estimators, self.learning_rate, self.max_depth, self.random_state = n_estimators, learning_rate, max_depth, random_state

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        rng = np.random.default_rng(self.random_state)
        weights = np.full(len(y), 1 / len(y))
        self.trees_, self.alphas_ = [], []
        self.priors_ = np.bincount(y, minlength=self.n_classes_) / len(y)
        for _ in range(self.n_estimators):
            tree = grow_tree(X, y, rng, self.max_depth, 1, None, weights=weights, n_classes=self.n_classes_)
            wrong = tree_predict(tree, X).argmax(1) != y
            error = float(weights @ wrong)
            if error >= 1 - 1 / self.n_classes_ - 1e-12:
                break
            alpha = self.learning_rate * (np.log((1 - max(error, 1e-12)) / max(error, 1e-12)) + np.log(self.n_classes_ - 1))
            self.trees_.append(tree)
            self.alphas_.append(float(alpha))
            if error < 1e-12:
                break
            weights *= np.exp(np.clip(alpha * wrong, -100, 100))
            weights /= weights.sum()
        return self

    def decision_function(self, X):
        X = self._check(X)
        if not self.trees_:
            return np.tile(self.priors_, (len(X), 1))
        scores = np.zeros((len(X), self.n_classes_))
        for tree, alpha in zip(self.trees_, self.alphas_):
            scores[np.arange(len(X)), tree_predict(tree, X).argmax(1)] += alpha
        return scores


class GradientBoosting(Classifier):
    """Descente fonctionnelle de l'entropie croisée, arbres sur résidus one-hot - softmax."""
    def __init__(self, n_estimators=25, learning_rate=0.15, max_depth=2, random_state=42):
        self.n_estimators, self.learning_rate, self.max_depth, self.random_state = n_estimators, learning_rate, max_depth, random_state

    def fit(self, X, y):
        X, y = self._prepare(X, y)
        rng = np.random.default_rng(self.random_state)
        self.init_ = np.log(np.bincount(y, minlength=self.n_classes_) / len(y))
        scores = np.tile(self.init_, (len(X), 1))
        self.stages_, self.history_ = [], []
        for _ in range(self.n_estimators):
            residual = np.eye(self.n_classes_)[y] - softmax(scores)
            stage = []
            for c in range(self.n_classes_):
                tree = grow_tree(X, residual[:, c], rng, self.max_depth, 2, None, regression=True)
                scores[:, c] += self.learning_rate * tree_predict(tree, X)
                stage.append(tree)
            self.stages_.append(stage)
            self.history_.append(cross_entropy(scores, y))
        return self

    def decision_function(self, X):
        X = self._check(X)
        scores = np.tile(self.init_, (len(X), 1))
        for stage in self.stages_:
            for c, tree in enumerate(stage):
                scores[:, c] += self.learning_rate * tree_predict(tree, X)
        return scores

    def predict_proba(self, X):
        return softmax(self.decision_function(X))
