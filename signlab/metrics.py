"""Métriques calculées sur toutes les observations, ordre des classes explicite."""

import numpy as np


def classification_metrics(y, predicted, class_names):
    y, predicted = np.asarray(y, dtype=int), np.asarray(predicted, dtype=int)
    k = len(class_names)
    if y.ndim != 1 or predicted.shape != y.shape or not len(y):
        raise ValueError("Cibles et prédictions incompatibles.")
    if min(y.min(), predicted.min()) < 0 or max(y.max(), predicted.max()) >= k:
        raise ValueError("Indice de classe hors limites.")
    confusion = np.zeros((k, k), dtype=int)
    np.add.at(confusion, (y, predicted), 1)
    tp = np.diag(confusion)
    precision = np.divide(tp, confusion.sum(0), out=np.zeros(k, dtype=float), where=confusion.sum(0) != 0)
    recall = np.divide(tp, confusion.sum(1), out=np.zeros(k, dtype=float), where=confusion.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=precision + recall != 0)
    return {"accuracy": float(tp.sum() / len(y)), "macro_f1": float(f1.mean()),
            "confusion_matrix": confusion.tolist(), "n_samples": len(y),
            "per_class": [{"class": name, "precision": float(precision[i]), "recall": float(recall[i]),
                            "f1": float(f1[i]), "support": int(confusion[i].sum())} for i, name in enumerate(class_names)]}
