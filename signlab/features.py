"""Représentations d'images et transformations apprises sur le train uniquement."""

import numpy as np
from PIL import Image, ImageOps
from .models.base import matrix


def image_vector(image, image_size=16, features="pixels"):
    if not isinstance(image_size, int) or not 4 <= image_size <= 128:
        raise ValueError("image_size doit être un entier entre 4 et 128.")
    image = ImageOps.exif_transpose(image).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=float) / 255
    if features == "pixels":
        return rgb.ravel()
    if features != "hog":
        raise ValueError("features doit être 'pixels' ou 'hog'.")
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    bins = ((np.arctan2(gy, gx) % np.pi) * (8 / np.pi)).astype(int) % 8
    # HOG pédagogique : cellules 4x4, 8 orientations, normalisation L2 par cellule.
    histograms = []
    for row in range(0, image_size - 3, 4):
        for col in range(0, image_size - 3, 4):
            h = np.bincount(bins[row:row + 4, col:col + 4].ravel(),
                            weights=magnitude[row:row + 4, col:col + 4].ravel(), minlength=8)
            histograms.extend(h / np.sqrt(h @ h + 1e-10))
    return np.asarray(histograms)


class Preprocessor:
    def __init__(self, scale="standard", pca_components=24):
        self.scale, self.pca_components = scale, pca_components

    def fit(self, X):
        X = matrix(X)
        self.n_features_in_ = X.shape[1]
        self.center_ = X.mean(0)
        self.components_ = None
        self.explained_variance_ratio_ = []
        count = self.pca_components
        if count is not None:
            if not isinstance(count, int) or count < 1:
                raise ValueError("pca_components doit être positif ou null.")
            # SVD déterministe ; aucun accès à la validation ou au test.
            _, s, v = np.linalg.svd(X - self.center_, full_matrices=False)
            k = min(count, max(1, len(X) - 1), X.shape[1])
            self.components_ = v[:k].copy()
            self.explained_variance_ratio_ = (s[:k] ** 2 / max(float(s @ s), 1e-20)).tolist()
            Z = (X - self.center_) @ self.components_.T
        else:
            Z = X
        if self.scale == "standard":
            self.offset_, raw_scale = Z.mean(0), Z.std(0)
        elif self.scale == "minmax":
            self.offset_, raw_scale = Z.min(0), np.ptp(Z, axis=0)
        elif self.scale == "none":
            self.offset_, raw_scale = np.zeros(Z.shape[1]), np.ones(Z.shape[1])
        else:
            raise ValueError("Échelle inconnue.")
        self.divisor_ = np.where(raw_scale > 1e-10, raw_scale, 1)
        return self

    def transform(self, X):
        X = matrix(X)
        if not hasattr(self, "offset_"):
            raise ValueError("Prétraitement non ajusté.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Dimensions incompatibles avec le prétraitement sauvegardé.")
        Z = (X - self.center_) @ self.components_.T if self.components_ is not None else X
        Z = (Z - self.offset_) / self.divisor_
        return np.clip(Z, 0, 1) if self.scale == "minmax" else Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)
