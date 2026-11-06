"""Artefacts JSON typés, sans pickle ni import arbitraire à la lecture."""

import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .features import Preprocessor, image_vector
from .registry import SERIALIZABLE_CLASSES


def encode(value):
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if "__ndarray__" in value:
            dtype = np.dtype(value["dtype"])
            if dtype.kind not in "biufUS":
                raise ValueError("Type de tableau interdit dans l'artefact.")
            return np.asarray(value["__ndarray__"], dtype=dtype)
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(encode(payload), ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        handle.write(content)
        tmp_path = Path(handle.name)
    try:
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 200 * 1024 * 1024:
        raise ValueError("Artefact supérieur à 200 Mo.")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_bundle(path, model, preprocessor, metadata):
    write_json(path, {"schema_version": 1, "model_type": type(model).__name__,
                      "model_state": model.__dict__, "preprocessor_state": preprocessor.__dict__,
                      "metadata": metadata})


def load_bundle(path):
    payload = read_json(path)
    if payload.get("schema_version") != 1 or payload.get("model_type") not in SERIALIZABLE_CLASSES:
        raise ValueError("Format ou type de modèle non pris en charge.")
    model = SERIALIZABLE_CLASSES[payload["model_type"]].__new__(SERIALIZABLE_CLASSES[payload["model_type"]])
    model.__dict__.update(decode(payload["model_state"]))
    preprocessor = Preprocessor()
    preprocessor.__dict__.update(decode(payload["preprocessor_state"]))
    metadata = payload["metadata"]
    required = {"class_names", "image_size", "features", "dataset_fingerprint", "model_name"}
    if not required.issubset(metadata) or not hasattr(model, "classes_"):
        raise ValueError("Artefact incomplet.")
    if (model.n_features_in_ != len(preprocessor.offset_)
            or not np.array_equal(model.classes_, np.arange(len(metadata["class_names"])))):
        raise ValueError("Artefact incohérent : dimensions modèle/prétraitement.")
    return model, preprocessor, metadata


def predict_image(bundle, image):
    model, preprocessor, metadata = bundle
    raw = image_vector(image, metadata["image_size"], metadata["features"])[None, :]
    X = preprocessor.transform(raw)
    scores = model.decision_function(X)[0]
    label = int(model.classes_[np.argmax(scores)])
    return {"class": metadata["class_names"][label], "class_id": label,
            "model": metadata["model_name"], "scores": scores.tolist(),
            "class_order": metadata["class_names"],
            "score_notice": "Scores propres au modèle, non calibrés ; ne pas les lire comme des probabilités fiables.",
            "data_kind": metadata.get("data_kind", "real"),
            "limitations": "Classification de 3 catégories ; absence de détection de panneaux ou de classe inconnue."}
