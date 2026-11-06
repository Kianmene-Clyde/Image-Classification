"""Inventaire, traçabilité, déduplication et partitions groupées sans fuite."""

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from .features import image_vector

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    records: list
    issues: list
    class_names: list
    fingerprint: str


def load_dataset(config):
    root = Path(config["data_dir"]).resolve()
    names = config["class_names"]
    if not root.is_dir():
        raise ValueError(f"Dataset absent : {root}. Remplissez les dossiers indiqués dans le README.")
    metadata = {}
    if config.get("metadata_csv"):
        with open(config["metadata_csv"], encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if "relative_path" not in row:
                    raise ValueError("Le CSV doit contenir relative_path,group,source,license.")
                metadata[row["relative_path"].replace("\\", "/")] = row
    records, vectors, labels, issues = [], [], [], []
    exact, hashes, smalls = {}, [], []
    aliases = {}
    def canonical(group):
        while group in aliases:
            group = aliases[group]
        return group

    def merge_groups(groups):
        groups = {canonical(group) for group in groups}
        representative = min(groups)
        for group in groups - {representative}:
            aliases[group] = representative
        return representative
    buckets = {}
    max_samples = int(config.get("max_samples", 20000))
    for label, name in enumerate(names):
        folder = root / name
        if not folder.is_dir():
            raise ValueError(f"Dossier de classe absent : {folder}")
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                issues.append({"path": str(path), "reason": "Lien hors dataset ignoré"})
                continue
            relative = path.relative_to(root).as_posix()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as source:
                        source.load()
                        image = ImageOps.exif_transpose(source).convert("RGB")
                        normalized_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                        thumbnail = np.asarray(image.resize((16, 16)), dtype=float) / 255
                        gray = np.asarray(image.convert("L").resize((9, 8)), dtype=int)
                        bits = (gray[:, 1:] > gray[:, :-1]).ravel()
                        dhash = int.from_bytes(np.packbits(bits).tobytes(), "big")
                        vector = image_vector(image, config["image_size"], config["features"])
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
                issues.append({"path": relative, "reason": f"Image rejetée : {exc}"})
                continue
            if normalized_hash in exact:
                previous = exact[normalized_hash]
                if previous["label"] != label:
                    raise ValueError(f"Même image avec labels contradictoires : {previous['path']} et {relative}")
                duplicate_group = metadata.get(relative, {}).get("group")
                if duplicate_group:
                    merge_groups({duplicate_group, previous["group"]})
                issues.append({"path": relative, "reason": "Doublon exact retiré", "duplicate_of": previous["path"]})
                continue
            if len(records) >= max_samples:
                raise ValueError(f"Dataset supérieur à max_samples={max_samples}. Augmentez la limite explicitement ; aucun sous-échantillonnage silencieux.")
            row = metadata.get(relative, {})
            group = canonical(row.get("group") or f"image:{normalized_hash}")
            candidate_indices = set()
            for chunk in range(4):
                candidate_indices.update(buckets.get((chunk, (dhash >> (16 * chunk)) & 65535), []))
            similar = []
            threshold = int(config.get("near_duplicate_distance", 2))
            for index in sorted(candidate_indices):
                if (dhash ^ hashes[index]).bit_count() <= threshold and np.mean((thumbnail - smalls[index]) ** 2) < 0.003:
                    similar.append(index)
            if similar:
                # Union des groupes, y compris métadonnées : une seule composante par scène liée.
                group = merge_groups({records[i]["group"] for i in similar} | {group})
                issues.append({"path": relative, "reason": "Quasi-doublon regroupé", "similar_to": [records[i]["path"] for i in similar[:10]]})
            record = {"path": relative, "label": label, "class": name, "group": group,
                      "sha256_pixels": normalized_hash, "sha256_file": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "source": row.get("source", "non documentée"), "license": row.get("license", "non documentée")}
            records.append(record)
            exact[normalized_hash] = record
            hashes.append(dhash)
            smalls.append(thumbnail)
            for chunk in range(4):
                buckets.setdefault((chunk, (dhash >> (16 * chunk)) & 65535), []).append(len(records) - 1)
            vectors.append(vector)
            labels.append(label)
    counts = np.bincount(labels, minlength=len(names))
    if not len(records) or np.any(counts == 0):
        raise ValueError("Dataset vide ou classe sans image valide : " + ", ".join(f"{n}={counts[i]}" for i, n in enumerate(names)) + ". Ajoutez vos images aux dossiers vides.")
    for record in records:
        record["group"] = canonical(record["group"])
    X = np.array(vectors, dtype=float)
    if X.shape[1] > int(config.get("max_features", 4096)):
        raise ValueError("Trop de variables ; réduisez image_size, utilisez HOG ou relevez max_features.")
    fingerprint = hashlib.sha256("\n".join(f"{r['path']}:{r['sha256_file']}:{r['label']}:{r['group']}" for r in records).encode()).hexdigest()
    return Dataset(X, np.array(labels), records, issues, names, fingerprint)


def group_split(dataset, validation_fraction=0.2, test_fraction=0.2, seed=42):
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1 and validation_fraction + test_fraction < 1):
        raise ValueError("Fractions train/validation/test invalides.")
    groups = {}
    for i, record in enumerate(dataset.records):
        groups.setdefault(record["group"], []).append(i)
    k = len(dataset.class_names)
    group_ids = sorted(groups)
    group_counts = np.array([np.bincount(dataset.y[groups[g]], minlength=k) for g in group_ids])
    if np.any((group_counts > 0).sum(axis=0) < 3):
        raise ValueError("Chaque classe doit avoir au moins 3 groupes indépendants pour train/validation/test.")
    fractions = np.array([1 - validation_fraction - test_fraction, validation_fraction, test_fraction])
    target = fractions[:, None] * np.bincount(dataset.y, minlength=k)
    divisor = np.maximum(target, 1)
    sizes = group_counts.sum(axis=1)
    best = None
    rng = np.random.default_rng(seed)
    # Recherches déterministes d'une allocation approchant la stratification sans casser les groupes.
    for _ in range(12 if len(group_ids) > 1000 else 100):
        counts = np.zeros((3, k))
        assignment = [[] for _ in range(3)]
        order = rng.permutation(len(group_ids))
        order = sorted(order, key=lambda j: -sizes[j])
        for j in order:
            delta = group_counts[j]
            costs = np.sum(delta * (2 * (counts - target) + delta) / divisor, axis=1)
            part = int(np.argmin(costs))
            counts[part] += group_counts[j]
            assignment[part].extend(groups[group_ids[j]])
        if (counts > 0).all():
            cost = float(np.sum((counts - target) ** 2 / np.maximum(target, 1)))
            if best is None or cost < best[0]:
                best = cost, assignment
            if cost < 1e-12:
                break
    if best is None:
        raise ValueError("Impossible de conserver chaque classe dans les 3 partitions groupées ; ajoutez des groupes indépendants.")
    return {name: np.array(sorted(indices), dtype=int) for name, indices in zip(("train", "validation", "test"), best[1])}


def limit_training_groups(dataset, indices, limit, seed):
    """Sous-cohorte commune, reproductible et proportionnelle ; groupes indivisibles."""
    if limit is None or len(indices) <= limit:
        return indices
    names = len(dataset.class_names)
    if limit < names:
        raise ValueError("train_limit doit permettre au moins une image par classe.")
    groups = {}
    for index in indices:
        groups.setdefault(dataset.records[index]["group"], []).append(int(index))
    targets = np.bincount(dataset.y[indices], minlength=names) * limit / len(indices)
    counts = np.zeros(names, dtype=int)
    chosen = []
    rng = np.random.default_rng(seed)
    eligible = [(g, np.bincount(dataset.y[g], minlength=names)) for g in groups.values() if len(g) <= limit]
    for label in np.argsort(targets):
        if counts[label]:
            continue
        options = [(group, distribution) for group, distribution in eligible
                   if distribution[label] and len(chosen) + len(group) <= limit]
        if not options:
            raise ValueError("Impossible de garder toutes les classes en respectant train_limit et les groupes.")
        group, distribution = min(options, key=lambda option: len(option[0]))
        chosen.extend(group)
        counts += distribution
    selected = set(chosen)
    leftovers = []
    for i in rng.permutation(len(eligible)):
        group, distribution = eligible[i]
        if group[0] in selected:
            continue
        if np.any((counts + distribution > np.ceil(targets)) & (distribution > 0)):
            leftovers.append((group, distribution))
        elif len(chosen) + len(group) <= limit:
            chosen.extend(group)
            counts += distribution
    for group, _ in leftovers:
        if len(chosen) + len(group) <= limit:
            chosen.extend(group)
    return np.array(sorted(chosen), dtype=int)
