"""Sélection sur validation ; évaluation du test dans une commande distincte."""

import csv
from datetime import datetime, timezone
from pathlib import Path
import platform
import time

import numpy as np
import PIL
from . import __version__
from .artifacts import load_bundle, read_json, save_bundle, write_json
from .data import load_dataset, group_split, limit_training_groups
from .features import Preprocessor
from .metrics import classification_metrics
from .plots import line_plot, confusion_plot
from .registry import CATALOG, build_model


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    for key in ("data_dir", "metadata_csv"):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    names = config.get("class_names", [])
    if len(names) < 2 or len(set(names)) != len(names) or any(not isinstance(n, str) or Path(n).name != n or n in {".", ".."} for n in names):
        raise ValueError("class_names doit contenir au moins deux noms de dossiers distincts.")
    if not config.get("seeds") or not all(isinstance(s, int) for s in config["seeds"]):
        raise ValueError("Une liste de graines entières est nécessaire.")
    size = config.get("image_size")
    if not isinstance(size, int) or not 4 <= size <= 128 or config.get("features") not in {"pixels", "hog"}:
        raise ValueError("Choisissez pixels ou hog et une taille d'image entière entre 4 et 128.")
    limit = config.get("train_limit")
    if limit is not None and (not isinstance(limit, int) or limit < len(names)):
        raise ValueError("train_limit doit être null ou un entier au moins égal au nombre de classes.")
    config.setdefault("data_kind", "real")
    if config["data_kind"] not in {"real", "synthetic"}:
        raise ValueError("data_kind doit être real ou synthetic.")
    return config


def _models(config, requested):
    names = list(requested or config["models"])
    if not names or len(set(names)) != len(names):
        raise ValueError("Liste de modèles vide ou dupliquée.")
    unknown = set(names) - CATALOG.keys()
    if unknown:
        raise ValueError(f"Modèles inconnus : {sorted(unknown)}")
    return names


def benchmark(config, output_dir, models=None, quick=False, progress=print):
    names = _models(config, models)
    dataset = load_dataset(config)
    split = group_split(dataset, config["validation_fraction"], config["test_fraction"], config["split_seed"])
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Le dossier de résultat doit être nouveau ou vide, pour préserver les expériences.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    write_json(out / "config.json", config)
    write_json(out / "dataset.json", {"fingerprint": dataset.fingerprint, "records": dataset.records, "issues": dataset.issues,
                                      "class_names": dataset.class_names, "data_kind": config["data_kind"]})
    write_json(out / "splits.json", {key: value.tolist() for key, value in split.items()})
    write_json(out / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
                                          "numpy": np.__version__, "Pillow": PIL.__version__, "signlab": __version__,
                                          "utc": datetime.now(timezone.utc).isoformat()})
    train, validation = split["train"], split["validation"]
    train = limit_training_groups(dataset, train, config.get("train_limit"), config["split_seed"])
    write_json(out / "training_indices.json", train.tolist())
    seeds = config["seeds"][:1] if quick else config["seeds"]
    rows, trials, failures = [], [], []
    preprocessing_cache = {}
    for name in names:
        spec = CATALOG[name]
        progress(f"[{name}] sélection sur validation ({'rapide' if quick else str(len(seeds)) + ' graines'})")
        candidates = spec.candidates[:1] if quick else spec.candidates
        scale_key = (spec.scale, config.get("pca_components"))
        if scale_key not in preprocessing_cache:
            prep = Preprocessor(*scale_key)
            Xtrain = prep.fit_transform(dataset.X[train])
            Xval = prep.transform(dataset.X[validation])
            preprocessing_cache[scale_key] = (prep, Xtrain, Xval)
        prep, Xtrain, Xval = preprocessing_cache[scale_key]
        best = None
        for params_base in candidates:
            params = dict(params_base)
            if name.startswith("kernel_svm"):
                params["max_samples"] = config.get("kernel_max_samples", 2500)
            candidate_scores, candidate_trials = [], []
            first_model = None
            try:
                for seed in seeds:
                    model = build_model(name, params, seed)
                    start = time.perf_counter()
                    model.fit(Xtrain, dataset.y[train])
                    elapsed = time.perf_counter() - start
                    start = time.perf_counter()
                    prediction = model.predict(Xval)
                    latency = (time.perf_counter() - start) / len(Xval) * 1000
                    metrics = classification_metrics(dataset.y[validation], prediction, dataset.class_names)
                    train_metrics = classification_metrics(dataset.y[train], model.predict(Xtrain), dataset.class_names)
                    trial = {"model": name, "params": params, "seed": seed, "validation": metrics,
                             "train": train_metrics, "fit_seconds": elapsed, "latency_ms_per_image": latency,
                             "history": getattr(model, "history_", []), "diagnostics": getattr(model, "diagnostics_", None)}
                    candidate_trials.append(trial)
                    candidate_scores.append(metrics["macro_f1"])
                    if first_model is None:
                        first_model = model
                trials.extend(candidate_trials)
                mean_score = float(np.mean(candidate_scores))
                if best is None or mean_score > best["validation_macro_f1_mean"] + 1e-12:
                    best = {"model": name, "family": spec.family, "params": params, "seed": seeds[0],
                            "validation_macro_f1_mean": mean_score, "validation_macro_f1_std": float(np.std(candidate_scores)),
                            "validation_accuracy_mean": float(np.mean([t["validation"]["accuracy"] for t in candidate_trials])),
                            "train_macro_f1_mean": float(np.mean([t["train"]["macro_f1"] for t in candidate_trials])),
                            "fit_seconds_mean": float(np.mean([t["fit_seconds"] for t in candidate_trials])),
                            "latency_ms_per_image_mean": float(np.mean([t["latency_ms_per_image"] for t in candidate_trials])),
                            "n_seeds": len(seeds), "estimator": first_model, "trial": candidate_trials[0]}
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                failures.append({"model": name, "params": params, "error": str(exc)})
                progress(f"  candidat non exécuté : {exc}")
        write_json(out / "trials.json", trials)
        write_json(out / "failures.json", failures)
        if best is None:
            continue
        model, trial = best.pop("estimator"), best.pop("trial")
        artifact = out / "models" / f"{name}.json"
        metadata = {"model_name": name, "class_names": dataset.class_names, "features": config["features"],
                    "image_size": config["image_size"], "dataset_fingerprint": dataset.fingerprint,
                    "params": best["params"], "seed": seeds[0], "data_kind": config["data_kind"],
                    "partition": "train", "probability_calibration": "not_performed", "signlab_version": __version__}
        save_bundle(artifact, model, prep, metadata)
        best["artifact_bytes"] = artifact.stat().st_size
        best["artifact"] = f"models/{name}.json"
        rows.append(best)
        confusion_plot(out / "figures" / f"{name}_validation.svg", trial["validation"]["confusion_matrix"], dataset.class_names,
                       f"{name} - validation ({config['data_kind']})")
        if trial["history"]:
            line_plot(out / "figures" / f"{name}_loss.svg", {"train": trial["history"]}, f"{name} - fonction de coût d'entraînement")
    rows.sort(key=lambda r: (-r["validation_macro_f1_mean"], names.index(r["model"])))
    if not rows:
        raise ValueError(f"Aucun modèle entraîné. Voir {out / 'failures.json'}.")
    write_json(out / "comparison.json", rows)
    with (out / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [k for k in rows[0] if k != "params"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    selection = {"model": rows[0]["model"], "artifact": rows[0]["artifact"], "criterion": "mean validation macro_f1",
                 "test_accessed": False, "quick": quick, "data_kind": config["data_kind"],
                 "warning": "Validation utilisée pour la sélection : son score est optimiste. Test encore réservé."}
    write_json(out / "selection.json", selection)
    _summary(out, rows, failures, config, {**split, "train_effectif": train})
    return rows


def _summary(out, rows, failures, config, split):
    text = [f"# Comparaison sur validation ({config['data_kind']})", "",
            "Aucun score de test n'a été utilisé pour sélectionner les modèles.",
            "Le modèle sauvegardé emploie la première graine déclarée, sans choisir la meilleure graine.", "",
            f"Partitions : {', '.join(f'{k}={len(v)}' for k, v in split.items())}.", "",
            "| Modèle | Macro-F1 validation, moyenne ± écart-type | Temps ajustement (s) |",
            "|---|---:|---:|"]
    for row in rows:
        text.append(f"| {row['model']} | {row['validation_macro_f1_mean']:.4f} ± {row['validation_macro_f1_std']:.4f} | {row['fit_seconds_mean']:.3f} |")
    text.extend(["", "Les graines mesurent la variabilité de l'optimisation sur une partition fixe, pas l'incertitude de généralisation.",
                 "La latence mesure le batch de validation, hors lecture d'image et prétraitement.",
                 f"Candidats non exécutés : {len(failures)}. Consulter failures.json.", "",
                 "Les résultats historiques du rapport initial ne sont pas comparables à ce protocole sans reproduction."])
    if config["data_kind"] == "synthetic":
        text.extend(["", "DONNÉES SYNTHÉTIQUES : vérification technique uniquement, aucune preuve de performance sur des panneaux réels."])
    (out / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def finalize(run_dir, all_models=False):
    out = Path(run_dir)
    if (out / "test_results.json").exists():
        raise ValueError("Test déjà évalué pour cette expérience. Préservez ce résultat ; ne réglez pas les modèles à partir du test.")
    config = read_json(out / "config.json")
    dataset = load_dataset(config)
    original = read_json(out / "dataset.json")
    if dataset.fingerprint != original["fingerprint"]:
        raise ValueError("Dataset modifié depuis l'entraînement. Créez une nouvelle expérience.")
    indices = np.array(read_json(out / "splits.json")["test"], dtype=int)
    selection = read_json(out / "selection.json")
    rows = read_json(out / "comparison.json") if all_models else [selection]
    results = []
    for row in rows:
        model, prep, metadata = load_bundle(out / row["artifact"])
        if (metadata["dataset_fingerprint"] != dataset.fingerprint
                or metadata["class_names"] != dataset.class_names
                or metadata["model_name"] != row["model"]):
            raise ValueError(f"Artefact incohérent pour {row['model']}. Rechargez les fichiers de l'expérience d'origine.")
        X = prep.transform(dataset.X[indices])
        predicted = model.predict(X)
        metrics = classification_metrics(dataset.y[indices], predicted, dataset.class_names)
        errors = [{"path": dataset.records[i]["path"], "true": dataset.class_names[dataset.y[i]],
                   "predicted": dataset.class_names[int(p)]} for i, p in zip(indices, predicted) if p != dataset.y[i]]
        results.append({"model": row["model"], "metrics": metrics, "errors": errors, "data_kind": config["data_kind"]})
        confusion_plot(out / "figures" / f"{row['model']}_test.svg", metrics["confusion_matrix"], dataset.class_names,
                       f"{row['model']} - test final ({config['data_kind']})")
    write_json(out / "test_results.json", {"selection_frozen": selection, "results": results,
                                            "warning": "Ne pas modifier les choix à partir de ces scores. Une nouvelle itération requiert un nouveau test indépendant."})
    return results


def diagnose(config, model_name, output_dir, quick=False):
    """Courbe de taille et complexité sur train/validation ; aucun accès au test."""
    if model_name not in CATALOG:
        raise ValueError("Modèle inconnu.")
    dataset = load_dataset(config)
    split = group_split(dataset, config["validation_fraction"], config["test_fraction"], config["split_seed"])
    train, val = split["train"], split["validation"]
    train = limit_training_groups(dataset, train, config.get("train_limit"), config["split_seed"])
    rng = np.random.default_rng(config["split_seed"])
    ordered = {c: rng.permutation(train[dataset.y[train] == c]) for c in range(len(dataset.class_names))}
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Dossier diagnostics non vide.")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        indices = np.concatenate([items[:max(1, int(len(items) * fraction))] for items in ordered.values()])
        for number, params in enumerate(CATALOG[model_name].candidates[:1] if quick else CATALOG[model_name].candidates):
            for seed in config["seeds"][:1] if quick else config["seeds"]:
                prep = Preprocessor(CATALOG[model_name].scale, config.get("pca_components"))
                X = prep.fit_transform(dataset.X[indices])
                model = build_model(model_name, params, seed)
                start = time.perf_counter()
                model.fit(X, dataset.y[indices])
                results.append({"fraction": fraction, "n_train": len(indices), "candidate": number, "params": params, "seed": seed,
                                "fit_seconds": time.perf_counter() - start,
                                "train": classification_metrics(dataset.y[indices], model.predict(X), dataset.class_names),
                                "validation": classification_metrics(dataset.y[val], model.predict(prep.transform(dataset.X[val])), dataset.class_names),
                                "history": getattr(model, "history_", [])})
    write_json(out / "learning_curves.json", {"model": model_name, "data_kind": config["data_kind"], "test_accessed": False, "results": results})
    for number in sorted({row["candidate"] for row in results}):
        candidate_rows = [row for row in results if row["candidate"] == number]
        line_plot(out / f"learning_curve_{number}.svg", {
            part: [float(np.mean([r[part]["macro_f1"] for r in candidate_rows if r["fraction"] == fraction]))
                   for fraction in (0.25, 0.5, 0.75, 1.0)] for part in ("train", "validation")},
            f"{model_name} - taille train 25 / 50 / 75 / 100 % ({config['data_kind']})", "Macro-F1")
    return results
