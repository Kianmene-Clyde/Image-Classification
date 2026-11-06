"""Commandes du laboratoire."""
import argparse
import json
from pathlib import Path
import sys
from PIL import Image
from .artifacts import load_bundle, predict_image, write_json
from .data import load_dataset, group_split
from .experiment import benchmark, diagnose, finalize, load_config
from .registry import CATALOG


def parser():
    p = argparse.ArgumentParser(prog="signlab", description="Comparer et comprendre les classifieurs de panneaux.")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog", help="Lister les modèles et leurs hypothèses")
    for command in ("audit-data", "benchmark", "diagnose"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--config", default="configs/default.json")
        if command == "audit-data":
            cmd.add_argument("--output", default="runs/data-audit.json")
        else:
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--quick", action="store_true", help="Un candidat et une graine : exploration uniquement")
            if command == "benchmark":
                cmd.add_argument("--models", nargs="+", choices=list(CATALOG))
            else:
                cmd.add_argument("--model", required=True, choices=list(CATALOG))
    final = sub.add_parser("finalize")
    final.add_argument("--run", required=True)
    final.add_argument("--all-models", action="store_true", help="Évaluer tous les modèles déjà figés, sans nouvelle sélection")
    pred = sub.add_parser("predict")
    pred.add_argument("--artifact", required=True)
    pred.add_argument("--image", required=True)
    server = sub.add_parser("serve")
    server.add_argument("--run")
    server.add_argument("--port", type=int, default=8000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "catalog":
            for name, spec in CATALOG.items():
                print(f"{name:20} {spec.family:14} {spec.description}")
        elif args.command == "audit-data":
            config = load_config(args.config)
            dataset = load_dataset(config)
            split = group_split(dataset, config["validation_fraction"], config["test_fraction"], config["split_seed"])
            write_json(args.output, {"fingerprint": dataset.fingerprint, "class_names": dataset.class_names,
                                     "records": dataset.records, "issues": dataset.issues,
                                     "splits": {k: v.tolist() for k, v in split.items()}})
            print(f"{len(dataset.y)} images valides ; audit écrit dans {args.output}")
        elif args.command == "benchmark":
            rows = benchmark(load_config(args.config), args.output, args.models, args.quick)
            print(f"{len(rows)} modèles comparés. Résultats : {Path(args.output).resolve()}")
        elif args.command == "diagnose":
            diagnose(load_config(args.config), args.model, args.output, args.quick)
            print(f"Courbes train/validation : {args.output}")
        elif args.command == "finalize":
            print(json.dumps(finalize(args.run, args.all_models), ensure_ascii=False, indent=2))
        elif args.command == "predict":
            with Image.open(args.image) as image:
                image.load()
                result = predict_image(load_bundle(args.artifact), image)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "serve":
            from .server import serve
            serve(args.run, args.port)
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
