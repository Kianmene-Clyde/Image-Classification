"""Fixtures techniques séparées des dossiers réels ; ne simulent pas un vrai benchmark."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw
from signlab.artifacts import write_json


def create_fixture(root, per_class=24, seed=123):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = ["stop", "sens_interdit", "limitation_50"]
    for label, name in enumerate(names):
        folder = root / name
        folder.mkdir(exist_ok=True)
        for i in range(per_class):
            background = rng.integers(20, 180, size=(32, 32, 3), dtype=np.uint8)
            image = Image.fromarray(background)
            draw = ImageDraw.Draw(image)
            x, y = int(rng.integers(4, 9)), int(rng.integers(4, 9))
            if label == 0:
                draw.regular_polygon((16, 16, 11), 8, fill=(205, 25, 40), outline="white")
                draw.text((6, 12), "STOP", fill="white")
            elif label == 1:
                draw.ellipse((x, y, x + 20, y + 20), fill=(210, 25, 40))
                draw.rectangle((x + 3, y + 8, x + 17, y + 12), fill="white")
            else:
                draw.ellipse((x, y, x + 20, y + 20), fill="white", outline=(210, 25, 40), width=3)
                draw.text((x + 5, y + 5), "50", fill="black")
            image.save(folder / f"synthetic_{i:03}.png")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/synthetic-fixture")
    parser.add_argument("--per-class", type=int, default=24)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Choisissez un nouveau dossier de fixtures.")
    names = create_fixture(out / "images", args.per_class)
    config = json.loads(Path("configs/default.json").read_text(encoding="utf-8"))
    config.update(data_dir="images", class_names=names, data_kind="synthetic", image_size=8,
                  pca_components=8, near_duplicate_distance=-1)
    write_json(out / "config.json", config)
    print(f"Fixture SYNTHÉTIQUE uniquement : {out / 'config.json'}")


if __name__ == "__main__":
    main()
