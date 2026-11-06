from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
from PIL import Image
from scripts.synthetic_demo import create_fixture
from signlab.artifacts import read_json, predict_image, load_bundle
from signlab.data import load_dataset, group_split, limit_training_groups
from signlab.experiment import benchmark, finalize
from signlab.features import Preprocessor
from signlab.metrics import classification_metrics
from signlab.server import create_server


def fixture_config(root):
    return {"data_dir": str(root), "class_names": ["stop", "sens_interdit", "limitation_50"],
            "image_size": 8, "features": "pixels", "pca_components": 6, "near_duplicate_distance": -1,
            "validation_fraction": 0.2, "test_fraction": 0.2, "split_seed": 42,
            "seeds": [17, 42], "models": ["majority", "logistic", "decision_tree"], "data_kind": "synthetic"}


class PipelineTests(unittest.TestCase):
    def test_empty_placeholders_raise_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in fixture_config(root)["class_names"]:
                (root / name).mkdir()
            with self.assertRaisesRegex(ValueError, "Dataset vide"):
                load_dataset(fixture_config(root))

    def test_preprocessing_train_only_and_constant_feature(self):
        train = np.array([[2, 4], [2, 6], [2, 8]], dtype=float)
        test = np.array([[999, 1000]], dtype=float)
        pre = Preprocessor("standard", None).fit(train)
        before = pre.offset_.copy()
        transformed = pre.transform(test)
        np.testing.assert_array_equal(before, pre.offset_)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertGreater(transformed[0, 1], 100)

    def test_deduplication_and_group_disjointness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_fixture(root, per_class=10)
            image = root / "stop" / "synthetic_000.png"
            (root / "stop" / "copy.png").write_bytes(image.read_bytes())
            (root / "stop" / "broken.png").write_bytes(b"not an image")
            dataset = load_dataset(fixture_config(root))
            self.assertEqual(len(dataset.y), 30)
            self.assertEqual(len(dataset.issues), 2)
            split = group_split(dataset)
            groups = {name: {dataset.records[i]["group"] for i in idx} for name, idx in split.items()}
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["validation"] & groups["test"])
            self.assertEqual(sum(map(len, split.values())), len(dataset.y))
            for indices in split.values():
                self.assertEqual(set(dataset.y[indices]), {0, 1, 2})
            subset = limit_training_groups(dataset, split["train"], 9, seed=42)
            self.assertEqual(len(subset), 9)
            self.assertEqual(set(dataset.y[subset]), {0, 1, 2})
            self.assertTrue(set(subset).issubset(split["train"]))
            (root / "sens_interdit" / "contradiction.png").write_bytes(image.read_bytes())
            with self.assertRaisesRegex(ValueError, "contradictoires"):
                load_dataset(fixture_config(root))

    def test_metrics_known_confusion(self):
        metrics = classification_metrics([0, 0, 1, 2], [0, 1, 1, 2], ["a", "b", "c"])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 2 / 3 + 1) / 3)

    def test_full_pipeline_finalization_and_http(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_fixture(root / "images", per_class=12)
            config = fixture_config(root / "images")
            run = root / "run"
            rows = benchmark(config, run, quick=False, progress=lambda *_: None)
            self.assertEqual(len(rows), 3)
            self.assertFalse(read_json(run / "selection.json")["test_accessed"])
            self.assertFalse((run / "test_results.json").exists())
            selection = read_json(run / "selection.json")
            with Image.open(root / "images" / "stop" / "synthetic_000.png") as image:
                prediction = predict_image(load_bundle(run / selection["artifact"]), image)
            self.assertEqual(prediction["data_kind"], "synthetic")
            results = finalize(run, all_models=True)
            self.assertEqual(len(results), 3)
            with self.assertRaisesRegex(ValueError, "déjà évalué"):
                finalize(run)
            server = create_server(run, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                self.assertEqual(json.load(urlopen(base + "/api/health"))["trained_models"], 3)
                page = urlopen(base + "/").read().decode()
                self.assertIn("SignLab", page)
                for asset, media_type in (("app.css", "text/css"), ("app.js", "text/javascript")):
                    self.assertIn(asset, page)
                    with urlopen(base + "/" + asset) as response:
                        self.assertEqual(response.headers.get_content_type(), media_type)
                        self.assertTrue(response.read())
                raw = (root / "images" / "stop" / "synthetic_000.png").read_bytes()
                response = json.load(urlopen(Request(base + "/api/predict?model=logistic", raw, {"Content-Type": "image/png"})))
                self.assertIn(response["class"], config["class_names"])
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + "/api/predict?model=logistic", b"invalid", {"Content-Type": "image/png"}))
                self.assertEqual(error.exception.code, 400)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + "/api/predict?model=logistic", raw, {"Content-Type": "image/png", "Origin": "https://example.com"}))
                self.assertEqual(error.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
