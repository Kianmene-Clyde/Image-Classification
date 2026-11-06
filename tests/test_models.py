import tempfile
import unittest
from pathlib import Path
import numpy as np
from signlab.artifacts import load_bundle, save_bundle
from signlab.features import Preprocessor
from signlab.models.base import softmax
from signlab.models.linear import Logistic, LinearSVM
from signlab.models.nonlinear import MLP, KernelSVM, kmeans
from signlab.registry import CATALOG, build_model


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        # Directions distinctes : les modèles de fréquences ne peuvent pas séparer
        # des classes qui diffèrent seulement par leur intensité totale.
        cls.X = np.vstack([rng.normal(0, 0.15, (18, 3)) + v for v in ((2, 0, 0), (0, 2, 0), (0, 0, 2))])
        cls.y = np.repeat([0, 1, 2], 18)

    def test_all_classifiers_learn_and_roundtrip(self):
        for name in CATALOG:
            if name.startswith("native_"):
                continue
            with self.subTest(model=name), tempfile.TemporaryDirectory() as tmp:
                pre = Preprocessor(CATALOG[name].scale, None)
                X = pre.fit_transform(self.X)
                model = build_model(name).fit(X, self.y)
                predicted = model.predict(X)
                self.assertEqual(predicted.shape, self.y.shape)
                self.assertTrue(np.isfinite(model.decision_function(X)).all())
                if name != "majority":
                    self.assertGreaterEqual(float(np.mean(predicted == self.y)), 0.8)
                path = Path(tmp) / "model.json"
                save_bundle(path, model, pre, {"class_names": ["a", "b", "c"], "image_size": 4,
                                              "features": "pixels", "dataset_fingerprint": "test", "model_name": name})
                restored, restored_pre, _ = load_bundle(path)
                np.testing.assert_array_equal(restored.predict(restored_pre.transform(self.X)), predicted)
                np.testing.assert_allclose(restored.decision_function(X), model.decision_function(X), atol=1e-12)

    def test_labels_and_invalid_inputs(self):
        for name in ("logistic", "knn", "mlp", "random_forest"):
            with self.subTest(model=name):
                model = build_model(name)
                with self.assertRaises(ValueError):
                    model.predict(self.X)
                model.fit(self.X, np.array(["a", "b", "c"])[self.y])
                self.assertTrue(set(model.predict(self.X)) <= {"a", "b", "c"})
                with self.assertRaises(ValueError):
                    model.predict([[float("nan")] * 3])
                with self.assertRaises(ValueError):
                    model.predict([[1, 2]])
                with self.assertRaises(ValueError):
                    model.fit(self.X, np.zeros(len(self.X)))

    def test_softmax_extreme_values(self):
        probs = softmax(np.array([[10000, 9999], [-10000, -9999]]))
        np.testing.assert_allclose(probs.sum(1), 1)
        self.assertTrue(np.isfinite(probs).all())

    def test_reproducible_rng_does_not_touch_global_state(self):
        np.random.seed(219)
        expected = np.random.random(4)
        np.random.seed(219)
        first = build_model("mlp", seed=7).fit(self.X, self.y)
        np.testing.assert_array_equal(np.random.random(4), expected)
        second = build_model("mlp", seed=7).fit(self.X, self.y)
        np.testing.assert_allclose(first.W1_, second.W1_, atol=0)

    def test_xor_nonlinear_models_and_linear_limit(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        self.assertLessEqual(build_model("logistic").fit(X, y).score(X, y), 0.75)
        for model in (KernelSVM(C=10, gamma=2), MLP(hidden_units=12, epochs=1500, learning_rate=0.15)):
            with self.subTest(model=type(model).__name__):
                self.assertEqual(model.fit(X, y).score(X, y), 1.0)

    def test_svm_dual_constraints(self):
        model = KernelSVM(C=2, gamma=1).fit(self.X, self.y)
        for diagnostic in model.diagnostics_:
            self.assertLess(diagnostic["dual_equality_error"], 1e-7)
            self.assertLess(diagnostic["max_kkt_violation"], 0.03)

    def test_kmeans_degenerate_data(self):
        centers = kmeans(np.ones((8, 3)), 5)
        self.assertTrue(np.isfinite(centers).all())


class GradientTests(unittest.TestCase):
    def test_logistic_gradient_finite_differences(self):
        rng = np.random.default_rng(9)
        X, y = rng.normal(size=(5, 3)), np.array([0, 1, 2, 1, 0])
        model = Logistic(l2=0.03)
        params = [rng.normal(size=(3, 3)), rng.normal(size=3)]
        _, dw, db = model.loss_gradient(X, y, *params)
        self.check_gradients(lambda: model.loss_gradient(X, y, *params)[0], params, (dw, db))

    def test_mlp_gradient_finite_differences(self):
        rng = np.random.default_rng(5)
        X, y = rng.normal(size=(4, 2)), np.array([0, 1, 1, 0])
        model = MLP(l2=0.02)
        params = [rng.normal(size=(2, 3)) * 0.2, rng.normal(size=3) * 0.2,
                  rng.normal(size=(3, 2)) * 0.2, rng.normal(size=2) * 0.2]
        _, gradients = model.loss_gradients(X, y, *params)
        self.check_gradients(lambda: model.loss_gradients(X, y, *params)[0], params, gradients)

    def test_hinge_gradient_away_from_kinks(self):
        rng = np.random.default_rng(4)
        X, y = rng.normal(size=(5, 2)), np.array([0, 1, 2, 1, 0])
        model = LinearSVM(l2=0.03)
        params = [rng.normal(size=(2, 3)) * 0.1, rng.normal(size=3) * 0.1]
        _, dw, db = model.loss_gradient(X, y, *params)
        self.check_gradients(lambda: model.loss_gradient(X, y, *params)[0], params, (dw, db))

    def check_gradients(self, loss, params, analytic):
        eps = 1e-6
        for p, expected in zip(params, analytic):
            numerical = np.zeros_like(p)
            for index in np.ndindex(p.shape):
                original = p[index]
                p[index] = original + eps
                plus = loss()
                p[index] = original - eps
                minus = loss()
                p[index] = original
                numerical[index] = (plus - minus) / (2 * eps)
            np.testing.assert_allclose(numerical, expected, atol=2e-7, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
