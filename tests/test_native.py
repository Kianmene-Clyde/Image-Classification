import unittest
import numpy as np
from signlab.native import library_path, NativeClassifier


@unittest.skipUnless(library_path().is_file(), "Bibliothèque native non compilée ; exécuter scripts/build_native.py")
class NativeTests(unittest.TestCase):
    def test_four_models_learn_separable_data(self):
        rng = np.random.default_rng(8)
        X = np.vstack([rng.normal(-1, 0.1, (15, 2)), rng.normal(1, 0.1, (15, 2))])
        y = np.repeat([0, 1], 15)
        for kind in range(4):
            with self.subTest(kind=kind):
                model = NativeClassifier(kind=kind, hidden_units=8, epochs=500).fit(X, y)
                self.assertGreater(model.score(X, y), 0.95)
                self.assertTrue(np.isfinite(model.native_state_).all())
