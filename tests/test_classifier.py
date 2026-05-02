# -*- coding: utf-8 -*-
import unittest
import numpy as np

from gmdhpy.gmdh import Classifier


class ClassifierExtraTestCase(unittest.TestCase):

    def test_only_binary_supported(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(-1, 1, size=(60, 4))
        y = rng.randint(0, 3, size=60)  # 3 classes
        model = Classifier(
            ref_functions="linear_cov",
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        self.assertRaises(ValueError, model.fit, x, y)

    def test_predict_returns_original_labels(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(-1, 1, size=(80, 4))
        y = np.array(["neg", "pos"])[(x[:, 0] + x[:, 1] > 0).astype(int)]
        model = Classifier(
            ref_functions="linear_cov",
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(x, y)

        pred = model.predict(x)
        self.assertEqual(pred.shape, (x.shape[0],))
        self.assertTrue(set(np.unique(pred)).issubset({"neg", "pos"}))

        proba = model.predict_proba(x)
        expected = model.le.inverse_transform((proba >= 0.5).astype(int))
        np.testing.assert_array_equal(pred, expected)

        # custom threshold: 1.0 forces every sample below threshold -> negative class
        pred_all_neg = model.predict(x, threshold=1.0)
        self.assertTrue(np.all(pred_all_neg == "neg"))


if __name__ == "__main__":
    unittest.main()