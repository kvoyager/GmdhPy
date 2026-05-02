# -*- coding: utf-8 -*-
import unittest
import numpy as np

from gmdhpy.gmdh import Regressor
from gmdhpy.neuron import RefFunctionType, CriterionType


def _make_regression_data(n_samples=80, n_features=4, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, size=(n_samples, n_features))
    y = (
        2.0 * x[:, 0]
        - 1.5 * x[:, 1] * x[:, 2]
        + 0.5 * x[:, 3] ** 2
        + 0.1 * rng.randn(n_samples)
    )
    return x, y


class RegressorTestCase(unittest.TestCase):

    def setUp(self):
        self.x, self.y = _make_regression_data()

    def test_fit_predict_basic(self):
        model = Regressor(
            ref_functions="linear_cov",
            max_layer_count=3,
            criterion_minimum_width=2,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertTrue(model.valid)
        pred = model.predict(self.x)
        self.assertEqual(pred.shape[0], self.x.shape[0])

    def test_predict_before_fit_raises(self):
        model = Regressor(verbose=0)
        self.assertRaises(ValueError, model.predict, self.x)

    def test_set_of_ref_functions(self):
        model = Regressor(
            ref_functions=("linear", "linear_cov", "quadratic", "cubic"),
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertTrue(model.valid)

    def test_ref_function_enum_passthrough(self):
        model = Regressor(
            ref_functions=RefFunctionType.rfLinear,
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertTrue(model.valid)

    def test_normalize_false(self):
        model = Regressor(
            ref_functions="linear_cov",
            normalize=False,
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        pred = model.predict(self.x)
        self.assertEqual(pred.shape[0], self.x.shape[0])

    def test_admix_features_false(self):
        model = Regressor(
            ref_functions="quadratic",
            admix_features=False,
            max_layer_count=3,
            criterion_minimum_width=2,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertTrue(model.valid)

    def test_validation_data_path(self):
        x_train, y_train = self.x[:60], self.y[:60]
        x_val, y_val = self.x[60:], self.y[60:]
        model = Regressor(
            ref_functions="linear_cov",
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(x_train, y_train, validation_data=(x_val, y_val))
        self.assertTrue(model.valid)

    def test_layer_err_avg(self):
        model = Regressor(
            ref_functions="linear_cov",
            layer_err_criterion="avg",
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertTrue(model.valid)

    def test_layer_err_invalid_raises(self):
        model = Regressor(
            ref_functions="linear_cov",
            layer_err_criterion="bogus",
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        self.assertRaises(NotImplementedError, model.fit, self.x, self.y)

    def test_keep_partial_neurons(self):
        model = Regressor(
            ref_functions="linear_cov",
            keep_partial_neurons=True,
            max_layer_count=3,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertTrue(model.valid)

    def test_manual_best_neurons_selection(self):
        model = Regressor(
            ref_functions="linear_cov",
            manual_best_neurons_selection=True,
            min_best_neurons_count=3,
            max_best_neurons_count=8,
            max_layer_count=3,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertTrue(model.valid)

    def test_n_jobs_max(self):
        model = Regressor(
            ref_functions="linear_cov",
            n_jobs="max",
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertTrue(model.valid)

    def test_n_jobs_invalid_string_raises(self):
        self.assertRaises(
            ValueError, Regressor, ref_functions="linear_cov", n_jobs="foo"
        )

    def test_describe_methods(self):
        model = Regressor(
            ref_functions="linear_cov",
            feature_names=["a", "b", "c", "d"],
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        s = model.describe()
        self.assertIn("Model", s)
        s = model.describe_layer(0)
        self.assertIsInstance(s, str)
        s = model.describe_neuron(0, 0)
        self.assertIsInstance(s, str)

    def test_feature_names_as_ndarray(self):
        names = np.array(["a", "b", "c", "d"])
        model = Regressor(
            ref_functions="linear_cov",
            feature_names=names,
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertIsInstance(model.feature_names, list)

    def test_get_selected_features(self):
        model = Regressor(
            ref_functions="linear_cov",
            feature_names=["a", "b", "c", "d"],
            max_layer_count=3,
            criterion_minimum_width=2,
            verbose=0,
        )
        model.fit(self.x, self.y)
        selected = model.get_selected_features_indices()
        self.assertIsInstance(selected, list)
        unselected = model.get_unselected_features_indices()
        self.assertIsInstance(unselected, list)
        # Names should be retrievable
        self.assertIsInstance(model.get_selected_features(), str)
        unsel = model.get_unselected_features()
        self.assertIsInstance(unsel, str)

    def test_get_features_no_names(self):
        model = Regressor(
            ref_functions="linear_cov",
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        s = model.get_selected_features()
        self.assertIsInstance(s, str)

    def test_predict_neuron_output(self):
        model = Regressor(
            ref_functions="linear_cov",
            max_layer_count=3,
            criterion_minimum_width=2,
            verbose=0,
        )
        model.fit(self.x, self.y)
        out = model.predict_neuron_output(self.x, 0, 0)
        self.assertEqual(out.shape[0], self.x.shape[0])

    def test_predict_neuron_output_invalid_indices(self):
        model = Regressor(
            ref_functions="linear_cov",
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertRaises(ValueError, model.predict_neuron_output, self.x, 99, 0)
        self.assertRaises(ValueError, model.predict_neuron_output, self.x, 0, 99)
        self.assertRaises(ValueError, model.predict_neuron_output, self.x, -1, 0)

    def test_bias_retrain_criterion(self):
        model = Regressor(
            ref_functions="linear_cov",
            criterion_type="bias_retrain",
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertTrue(model.valid)

    def test_validate_bias_criterion(self):
        model = Regressor(
            ref_functions="linear_cov",
            criterion_type=CriterionType.cmpComb_validate_bias,
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertTrue(model.valid)

    def test_seq_type_random(self):
        model = Regressor(
            ref_functions="linear_cov",
            seq_type="random",
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y)
        self.assertTrue(model.valid)

    def test_str_returns_description(self):
        model = Regressor(verbose=0)
        self.assertIn("polynomial", str(model))

    def test_verbose_in_fit(self):
        model = Regressor(
            ref_functions="linear_cov",
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(self.x, self.y, verbose=0)
        self.assertTrue(model.valid)


if __name__ == "__main__":
    unittest.main()
