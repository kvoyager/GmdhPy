# -*- coding: utf-8 -*-
import unittest
import numpy as np

from gmdhpy.neuron import (
    RefFunctionType,
    CriterionType,
    PolynomNeuron,
    Layer,
    LayerCreationError,
    fit_layer,
    FitLayerData,
)


class _DummyModel:
    """Minimal model stub for Layer constructor."""

    def __init__(self, l_count=4, n_features=4):
        self.l_count = l_count
        self.n_features = n_features


class RefFunctionTypeNamesTestCase(unittest.TestCase):

    def test_get_name_for_each(self):
        self.assertEqual(RefFunctionType.get_name(RefFunctionType.rfLinear), "Linear")
        self.assertEqual(
            RefFunctionType.get_name(RefFunctionType.rfLinearCov), "LinearCov"
        )
        self.assertEqual(
            RefFunctionType.get_name(RefFunctionType.rfQuadratic), "Quadratic"
        )
        self.assertEqual(RefFunctionType.get_name(RefFunctionType.rfCubic), "Cubic")
        self.assertEqual(RefFunctionType.get_name(RefFunctionType.rfUnknown), "Unknown")

    def test_get_passthrough(self):
        self.assertEqual(
            RefFunctionType.get(RefFunctionType.rfLinear),
            RefFunctionType.rfLinear,
        )


class CriterionTypeNamesTestCase(unittest.TestCase):

    def test_get_name_each(self):
        self.assertEqual(
            CriterionType.get_name(CriterionType.cmpValidate),
            "validate error comparison",
        )
        self.assertEqual(
            CriterionType.get_name(CriterionType.cmpBias), "bias error comparison"
        )
        self.assertEqual(
            CriterionType.get_name(CriterionType.cmpComb_validate_bias),
            "bias and validate error comparison",
        )
        self.assertEqual(
            CriterionType.get_name(CriterionType.cmpComb_bias_retrain),
            "bias error comparison with retrain",
        )

    def test_get_passthrough(self):
        self.assertEqual(
            CriterionType.get(CriterionType.cmpBias),
            CriterionType.cmpBias,
        )

    def test_get_alias(self):
        self.assertEqual(
            CriterionType.get("bias_refit"), CriterionType.cmpComb_bias_retrain
        )


class PolynomNeuronExtraTestCase(unittest.TestCase):

    def test_invalid_loss_raises(self):
        self.assertRaises(
            ValueError,
            PolynomNeuron,
            0,
            0,
            1,
            RefFunctionType.rfLinear,
            0,
            model_class="regression",
            loss="bogus",
        )

    def test_invalid_ftype_raises(self):
        self.assertRaises(
            ValueError,
            PolynomNeuron,
            0,
            0,
            1,
            RefFunctionType.rfUnknown,
            0,
            model_class="regression",
            loss="mse",
        )

    def test_get_name_short_name(self):
        for ftype, long_name, short_name in [
            (RefFunctionType.rfLinear, "w0 + w1*xi + w2*xj", "linear"),
            (
                RefFunctionType.rfLinearCov,
                "w0 + w1*xi + w2*xj + w3*xi*xj",
                "linear cov",
            ),
            (RefFunctionType.rfQuadratic, "full polynom 2nd degree", "quadratic"),
            (RefFunctionType.rfCubic, "full polynom 3rd degree", "cubic"),
        ]:
            n = PolynomNeuron(0, 0, 1, ftype, 0, model_class="regression", loss="mse")
            self.assertEqual(n.get_name(), long_name)
            self.assertEqual(n.get_short_name(), short_name)

    def test_repr(self):
        n = PolynomNeuron(
            0, 0, 1, RefFunctionType.rfLinear, 0, model_class="regression", loss="mse"
        )
        self.assertIn("PolynomModel", repr(n))

    def test_get_error_branches(self):
        n = PolynomNeuron(
            0, 0, 1, RefFunctionType.rfLinear, 0, model_class="regression", loss="mse"
        )
        n.validate_err = 0.2
        n.bias_err = 0.3
        self.assertAlmostEqual(n.get_error(CriterionType.cmpValidate), 0.2)
        self.assertAlmostEqual(n.get_error(CriterionType.cmpBias), 0.3)
        self.assertAlmostEqual(n.get_error(CriterionType.cmpComb_validate_bias), 0.25)
        self.assertAlmostEqual(n.get_error(CriterionType.cmpComb_bias_retrain), 0.3)
        # default get_error() with no arg -> sys.float_info.max-ish path
        n.validate_err = 0.42
        self.assertAlmostEqual(n.get_error(CriterionType.cmpValidate), 0.42)

    def test_need_bias_stuff(self):
        n = PolynomNeuron(
            0, 0, 1, RefFunctionType.rfLinear, 0, model_class="regression", loss="mse"
        )
        self.assertFalse(n.need_bias_stuff(CriterionType.cmpValidate))
        self.assertTrue(n.need_bias_stuff(CriterionType.cmpBias))

    def test_describe(self):
        n = PolynomNeuron(
            0, 0, 1, RefFunctionType.rfLinear, 0, model_class="regression", loss="mse"
        )
        n.w = np.array([1.0, 2.0, 3.0])
        s = n.describe([], [])
        self.assertIn("PolynomModel", s)
        self.assertIn("w0", s)

    def test_get_features_name_first_layer(self):
        n = PolynomNeuron(
            0, 0, 1, RefFunctionType.rfLinear, 0, model_class="regression", loss="mse"
        )
        s = n.get_features_name(0, ["x", "y"], [])
        self.assertIn("inp_0", s)

    def test_get_features_name_higher_layer(self):
        n = PolynomNeuron(
            1, 0, 1, RefFunctionType.rfLinear, 0, model_class="regression", loss="mse"
        )
        layers = [Layer(_DummyModel(), 0)]
        layers[0].append(
            PolynomNeuron(0, 0, 1, RefFunctionType.rfLinear, 0, "regression", "mse")
        )
        layers[0].append(
            PolynomNeuron(0, 0, 1, RefFunctionType.rfLinear, 1, "regression", "mse")
        )
        # input_index < neurons_num -> previous-layer-neuron branch
        s_prev = n.get_features_name(0, ["a", "b"], layers)
        self.assertIn("prev_layer_neuron", s_prev)
        # input_index >= neurons_num -> input-feature branch
        s_inp = n.get_features_name(2, ["a", "b"], layers)
        self.assertIn("inp_", s_inp)

    def test_fit_full_path(self):
        rng = np.random.RandomState(0)
        train_x = rng.uniform(-1, 1, size=(40, 3))
        train_y = train_x[:, 0] + 0.5 * train_x[:, 1]
        validate_x = rng.uniform(-1, 1, size=(20, 3))
        validate_y = validate_x[:, 0] + 0.5 * validate_x[:, 1]
        n = PolynomNeuron(
            0, 0, 1, RefFunctionType.rfLinear, 0, model_class="regression", loss="mse"
        )
        params = {"l2": 0.5, "layer_index": 0, "criterion_type": CriterionType.cmpBias}
        n.fit(train_x, train_y, validate_x, validate_y, params)
        self.assertTrue(n.valid)
        self.assertGreater(n.bias_err, 0.0)


class LayerTestCase(unittest.TestCase):

    def test_repr(self):
        layer = Layer(_DummyModel(), 3)
        self.assertEqual(repr(layer), "Layer 3")

    def test_add_and_delete_neuron(self):
        layer = Layer(_DummyModel(), 0)
        layer.add_neuron(0, 1, RefFunctionType.rfLinear, "regression", "mse")
        layer.add_neuron(0, 2, RefFunctionType.rfLinear, "regression", "mse")
        layer.add_neuron(1, 2, RefFunctionType.rfLinear, "regression", "mse")
        self.assertEqual(len(layer), 3)
        self.assertIn(0, layer.input_index_set)
        layer.delete(0)
        self.assertEqual(len(layer), 2)
        # after delete, neuron_index must be re-numbered
        for i, neuron in enumerate(layer):
            self.assertEqual(neuron.neuron_index, i)

    def test_describe(self):
        layer = Layer(_DummyModel(), 0)
        layer.add_neuron(0, 1, RefFunctionType.rfLinear, "regression", "mse")
        layer[0].w = np.array([0.1, 0.2, 0.3])
        s = layer.describe([], [layer])
        self.assertIn("Layer 0", s)


class LayerCreationErrorTestCase(unittest.TestCase):

    def test_carries_layer_index(self):
        err = LayerCreationError("boom", 7)
        self.assertEqual(err.layer_index, 7)
        self.assertIn("boom", str(err))


class FitLayerFunctionTestCase(unittest.TestCase):

    def test_fit_layer_runs_each_neuron(self):
        rng = np.random.RandomState(1)
        train_x = rng.uniform(-1, 1, size=(30, 3))
        train_y = train_x[:, 0] + 0.5 * train_x[:, 1]
        validate_x = rng.uniform(-1, 1, size=(15, 3))
        validate_y = validate_x[:, 0] + 0.5 * validate_x[:, 1]

        layer = Layer(_DummyModel(), 0)
        layer.add_neuron(0, 1, RefFunctionType.rfLinear, "regression", "mse")
        layer.add_neuron(0, 2, RefFunctionType.rfLinear, "regression", "mse")

        data = FitLayerData(
            layer,
            train_x,
            train_y,
            validate_x,
            validate_y,
            {"l2": 0.5, "layer_index": 0, "criterion_type": CriterionType.cmpValidate},
        )
        result = fit_layer(data)
        self.assertEqual(len(result), 2)
        for n in result:
            self.assertTrue(n.valid)


if __name__ == "__main__":
    unittest.main()
