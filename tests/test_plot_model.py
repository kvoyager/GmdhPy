# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import graphviz

    # Ensure it's the real PyPI `graphviz` package, not a shadowing module
    graphviz.Digraph
    graphviz.Graph
    from gmdhpy.plot_model import PlotModel

    HAS_GRAPHVIZ = True
except (ImportError, AttributeError):
    HAS_GRAPHVIZ = False

from gmdhpy.gmdh import Regressor


@unittest.skipUnless(HAS_GRAPHVIZ, "graphviz Python package not installed")
class PlotModelTestCase(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(0)
        self.x = rng.uniform(-1, 1, size=(60, 4))
        self.y = self.x[:, 0] + 0.5 * self.x[:, 1] * self.x[:, 2] + 0.1 * rng.randn(60)

    def _trained_model(self, feature_names=None):
        model = Regressor(
            ref_functions="linear_cov",
            feature_names=feature_names,
            max_layer_count=3,
            criterion_minimum_width=2,
            verbose=0,
        )
        model.fit(self.x, self.y)
        return model

    def test_plot_no_layers_short_circuits(self):
        empty_model = Regressor(verbose=0)
        plotter = PlotModel(empty_model, "unused")
        # Should return early without raising
        plotter.plot()

    def test_plot_with_features(self):
        model = self._trained_model(feature_names=["a", "b", "c", "d"])
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "graph")
            plotter = PlotModel(model, target, plot_neuron_name=True)
            # Avoid actually invoking the graphviz binary
            with mock.patch.object(plotter.g, "render") as render_mock:
                plotter.plot()
                render_mock.assert_called_once()

    def test_plot_without_feature_names(self):
        model = self._trained_model()
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "graph")
            plotter = PlotModel(model, target)
            with mock.patch.object(plotter.g, "render") as render_mock:
                plotter.plot()
                render_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
