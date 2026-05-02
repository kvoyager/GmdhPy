# -*- coding: utf-8 -*-
import unittest
from unittest import mock
import numpy as np
import pandas as pd

from gmdhpy.gmdh import Regressor, MultilayerGMDH
from gmdhpy.data_preprocessing import (
    set_split_types,
    SequenceTypeSet,
    train_preprocessing,
)


class MultilayerGMDHAliasTestCase(unittest.TestCase):

    def test_alias_is_regressor(self):
        self.assertIs(MultilayerGMDH, Regressor)


class SetSplitTypesInvalidTestCase(unittest.TestCase):

    def test_invalid_seq_type_raises(self):
        # Pass a totally bogus value to trigger the "Unknown type" branch
        class _Bogus:
            pass

        self.assertRaises(ValueError, set_split_types, _Bogus(), 10)


class TrainPreprocessingDataFrameYTestCase(unittest.TestCase):

    def test_dataframe_y(self):
        x = np.zeros((4, 2))
        df_y = pd.DataFrame({"y": [0, 1, 0, 1]})
        _, y = train_preprocessing(x, df_y, None)
        self.assertEqual(y.shape, (4,))


class PlotLayerErrorTestCase(unittest.TestCase):

    def test_plot_layer_error_runs(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(-1, 1, size=(60, 4))
        y = x[:, 0] + 0.5 * x[:, 1]
        model = Regressor(
            ref_functions="linear_cov",
            max_layer_count=3,
            criterion_minimum_width=2,
            verbose=0,
        )
        model.fit(x, y)
        # Don't actually open a window
        with mock.patch("gmdhpy.gmdh.plt.show"):
            model.plot_layer_error()


class FitWithFeatureNamesTestCase(unittest.TestCase):

    def test_get_unselected_with_no_unselected(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(-1, 1, size=(40, 2))
        y = x[:, 0] + 0.5 * x[:, 1]
        model = Regressor(
            ref_functions="linear_cov",
            feature_names=["a", "b"],
            max_layer_count=2,
            criterion_minimum_width=1,
            verbose=0,
        )
        model.fit(x, y)
        # With only 2 features, both will likely be selected -> "No unselected"
        msg = model.get_unselected_features()
        self.assertIsInstance(msg, str)


if __name__ == "__main__":
    unittest.main()
