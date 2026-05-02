# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd

from gmdhpy.data_preprocessing import (
    SequenceTypeSet,
    DataSetType,
    train_preprocessing,
    predict_preprocessing,
    set_split_types,
    split_dataset,
)


class SequenceTypeSetTestCase(unittest.TestCase):

    def test_get_string_aliases(self):
        self.assertEqual(SequenceTypeSet.get("mode1"), SequenceTypeSet.sqMode1)
        self.assertEqual(SequenceTypeSet.get("mode2"), SequenceTypeSet.sqMode2)
        self.assertEqual(SequenceTypeSet.get("mode3_1"), SequenceTypeSet.sqMode3_1)
        self.assertEqual(SequenceTypeSet.get("mode3_2"), SequenceTypeSet.sqMode3_2)
        self.assertEqual(SequenceTypeSet.get("mode4_1"), SequenceTypeSet.sqMode4_1)
        self.assertEqual(SequenceTypeSet.get("mode4_2"), SequenceTypeSet.sqMode4_2)
        self.assertEqual(SequenceTypeSet.get("random"), SequenceTypeSet.sqRandom)

    def test_get_passthrough(self):
        self.assertEqual(
            SequenceTypeSet.get(SequenceTypeSet.sqMode1),
            SequenceTypeSet.sqMode1,
        )

    def test_get_invalid_raises(self):
        self.assertRaises(ValueError, SequenceTypeSet.get, "nope")

    def test_is_mode_type_predicates(self):
        self.assertTrue(SequenceTypeSet.is_mode1_type(SequenceTypeSet.sqMode1))
        self.assertTrue(SequenceTypeSet.is_mode1_type(SequenceTypeSet.sqMode3_1))
        self.assertTrue(SequenceTypeSet.is_mode1_type(SequenceTypeSet.sqMode4_1))
        self.assertFalse(SequenceTypeSet.is_mode1_type(SequenceTypeSet.sqMode2))

        self.assertTrue(SequenceTypeSet.is_mode2_type(SequenceTypeSet.sqMode2))
        self.assertTrue(SequenceTypeSet.is_mode2_type(SequenceTypeSet.sqMode3_2))
        self.assertTrue(SequenceTypeSet.is_mode2_type(SequenceTypeSet.sqMode4_2))
        self.assertFalse(SequenceTypeSet.is_mode2_type(SequenceTypeSet.sqMode1))


class SetSplitTypesTestCase(unittest.TestCase):

    def test_random_split_returns_full_array(self):
        np.random.seed(42)
        seq_types = set_split_types(SequenceTypeSet.sqRandom, 50)
        self.assertEqual(seq_types.shape[0], 50)
        # Both classes appear
        self.assertIn(DataSetType.dsTrain, seq_types)
        self.assertIn(DataSetType.dsValidate, seq_types)

    def test_mode1_last_is_validate(self):
        seq_types = set_split_types(SequenceTypeSet.sqMode1, 10)
        self.assertEqual(seq_types[-1], DataSetType.dsValidate)

    def test_mode2_last_is_train(self):
        seq_types = set_split_types(SequenceTypeSet.sqMode2, 10)
        self.assertEqual(seq_types[-1], DataSetType.dsTrain)

    def test_mode3_1_period(self):
        seq_types = set_split_types(SequenceTypeSet.sqMode3_1, 9)
        # mode3_1 has period 3 starting from end
        self.assertEqual(seq_types[-1], DataSetType.dsValidate)

    def test_mode4_1_period(self):
        seq_types = set_split_types(SequenceTypeSet.sqMode4_1, 12)
        self.assertEqual(seq_types[-1], DataSetType.dsValidate)

    def test_mode3_2_period(self):
        seq_types = set_split_types(SequenceTypeSet.sqMode3_2, 9)
        self.assertEqual(seq_types[-1], DataSetType.dsTrain)

    def test_mode4_2_period(self):
        seq_types = set_split_types(SequenceTypeSet.sqMode4_2, 12)
        self.assertEqual(seq_types[-1], DataSetType.dsTrain)


class SplitDatasetTestCase(unittest.TestCase):

    def test_split_balanced_lengths(self):
        x = np.arange(20).reshape(10, 2).astype(float)
        y = np.arange(10).astype(float)
        tr_x, tr_y, va_x, va_y = split_dataset(x, y, SequenceTypeSet.sqMode1)
        self.assertEqual(tr_x.shape[0] + va_x.shape[0], 10)
        self.assertEqual(tr_y.shape[0], tr_x.shape[0])
        self.assertEqual(va_y.shape[0], va_x.shape[0])


class TrainPreprocessingTestCase(unittest.TestCase):

    def test_dataframe_inputs_converted(self):
        df_x = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [5.0, 6, 7, 8]})
        s_y = pd.Series([0, 1, 0, 1])
        x, y = train_preprocessing(df_x, s_y, None)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(x.shape, (4, 2))
        self.assertEqual(y.shape, (4,))

    def test_y_2d_column_squeezed(self):
        x = np.zeros((4, 2))
        y = np.array([[0], [1], [0], [1]])
        _, y_out = train_preprocessing(x, y, None)
        self.assertEqual(y_out.shape, (4,))

    def test_y_2d_row_squeezed(self):
        x = np.zeros((4, 2))
        y = np.array([[0, 1, 0, 1]])
        _, y_out = train_preprocessing(x, y, None)
        self.assertEqual(y_out.shape, (4,))

    def test_y_invalid_2d_shape_raises(self):
        x = np.zeros((4, 2))
        y = np.zeros((4, 3))
        self.assertRaises(ValueError, train_preprocessing, x, y, None)

    def test_x_transposed_when_needed(self):
        x = np.zeros((2, 4))  # features x samples
        y = np.zeros(4)
        x_out, _ = train_preprocessing(x, y, None)
        self.assertEqual(x_out.shape, (4, 2))

    def test_x_sample_mismatch_raises(self):
        x = np.zeros((5, 2))
        y = np.zeros(4)
        self.assertRaises(ValueError, train_preprocessing, x, y, None)

    def test_x_dim_not_2_raises(self):
        x = np.zeros(8)
        y = np.zeros(4)
        self.assertRaises(ValueError, train_preprocessing, x, y, None)

    def test_x_too_few_features_raises(self):
        x = np.zeros((10, 1))
        y = np.zeros(10)
        self.assertRaises(ValueError, train_preprocessing, x, y, None)

    def test_too_few_samples_raises(self):
        x = np.zeros((1, 3))
        y = np.zeros(1)
        self.assertRaises(ValueError, train_preprocessing, x, y, None)

    def test_feature_names_size_mismatch_raises(self):
        x = np.zeros((4, 2))
        y = np.zeros(4)
        self.assertRaises(ValueError, train_preprocessing, x, y, ["a", "b", "c"])

    def test_list_inputs_converted(self):
        x = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [0, 1, 0, 1]
        x_out, y_out = train_preprocessing(x, y, None)
        self.assertIsInstance(x_out, np.ndarray)
        self.assertIsInstance(y_out, np.ndarray)


class PredictPreprocessingTestCase(unittest.TestCase):

    def test_dataframe_input(self):
        df = pd.DataFrame({"a": [1.0, 2, 3], "b": [4.0, 5, 6]})
        x, n = predict_preprocessing(df, 2)
        self.assertEqual(x.shape, (3, 2))
        self.assertEqual(n, 3)

    def test_transpose_when_features_first(self):
        x = np.zeros((3, 5))  # n_features=3, samples=5
        x_out, n = predict_preprocessing(x, 3)
        self.assertEqual(x_out.shape, (5, 3))
        self.assertEqual(n, 5)

    def test_dim_mismatch_raises(self):
        x = np.zeros((4, 2))
        self.assertRaises(ValueError, predict_preprocessing, x, 5)

    def test_dim_not_2_raises(self):
        x = np.zeros(6)
        self.assertRaises(ValueError, predict_preprocessing, x, 2)


if __name__ == "__main__":
    unittest.main()
