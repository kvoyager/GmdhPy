__author__ = 'Konstantin Kolokolov'

import numpy as np
import pandas as pd


def train_preprocessing(data_x, data_y, feature_names):
    """process of train input data: transform to numpy matrix, transpose etc"""

    if isinstance(data_x, pd.DataFrame):
        data_x = data_x.as_matrix()

    if isinstance(data_y, pd.DataFrame) or isinstance(data_y, pd.Series):
        data_y = data_y.as_matrix()

    if not isinstance(data_y, (np.ndarray, np.generic)):
        data_y = np.asarray(data_y)
    # if len(data_y.shape) != 1:
    #     if data_y.shape[0] == 1:
    #         data_y = data_y[0, :]
    #     elif data_y.shape[1] == 1:
    #         data_y = data_y[:, 0]
    #     else:
    #         raise ValueError('data_y dimension should be 1 or (n, 1) or (1, n)')
    data_len = data_y.shape[0]

    if not isinstance(data_x, (np.ndarray, np.generic)):
        data_x = np.asarray(data_x)

    if len(data_x.shape) != 2:
        raise ValueError('data_x dimension has to be 2. it has to be a 2D numpy array: number of features x '
                         'number of observations')

    if data_x.shape[0] != data_len:
        # try to check if transpose matrix is suitable
        if data_x.shape[1] == data_len:
            # ok, need to transpose data_x
            data_x = data_x.transpose()
        else:
            raise ValueError('number of examples in data_x is not equal to number of examples in data_y')

    if data_x.shape[1] < 2:
        raise ValueError('Error: number of features should be not less than two')

    if data_x.shape[0] < 2:
        raise ValueError('Error: number of samples should be not less than two')

    if feature_names is not None:
        feature_names_len = len(feature_names)
        if feature_names_len > 0 and feature_names_len != data_x.shape[1]:
            raise ValueError('Error: size of feature_names list is not equal to number of features')

    return data_x, data_y, data_len


def predict_preprocessing(data_x, n_features):
    """process of predict input data: transform to numpy matrix, transpose etc"""

    if isinstance(data_x, pd.DataFrame):
        data_x = data_x.as_matrix()

    if not isinstance(data_x, (np.ndarray, np.generic)):
        data_x = np.asarray(data_x)

    if len(data_x.shape) != 2:
        raise ValueError('data_x dimension has to be 2. it has to be a 2D numpy array: number of features x '
                         'number of observations')

    if data_x.shape[1] != n_features:
        # try to check if transpose matrix is suitable
        if data_x.shape[0] == n_features:
            # ok, need to transpose data_x
            data_x = data_x.transpose()
        else:
            raise ValueError('number of features in data_x is not equal to number of features in trained model')

    data_len = data_x.shape[0]
    return data_x, data_len
