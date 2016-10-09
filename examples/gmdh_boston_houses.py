#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Konstantin Kolokolov'

from sklearn.datasets import load_boston
from sklearn import metrics
from gmdhpy.gmdh import MultilayerGMDH
from gmdhpy.plot_gmdh import PlotGMDH


if __name__ == '__main__':

    boston = load_boston()

    n_samples = boston.data.shape[0]

    train_data_is_the_first_half = False
    n = n_samples // 2
    if train_data_is_the_first_half:
        train_x = boston.data[:n]
        train_y = boston.target[:n]
        exam_x = boston.data[n:]
        exam_y = boston.target[n:]
    else:
        train_x = boston.data[n:]
        train_y = boston.target[n:]
        exam_x = boston.data[:n]
        exam_y = boston.target[:n]

    gmdh = MultilayerGMDH(ref_functions=('linear_cov',),
                          criterion_type='validate_bias',
                          feature_names=boston.feature_names,
                          criterion_minimum_width=5,
                          admix_features=True,
                          max_layer_count=30,
                          normalize=True,
                          stop_train_epsilon_condition=0.001,
                          layer_err_criterion='avg',
                          alpha=0.5,
                          keep_partial_models=False)
    gmdh.fit(train_x, train_y)

    # Now predict the value of the second half:
    # predict with GMDH
    y_pred = gmdh.predict(exam_x)
    mse = metrics.mean_squared_error(exam_y, y_pred)
    mae = metrics.mean_absolute_error(exam_y, y_pred)

    print("mse error on validate set: {mse:0.2f}".format(mse=mse))
    print("mae error on validate set: {mae:0.2f}".format(mae=mae))

    PlotGMDH(gmdh, filename='boston_house_model', plot_model_name=True, view=True)

    y_pred = gmdh.predict(exam_x)

    gmdh.plot_layer_error()
