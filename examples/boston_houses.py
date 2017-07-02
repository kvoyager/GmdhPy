# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.datasets import load_boston
from sklearn import metrics
from gmdhpy.gmdh import Regressor
from gmdhpy.plot_model import PlotModel


if __name__ == '__main__':

    boston = load_boston()

    n_samples = boston.data.shape[0]

    train_data_is_the_first_half = False
    n = n_samples // 2
    if train_data_is_the_first_half:
        train_x = boston.data[:n]
        train_y = boston.target[:n]
        test_x = boston.data[n:]
        test_y = boston.target[n:]
    else:
        train_x = boston.data[n:]
        train_y = boston.target[n:]
        test_x = boston.data[:n]
        test_y = boston.target[:n]

    model = Regressor(ref_functions=('linear_cov',),
                      criterion_type='validate',
                      feature_names=boston.feature_names,
                      criterion_minimum_width=5,
                      stop_train_epsilon_condition=0.001,
                      layer_err_criterion='top',
                      l2=0.5,
                      n_jobs='max')
    model.fit(train_x, train_y)

    # Now predict the value of the second half:
    y_pred = model.predict(test_x)
    mse = metrics.mean_squared_error(test_y, y_pred)
    mae = metrics.mean_absolute_error(test_y, y_pred)

    print("mse error on test set: {mse:0.2f}".format(mse=mse))
    print("mae error on test set: {mae:0.2f}".format(mae=mae))

    y_pred = model.predict(test_x)

    print(model.get_selected_features_indices())
    print(model.get_unselected_features_indices())

    print("Selected features: {}".format(model.get_selected_features()))
    print("Unselected features: {}".format(model.get_unselected_features()))

    PlotModel(model, filename='boston_house_model', plot_neuron_name=True, view=True).plot()
