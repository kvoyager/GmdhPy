# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from gmdhpy.gmdh import Regressor

def f(x):
    """ function to approximate by group method of data handling algorithm"""
    return x * np.sin(x) + x + 0.25*x**2 - 0.04*x**3

if __name__ == '__main__':

    # generate points
    x = np.linspace(-2, 10, 200)
    n_samples = x.shape[0]

    # add random noise
    eps = 1.5
    eps_data = np.random.uniform(-eps, eps, (n_samples,))
    y = f(x)
    train_y = y[:] + eps_data[:]

    train_x = np.vstack((x, np.power(x, 2)))
    model = Regressor(ref_functions=('linear_cov', 'quad'),
                      manual_best_neurons_selection=True,
                      min_best_neurons_count=30,
                      n_jobs='max')

    # train model
    model.fit(train_x, train_y)

    # predict with GMDH
    y_pred = model.predict(train_x)

    plt.plot(x, y, label="ground truth")
    plt.scatter(x, train_y, label="training points")
    plt.plot(x, y_pred, label="fit")
    plt.legend(loc='lower left')

    plt.show()
