#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from gmdhpy.gmdh import MultilayerGMDH

__author__ = 'Konstantin Kolokolov'


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
    gmdh = MultilayerGMDH(ref_functions=('linear_cov', 'quad'), n_jobs='max',
                          manual_best_models_selection=True, min_best_models_count=30)

    # train model
    gmdh.fit(train_x, train_y)

    # predict with GMDH
    y_pred = gmdh.predict(train_x)

    plt.plot(x, y, label="ground truth")
    plt.scatter(x, train_y, label="training points")
    plt.plot(x, y_pred, label="fit")
    plt.legend(loc='lower left')

    plt.show()
