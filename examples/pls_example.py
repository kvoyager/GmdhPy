#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from gmdhpy.gmdh import MultilayerGMDH

__author__ = 'Konstantin Kolokolov'


def f1(x):
    """ function to approximate by group method of data handling algorithm"""
    return x * np.sin(x) + x + 0.25*x**2 - 0.04*x**3

def f2(x):
    """ function to approximate by group method of data handling algorithm"""
    return x * np.sin(x+0.4) + x + 0.25*x**2 - 0.04*x**3

if __name__ == '__main__':

    # generate points
    x = np.linspace(-2, 10, 200)
    n_samples = x.shape[0]

    # add random noise
    eps = 0.5
    np.random.seed(29)
    eps_data1 = np.random.uniform(-eps, eps, (n_samples,))
    eps_data2 = np.random.uniform(-eps, eps, (n_samples,))
    y1 = f1(x)
    y2 = f2(x)
    train_y = np.empty((n_samples, 2), dtype=np.double)
    train_y[:, 0] =  y1 + eps_data1
    train_y[:, 1] =  y2 + eps_data2

    train_x = np.vstack((x, np.power(x, 2)))
    gmdh = MultilayerGMDH(ref_functions=('linear_cov', 'quad'),
                          manual_best_models_selection=True,
                          min_best_models_count=30,
                          model_type='pls')

    # train model
    gmdh.fit(train_x, train_y)

    # predict with GMDH
    y_pred = gmdh.predict(train_x)

    plt.plot(x, y1, label="ground truth y1", color='c')
    plt.plot(x, y2, label="ground truth y2", color='b')
    plt.scatter(x, train_y[:, 0], label="training points y1", color='c')
    plt.scatter(x, train_y[:, 1], label="training points y2", color='b')
    plt.plot(x, y_pred[:, 0], label="fit y1", color=cm.hot(0.6), linestyle='--')
    plt.plot(x, y_pred[:, 1], label="fit y2", color=cm.hot(0.5), linestyle='--' )
    plt.legend(loc='lower left')

    plt.show()
