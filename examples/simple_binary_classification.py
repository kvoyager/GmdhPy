# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from gmdhpy.gmdh import Classifier
from gmdhpy.plot_model import PlotModel
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    n_informative=5
    n_clusters_per_class=3
    plt.title("{} informative feature, {} cluster per class".format(n_informative, n_clusters_per_class), fontsize='small')
    data_x, data_y = make_classification(n_samples = 1000, n_features=9, n_redundant=3, n_informative=n_informative,
                                 n_clusters_per_class=n_clusters_per_class, random_state=27)
    label_y = np.where(data_y == 0, 'A', 'B')

    train_x, test_x, train_y, test_y = train_test_split(data_x, label_y,
                                                        test_size=0.33,
                                                        random_state=42)

    train_x, validate_x, train_y, validate_y = train_test_split(train_x, train_y,
                                                                test_size=0.5,
                                                                random_state=42)

    model = Classifier(ref_functions=('linear_cov'),
                       criterion_type='validate',
                       criterion_minimum_width=5,
                       max_layer_count=50)

    model.fit(train_x, train_y, validation_data=(validate_x, validate_y))
    pred_y = model.predict_proba(test_x)

    roc_auc = roc_auc_score(model.le.transform(test_y), pred_y)

    print('ROC AUC: {}'.format(roc_auc))

    plt.scatter(data_x[:, 0], data_x[:, 1], marker='o', c=data_y)

    plt.show()

    print(model.get_selected_features_indices())
    print(model.get_unselected_features_indices())

    print("Selected features: {}".format(model.get_selected_features()))
    print("Unselected features: {}".format(model.get_unselected_features()))

    PlotModel(model, filename='boston_house_model', plot_neuron_name=True, view=True).plot()