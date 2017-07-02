# -*- coding: utf-8 -*-
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from gmdhpy.gmdh import Classifier

class NeuronTestCase(unittest.TestCase):

    def fit_classification_model(self, train_x, test_x, train_y, test_y,
                                 params):
        model = Classifier(**params)
        model.fit(train_x, train_y)
        pred_y = model.predict_proba(test_x)
        roc_auc = roc_auc_score(model.le.transform(test_y), pred_y)
        return roc_auc

    def test_plain_classification(self):
        tol_places = 4
        data_x, data_y = make_classification(n_samples=100, n_features=7,
                                             n_redundant=0, n_informative=7,
                                             n_clusters_per_class=2,
                                             random_state=3227)
        label_y = np.where(data_y == 0, 'A', 'B')

        train_x, test_x, train_y, test_y = train_test_split(data_x, label_y,
                                                            test_size=0.33,
                                                            random_state=3227)

        # case with linear_cov
        params = {
            'ref_functions': ('linear_cov',),
            'criterion_type': 'validate',
            'criterion_minimum_width': 5,
            'max_layer_count': 5,
            'verbose': 0
        }
        roc_auc = self.fit_classification_model(train_x, test_x,
                                                train_y, test_y, params)
        self.assertAlmostEqual(roc_auc, 0.65, places=tol_places)

        # case with quad
        params = {
            'ref_functions': ('quadratic',),
            'criterion_type': 'validate',
            'criterion_minimum_width': 5,
            'max_layer_count': 5,
            'verbose': 0
        }
        roc_auc = self.fit_classification_model(train_x, test_x,
                                                train_y, test_y, params)
        self.assertAlmostEqual(roc_auc, 0.892307692308, places=tol_places)


        # case with mix
        params = {
            'ref_functions': ('linear_cov', 'quadratic', 'cubic'),
            'criterion_type': 'validate',
            'criterion_minimum_width': 5,
            'max_layer_count': 5,
            'admix_features': False,
            'verbose': 0
        }
        roc_auc = self.fit_classification_model(train_x, test_x,
                                                train_y, test_y, params)
        self.assertAlmostEqual(roc_auc, 0.907692307692, places=tol_places)


        # test bias criterion
        params = {
            'ref_functions': 'linear_cov',
            'criterion_type': 'bias',
            'criterion_minimum_width': 3,
            'max_layer_count': 6,
            'admix_features': False,
            'verbose': 0,
            'n_jobs': 'max'
        }
        roc_auc = self.fit_classification_model(train_x, test_x,
                                                train_y, test_y, params)
        self.assertAlmostEqual(roc_auc, 0.634615384615, places=tol_places)

        # test bias_retrain criterion
        params = {
            'ref_functions': 'linear_cov',
            'criterion_type': 'bias_retrain',
            'criterion_minimum_width': 3,
            'max_layer_count': 6,
            'admix_features': False,
            'verbose': 0,
            'n_jobs': 'max'
        }
        roc_auc = self.fit_classification_model(train_x, test_x,
                                                train_y, test_y, params)
        self.assertAlmostEqual(roc_auc, 0.588461538462, places=tol_places)

    def test_classification_with_validation(self):
        tol_places = 4
        data_x, data_y = make_classification(n_samples=100, n_features=7,
                                             n_redundant=0, n_informative=7,
                                             n_clusters_per_class=2,
                                             random_state=3227)
        label_y = np.where(data_y == 0, 'A', 'B')

        train_x, test_x, train_y, test_y = train_test_split(data_x, label_y,
                                                            test_size=0.25,
                                                            random_state=3227)

        train_x, validate_x, train_y, validate_y = train_test_split(
            train_x, train_y, test_size=0.5, random_state=3227)

        params = {
            'ref_functions': ('linear_cov',),
            'criterion_type': 'bias_retrain',
            'criterion_minimum_width': 5,
            'max_layer_count': 5,
            'verbose': 0,
            'n_jobs': 'max'
        }
        model = Classifier(**params)
        model.fit(train_x, train_y, validation_data=(validate_x, validate_y))
        pred_y = model.predict_proba(test_x)
        roc_auc = roc_auc_score(model.le.transform(test_y), pred_y)
        self.assertAlmostEqual(roc_auc, 0.76, places=tol_places)

        no1 = model.predict_neuron_output(test_x, 0, 0)
        no2 = model.predict_neuron_output(test_x, 1, 0)
