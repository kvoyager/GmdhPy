# -*- coding: utf-8 -*-
import unittest

import numpy as np
from gmdhpy.neuron import RefFunctionType, CriterionType, PolynomNeuron


class NeuronTestCase(unittest.TestCase):

    def test_neuron_ref_function_getter(self):
        try:
            RefFunctionType.get('linear')
            RefFunctionType.get('linear_cov')
            RefFunctionType.get('lcov')
            RefFunctionType.get('quadratic')
            RefFunctionType.get('quad')
            RefFunctionType.get('cubic')
        except ValueError:
            self.assertTrue(False)

        def incorrect_ref_func():
            return RefFunctionType.get('bessel')

        self.assertRaises(ValueError, incorrect_ref_func)

    def test_neuron_criterion_type(self):
        try:
            CriterionType.get('validate')
            CriterionType.get('bias')
            CriterionType.get('validate_bias')
            CriterionType.get('bias_retrain')
        except ValueError:
            self.assertTrue(False)

        def incorrect_type():
            return CriterionType.get('plain')

        self.assertRaises(ValueError, incorrect_type)

    def test_neuron_transfer(self):
        neuron = PolynomNeuron(0, 0, 1, RefFunctionType.rfLinear, 0,
                               model_class='regression', loss='mse')
        neuron.w = np.array([1.5, 8, 2.7])
        res = neuron.transfer(1.5, -2.3, neuron.w)
        self.assertAlmostEqual(res, 7.29)

        neuron = PolynomNeuron(0, 0, 1, RefFunctionType.rfLinearCov, 0,
                               model_class='regression', loss='mse')
        neuron.w = np.array([1.5, 8, 2.7, 2.5])
        res = neuron.transfer(1.5, -2.3, neuron.w)
        self.assertAlmostEqual(res, -1.335)

        neuron = PolynomNeuron(0, 0, 1, RefFunctionType.rfQuadratic, 0,
                               model_class='regression', loss='mse')
        neuron.w = np.array([1.5, 8, 2.7, 2.5, 1.5, -2.3])
        res = neuron.transfer(1.5, -2.3, neuron.w)
        self.assertAlmostEqual(res, -10.127)

        neuron = PolynomNeuron(0, 0, 1, RefFunctionType.rfCubic, 0,
                               model_class='regression', loss='mse')
        neuron.w = np.array([1.5, 1, 2.7, 2.5, 1.5, -1.3, 1.0, 3.0, -1.0, 2.0])
        res = neuron.transfer(1.5, -1.5, neuron.w)
        self.assertAlmostEqual(res, -23.1)

        neuron = PolynomNeuron(0, 0, 1, RefFunctionType.rfLinear, 0,
                               model_class='classification', loss='logloss')
        neuron.w = np.array([1.5, 8, 2.7])
        res = neuron.transfer(1.5, -2.3, neuron.w)
        self.assertAlmostEqual(res, 0.999318137201)

    def test_neuron_error(self):
        neuron = PolynomNeuron(0, 0, 1, RefFunctionType.rfLinear, 0,
                               model_class='regression', loss='mse')
        res = neuron.loss_function(np.array([1.5, 1.6, 1.3]),
                                   np.array([1.0, 1.0, 1.0]))
        self.assertAlmostEqual(res, 0.7)

        neuron = PolynomNeuron(0, 0, 1, RefFunctionType.rfLinear, 0,
                               model_class='classification', loss='logloss')
        res = neuron.loss_function(['A', 'A', 'B'],
                                   np.array([0.9, 0.95, 0.1]))
        self.assertAlmostEqual(res, 2.53363415318)

if __name__ == '__main__':
    unittest.main()