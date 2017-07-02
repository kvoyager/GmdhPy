# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
from enum import Enum
import numpy as np
from sklearn import linear_model
from sklearn.metrics import log_loss
from collections import namedtuple

FitLayerData = namedtuple('FitLayerData',
                         ['sublayer', 'train_x', 'train_y',
                          'validate_x', 'validate_y', 'params'])


class RefFunctionType(Enum):
    rfUnknown = -1
    rfLinear = 0
    rfLinearCov = 1
    rfQuadratic = 2
    rfCubic = 3

    @classmethod
    def get_name(cls, value):
        if value == cls.rfUnknown:
            return 'Unknown'
        elif value == cls.rfLinear:
            return 'Linear'
        elif value == cls.rfLinearCov:
            return 'LinearCov'
        elif value == cls.rfQuadratic:
            return 'Quadratic'
        elif value == cls.rfCubic:
            return 'Cubic'
        elif value == cls.rfHarmonic:
            return 'Harmonic'
        else:
            return 'Unknown'

    @staticmethod
    def get(arg):
        if isinstance(arg, RefFunctionType):
            return arg
        if arg == 'linear':
            return RefFunctionType.rfLinear
        elif arg in ('linear_cov', 'lcov'):
            return RefFunctionType.rfLinearCov
        elif arg in ('quadratic', 'quad'):
            return RefFunctionType.rfQuadratic
        elif arg == 'cubic':
            return RefFunctionType.rfCubic
        else:
            raise ValueError(arg)


class CriterionType(Enum):
    cmpValidate = 1
    cmpBias = 2
    cmpComb_validate_bias = 4
    cmpComb_bias_retrain = 5

    @classmethod
    def get_name(cls, value):
        if value == cls.cmpValidate:
            return 'validate error comparison'
        elif value == cls.cmpBias:
            return 'bias error comparison'
        elif value == cls.cmpComb_validate_bias:
            return 'bias and validate error comparison'
        elif value == cls.cmpComb_bias_retrain:
            return 'bias error comparison with retrain'
        else:
            return 'Unknown'

    @staticmethod
    def get(arg):
        if isinstance(arg, CriterionType):
            return arg
        elif arg == 'validate':
            return CriterionType.cmpValidate
        elif arg == 'bias':
            return CriterionType.cmpBias
        elif arg == 'validate_bias':
            return CriterionType.cmpComb_validate_bias
        elif arg in ('bias_retrain', 'bias_refit') :
            return CriterionType.cmpComb_bias_retrain
        else:
            raise ValueError(arg)


# *****************************************************************************
#   Base neuron class
# *****************************************************************************
class Neuron(object):
    """Base class for neuron
    """

    def __init__(self, layer_index, u1_index, u2_index, neuron_index):
        self.layer_index = layer_index
        self.neuron_index = neuron_index
        self.u1_index = u1_index
        self.u2_index = u2_index
        self.ref_function_type = RefFunctionType.rfUnknown
        self.valid = True
        self.train_err = sys.float_info.max	            # neuron error on train data set
        self.validate_err = sys.float_info.max	        # neuron error on validate data set
        self.bias_err = sys.float_info.max	            # bias neuron error
        self.transfer = None                            # transfer function

    def need_bias_stuff(self, criterion_type):
        if criterion_type == CriterionType.cmpValidate:
            return False
        return True

    def get_error(self, criterion_type):
        """Compute error of the neuron according to specified criterion
        """
        if criterion_type == CriterionType.cmpValidate:
            return self.validate_err
        elif criterion_type == CriterionType.cmpBias:
            return self.bias_err
        elif criterion_type == CriterionType.cmpComb_validate_bias:
            return 0.5*self.bias_err + 0.5*self.validate_err
        elif criterion_type == CriterionType.cmpComb_bias_retrain:
            return self.bias_err
        else:
            return sys.float_info.max

    def get_regularity_err(self, x, y):
        raise NotImplementedError

    def get_bias_err(self, train_x, validate_x, train_y, validate_y):
        raise NotImplementedError

    def get_features_name(self, input_index, feature_names, layers):
        if self.layer_index == 0:
            s = 'index=inp_{0}'.format(input_index)
            if len(feature_names) > 0:
                s += ', {0}'.format(feature_names[input_index])
        else:
            neurons_num = len(layers[self.layer_index-1])
            if input_index < neurons_num:
                s = 'index=prev_layer_neuron_{0}'.format(input_index)
            else:
                s = 'index=inp_{0}'.format(input_index - neurons_num)
                if len(feature_names) > 0:
                    s += ', {0}'.format(feature_names[input_index - neurons_num])
        return s

    def linear_activation(self, x):
        return x

    def sigmoid_activation(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def get_name(self):
        raise NotImplementedError

    def get_short_name(self):
        raise NotImplementedError


# *****************************************************************************
#   Polynomial neuron class
# *****************************************************************************

class PolynomNeuron(Neuron):
    """Polynomial neuron class
    """

    def __init__(self, layer_index, u1_index, u2_index, ftype, neuron_index, model_class, loss):
        super(PolynomNeuron, self).__init__(layer_index, u1_index, u2_index, neuron_index)
        self.ftype = ftype
        self.fw_size = 0
        self.set_type(ftype)
        self.w = None
        self.wt = None
        self.valid = False
        self.bias_err = 0
        self.train_err = 0
        self.validate_err = 0
        self.model_class = model_class

        if model_class=='classification':
            self.fit_function = self._fit_classifier
            self.activation = self.sigmoid_activation
        else:
            self.fit_function = self._fit_regressor
            self.activation = self.linear_activation

        if loss == 'mse':
            self.loss_function = self._mse
            self.loss_norm = self._mse_norm
        elif loss == 'logloss':
            self.loss_function = log_loss
            self.loss_norm = self._logloss_norm
        else:
            raise ValueError('Unexpected loss function type: {}'.format(loss))

    def _transfer_linear(self, u1, u2, w):
        return self.activation(w[0] + w[1]*u1 + w[2]*u2)

    def _transfer_linear_cov(self, u1, u2, w):
        return self.activation(w[0] + u1*(w[1] + w[3]*u2) + w[2]*u2)

    def _transfer_quadratic(self, u1, u2, w):
        return self.activation(w[0] + u1*(w[1] + w[3]*u2 + w[4]*u1) + u2*(w[2] + w[5]*u2))

    def _transfer_cubic(self, u1, u2, w):
        u1_sq = u1*u1
        u2_sq = u2*u2
        return self.activation(w[0] + w[1]*u1 + w[2]*u2 + w[3]*u1*u2 + w[4]*u1_sq + w[5]*u2_sq + \
            w[6]*u1*u1_sq + w[7]*u1_sq*u2 + w[8]*u1*u2_sq + w[9]*u2*u2_sq)

    def set_type(self, new_type):
        self.ref_function_type = new_type
        if new_type == RefFunctionType.rfLinear:
            self.transfer = self._transfer_linear
            self.fw_size = 3
        elif new_type == RefFunctionType.rfLinearCov:
            self.transfer = self._transfer_linear_cov
            self.fw_size = 4
        elif new_type == RefFunctionType.rfQuadratic:
            self.transfer = self._transfer_quadratic
            self.fw_size = 6
        elif new_type == RefFunctionType.rfCubic:
            self.transfer = self._transfer_cubic
            self.fw_size = 10
        else:
            raise ValueError('Unknown type of neuron: {}'.format(new_type))

    def _mse(self, y, yp):
        return ((y - yp) ** 2).sum()

    def _mse_norm(self, y):
        return (y ** 2).sum()

    def _logloss_norm(self, y):
        return np.absolute(y).sum()

    def get_regularity_err(self, x, y):
        """Calculation of regularity error
        """
        x1 = x[:, self.u1_index]
        x2 = x[:, self.u2_index]
        yp = self.transfer(x1, x2, self.w)
        err = self.loss_function(y, yp) / self.loss_norm(y)
        return err

    def get_sub_bias_err(self, x, wa, wb):
        """Helper function for calculation of unbiased error
        """
        x1 = x[:, self.u1_index]
        x2 = x[:, self.u2_index]
        yta = self.transfer(x1, x2, wa)
        ytb = self.transfer(x1, x2, wb)

        s = ((yta - ytb) ** 2).sum()
        return s

    def get_bias_err(self, train_x, validate_x, train_y, validate_y):
        """Calculation of unbiased error
        """
        s = self.get_sub_bias_err(train_x, self.w, self.wt) + \
            self.get_sub_bias_err(validate_x, self.w, self.wt)
        s2 = (train_y ** 2).sum() + (validate_y ** 2).sum()
        err = s/s2
        return err

    def get_name(self):
        if self.ftype == RefFunctionType.rfLinear:
            return 'w0 + w1*xi + w2*xj'
        elif self.ftype == RefFunctionType.rfLinearCov:
            return 'w0 + w1*xi + w2*xj + w3*xi*xj'
        elif self.ftype == RefFunctionType.rfQuadratic:
            return 'full polynom 2nd degree'
        elif self.ftype == RefFunctionType.rfCubic:
            return 'full polynom 3rd degree'
        else:
            return 'Unknown'

    def get_short_name(self):
        if self.ftype == RefFunctionType.rfLinear:
            return 'linear'
        elif self.ftype == RefFunctionType.rfLinearCov:
            return 'linear cov'
        elif self.ftype == RefFunctionType.rfQuadratic:
            return 'quadratic'
        elif self.ftype == RefFunctionType.rfCubic:
            return 'cubic'
        else:
            return 'Unknown'

    def __repr__(self):
        return 'PolynomModel {0} - {1}'.format(self.neuron_index, RefFunctionType.get_name(self.ref_function_type))

    def describe(self, features, layers):
        s = ['PolynomModel {0} - {1}'.format(self.neuron_index, RefFunctionType.get_name(self.ref_function_type)),
            'u1: {0}'.format(self.get_features_name(self.u1_index, features, layers)),
            'u2: {0}'.format(self.get_features_name(self.u2_index, features, layers)),
            'train error: {0}'.format(self.train_err),
            'validate error: {0}'.format(self.validate_err),
            'bias error: {0}'.format(self.bias_err),
            '; '.join(['w{0}={1}'.format(n, self.w[n]) for n in range(self.w.shape[0])]),
            '||w||^2={ww}'.format(ww=self.w.mean())
        ]
        return '\n'.join(s)


    def get_polynom_inputs(self, ftype, u1_index, u2_index, source):
        """
        function set matrix value required to calculate polynom neuron coefficient
        by multiple linear regression
        """
        u1x = source[:, u1_index]
        u2x = source[:, u2_index]
        a = np.empty((source.shape[0], self.fw_size), dtype=np.double)

        a[:, 0] = 1
        a[:, 1] = u1x
        a[:, 2] = u2x

        if ftype in (RefFunctionType.rfLinearCov,
                     RefFunctionType.rfQuadratic,
                     RefFunctionType.rfCubic):
            a[:, 3] = u1x * u2x

        if ftype in (RefFunctionType.rfQuadratic,
                     RefFunctionType.rfCubic):
            a[:, 3] = u1x * u2x
            a[:, 4] = u1x * u1x
            a[:, 5] = u2x * u2x

        if RefFunctionType.rfCubic == ftype:
            a[:, 3] = u1x * u2x
            a[:, 4] = u1x * u1x
            a[:, 5] = u2x * u2x
            a[:, 6] = a[:, 4] * u1x
            a[:, 7] = a[:, 4] * u2x
            a[:, 8] = a[:, 5] * u1x
            a[:, 9] = a[:, 6] * u2x
        return a

    def _fit_regressor(self, x, y, params):
        a = self.get_polynom_inputs(self.ftype, self.u1_index, self.u2_index, x)
        reg = linear_model.Ridge(alpha=params['l2'], solver='lsqr')
        a2 = a[:, 1:]
        reg.fit(a2, y)
        w = np.empty((len(reg.coef_) + 1,), dtype=np.double)
        w[0] = reg.intercept_
        w[1:] = reg.coef_
        return w

    def _fit_classifier(self, x, y, params):
        a = self.get_polynom_inputs(self.ftype, self.u1_index, self.u2_index, x)
        clf = linear_model.LogisticRegression(C=1.0/params['l2'])
        a2 = a[:, 1:]
        clf.fit(a2, y)
        w = np.empty((clf.coef_.shape[1] + 1,), dtype=np.double)
        w[0] = clf.intercept_
        w[1:] = clf.coef_[0, :]
        return w

    def fit(self, train_x, train_y, validate_x, validate_y, params):
        """
        Train the neuron using train and validate sets
        """
        self.w = self.fit_function(train_x, train_y, params)
        if self.need_bias_stuff(params['criterion_type']):
            self.wt = self.fit_function(validate_x, validate_y, params)

        self.bias_err = 0
        self.valid = True
        # calculate neuron errors
        if self.need_bias_stuff(params['criterion_type']):
            self.bias_err = self.get_bias_err(train_x, validate_x, train_y, validate_y)

        self.train_err = self.get_regularity_err(train_x, train_y)
        self.validate_err = self.get_regularity_err(validate_x, validate_y)


#***********************************************************************************************************************
#   Network layer
#***********************************************************************************************************************
class LayerCreationError(Exception):
    """raised when error happens while layer creation
    """
    def __init__(self, message, layer_index):
        # Call the base class constructor with the parameters it needs
        super(LayerCreationError, self).__init__(message)
        self.layer_index = layer_index


class Layer(list):
    """Layer class of multilayered group method of data handling algorithm
    """

    def __init__(self, model, layer_index, *args):
        list.__init__(self, *args)
        self.layer_index = layer_index
        self.l_count = model.l_count
        self.n_features = model.n_features
        self.err = sys.float_info.max
        self.train_err = sys.float_info.max
        self.valid = True
        self.input_index_set = set([])

    def add_neuron(self, index_u1, index_u2, ftype, model_class, loss):
        """Add polynomial neuron to the layer
        """
        self.add(PolynomNeuron(self.layer_index, index_u1, index_u2, ftype, len(self), model_class, loss))

    def __repr__(self):
        return 'Layer {0}'.format(self.layer_index)

    def describe(self, features, layers):

        s = ['*' * 50,
             'Layer {0}'.format(self.layer_index),
             '*' * 50,
        ]
        for neuron in self:
            s.append(neuron.describe(features, layers))
        return '\n'.join(s)

    def add(self, neuron):
        neuron.neuron_index = len(self)
        self.append(neuron)
        self.input_index_set.add(neuron.u1_index)
        self.input_index_set.add(neuron.u2_index)

    def delete(self, index):
        self.pop(index)
        for n in range(index, len(self)):
            self[n].neuron_index = n
        self.input_index_set.clear()
        for neuron in self:
            self.input_index_set.add(neuron.u1_index)
            self.input_index_set.add(neuron.u2_index)


def fit_layer(fit_layer_data):
    sublayer = fit_layer_data.sublayer
    for neuron in sublayer:
        neuron.fit(fit_layer_data.train_x,
                   fit_layer_data.train_y,
                   fit_layer_data.validate_x,
                   fit_layer_data.validate_y,
                   fit_layer_data.params)
    return sublayer





