__author__ = 'Konstantin Kolokolov'

from enum import Enum
import numpy as np
import statsmodels.api as sm
from numpy import ndarray
import sys
import math


class RefFunctionType(Enum):
    rfUnknown = -1
    rfLinear = 0
    rfLinearPerm = 1
    rfQuadratic = 2
    rfCubic = 3
    #rfHarmonic = 4

    @classmethod
    def get_name(cls, value):
        if value == cls.rfUnknown:
            return 'Unknown'
        elif value == cls.rfLinear:
            return 'Linear'
        elif value == cls.rfLinearPerm:
            return 'LinearPerm'
        elif value == cls.rfQuadratic:
            return 'Quadratic'
        elif value == cls.rfCubic:
            return 'Cubic'
        elif value == cls.rfHarmonic:
            return 'Harmonic'
        else:
            return 'Unknown'


class AlgorithmType(Enum):
    Multilayer = 0
    HarmonicTrend = 1


class SequenceTypeSet(Enum):
    """
    Divide data set to train and test class, see MultilayerGMDHparam class for explanation
    """
    sqCustom = 0
    sqMode1 = 1
    sqMode2 = 2
    sqMode3_1 = 3
    sqMode3_2 = 4
    sqMode4_1 = 5
    sqMode4_2 = 6
    sqRandom = 7

    @classmethod
    def is_mode1_type(cls, seq_type):
        if seq_type == cls.sqMode1 or seq_type == cls.sqMode3_1 or seq_type == cls.sqMode4_1:
            return True
        else:
            return False

    @classmethod
    def is_mode2_type(cls, seq_type):
        if seq_type == cls.sqMode2 or seq_type == cls.sqMode3_2 or seq_type == cls.sqMode4_2:
            return True
        else:
            return False


class SequenceTypeError(Exception):
    """raised when unknown data type sequence applied
    """
    pass


class DataSetType(Enum):
    dsTrain = 0
    dsTest = 1


class CriterionType(Enum):
    cmpTest = 1
    cmpBias = 2
    cmpComb_train_bias = 3
    cmpComb_test_bias = 4
    cmpComb_bias_retrain = 5

    @classmethod
    def get_name(cls, value):
        if value == cls.cmpTest:
            return 'test error comparison'
        elif value == cls.cmpBias:
            return 'bias error comparison'
        elif value == cls.cmpComb_train_bias:
            return 'bias and train error comparison'
        elif value == cls.cmpComb_test_bias:
            return 'bias and test error comparison'
        elif value == cls.cmpComb_bias_retrain:
            return 'bias error comparison with retrain'
        else:
            return 'Unknown'


# **********************************************************************************************************************
#   Model class
# **********************************************************************************************************************
class Model:
    """base class for GMDH model
    """

    def __init__(self, gmdh, layer_index, u1_index, u2_index, model_index):
        self.layer_index = layer_index
        self.model_index = model_index
        self.u1_index = u1_index
        self.u2_index = u2_index
        self.gmdh = gmdh
        self.ref_function_type = RefFunctionType.rfUnknown
        self.valid = True
        self.train_err = sys.float_info.max	            # model error on train data set
        self.test_err = sys.float_info.max	            # model error on test data set
        self.bias_err = sys.float_info.max	            # bias model error
        self.transfer = self._transfer_dummy            # transfer function

    @classmethod
    def _transfer_dummy(cls, u1, u2, w):
        return 0

    def get_error(self):
        """Compute error of the model according to specified criterion
        """
        if self.gmdh.param.criterion_type == CriterionType.cmpTest:
            return self.test_err
        elif self.gmdh.param.criterion_type == CriterionType.cmpBias:
            return self.bias_err
        elif self.gmdh.param.criterion_type == CriterionType.cmpComb_train_bias:
            return 0.5*self.bias_err + 0.5*self.train_err
        elif self.gmdh.param.criterion_type == CriterionType.cmpComb_test_bias:
            return 0.5*self.bias_err + 0.5*self.test_err
        elif self.gmdh.param.criterion_type == CriterionType.cmpComb_bias_retrain:
            return self.bias_err
        else:
            return sys.float_info.max

    def mse(self, x, y, w):
        """Calculation of error using MSE criterion
        """
        sy = 0
        data_len = x.shape[0]
        yt = np.empty((data_len,), dtype=np.double)
        x1 = x[:, self.u1_index]
        x2 = x[:, self.u2_index]
        for m in range(0, data_len):
            yt[m] = self.transfer(x1[m], x2[m], w)
        #vtransfer = np.vectorize(self.transfer, otypes=[np.float])
        #vtransfer.excluded.add(2)
        #yt = vtransfer(x1, x2, w)

        s = ((y - yt)**2).mean()
        err = math.sqrt(s)
        return err

    def bias(self, train_x, test_x, train_y, test_y):
        """Calculation of error using of bias criterion
        """
        s = 0
        sy = 0

        n_train = train_x.shape[0]
        n_test = test_x.shape[0]
        data_len = n_train + n_test
        x1 = train_x[:, self.u1_index]
        x2 = train_x[:, self.u2_index]
        yta = np.empty((n_train,), dtype=np.double)
        ytb = np.empty((n_train,), dtype=np.double)
        for m in range(0, n_train):
            yta[m] = self.transfer(x1[m], x2[m], self.w)
            ytb[m] = self.transfer(x1[m], x2[m], self.wt)
            #sy += train_y[m]**2
        s = ((yta - ytb)**2).mean()

        x1 = test_x[:, self.u1_index]
        x2 = test_x[:, self.u2_index]
        yta = np.empty((n_test,), dtype=np.double)
        ytb = np.empty((n_test,), dtype=np.double)
        for m in range(0, n_test):
            yta[m] = self.transfer(x1[m], x2[m], self.w)
            ytb[m] = self.transfer(x1[m], x2[m], self.wt)
            #sy += self.test_y[m]**2
        s += ((yta - ytb)**2).mean()

        #err = math.sqrt(s / sy / data_len)
        err = math.sqrt(s / data_len)
        return err

    def get_features_name(self, input_index):
        if self.layer_index == 0:
            s = 'index=inp_{0}'.format(input_index)
            if len(self.gmdh.feature_names) > 0:
                s += ', {0}'.format(self.gmdh.feature_names[input_index])
        else:
            models_num = len(self.gmdh.layers[self.layer_index-1])
            if input_index < models_num:
                s = 'index=prev_layer_model_{0}'.format(input_index)
            else:
                s = 'index=inp_{0}'.format(input_index - models_num)
                if len(self.gmdh.feature_names) > 0:
                    s += ', {0}'.format(self.gmdh.feature_names[input_index - models_num])
        return s

    def get_name(self):
        return 'Not defined'

    def get_short_name(self):
        return 'Not defined'


# **********************************************************************************************************************
#   Polynomial model class
# **********************************************************************************************************************
class PolynomModel(Model):
    """Polynomial GMDH model class
    """

    def __init__(self, gmdh, layer_index, u1_index, u2_index, ftype, model_index):
        super(PolynomModel, self).__init__(gmdh, layer_index, u1_index, u2_index, model_index)
        self.ftype = ftype
        self.fw_size = 0
        self.set_type(ftype)
        self.w = np.array([self.fw_size], dtype=np.double)
        self.wt = np.array([self.fw_size], dtype=np.double)

    @classmethod
    def _transfer_linear(cls, u1, u2, w):
        return w[0] + w[1]*u1 + w[2]*u2
    
    @classmethod
    def _transfer_linear_perm(cls, u1, u2, w):
        return w[0] + u1*(w[1] + w[3]*u2) + w[2]*u2
    
    @classmethod
    def _transfer_quadratic(cls, u1, u2, w):
        return w[0] + u1*(w[1] + w[3]*u2 + w[4]*u1) + u2*(w[2] + w[5]*u2)
    
    @classmethod
    def _transfer_cubic(cls, u1, u2, w):
        u1_sq = u1*u1
        u2_sq = u2*u2
        return w[0] + w[1]*u1 + w[2]*u2 + w[3]*u1*u2 + w[4]*u1_sq + w[5]*u2_sq + \
            w[6]*u1*u1_sq + w[7]*u1_sq*u2 + w[8]*u1*u2_sq + w[9]*u2*u2_sq

    def set_type(self, new_type):
        self.ref_function_type = new_type
        if new_type == RefFunctionType.rfLinear:
            self.transfer = self._transfer_linear
            self.fw_size = 3
        elif new_type == RefFunctionType.rfLinearPerm:
            self.transfer = self._transfer_linear_perm
            self.fw_size = 4
        elif new_type == RefFunctionType.rfQuadratic:
            self.transfer = self._transfer_quadratic
            self.fw_size = 6
        elif new_type == RefFunctionType.rfCubic:
            self.transfer = self._transfer_cubic
            self.fw_size = 10
        else:
            self.transfer = self._transfer_dummy
            self.fw_size = 0

    def get_name(self):
        if self.ftype == RefFunctionType.rfLinear:
            return 'w0 + w1*xi + w2*xj'
        elif self.ftype == RefFunctionType.rfLinearPerm:
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
        elif self.ftype == RefFunctionType.rfLinearPerm:
            return 'linear perm'
        elif self.ftype == RefFunctionType.rfQuadratic:
            return 'quadratic'
        elif self.ftype == RefFunctionType.rfCubic:
            return 'cubic'
        else:
            return 'Unknown'

    def __repr__(self):
        s = 'PolynomModel {0} - {1}\n'.format(self.model_index, RefFunctionType.get_name(self.ref_function_type))
        s += 'u1: {0}\n'.format(self.get_features_name(self.u1_index))
        s += 'u2: {0}\n'.format(self.get_features_name(self.u2_index))
        s += 'train error: {0}\n'.format(self.train_err)
        s += 'test error: {0}\n'.format(self.test_err)
        s += 'bias error: {0}\n'.format(self.bias_err)
        for n in range(0, self.w.shape[0]):
            s += 'w{0}={1}'.format(n, self.w[n])
            if n < self.w.shape[0] - 1:
                s += '; '
            else:
                s += '\n'
        return s


#***********************************************************************************************************************
#   GMDH layer
#***********************************************************************************************************************
class LayerCreationError(Exception):
    """raised when error happens while layer creation
    """
    def __init__(self, message, layer_index):
        # Call the base class constructor with the parameters it needs
        super(LayerCreationError, self).__init__(message)
        self.layer_index = layer_index


class Layer(list):
    """Layer class of multilayer group method of data handling algorithm
    """

    def __init__(self, gmdh, layer_index, *args):
        list.__init__(self, *args)
        self.gmdh = gmdh
        self.layer_index = layer_index
        self.l_count = gmdh.l_count
        self.n_features = gmdh.n_features
        self.err = sys.float_info.max
        self.valid = True
        self.input_index_set = set([])

    def add_polynomial_model(self, index_u1, index_u2, ftype):
        """Add polynomial model to the layer
        """
        self.add(PolynomModel(self.gmdh, self.layer_index, index_u1, index_u2, ftype, len(self)))

    def __repr__(self):
        st = '*********************************************\n'
        s = st
        s += 'Layer {0}\n'.format(self.layer_index)
        s += st
        for n, model in enumerate(self):
            s += '\n'
            s += model.__repr__()
            if n == len(self) - 1:
                s += '\n'
        return s

    def add(self, model):
        model.model_index = len(self)
        self.append(model)
        self.input_index_set.add(model.u1_index)
        self.input_index_set.add(model.u2_index)

    def delete(self, index):
        self.pop(index)
        for n in range(index, len(self)):
            self[n].model_index = n
        self.input_index_set.clear()
        for model in self:
            self.input_index_set.add(model.u1_index)
            self.input_index_set.add(model.u2_index)


'''
#***********************************************************************************************************************
#   GMDH layers
#***********************************************************************************************************************

class Layers(list):
    """
    Layers class of multilayer group method of data handling algorithm
    """

    def __init__(self, *args):
        list.__init__(self, *args)

    #def delete(self, index: int):
    #    self.pop(index)

    def append(self, layer: Layer):
'''
