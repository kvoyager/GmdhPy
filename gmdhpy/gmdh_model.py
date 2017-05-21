__author__ = 'Konstantin Kolokolov'

from enum import Enum
import numpy as np
import types

def _pickle_method(method):
     # if method.im_self is None:
     #    return getattr, (method.im_class, method.im_func. func_name)
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'):
        #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_%s%s' % (cls_name, func_name)
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    if obj and func_name in obj.__dict__:
        cls, obj = obj, None # if func_name is classmethod
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


def _transfer_dummy(cls, u1, u2, w):
    return 0


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

    @staticmethod
    def get(arg):
        if isinstance(arg, SequenceTypeSet):
            return arg
        elif arg == 'custom':
            return SequenceTypeSet.sqCustom
        elif arg == 'mode1':
            return SequenceTypeSet.sqMode1
        elif arg == 'mode2':
            return SequenceTypeSet.sqMode2
        elif arg == 'mode3_1':
            return SequenceTypeSet.sqMode3_1
        elif arg == 'mode3_2':
            return SequenceTypeSet.sqMode3_2
        elif arg == 'mode4_1':
            return SequenceTypeSet.sqMode4_1
        elif arg == 'mode4_2':
            return SequenceTypeSet.sqMode4_2
        elif arg == 'random':
            return SequenceTypeSet.sqRandom
        else:
            raise ValueError(arg)


class DataSetType(Enum):
    dsTrain = 0
    dsTest = 1


class CriterionType(Enum):
    cmpTest = 1
    cmpBias = 2
    cmpComb_test_bias = 4
    cmpComb_bias_retrain = 5

    @classmethod
    def get_name(cls, value):
        if value == cls.cmpTest:
            return 'test error comparison'
        elif value == cls.cmpBias:
            return 'bias error comparison'
        elif value == cls.cmpComb_test_bias:
            return 'bias and test error comparison'
        elif value == cls.cmpComb_bias_retrain:
            return 'bias error comparison with retrain'
        else:
            return 'Unknown'

    @staticmethod
    def get(arg):
        if isinstance(arg, CriterionType):
            return arg
        elif arg == 'test':
            return CriterionType.cmpTest
        elif arg == 'bias':
            return CriterionType.cmpBias
        elif arg == 'test_bias':
            return CriterionType.cmpComb_test_bias
        elif arg == 'bias_retrain':
            return CriterionType.cmpComb_bias_retrain
        else:
            raise ValueError(arg)


# **********************************************************************************************************************
#   Model class
# **********************************************************************************************************************
class Model(object):
    """base class for GMDH model
    """

    def __init__(self, gmdh, layer_index, u1_index, u2_index, model_index):
        self.layer_index = layer_index
        self.model_index = model_index
        self.u1_index = u1_index
        self.u2_index = u2_index
        self.criterion_type = gmdh.param.criterion_type
        self.feature_names = gmdh.feature_names
        self.layers = gmdh.layers
        self.ref_function_type = RefFunctionType.rfUnknown
        self.valid = True
        self.train_err = sys.float_info.max	            # model error on train data set
        self.test_err = sys.float_info.max	            # model error on test data set
        self.bias_err = sys.float_info.max	            # bias model error
        self.transfer = _transfer_dummy            # transfer function

    # @classmethod
    # def _transfer_dummy(cls, u1, u2, w):
    #     return 0

    def need_bias_stuff(self):
        if self.criterion_type == CriterionType.cmpTest:
            return False
        return True

    def get_error(self):
        """Compute error of the model according to specified criterion
        """
        if self.criterion_type == CriterionType.cmpTest:
            return self.test_err
        elif self.criterion_type == CriterionType.cmpBias:
            return self.bias_err
        elif self.criterion_type == CriterionType.cmpComb_test_bias:
            return 0.5*self.bias_err + 0.5*self.test_err
        elif self.criterion_type == CriterionType.cmpComb_bias_retrain:
            return self.bias_err
        else:
            return sys.float_info.max

    @staticmethod
    def get_regularity_err(u1_index, u2_index, transfer, x, y, w):
        raise NotImplementedError

    def get_bias_err(u1_index, u2_index, transfer, train_x, test_x, train_y, test_y, w, wt):
        raise NotImplementedError

    def get_features_name(self, input_index):
        if self.layer_index == 0:
            s = 'index=inp_{0}'.format(input_index)
            if self.feature_names is not None and len(self.feature_names) > 0:
                s += ', {0}'.format(self.feature_names[input_index])
        else:
            models_num = len(self.layers[self.layer_index-1])
            if input_index < models_num:
                s = 'index=prev_layer_model_{0}'.format(input_index)
            else:
                s = 'index=inp_{0}'.format(input_index - models_num)
                if self.feature_names is not None and len(self.feature_names) > 0:
                    s += ', {0}'.format(self.feature_names[input_index - models_num])
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

    def copy_result(self, source):
        self.w = np.copy(source[1])
        self.wt = np.copy(source[2])
        self.valid = source[3]
        self.bias_err = source[4]
        self.train_err = source[5]
        self.test_err = source[6]


    def _transfer_linear(self, u1, u2, w):
        return w[0] + w[1]*u1 + w[2]*u2

    def _transfer_linear_cov(self, u1, u2, w):
        return w[0] + u1*(w[1] + w[3]*u2) + w[2]*u2

    def _transfer_quadratic(self, u1, u2, w):
        return w[0] + u1*(w[1] + w[3]*u2 + w[4]*u1) + u2*(w[2] + w[5]*u2)

    def _transfer_cubic(self, u1, u2, w):
        u1_sq = u1*u1
        u2_sq = u2*u2
        return w[0] + w[1]*u1 + w[2]*u2 + w[3]*u1*u2 + w[4]*u1_sq + w[5]*u2_sq + \
            w[6]*u1*u1_sq + w[7]*u1_sq*u2 + w[8]*u1*u2_sq + w[9]*u2*u2_sq

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
            self.transfer = _transfer_dummy
            self.fw_size = 0

    @staticmethod
    def get_regularity_err(u1_index, u2_index, transfer, x, y, w):
        """Calculation of regularity error
        """
        data_len = x.shape[0]
        x1 = x[:, u1_index]
        x2 = x[:, u2_index]
        yt = np.empty((data_len,), dtype=np.double)

        for m in range(0, data_len):
            yt[m] = transfer(x1[m], x2[m], w)

        s = ((y - yt) ** 2).sum()
        s2 = (y ** 2).sum()
        err = s/s2
        return err

    @staticmethod
    def get_sub_bias_err(u1_index, u2_index, transfer, x, len, w, wt):
        """Helper function for calculation of unbiased error
        """
        x1 = x[:, u1_index]
        x2 = x[:, u2_index]
        yta = np.empty((len,), dtype=np.double)
        ytb = np.empty((len,), dtype=np.double)

        for m in range(0, len):
            yta[m] = transfer(x1[m], x2[m], w)
            ytb[m] = transfer(x1[m], x2[m], wt)

        s = ((yta - ytb) ** 2).sum()
        return s

    @staticmethod
    def get_bias_err(u1_index, u2_index, transfer, train_x, test_x, train_y, test_y, w, wt):
        """Calculation of unbiased error
        """
        s = 0
        n_train = train_x.shape[0]
        n_test = test_x.shape[0]
        s += PolynomModel.get_sub_bias_err(u1_index, u2_index, transfer, train_x, n_train, w, wt)
        s += PolynomModel.get_sub_bias_err(u1_index, u2_index, transfer, test_x, n_test, w, wt)
        s2 = (train_y ** 2).sum() + (test_y ** 2).sum()
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
        s += '||w||^2={ww}'.format(ww=self.w.mean())
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
    """Layer class of multilayered group method of data handling algorithm
    """

    def __init__(self, gmdh, layer_index, *args):
        list.__init__(self, *args)
        self.layer_index = layer_index
        self.l_count = gmdh.l_count
        self.n_features = gmdh.n_features
        self.err = sys.float_info.max
        self.train_err = sys.float_info.max
        self.valid = True
        self.input_index_set = set([])

    def add_polynomial_model(self, gmdh, index_u1, index_u2, ftype):
        """Add polynomial model to the layer
        """
        self.add(PolynomModel(gmdh, self.layer_index, index_u1, index_u2, ftype, len(self)))

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


import sys
if sys.version_info.major == 2:
    import copy_reg as cpr
    cpr.pickle(types.MethodType, _pickle_method, _unpickle_method)



