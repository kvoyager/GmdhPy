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
        self.transfer = None            # transfer function

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
            if len(self.feature_names) > 0:
                s += ', {0}'.format(self.feature_names[input_index])
        else:
            models_num = len(self.layers[self.layer_index-1])
            if input_index < models_num:
                s = 'index=prev_layer_model_{0}'.format(input_index)
            else:
                s = 'index=inp_{0}'.format(input_index - models_num)
                if len(self.feature_names) > 0:
                    s += ', {0}'.format(self.feature_names[input_index - models_num])
        return s

    def get_name(self):
        return 'Not defined'

    def get_short_name(self):
        return 'Not defined'

    def fit(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError



import sys
if sys.version_info.major == 2:
    import copy_reg as cpr
    cpr.pickle(types.MethodType, _pickle_method, _unpickle_method)



