from enum import Enum

def get_y_components(data_y):
    return 1 if len(data_y.shape) == 1 else data_y.shape[1]

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
    Divide data set to train and validate class, see MultilayerGMDHparam class for explanation
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
    dsValidate = 1


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
        elif arg == 'bias_retrain':
            return CriterionType.cmpComb_bias_retrain
        else:
            raise ValueError(arg)

class LayerCreationError(Exception):
    """raised when error happens while layer creation
    """
    def __init__(self, message, layer_index):
        # Call the base class constructor with the parameters it needs
        super(LayerCreationError, self).__init__(message)
        self.layer_index = layer_index
