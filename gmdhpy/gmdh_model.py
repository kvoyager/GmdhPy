__author__ = 'Konstantin Kolokolov'


import sys
from gmdhpy.utils import RefFunctionType, CriterionType


# **********************************************************************************************************************
#   Model class
# **********************************************************************************************************************
class Model(object):
    """base class for GMDH model
    """

    def __init__(self, gmdh, layer_index, u1_index, u2_index, model_index):
        self.layer_index = layer_index
        self.model_index = model_index
        self._u1_index = u1_index
        self._u2_index = u2_index
        self.criterion_type = gmdh.param.criterion_type
        self.feature_names = gmdh.feature_names
        self.layers = gmdh.layers
        self.ref_function_type = RefFunctionType.rfUnknown
        self.valid = True
        self.train_err = sys.float_info.max	            # model error on train data set
        self.validate_err = sys.float_info.max	            # model error on validate data set
        self.bias_err = sys.float_info.max	            # bias model error
        self.f = None
        self.fm = None

    def get_u1_index(self):
        return self._u1_index

    def set_u1_index(self, value):
        self._u1_index = value
        if self.f is not None:
            self.f.u1_index = value
        if self.fm is not None:
            self.fm.u1_index = value

    def get_u2_index(self):
        return self._u2_index

    def set_u2_index(self, value):
        self._u2_index = value
        if self.f is not None:
            self.f.u2_index = value
        if self.f is not None:
            self.f.u2_index = value

    u1_index = property(get_u1_index, set_u1_index)
    u2_index = property(get_u2_index, set_u2_index)

    def need_bias_stuff(self):
        if self.criterion_type == CriterionType.cmpValidate:
            return False
        return True

    def get_error(self):
        """Compute error of the model according to specified criterion
        """
        if self.criterion_type == CriterionType.cmpValidate:
            return self.validate_err
        elif self.criterion_type == CriterionType.cmpBias:
            return self.bias_err
        elif self.criterion_type == CriterionType.cmpComb_validate_bias:
            return 0.5*self.bias_err + 0.5*self.validate_err
        elif self.criterion_type == CriterionType.cmpComb_bias_retrain:
            return self.bias_err
        else:
            return sys.float_info.max

    def get_regularity_err(self, x, y):
        """Calculation of regularity error (similar to RMSPE)
        """
        yt = self.transform(x)
        s = ((y - yt) ** 2).sum()
        s2 = (y ** 2).sum()
        err = s / s2
        return err

    def get_sub_bias_err(self, x):
        """Helper function for calculation of unbiased error
        """
        yta = self.f.transform(x)
        ytb = self.fm.transform(x)

        s = ((yta - ytb) ** 2).sum()
        return s

    def get_bias_err(self, train_x, validate_x, train_y, validate_y):
        """Calculation of unbiased error
        """
        s = 0

        s += self.get_sub_bias_err(train_x)
        s += self.get_sub_bias_err(validate_x)
        s2 = (train_y ** 2).sum() + (validate_y ** 2).sum()
        err = s / s2
        return err

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

    def fit(self, index, train_x, train_y, validate_x, validate_y, **fit_params):
        self.fit_params = fit_params
        raise NotImplementedError

    def _refit(self, data_x, data_y):
        raise NotImplementedError

    def transform(self, x):
        return self.f.transform(x)


