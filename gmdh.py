"""
Multilayer group method of data handling of Machine learning for Python
==================================

"""
__author__ = 'Konstantin Kolokolov'
__version__ = '0.1.0'

import timeit
import numpy as np
from numpy import ndarray
import sys
from gmdh_model import RefFunctionType, CriterionType, SequenceTypeSet, SequenceTypeError
from gmdh_model import DataSetType, Model, PolynomModel, Layer, LayerCreationError
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# **********************************************************************************************************************
#   implementation of multilayer GMDH algorithm
# **********************************************************************************************************************
class GMDHCheckError(Exception):
    """
    GMDHCheckError raised if some parameters did not pass check before training
    """
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(GMDHCheckError, self).__init__(message)


class MultilayerGMDHparam(object):
    """
    Parameters of GMDH algorithm
    ----------------------------
    admix_features - if set to true the original features will be added to the list of features of each layer
        default value is true

    criterion_type - criterion for selecting best models
    the following criteria are possible:
        cmpTest: the default value,
            models are compared on the basis of test error
        cmpBias: models are compared on the basis of bias error
        cmpComb_train_bias: combined criterion, models are compared on the basis of bias and train errors
        cmpComb_test_bias: combined criterion, models are compared on the basis of bias and test errors
        cmpComb_bias_retrain: firstly, models are compared on the basis of bias error, then models are retrain
            on the total data set (train and test)
    example of using:
        from gmdh import MultilayerGMDH, CriterionType
        gmdh = MultilayerGMDH()
        gmdh.param.criterion_type = CriterionType.cmpComb_bias_retrain

    seq_type - method to split data set to train and test
        sqMode1 - 	the default value
                    data set is split to train and test data sets in the following way:
                    ... train test train test train test ... train test.
                    The last point is chosen to belong to test set
        sqMode2 - 	data set is split to train and test data sets in the following way:
                    ... train test train test train test ... test train.
                    The last point is chosen to belong to train set
        sqMode3_1 - data set is split to train and test data sets in the following way:
                    ... train test train train test train train test ... train train test.
                    The last point is chosen to belong to test set
        sqMode4_1 - data set is split to train and test data sets in the following way:
                    ... train test train train train test ... test train train train test.
                    The last point is chosen to belong to test set
        sqMode3_2 - data set is split to train and test data sets in the following way:
                    ... train test test train test test train test ... test test train.
                    The last point is chosen to belong to train set
        sqMode4_2 - data set is split to train and test data sets in the following way:
                    ... train test test test train test ... train test test test train.
                    The last point is chosen to belong to train set
        sqRandom -  Random split data to train and test
        sqCustom -  custom way to split data to train and test
                    set_custom_seq_type has to be provided
                    Example:
                    def my_set_custom_sequence_type(seq_types):
                        r = np.random.uniform(-1, 1, seq_types.shape)
                        seq_types[:] = np.where(r > 0, DataSetType.dsTrain, DataSetType.dsTest)
                    gmdh.param.seq_type = SequenceTypeSet.sqCustom
                    gmdh.param.set_custom_seq_type = my_set_custom_sequence_type
    example of using:
        from gmdh import MultilayerGMDH, SequenceTypeSet
        gmdh = MultilayerGMDH()
        gmdh.param.seq_type = SequenceTypeSet.sqMode2

    max_layer_count - maximum number of layers,
        the default value is mostly infinite (sys.maxsize)

    criterion_minimum_width - minimum number of layers at the right required to evaluate optimal number of layer
        (the optimal model) according to the minimum of criteria. For example, if it is found that
         criterion value has minimum at layer with index 10, the algorithm will proceed till the layer
         with index 15
         the default value is 5

    manual_min_l_count_value - if this value set to False, the number of best models to be
        selected is determined automatically and it is equal number of original features.
        Otherwise the number of best models to be selected is determined as
        max(original features, min_l_count). min_l_count has to be provided
        For example, if you have N=10 features, the number of all generated models will be at least
        N*(N-1)/2=45, the number of selected best models will be 10, but you increase this number to
        20 by setting manual_min_l_count_value = True and min_l_count = 20
        Note: if min_l_count is larger than number of generated models of the layer it will be reduced
        to that number
    example of using:
        from gmdh import MultilayerGMDH
        gmdh = MultilayerGMDH()
        gmdh.param.manual_min_l_count_value = True
        gmdh.param.min_l_count = 20

    ref_function_types - set of reference functions, by default the set contains polynom
        of the second degree: y = w0 + w1*x1 + w2*x2 + w3*x1*x2
        you can add other reference functions in the following way:
        param.ref_function_types.add(RefFunctionType.rfLinear) - y = w0 + w1*x1 + w2*x2
        param.ref_function_types.add(RefFunctionType.rfQuadratic) - full polynom of the 2-nd degree
        param.ref_function_types.add(RefFunctionType.rfCubic) - full polynom of the 3-rd degree

    """
    def __init__(self):
        self.admix_features = True
        self.criterion_type = CriterionType.cmpTest
        self.seq_type = SequenceTypeSet.sqMode1
        self.set_custom_seq_type = self.__dummy_set_custom_sequence_type
        self.max_layer_count = sys.maxsize
        self.criterion_minimum_width = 5
        self.manual_min_l_count_value = False
        self.min_l_count = 0
        self.ref_function_types = set([RefFunctionType.rfLinearPerm])

    @classmethod
    def __dummy_set_custom_sequence_type(cls, seq_types):
        raise SequenceTypeError('param.set_custom_seq_type function is not provided')


class BaseMultilayerGMDH(object):
    """
    Base class for multilayer group method of data handling algorithm
    """

    def __init__(self):
        self.param = MultilayerGMDHparam()                  # parameters
        self.l_count = 0                                    # number of best models to be selected
        self.layers = []                                    # list of gmdh layers
        self.feature_names = []                             # name of inputs, used to print GMDH
        self.n_features = 0                                 # number of original features
        self.n_train = 0                                    # number of train samples
        self.n_test = 0                                     # number of test samples

        # array specified how the original data will be divided into train and test data sets
        self.seq_types = np.array([], dtype=DataSetType)
        self.data_x = np.array([], dtype=np.double)             # array of original data samples
        self.data_y = np.array([], dtype=np.double)             # array of original target samples
        self.input_train_x = np.array([], dtype=np.double)      # array of original train samples
        self.input_test_x = np.array([], dtype=np.double)       # array of original test samples
        self.train_y = np.array([], dtype=np.double)            # array of train targets
        self.test_y = np.array([], dtype=np.double)             # array of test targets

        self.train_x = np.array([], dtype=np.double)            # array of train samples for current layer
        self.test_x = np.array([], dtype=np.double)             # array of test samples for current layer
        self.layer_data_x = np.array([], dtype=np.double)       # array of total samples for current layer

        self.layer_err = np.array([], dtype=np.double)          # array of layer's errors
        self.valid = False

    def _select_best_models(self, layer):
        """
        Select l_count the best models from the list
        """

        if self.param.manual_min_l_count_value:
            layer.l_count = max(layer.l_count, self.param.min_l_count)
        # the number of selected best models can't be larger than the total number of models in the layer
        if self.param.admix_features and layer.layer_index == 0:
            layer.l_count = min(layer.l_count, 2*len(layer))
        else:
            layer.l_count = min(layer.l_count, len(layer))

        # check the validity of the models
        for n in range(0, len(layer)):
            if not layer[n].valid:
                layer.l_count = min(layer.l_count, n)
                break

        layer.valid = layer.l_count > 0
        # if layer.l_count == 0:
        #    raise LayerCreationError('Layer is not valid', layer.layer_index)

        # models are already sorted, the layer error is the error of the first model
        # (the model with smallest error according to the specified criterion)
        layer.err = layer[0].get_error()


class MultilayerGMDH(BaseMultilayerGMDH):
    """
    Multilayer group method of data handling algorithm with polynomial partial descriptions(models)

    Example of using:

        data_x - training data, numpy array shape [n_samples,n_features]
        data_y - target values, numpy array of shape [n_samples]
        exam_x - predicting data, numpy array of shape [exam_n_samples, n_features]

        from gmdh import MultilayerGMDH
        gmdh = MultilayerGMDH()
        gmdh.fit(data_x, data_y)
        exam_y = gmdh.predict(exam_x)

    Example of using with parameters specification:

        from gmdh import MultilayerGMDH, CriterionType
        gmdh = MultilayerGMDH()
        gmdh.param.criterion_type = CriterionType.cmpComb_bias_retrain
        gmdh.fit(data_x, data_y)
        exam_y = gmdh.predict(exam_x)

    Check MultilayerGMDHparam for details of parameters specification
    """

    def __init__(self):
        super(self.__class__, self).__init__()

    def __repr__(self):
        st = '*********************************************\n'
        s = st
        s += 'GMDH\n'
        s += st
        s += 'Number of layers: {0}\n'.format(len(self.layers))
        s += 'Max possible number of layers: {0}\n'.format(self.param.max_layer_count)

        s += 'Model selection criterion: {0}\n'.format(CriterionType.get_name(self.param.criterion_type))
        s += 'Number of features: {0}\n'.format(self.n_features)
        s += 'Include features to inputs list for each layer: {0}\n'.format(self.param.admix_features)
        s += 'Data size: {0}\n'.format(self.data_x.shape[0])
        s += 'Train data size: {0}\n'.format(self.n_train)
        s += 'Test data size: {0}\n'.format(self.n_test)
        s += 'Selected features by index: {0}\n'.format(self.get_selected_features())
        s += 'Selected features by name: {0}\n'.format(self.get_selected_n_featuresames())
        s += 'Unselected features by index: {0}\n'.format(self.get_unselected_features())
        s += 'Unselected features by name: {0}\n'.format(self.get_unselected_n_featuresames())
        s += '\n'
        for n, layer in enumerate(self.layers):
            s += layer.__repr__()
        return s

    @classmethod
    def set_matrix_a(cls, model, source, source_y, a, y):
        """
        function set matrix value required to calculate polynom model coefficient
        by multiple linear regression
        """
        m = source.shape[0]
        u1x = source[:, model.u1_index]
        u2x = source[:, model.u2_index]
        for mi in range(0, m):
            # compute inputs for model
            u1 = u1x[mi]
            u2 = u2x[mi]
            a[mi, 0] = 1
            a[mi, 1] = u1
            a[mi, 2] = u2
            if RefFunctionType.rfLinearPerm == model.ftype:
                a[mi, 3] = u1 * u2
            if RefFunctionType.rfQuadratic == model.ftype:
                a[mi, 3] = u1 * u2
                u1_sq = u1 * u1
                u2_sq = u2 * u2
                a[mi, 4] = u1_sq
                a[mi, 5] = u2_sq
            if RefFunctionType.rfCubic == model.ftype:
                a[mi, 3] = u1 * u2
                u1_sq = u1 * u1
                u2_sq = u2 * u2
                a[mi, 4] = u1_sq
                a[mi, 5] = u2_sq
                a[mi, 6] = u1_sq * u1
                a[mi, 7] = u1_sq * u2
                a[mi, 8] = u2_sq * u1
                a[mi, 9] = u2_sq * u2
        y[:] = source_y

    def _new_layer_with_all_models(self):
        """
        Generate new layer with all possible models
        """
        layers_count = len(self.layers)
        layer = Layer(self, layers_count)

        if layers_count == 0:
            # the first layer, number of inputs equals to the number of the original features
            n = self.n_features
        else:
            # all other layers: number of inputs equals to the number of selected models from the previous layer
            # plus number of the original features if param.admix_features is True
            n = self.layers[layers_count - 1].l_count
            if self.param.admix_features:
                n += self.n_features

        # number of all possible combination of input pairs is N = (n * (n-1)) / 2
        # add all models to the layer
        for u1 in range(0, n):
            for u2 in range(u1 + 1, n):

                # y = w0 + w1*x1 + w2*x2
                if RefFunctionType.rfLinear in self.param.ref_function_types:
                    layer.add_polynomial_model(u1, u2, RefFunctionType.rfLinear)

                # y = w0 + w1*x1 + w2*x2 + w3*x1*x2
                if RefFunctionType.rfLinearPerm in self.param.ref_function_types:
                    layer.add_polynomial_model(u1, u2, RefFunctionType.rfLinearPerm)

                # y = full polynom of the 2-nd degree
                if RefFunctionType.rfQuadratic in self.param.ref_function_types:
                    layer.add_polynomial_model(u1, u2, RefFunctionType.rfQuadratic)

                # y = full polynom of the 3-rd degree
                if RefFunctionType.rfCubic in self.param.ref_function_types:
                    layer.add_polynomial_model(u1, u2, RefFunctionType.rfCubic)

        if len(layer) == 0:
            raise LayerCreationError('Error creating layer. No functions were generated', layer.layer_index)

        return layer

    def _retrain(self, layer, model):
        """
        Train model on total (original) data set (train and test sets)
        """
        data_len = self.n_train + self.n_test
        a = np.empty((data_len, model.fw_size), dtype=np.double)
        y = np.empty((data_len,), dtype=np.double)
        self.set_matrix_a(model, self.layer_data_x, self.data_y, a, y)
        # Train the model using all data (train and test sets)
        try:
            model.w, resid_a, rank_a, sigma_a = np.linalg.lstsq(a, y)
        except:
            raise LayerCreationError('Error executing linalg on train data set', layer.layer_index)

    def _sub_calculate_model_weights(self, from_m, to_m, layer):
        """
        Calculate model coefficients
        """

        # declare Y variables of multiple linear regression for train and test targets
        ya = np.empty((self.n_train,), dtype=np.double)
        yb = np.empty((self.n_test,), dtype=np.double)
        min_rank = 2

        # loop for all models in the layer starting from 'from_m' to 'to_m'
        for n in range(from_m, to_m):
            model = layer[n]
            if isinstance(model, PolynomModel):
                # declare X variables of multiple linear regression for train and test targets
                a = np.empty((self.n_train, model.fw_size), dtype=np.double)
                b = np.empty((self.n_test, model.fw_size), dtype=np.double)

                # set X and Y variables of multiple linear regression
                self.set_matrix_a(model, self.train_x, self.train_y, a, ya)
                self.set_matrix_a(model, self.test_x,  self.test_y, b, yb)

                # Train the model using train and test sets
                try:
                    model.w, resid_a, rank_a, sigma_a = np.linalg.lstsq(a, ya)
                except:
                    raise LayerCreationError('Error executing linalg on train data set', layer.layer_index)
                try:
                    model.wt, resid_b, rank_b, sigma_b = np.linalg.lstsq(b, yb)
                except:
                    raise LayerCreationError('Error executing linalg on test data set' , layer.layer_index)
                model.valid = True
                if rank_a < min_rank or rank_b < min_rank:
                    model.valid = False
                    self.train_err = sys.float_info.max
                    self.test_err = sys.float_info.max
                    self.bias_err = sys.float_info.max
                else:
                    model.valid = True
                    # calculate model errors
                    model.bias_err = model.bias(self.train_x, self.test_x, self.train_y, self.test_y)
                    # if criterion is cmpComb_bias_retrain we need to retrain model on total data set
                    # before calculate train and test errors
                    if self.param.criterion_type == CriterionType.cmpComb_bias_retrain:
                        self._retrain(layer, model)
                    model.train_err = model.mse(self.train_x, self.train_y, model.w)
                    model.test_err = model.mse(self.test_x, self.test_y, model.wt)

    def _calculate_model_weights(self, layer):
        """
        Calculate model coefficients
        """
        #   parallel computing can be implemented here
        #   number of model to be paralleled = len(layer)
        self._sub_calculate_model_weights(0, len(layer), layer)
        return

    def _create_layer(self):
        """
        Create new layer, calculate models coefficients, select best models
        """

        # compute features for the layer to be created for train and test data sets
        # if the there are no previous layers just copy original features
        if len(self.layers) > 0:
            prev_layer = self.layers[-1]
            self.train_x = self._set_internal_data(prev_layer, self.input_train_x, self.train_x)
            self.test_x = self._set_internal_data(prev_layer, self.input_test_x, self.test_x)
            if self.param.criterion_type == CriterionType.cmpComb_bias_retrain:
                self.layer_data_x = self._set_internal_data(prev_layer, self.data_x, self.layer_data_x)
        else:
            self.train_x = np.array(self.input_train_x, copy=True)
            self.test_x = np.array(self.input_test_x, copy=True)
            self.layer_data_x = np.array(self.data_x, copy=True)
            prev_layer = None

        # create new layer with all possible models
        layer = self._new_layer_with_all_models()

        # calculate model coefficients (weights)
        self._calculate_model_weights(layer)

        # sort models in ascending error order according to specified criterion
        layer.sort(key=lambda x: x.get_error())

        # reset model indexes
        for n, model in enumerate(layer):
            model.model_index = n

        # select l_count best models from the list and check the models validity
        self._select_best_models(layer)

        # delete unused models keeping only l_count best models
        del layer[layer.l_count:]

        # add created layer
        self.layers.append(layer)

        return layer

    def _set_internal_data(self, layer, data, x):
        """Compute inputs(features) for the layer sdfas
        data - original features of algorithm , the dimensionality is (data size) x (number of original features)
        x is the output of selected models from the previous layer
        """

        data_m = data.shape[0]
        if layer is None:
            # we are dealing with the first layer, its features are original features of the algorithm
            # just copy them
            out_x = np.array(data, copy=True)
        else:
            # we are dealing with the second or higher number layer
            # its features are outputs of the previous layer
            # we need to compute them
            n = layer.l_count
            if self.param.admix_features:
                n += data.shape[1]
            out_x = np.zeros([data_m, n], dtype=np.double)
            for i in range(0, data_m):
                for j in range(0, layer.l_count):
                    # loop for selected best models
                    model = layer[j]
                    u1 = x[i, model.u1_index]
                    u2 = x[i, model.u2_index]
                    out_x[i, j] = model.transfer(u1, u2, model.w)
                # Important !!!
                # if parameter admix_features set to true we need to add original features to
                # the current features of the layer
                if self.param.admix_features:
                    for j in range(0, data.shape[1]):
                        # loop for original features
                        out_x[i, layer.l_count + j] = data[i, j]

        return out_x

    def _get_split_numbers(self):
        """
        Compute sizes of train and test data sets
        """
        self.n_train = 0
        self.n_test = 0
        for i in range(0, self.seq_types.shape[0]):
            if self.seq_types[i] == DataSetType.dsTest:
                self.n_test += 1
            elif self.seq_types[i] == DataSetType.dsTrain:
                self.n_train += 1
            else:
                raise SequenceTypeError('Unknown type of data division generated by set_custom_seq_type function')

    def _set_sequence_type(self, seq_type, data_len):
        """
        Set seq_types array that will be used to divide data set to train and test ones
        """
        self.seq_types = np.empty((data_len,), dtype=DataSetType)
        n = 0

        if seq_type == SequenceTypeSet.sqCustom:
            if self.param.set_custom_seq_type is None:
                raise SequenceTypeError('param.set_custom_seq_type function is not provided')
            self.param.set_custom_seq_type(self.seq_types)
            self._get_split_numbers()
            return

        elif seq_type == SequenceTypeSet.sqRandom:
            r = np.random.uniform(-1, 1, data_len)
            self.seq_types[:] = np.where(r > 0, DataSetType.dsTrain, DataSetType.dsTest)
            self._get_split_numbers()
            return

        elif seq_type == SequenceTypeSet.sqMode1:
            n = 2
        elif seq_type == SequenceTypeSet.sqMode3_1:
            n = 3
        elif seq_type == SequenceTypeSet.sqMode4_1:
            n = 4
        elif seq_type == SequenceTypeSet.sqMode2:
            n = 2
        elif seq_type == SequenceTypeSet.sqMode3_2:
            n = 3
        elif seq_type == SequenceTypeSet.sqMode4_2:
            n = 4
        else:
            raise SequenceTypeError('Unknown type of data division into train and test sequences')

        self.n_train = 0
        self.n_test = 0

        if SequenceTypeSet.is_mode1_type(seq_type):
            for i in range(data_len, 0, -1):
                if (data_len-i) % n == 0:
                    self.seq_types[i-1] = DataSetType.dsTest
                    self.n_test += 1
                else:
                    self.seq_types[i-1] = DataSetType.dsTrain
                    self.n_train += 1

        if SequenceTypeSet.is_mode2_type(seq_type):
            for i in range(data_len, 0, -1):
                if (data_len-i) % n == 0:
                    self.seq_types[i-1] = DataSetType.dsTrain
                    self.n_train += 1
                else:
                    self.seq_types[i-1] = DataSetType.dsTest
                    self.n_test += 1

    def _set_sequence(self, data, y, seq_type):
        """
        fill data according to sequence type, train or test
        """
        j = 0
        for i in range(0, self.seq_types.shape[0]):
            if self.seq_types[i] == seq_type:
                for k in range(0, data.shape[1]):
                    data[j, k] = self.data_x[i, k]
                y[j] = self.data_y[i]
                j += 1

    def _model_not_in_use(self, model):
        if model.layer_index == len(self.layers)-1:
            return model.model_index > 0
        else:
            next_layer = self.layers[model.layer_index+1]
            return model.model_index not in next_layer.input_index_set

    def _delete_unused_model(self, model):
        """
        Delete unused model from layer
        """
        if model.layer_index < len(self.layers)-1:
            next_layer = self.layers[model.layer_index+1]
            for next_layer_model in next_layer:
                if next_layer_model.u1_index >= model.model_index:
                    next_layer_model.u1_index -= 1
                if next_layer_model.u2_index >= model.model_index:
                    next_layer_model.u2_index -= 1
        layer = self.layers[model.layer_index]
        layer.l_count -= 1
        layer.delete(model.model_index)

    def _delete_unused_models(self):
        """
        Delete unused models from layers
        """
        layers_count = len(self.layers)
        if layers_count == 0:
            return

        layer = self.layers[layers_count-1]
        for model_index, model in reversed(list(enumerate(layer))):
            if model_index > 0:
                self._delete_unused_model(model)

        for layer in reversed(self.layers):
            for model in reversed(layer):
                if self._model_not_in_use(model):
                    self._delete_unused_model(model)

    def _train_gmdh(self):
        """
        Train multilayer GMDH algorithm
        """

        min_error = sys.float_info.max
        error_stopped_decrease = False
        del self.layers[:]
        self.valid = False
        error_min_index = 0

        while True:
            try:
                # create layer, calculating all possible models and then selecting the best ones
                # using specifying criterion
                layer = self._create_layer()

                # proceed until stop condition is fulfilled

                if layer.err < min_error:
                    # layer error has been decreased, memorize the layer index
                    error_min_index = layer.layer_index

                if layer.err > min_error and layer.layer_index > 0 and \
                   layer.layer_index - error_min_index >= self.param.criterion_minimum_width:
                    # layer error stopped decreasing
                    error_stopped_decrease = True

                min_error = min(min_error, layer.err)

                # if error does not decrease anymore or number of layers reached the limit
                # or the layer does not have any valid model - stop training
                if error_stopped_decrease or not (layer.layer_index < self.param.max_layer_count-1) or \
                        not layer.valid:

                    self.valid = True
                    break

            except LayerCreationError as e:
                print('%s, layer index %d' % (str(e), e.layer_index))
                # self.valid = True
                break

        if self.valid:
            self.layer_err.resize((len(self.layers),))
            for i in range(0, len(self.layers)):
                self.layer_err[i] = self.layers[i].err
            # delete unused layers keeping only error_min_index layers
            del self.layers[error_min_index+1:]
            # to be implemented - delete invalid models

            self._delete_unused_models()


    def _set_data(self, data_x, data_y):
        """
        Split train and test data sets from input data set and target
        """
        data_len = data_x.shape[0]
        self.n_features = data_x.shape[1]
        self._set_sequence_type(self.param.seq_type, data_len)
        self.l_count = self.n_features

        # allocate arrays for train and test sets
        self.input_train_x = np.empty((self.n_train, self.n_features), dtype=np.double)
        self.train_y = np.empty((self.n_train,), dtype=np.double)
        self.input_test_x = np.empty((self.n_test, self.n_features), dtype=np.double)
        self.test_y = np.empty((self.n_test,), dtype=np.double)

        # set train and test data sets
        self._set_sequence(self.input_train_x, self.train_y, DataSetType.dsTrain)
        self._set_sequence(self.input_test_x, self.test_y, DataSetType.dsTest)

        if isinstance(self.feature_names, np.ndarray):
            self.feature_names = self.feature_names.tolist()

    def _pre_check(self):
        """
        Check input data
        """

        if not isinstance(self.data_x, np.ndarray):
            raise ValueError("Error: samples has to be a 2D numpy array")

        if not isinstance(self.data_y, np.ndarray):
            raise ValueError("Error: targets have to be a 1D numpy array")

        if self.data_y.ndim != 1:
            raise ValueError('Error: targets have to be a 1D numpy array')

        if self.data_x.ndim != 2:
            raise ValueError('Error: samples data set has to be a 2D numpy array')

        if self.data_y.shape[0] != self.data_x.shape[0]:
            raise ValueError('Error: samples and targets has to be the same size')

        if self.data_x.shape[1] < 2:
            raise ValueError('Error: number of features has to be not less than two')

        if self.data_x.shape[0] < 2:
            raise ValueError('Error: number of samples has to be not less than two')

        feature_names_len = len(self.feature_names)
        if feature_names_len > 0 and feature_names_len != self.data_x.shape[1]:
            raise ValueError('Error: size of feature_names list does not equal to number of features')

    def _pre_fit_check(self):
        """
        Check internal arrays after split input data
        """
        if self.n_train == 0:
            raise ValueError('Error: train data set size is zero')
        if self.n_test == 0:
            raise ValueError('Error: test data set size is zero')

    def _predict_check(self, predict_data_x):
        """
        Check input data
        """

        if not isinstance(predict_data_x, np.ndarray):
            raise ValueError("Error: samples has to be a 2D numpy array")

        if predict_data_x.ndim != 2:
            raise ValueError('Error: samples data set has to be a 2D numpy array')

        if predict_data_x.shape[1] != self.n_features:
            raise ValueError('Error: number of features in data to predict has to equal to number\
             of features in fit samples')

        if not self.valid:
            raise ValueError('GMDH is not trained')


    # *************************************************************
    #                   public methods
    # *************************************************************
    def fit(self, data_x, data_y):
        """
        Fit multilayer group method of data handling algorithm (model)

        Parameters
        ----------

        data_x : numpy array or sparse matrix of shape [n_samples,n_features]
                 training data
        data_y : numpy array of shape [n_samples]
                 target values

        Returns
        -------
        self : returns an instance of self.

        Example of using
        ----------------
        from gmdh import MultilayerGMDH
        gmdh = MultilayerGMDH()
        gmdh.fit(data_x, data_y)

        """

        self.data_y = data_y
        self.data_x = data_x
        self._pre_check()
        self._set_data(data_x, data_y)
        self._pre_fit_check()
        self._train_gmdh()
        return self

    def predict(self, input_data_x):
        """
        Predict using multilayer group method of data handling algorithm (model)

        Parameters
        ----------

        input_data_x : numpy array of shape [predicted_n_samples, n_features]
                       samples

        Returns
        -------
        numpy array of shape [predicted_n_samples]
        Returns predicted values.

        Example of using:
        from gmdh import MultilayerGMDH, CriterionType
        gmdh = MultilayerGMDH()
        gmdh.fit(data_x, data_y)
        predict_y = gmdh.predict(exam_x)

        where

        data_x - training data, numpy array shape [n_samples, n_features]
        data_y - target values, numpy array of shape [n_samples]
        predict_x - samples to be predicted, numpy array of shape [predicted_n_samples, n_features]
        """

        # check dimensions
        # check validity of the model

        self._predict_check(input_data_x)

        data_len = input_data_x.shape[0]
        layer_data_x = np.array(input_data_x, copy=True)

        prev_layer = None
        # calculate outputs of all layers except the last one
        for n in range(0, len(self.layers)):
            layer_data_x = self._set_internal_data(prev_layer, input_data_x, layer_data_x)
            prev_layer = self.layers[n]

        # calculate output for the last layer
        # we choose the first (best) model of the last layer as output of multilayer gmdh
        output_y = np.zeros([data_len], dtype=np.double)
        model = self.layers[-1][0]
        for i in range(0, input_data_x.shape[0]):
            u1 = layer_data_x[i, model.u1_index]
            u2 = layer_data_x[i, model.u2_index]
            output_y[i] = model.transfer(u1, u2, model.w)

        return output_y

    def get_selected_features(self):
        """
        Return features that was selected as useful for model during fit

        Returns
        -------
        list
        """
        selected_features_set = set([])
        for n in range(0, len(self.layers)):
            layer = self.layers[n]
            if n == 0:
                for model in layer:
                    selected_features_set.add(model.u1_index)
                    selected_features_set.add(model.u2_index)
            else:
                if self.param.admix_features:
                    prev_layer = self.layers[n-1]
                    u1_index = model.u1_index - prev_layer.l_count
                    u2_index = model.u2_index - prev_layer.l_count
                    if u1_index >= 0:
                        selected_features_set.add(u1_index)
                    if u2_index >= 0:
                        selected_features_set.add(u2_index)
        return list(selected_features_set)

    def _get_n_featuresames(self, features_set):
        """
        Return names of features
        """
        s = ''
        for n, index in enumerate(features_set):
            if len(self.feature_names) > 0:
                s += '{0}'.format(self.feature_names[index])
                if n < len(features_set)-1:
                    s += ', '
            else:
                s += 'index=inp_{0} '.format(index)
        return s

    def get_unselected_features(self):
        """
        Return features that was not selected as useful for model during fit

        Returns
        -------
        list
        """
        selected_features_set = self.get_selected_features()
        features = []
        features.extend(range(0, self.n_features))
        return list(set(features) - set(selected_features_set))

    def get_unselected_n_featuresames(self):
        """
        Return names of features that was not selected as useful for model during fit

        Returns
        -------
        string
        """
        unselected_features = self.get_unselected_features()
        if len(unselected_features) == 0:
            return "No unselected features"
        else:
            return self._get_n_featuresames(unselected_features)

    def get_selected_n_featuresames(self):
        """
        Return names of features that was selected as useful for model during fit

        Returns
        -------
        string
        """
        return self._get_n_featuresames(self.get_selected_features())


    def plot_layer_error(self):
        """Plot layer error on test set vs layer index
        """

        y = self.layer_err
        x = range(0, y.shape[0])
        ax1 = plt.subplot(111)
        ax1.plot(x, y)
        ax1.set_title('Layer error on test set')
        plt.xlabel('layer index')
        plt.ylabel('mse')
        idx = len(self.layers)-1
        # plt.ion()
        plt.plot(x[idx], y[idx], 'rD')
        plt.show()

        #start_time = timeit.default_timer()
        #elapsed = timeit.default_timer() - start_time
        #print(elapsed)














