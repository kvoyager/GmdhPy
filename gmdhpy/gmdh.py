"""
Multilayered group method of data handling of Machine learning for Python
==================================

"""
__author__ = 'Konstantin Kolokolov'
__version__ = '0.1.1'

import numpy as np
import sys
from gmdhpy.gmdh_model import RefFunctionType, CriterionType, SequenceTypeSet
from gmdhpy.gmdh_model import DataSetType, PolynomModel, Layer, LayerCreationError
from gmdhpy.data_preprocessing import train_preprocessing, predict_preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
import multiprocessing as mp
import six
import time


# **********************************************************************************************************************
#   implementation of multilayered GMDH algorithm
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
        'test': the default value,
            models are compared on the basis of test error
        'bias': models are compared on the basis of bias error
        'test_bias': combined criterion, models are compared on the basis of bias and test errors
        'bias_retrain': firstly, models are compared on the basis of bias error, then models are retrain
            on the total data set (train and test)
    example of using:
        gmdh = MultilayerGMDH(criterion_type='bias_retrain')

    seq_type - method to split data set to train and test
        'mode1' - 	the default value
                    data set is split to train and test data sets in the following way:
                    ... train test train test train test ... train test.
                    The last point is chosen to belong to test set
        'mode2' - 	data set is split to train and test data sets in the following way:
                    ... train test train test train test ... test train.
                    The last point is chosen to belong to train set
        'mode3_1' - data set is split to train and test data sets in the following way:
                    ... train test train train test train train test ... train train test.
                    The last point is chosen to belong to test set
        'mode4_1' - data set is split to train and test data sets in the following way:
                    ... train test train train train test ... test train train train test.
                    The last point is chosen to belong to test set
        'mode3_2' - data set is split to train and test data sets in the following way:
                    ... train test test train test test train test ... test test train.
                    The last point is chosen to belong to train set
        'mode4_2' - data set is split to train and test data sets in the following way:
                    ... train test test test train test ... train test test test train.
                    The last point is chosen to belong to train set
        'random' -  Random split data to train and test
        'custom' -  custom way to split data to train and test
                    set_custom_seq_type has to be provided
                    Example:
                    def my_set_custom_sequence_type(seq_types):
                        r = np.random.uniform(-1, 1, seq_types.shape)
                        seq_types[:] = np.where(r > 0, DataSetType.dsTrain, DataSetType.dsTest)
                    MultilayerGMDH(seq_type='custom', set_custom_seq_type=my_set_custom_sequence_type)
    example of using:
        gmdh = MultilayerGMDH(seq_type='random')

    max_layer_count - maximum number of layers,
        the default value is mostly infinite (sys.maxsize)

    criterion_minimum_width - minimum number of layers at the right required to evaluate optimal number of layer
        (the optimal model) according to the minimum of criteria. For example, if it is found that
         criterion value has minimum at layer with index 10, the algorithm will proceed till the layer
         with index 15
         the default value is 5

    stop_train_epsilon_condition - the threshold to stop train. If the layer relative training error in compare
        with minimum layer error becomes smaller than stop_train_epsilon_condition the train is stopped. Default value is
        0.001

    manual_best_models_selection - if this value set to False, the number of best models to be
        selected is determined automatically and it is equal number of original features.
        Otherwise the number of best models to be selected is determined as
        max(original features, min_best_models_count). min_best_models_count has to be provided
        For example, if you have N=10 features, the number of all generated models will be at least
        N*(N-1)/2=45, the number of selected best models will be 10, but you increase this number to
        20 by setting manual_min_l_count_value = True and min_best_models_count = 20
        Note: if min_best_models_count is larger than number of generated models of the layer it will be reduced
        to that number
    example of using:
        gmdh = MultilayerGMDH(manual_best_models_selection=True, min_best_models_count=20)

    ref_function_types - set of reference functions, by default the set contains linear combination of two inputs
        and covariation: y = w0 + w1*x1 + w2*x2 + w3*x1*x2
        you can add other reference functions:
        'linear': y = w0 + w1*x1 + w2*x2
        'linear_cov': y = w0 + w1*x1 + w2*x2 + w3*x1*x2
        'quadratic': full polynom of the 2-nd degree
        'cubic': - full polynom of the 3-rd degree
        examples of using:
         - MultilayerGMDH(ref_functions='linear')
         - MultilayerGMDH(ref_functions=('linear_cov', 'quadratic', 'cubic', 'linear'))
         - MultilayerGMDH(ref_functions=('quadratic', 'linear'))

    normalize - scale and normalize features if set to True. Default value is True

    layer_err_criterion - criterion of layer error calculation: 'top' - the topmost best model error is chosen
        as layer error; 'avg' - the layer error is the average error of the selected best models
        default value is 'top'

    alpha - alpha value used in model train by Ridge regression (see sklearn linear_model.Ridge)
        default value is 0.5

    print_debug - print debug information while training, default = true

    n_jobs - number of parallel processes(threads) to train model, default 1. Use 'max' to train using
        all available threads.

    """
    def __init__(self):
        self.ref_function_types = set()
        self.admix_features = True
        self.criterion_type = CriterionType.cmpTest
        self.seq_type = SequenceTypeSet.sqMode1
        self.set_custom_seq_type = self.dummy_set_custom_sequence_type
        self.max_layer_count = sys.maxsize
        self.criterion_minimum_width = 5
        self.stop_train_epsilon_condition = 0.001
        self.manual_best_models_selection = False
        self.min_best_models_count = 0
        self.normalize = True
        self.layer_err_criterion = 'top'
        self.alpha = 0.5
        self.print_debug = True
        self.n_jobs = 1

    @classmethod
    def dummy_set_custom_sequence_type(cls, seq_types):
        raise ValueError('param.set_custom_seq_type function is not provided')


class BaseMultilayerGMDH(object):
    """
    Base class for multilayered group method of data handling algorithm
    """

    def __init__(self, seq_type, set_custom_seq_type, ref_functions, criterion_type, feature_names, max_layer_count,
                 admix_features, manual_best_models_selection, min_best_models_count, criterion_minimum_width,
                 stop_train_epsilon_condition, normalize, layer_err_criterion, alpha, print_debug, n_jobs):
        self.param = MultilayerGMDHparam()                  # parameters
        self.param.seq_type = SequenceTypeSet.get(seq_type)
        if set_custom_seq_type is not None:
            self.param.set_custom_seq_type = set_custom_seq_type

        if isinstance(ref_functions, RefFunctionType):
            self.param.ref_function_types.add(ref_functions)
        elif not isinstance(ref_functions, six.string_types):
            for ref_function in ref_functions:
                self.param.ref_function_types.add(RefFunctionType.get(ref_function))
        else:
            self.param.ref_function_types.add(RefFunctionType.get(ref_functions))

        self.param.criterion_type = CriterionType.get(criterion_type)
        self.feature_names = feature_names                  # name of inputs, used to print GMDH
        self.param.max_layer_count = max_layer_count
        self.param.admix_features = admix_features
        self.param.manual_best_models_selection = manual_best_models_selection
        self.param.min_best_models_count = min_best_models_count
        self.param.criterion_minimum_width = criterion_minimum_width
        self.param.stop_train_epsilon_condition = stop_train_epsilon_condition
        self.param.normalize = normalize
        self.param.layer_err_criterion = layer_err_criterion
        self.param.alpha = alpha
        self.param.print_debug = print_debug

        if isinstance(n_jobs, six.string_types):
            if n_jobs == 'max':
                self.param.n_jobs = mp.cpu_count()
            else:
                raise ValueError(n_jobs)
        else:
            self.param.n_jobs = max(1, min(mp.cpu_count(), n_jobs))


        self.l_count = 0                                    # number of best models to be selected
        self.layers = []                                    # list of gmdh layers
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
        self.train_layer_err = np.array([], dtype=np.double)    # array of layer's train errors
        self.valid = False

    def _select_best_models(self, layer):
        """
        Select l_count the best models from the list
        """

        if self.param.manual_best_models_selection:
            layer.l_count = max(layer.l_count, self.param.min_best_models_count)
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

    def _set_layer_errors(self, layer):
        """
        set layer errors
        """
        # models are already sorted, the layer error is the error of the first model
        # (the model with smallest error according to the specified criterion)

        if self.param.layer_err_criterion == 'top':
            layer.err = layer[0].get_error()
            layer.train_err = sys.float_info.max
        elif self.param.layer_err_criterion == 'avg':
            layer.err = 0
            layer.train_err = 0
        else:
            raise NotImplementedError

        den = 1.0/float(len(layer))
        for model in layer:
            if model.valid:
                if self.param.layer_err_criterion == 'avg':
                    layer.err += den*model.get_error()
                    layer.train_err += den*model.train_err
                elif self.param.layer_err_criterion == 'top':
                    layer.train_err = min(layer.train_err, model.train_err)
                else:
                    raise NotImplementedError


# **********************************************************************************************************************
#   MultilayerGMDH class
# **********************************************************************************************************************

def set_matrix_a(ftype, u1_index, u2_index, source, source_y, a, y):
    """
    function set matrix value required to calculate polynom model coefficient
    by multiple linear regression
    """
    m = source.shape[0]
    u1x = source[:, u1_index]
    u2x = source[:, u2_index]
    for mi in range(0, m):
        # compute inputs for model
        u1 = u1x[mi]
        u2 = u2x[mi]
        a[mi, 0] = 1
        a[mi, 1] = u1
        a[mi, 2] = u2
        if RefFunctionType.rfLinearCov == ftype:
            a[mi, 3] = u1 * u2
        if RefFunctionType.rfQuadratic == ftype:
            a[mi, 3] = u1 * u2
            u1_sq = u1 * u1
            u2_sq = u2 * u2
            a[mi, 4] = u1_sq
            a[mi, 5] = u2_sq
        if RefFunctionType.rfCubic == ftype:
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


def train_model(alpha, a, y):
    clf = linear_model.Ridge(alpha=alpha, solver='lsqr')
    a2 = a[:,1:]
    clf.fit(a2, y)
    w = np.empty((len(clf.coef_) + 1,), dtype=np.double)
    w[0] = clf.intercept_
    w[1:] = clf.coef_
    return w


def sub_calculate_model_weights(n, ftype, u1_index, u2_index, fw_size, need_bias_stuff, transfer,
                                mlst, n_train, n_test, train_x, train_y, test_x, test_y, layer_index, alpha):
    """
    Calculate model coefficients
    """

    # declare Y variables of multiple linear regression for train and test targets
    ya = np.empty((n_train,), dtype=np.double)
    yb = np.empty((n_test,), dtype=np.double)
    min_rank = 2
    rank_b = 10000000

    # declare X variables of multiple linear regression for train and test targets
    a = np.empty((n_train, fw_size), dtype=np.double)
    b = np.empty((n_test, fw_size), dtype=np.double)
    wt = None
    train_err = sys.float_info.max
    test_err = sys.float_info.max

    # set X and Y variables of multiple linear regression
    # Train the model using train and test sets
    try:
        set_matrix_a(ftype, u1_index, u2_index, train_x, train_y, a, ya)
        w = train_model(alpha, a, ya)
        rank_a = 10
        pass
    except:
        raise LayerCreationError('Error training model on train data set', layer_index)

    if need_bias_stuff:
        try:
            set_matrix_a(ftype, u1_index, u2_index, test_x, test_y, b, yb)
            wt = train_model(alpha, b, yb)
            rank_b = 10
        except:
            raise LayerCreationError('Error training model on train data set', layer_index)

    bias_err = 0
    if rank_a < min_rank or rank_b < min_rank:
        valid = False
    else:
        valid = True
        # calculate model errors
        if need_bias_stuff:
            bias_err = PolynomModel.get_bias_err(u1_index, u2_index, transfer, train_x, test_x, train_y, test_y, w, wt)

        train_err = PolynomModel.get_regularity_err(u1_index, u2_index, transfer, train_x, train_y, w)
        test_err = PolynomModel.get_regularity_err(u1_index, u2_index, transfer, test_x, test_y, w)

    if mlst is not None:
        mlst.append((n, w, wt, valid, bias_err, train_err, test_err))
    else:
        return (n, w, wt, valid, bias_err, train_err, test_err)


def sub_calculate_model_weights_helper(args):
    return sub_calculate_model_weights(*args)


class MultilayerGMDH(BaseMultilayerGMDH):
    """
    Multilayered group method of data handling algorithm with polynomial partial descriptions(models)

    Example of using:

        data_x - training data, numpy array of shape [n_samples,n_features]
        data_y - target values, numpy array of shape [n_samples]
        exam_x - predicting data, numpy array of shape [exam_n_samples, n_features]

        from gmdh import MultilayerGMDH
        gmdh = MultilayerGMDH()
        gmdh.fit(data_x, data_y)
        exam_y = gmdh.predict(exam_x)

    Example of using with parameters specification:

        gmdh = MultilayerGMDH(ref_functions=('linear_cov',),
                              criterion_type='test',
                              feature_names=boston.feature_names,
                              criterion_minimum_width=5,
                              admix_features=True,
                              max_layer_count=50,
                              normalize=True,
                              stop_train_epsilon_condition=0.001,
                              layer_err_criterion='avg',
                              alpha=0.5,
                              n_jobs=2)
        exam_y = gmdh.predict(exam_x)

    Check MultilayerGMDHparam for details of parameters specification
    """
    def __init__(self, seq_type=SequenceTypeSet.sqMode1, set_custom_seq_type=None,
                 ref_functions=RefFunctionType.rfLinearCov,
                 criterion_type=CriterionType.cmpTest, feature_names=None, max_layer_count=50,
                 admix_features=True, manual_best_models_selection=False, min_best_models_count=5, criterion_minimum_width=5,
                 stop_train_epsilon_condition=0.001, normalize=True, layer_err_criterion='top', alpha=0.5,
                 print_debug=True, n_jobs=1):
        super(self.__class__, self).__init__(seq_type, set_custom_seq_type,
                 ref_functions,
                 criterion_type, feature_names, max_layer_count,
                 admix_features, manual_best_models_selection, min_best_models_count, criterion_minimum_width,
                 stop_train_epsilon_condition, normalize, layer_err_criterion, alpha, print_debug, n_jobs)

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

    def _new_layer_with_all_models(self):
        """
        Generate new layer with all possible models
        """
        layers_count = len(self.layers)
        layer = Layer(self, layers_count)
        # manager = Manager()
        # layer = manager.Layer(self, 0)


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
                    layer.add_polynomial_model(self, u1, u2, RefFunctionType.rfLinear)

                # y = w0 + w1*x1 + w2*x2 + w3*x1*x2
                if RefFunctionType.rfLinearCov in self.param.ref_function_types:
                    layer.add_polynomial_model(self, u1, u2, RefFunctionType.rfLinearCov)

                # y = full polynom of the 2-nd degree
                if RefFunctionType.rfQuadratic in self.param.ref_function_types:
                    layer.add_polynomial_model(self, u1, u2, RefFunctionType.rfQuadratic)

                # y = full polynom of the 3-rd degree
                if RefFunctionType.rfCubic in self.param.ref_function_types:
                    layer.add_polynomial_model(self, u1, u2, RefFunctionType.rfCubic)

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
        set_matrix_a(model.ftype, model.u1_index, model.u2_index, self.layer_data_x, self.data_y, a, y)
        # Train the model using all data (train and test sets)
        try:
            model.w = train_model(self.param.alpha, a, y)
        except:
            raise LayerCreationError('Error training model on full data set', layer.layer_index)

    def _retrain_layer(self, layer):
        for model in layer:
            self._retrain(layer, model)


    def _calculate_model_weights(self, layer):
        """
        Calculate model coefficients
        """

        if self.param.n_jobs > 1:
            del self.mlst[:]
            job_args = [(idx, model.ftype, model.u1_index, model.u2_index, model.fw_size, model.need_bias_stuff(),
                         model.transfer,
                         self.mlst, self.n_train, self.n_test, self.train_x, self.train_y, self.test_x,
                         self.test_y, layer.layer_index, self.param.alpha)
                        for idx, model in enumerate(layer)]
            self.pool.map(sub_calculate_model_weights_helper, job_args)

            for mp in self.mlst:
                n = mp[0]
                layer[n].copy_result(mp)
            del self.mlst[:]
        else:
            for n, model in enumerate(layer):
                mp = sub_calculate_model_weights(n, model.ftype, model.u1_index, model.u2_index, model.fw_size,
                        model.need_bias_stuff(), model.transfer,
                        None, self.n_train, self.n_test, self.train_x, self.train_y, self.test_x, self.test_y,
                        layer.layer_index, self.param.alpha)
                layer[n].copy_result(mp)
                pass
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

        # calculate and set layer errors
        self._set_layer_errors(layer)

        # if criterion is cmpComb_bias_retrain we need to retrain model on total data set
        # before calculate train and test errors
        if self.param.criterion_type == CriterionType.cmpComb_bias_retrain:
            self._retrain_layer(layer)

        # add created layer
        self.layers.append(layer)

        return layer

    def _set_internal_data(self, layer, data, x):
        """Compute inputs(features) for the layer
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
                for j in range(0, min(len(layer), layer.l_count)):
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
                raise ValueError('Unknown type of data division generated by set_custom_seq_type function')

    def _set_sequence_type(self, seq_type, data_len):
        """
        Set seq_types array that will be used to divide data set to train and test ones
        """
        self.seq_types = np.empty((data_len,), dtype=DataSetType)
        n = 0

        if seq_type == SequenceTypeSet.sqCustom:
            if self.param.set_custom_seq_type is None:
                raise ValueError('param.set_custom_seq_type function is not provided')
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
            raise ValueError('Unknown type of data division into train and test sequences')

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

    def _clear_train_data(self):
        # array specified how the original data will be divided into train and test data sets
        self.seq_types = None
        self.data_x = None
        self.data_y = None
        self.input_train_x = None
        self.input_test_x = None
        self.train_y = None
        self.test_y = None
        self.train_x = None
        self.test_x = None
        self.layer_data_x = None


    def _train_gmdh(self):
        """
        Train multilayered GMDH algorithm
        """

        min_error = sys.float_info.max
        error_stopped_decrease = False
        del self.layers[:]
        self.valid = False
        error_min_index = 0
        if self.param.n_jobs > 1:
            self.pool = Pool(processes=self.param.n_jobs)
            manager = Manager()
            self.mlst = manager.list()

        while True:
            try:
                # create layer, calculating all possible models and then selecting the best ones
                # using specifying criterion
                t0 = time.time()
                layer = self._create_layer()
                t1 = time.time()
                total_time = (t1-t0)
                if self.param.print_debug:
                    print("train layer{lnum} in {time:0.2f} sec".format(lnum=layer.layer_index, time=total_time))

                # proceed until stop condition is fulfilled

                if layer.err < min_error:
                    # layer error has been decreased, memorize the layer index
                    error_min_index = layer.layer_index

                if layer.err > min_error and layer.layer_index > 0 and \
                   layer.layer_index - error_min_index >= self.param.criterion_minimum_width:
                    # layer error stopped decreasing
                    error_stopped_decrease = True

                if layer.layer_index > 0 and layer.err < min_error and min_error > 0:
                    if (min_error-layer.err)/min_error < self.param.stop_train_epsilon_condition:
                        # layer relative error decrease value is below stop condition
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
            self.train_layer_err.resize((len(self.layers),))
            for i in range(0, len(self.layers)):
                self.layer_err[i] = self.layers[i].err
                self.train_layer_err[i] = self.layers[i].train_err
            # delete unused layers keeping only error_min_index layers
            del self.layers[error_min_index+1:]
            # to be implemented - delete invalid models

            self._delete_unused_models()

        if self.param.n_jobs > 1:
            self.pool = None
            manager = None
            self.mlst = None

        self._clear_train_data()


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


    @staticmethod
    def aux_check_set_crossover(d1, d2):
        def is_contain_row(array, row):
            idx = (np.abs(array-row)).argmin()
            if idx < 0 or idx >= array.shape[0]:
                return False
            frow = array[idx, :]
            return np.array_equal(row, frow)

        result = np.empty((d2.shape[0],), dtype=bool)
        for i in range(d2.shape[0]):
            row = d2[i, :]
            result[i] = is_contain_row(d1, row)
        return result

    def _pre_fit_check(self):
        """
        Check internal arrays after split input data
        """
        if self.n_train == 0:
            raise ValueError('Error: train data set size is zero')
        if self.n_test == 0:
            raise ValueError('Error: test data set size is zero')


    # *************************************************************
    #                   public methods
    # *************************************************************
    def fit(self, data_x, data_y):
        """
        Fit multilayered group method of data handling algorithm (model)

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

        data_x, data_y, self.data_len = train_preprocessing(data_x, data_y, self.feature_names)
        self.data_y = data_y
        if self.param.normalize:
            self.scaler  = StandardScaler()
            self.data_x = self.scaler.fit_transform(data_x)
        else:
            self.data_x = data_x
        self._set_data(self.data_x, self.data_y)
        self._pre_fit_check()
        self._train_gmdh()
        return self

    def predict(self, input_data_x):
        """
        Predict using multilayered group method of data handling algorithm (model)

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

        data_x - training data, numpy array of shape [n_samples, n_features]
        data_y - target values, numpy array of shape [n_samples]
        predict_x - samples to be predicted, numpy array of shape [predicted_n_samples, n_features]
        """

        # check dimensions
        # check validity of the model
        input_data_x, data_len = predict_preprocessing(input_data_x, self.n_features)

        if self.param.normalize:
            input_data_x = np.array(self.scaler.transform(input_data_x), copy=True)
        layer_data_x = None

        prev_layer = None
        # calculate outputs of all layers except the last one
        for n in range(0, len(self.layers)):
            layer_data_x = self._set_internal_data(prev_layer, input_data_x, layer_data_x)
            prev_layer = self.layers[n]

        # calculate output for the last layer
        # we choose the first (best) model of the last layer as output of multilayered gmdh
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

        fig = plt.figure()
        y = self.layer_err
        x = range(0, y.shape[0])
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y, 'b')
        ax1.set_title('Layer error on test set')
        plt.xlabel('layer index')
        plt.ylabel('error')
        idx = len(self.layers)-1
        plt.plot(x[idx], y[idx], 'rD')
        plt.show()















