"""
Multilayered group method of data handling of Machine learning for Python
==================================

"""
__author__ = 'Konstantin Kolokolov'
__version__ = '0.1.1'

import numpy as np
import sys
from gmdhpy.utils import RefFunctionType, CriterionType, SequenceTypeSet, DataSetType, LayerCreationError

from gmdhpy.layer import Layer
from gmdhpy.data_preprocessing import train_preprocessing, predict_preprocessing
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager, Process, Queue
import multiprocessing as mp
import six
import time
from gmdhpy.utils import get_y_components



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
        'validate': the default value,
            models are compared on the basis of validate error
        'bias': models are compared on the basis of bias error
        'validate_bias': combined criterion, models are compared on the basis of bias and validate errors
        'bias_retrain': firstly, models are compared on the basis of bias error, then models are retrain
            on the total data set (train and validate)
    example of using:
        gmdh = MultilayerGMDH(criterion_type='bias_retrain')

    seq_type - method to split data set to train and validate
        'mode1' - 	the default value
                    data set is split to train and validate data sets in the following way:
                    ... train validate train validate train validate ... train validate.
                    The last point is chosen to belong to validate set
        'mode2' - 	data set is split to train and validate data sets in the following way:
                    ... train validate train validate train validate ... validate train.
                    The last point is chosen to belong to train set
        'mode3_1' - data set is split to train and validate data sets in the following way:
                    ... train validate train train validate train train validate ... train train validate.
                    The last point is chosen to belong to validate set
        'mode4_1' - data set is split to train and validate data sets in the following way:
                    ... train validate train train train validate ... validate train train train validate.
                    The last point is chosen to belong to validate set
        'mode3_2' - data set is split to train and validate data sets in the following way:
                    ... train validate validate train validate validate train validate ... validate validate train.
                    The last point is chosen to belong to train set
        'mode4_2' - data set is split to train and validate data sets in the following way:
                    ... train validate validate validate train validate ... train validate validate validate train.
                    The last point is chosen to belong to train set
        'random' -  Random split data to train and validate
        'custom' -  custom way to split data to train and validate
                    set_custom_seq_type has to be provided
                    Example:
                    def my_set_custom_sequence_type(seq_types):
                        r = np.random.uniform(-1, 1, seq_types.shape)
                        seq_types[:] = np.where(r > 0, DataSetType.dsTrain, DataSetType.dsValidate)
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
        selected is determined automatically and it is equal to the number of original features.
        Otherwise the number of best models to be selected is determined as
        max(original features, min_best_models_count) but not more than max_best_models_count.
        min_best_models_count (default 5) or max_best_models_count (default inf) have to be provided.
        For example, if you have N=10 features, the number of all generated models will be
        N*(N-1)/2=45, the number of selected best models will be 10, but you can increase this number to
        20 by setting manual_min_l_count_value = True and min_best_models_count = 20.
        If you have N=100 features, the number of all generated models will be
        N*(N-1)/2=4950, by default the number of partial models passed to the second layer is equal to the number of
        features = 100. If you want to reduce this number for some smaller number, 50 for example, set
        manual_best_models_selection=True and max_best_models_count=50.
        Note: if min_best_models_count is larger than number of generated models of the layer it will be reduced
        to that number
    example of using:
        gmdh = MultilayerGMDH(manual_best_models_selection=True, min_best_models_count=20)
        or
        gmdh = MultilayerGMDH(manual_best_models_selection=True, max_best_models_count=50)

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
        self.criterion_type = CriterionType.cmpValidate
        self.seq_type = SequenceTypeSet.sqMode1
        self.set_custom_seq_type = self.dummy_set_custom_sequence_type
        self.max_layer_count = sys.maxsize
        self.criterion_minimum_width = 5
        self.stop_train_epsilon_condition = 0.001
        self.manual_best_models_selection = False
        self.min_best_models_count = 0
        self.max_best_models_count = 0
        self.normalize = True
        self.layer_err_criterion = 'top'
        self.alpha = 0.5
        self.print_debug = True
        self.n_jobs = 1
        self.keep_partial_models = False
        self.model_type = None

    @classmethod
    def dummy_set_custom_sequence_type(cls, seq_types):
        raise ValueError('param.set_custom_seq_type function is not provided')


class BaseMultilayerGMDH(object):
    """
    Base class for multilayered group method of data handling algorithm
    """

    def __init__(self, seq_type, set_custom_seq_type, ref_functions, criterion_type, feature_names, max_layer_count,
                 admix_features, manual_best_models_selection, min_best_models_count, max_best_models_count,
                 criterion_minimum_width, stop_train_epsilon_condition, normalize, layer_err_criterion, alpha,
                 print_debug, keep_partial_models, model_type, n_jobs):
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
        self.param.max_best_models_count = max_best_models_count
        self.param.criterion_minimum_width = criterion_minimum_width
        self.param.stop_train_epsilon_condition = stop_train_epsilon_condition
        self.param.normalize = normalize
        self.param.layer_err_criterion = layer_err_criterion
        self.param.alpha = alpha
        self.param.print_debug = print_debug
        self.keep_partial_models = keep_partial_models
        self.param.model_type = model_type
        self.fit_params = {}

        if isinstance(n_jobs, six.string_types):
            if n_jobs == 'max':
                self.param.n_jobs = mp.cpu_count()
            else:
                raise ValueError(n_jobs)
        else:
            self.param.n_jobs = max(1, min(mp.cpu_count(), n_jobs))

        # parallel processing is temporally disabled
        self.param.n_jobs = 1


        self.l_count = 0                                    # number of best models to be selected
        self.layers = []                                    # list of gmdh layers
        self.n_features = 0                                 # number of original features
        self.n_train = 0                                    # number of train samples
        self.n_validate = 0                                     # number of validate samples

        # array specified how the original data will be divided into train and validate data sets
        self.seq_types = np.array([], dtype=DataSetType)
        self.data_x = np.array([], dtype=np.double)             # array of original data samples
        self.data_y = np.array([], dtype=np.double)             # array of original target samples
        self.input_train_x = np.array([], dtype=np.double)      # array of original train samples
        self.input_validate_x = np.array([], dtype=np.double)       # array of original validate samples
        self.train_y = np.array([], dtype=np.double)            # array of train targets
        self.validate_y = np.array([], dtype=np.double)             # array of validate targets

        self.train_x = np.array([], dtype=np.double)            # array of train samples for current layer
        self.validate_x = np.array([], dtype=np.double)             # array of validate samples for current layer
        self.layer_data_x = np.array([], dtype=np.double)       # array of total samples for current layer

        self.layer_err = np.array([], dtype=np.double)          # array of layer's errors
        self.train_layer_err = np.array([], dtype=np.double)    # array of layer's train errors
        self.valid = False
        self.y_components = None                                # number of components in Y

    def _select_best_models(self, layer):
        """
        Select l_count the best models from the list
        """

        if self.param.manual_best_models_selection:
            layer.l_count = max(layer.l_count, self.param.min_best_models_count)
            layer.l_count = min(layer.l_count, self.param.max_best_models_count)
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


    def _y_alloc(self, n):
        if self.y_components == 1:
            y = np.empty((n,), dtype=np.double)
        else:
            y = np.empty((n, self.y_components), dtype=np.double)
        return y

# **********************************************************************************************************************
#   MultilayerGMDH class
# **********************************************************************************************************************







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
                              criterion_type='validate',
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
                 criterion_type=CriterionType.cmpValidate, feature_names=None, max_layer_count=50,
                 admix_features=True, manual_best_models_selection=False, min_best_models_count=5,
                 max_best_models_count=10000000, criterion_minimum_width=5,
                 stop_train_epsilon_condition=0.001, normalize=True, layer_err_criterion='top', alpha=0.5,
                 print_debug=True, keep_partial_models=False, model_type='polynom', n_jobs=1):
        super(self.__class__, self).__init__(seq_type, set_custom_seq_type,
                 ref_functions,
                 criterion_type, feature_names, max_layer_count,
                 admix_features, manual_best_models_selection, min_best_models_count, max_best_models_count,
                 criterion_minimum_width, stop_train_epsilon_condition, normalize, layer_err_criterion, alpha,
                 print_debug, keep_partial_models, model_type, n_jobs)

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
        s += 'Data size: {0}\n'.format(self.n_train + self.n_validate)
        s += 'Train data size: {0}\n'.format(self.n_train)
        s += 'Validate data size: {0}\n'.format(self.n_validate)
        s += 'Selected features by index: {0}\n'.format(self.get_selected_features())
        s += 'Selected features by name: {0}\n'.format(self.get_selected_n_features_names())
        s += 'Unselected features by index: {0}\n'.format(self.get_unselected_features())
        s += 'Unselected features by name: {0}\n'.format(self.get_unselected_n_features_names())
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
            # if self.param.admix_features:
            #     n += self.n_features

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

    def _retrain_layer(self, layer):
        for model in layer:
            model._refit(self.layer_data_x, self.data_y)

    def _fit_model_range(self, idx_from, idx_to):
        pass


    def _fit_model(self, layer):
        """
        Calculate model coefficients
        """

        def get_data_partitions(n_sample, n):
            q, r = divmod(n_sample, n)
            indices = [q * i + min(i, r) for i in range(n + 1)]
            return [(indices[i], indices[i + 1]) for i in range(n)]

        self.param.n_jobs = 1
        # multiprocessing is disabled temporary

        if self.param.n_jobs > 1:

            data_partitions = get_data_partitions(len(layer), self.param.n_jobs)
            # for n in range(self.param.n_jobs):
            #     idx_from, idx_to = data_partitions[n]
            #
            #     p = Process(target=self._fit_model_range, args=(idx_from, idx_to))
            #     p.start()
            #
            pass
        else:
            for n, model in enumerate(layer):
                model.fit(n, self.train_x, self.train_y, self.validate_x, self.validate_y, alpha=self.param.alpha)
                pass
        return

    def _create_layer(self):
        """
        Create new layer, calculate models coefficients, select best models
        """

        # compute features for the layer to be created for train and validate data sets
        # if the there are no previous layers just copy original features
        if len(self.layers) > 0:
            prev_layer = self.layers[-1]
            self.train_x = self._set_internal_data(prev_layer, self.input_train_x, self.train_x)
            self.validate_x = self._set_internal_data(prev_layer, self.input_validate_x, self.validate_x)
            if self.param.criterion_type == CriterionType.cmpComb_bias_retrain:
                self.layer_data_x = self._set_internal_data(prev_layer, self.data_x, self.layer_data_x)
        else:
            self.train_x = np.array(self.input_train_x, copy=True)
            self.validate_x = np.array(self.input_validate_x, copy=True)
            self.layer_data_x = np.array(self.data_x, copy=True)
            prev_layer = None

        # create new layer with all possible models
        layer = self._new_layer_with_all_models()

        # calculate model coefficients (weights)
        self._fit_model(layer)

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
        # before calculate train and validate errors
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

        if layer is None:
            # we are dealing with the first layer, its features are original features of the algorithm
            # just copy them
            out_x = np.array(data, copy=True)
        else:
            # we are dealing with the second or higher number layer
            # its features are outputs of the previous layer
            # we need to compute them

            models_output = []
            for j in range(0, min(len(layer), layer.l_count)):
                # loop for selected best models
                model = layer[j]
                y = model.transform(x)
                models_output.append(y)

            if self.param.admix_features:
                # if parameter admix_features set to true we need to add original features to
                # the current features of the layer
                models_output.append(data)

            out_x = np.hstack(models_output)

        return out_x

    def _get_split_numbers(self):
        """
        Compute sizes of train and validate data sets
        """
        self.n_train = 0
        self.n_validate = 0
        for i in range(0, self.seq_types.shape[0]):
            if self.seq_types[i] == DataSetType.dsValidate:
                self.n_validate += 1
            elif self.seq_types[i] == DataSetType.dsTrain:
                self.n_train += 1
            else:
                raise ValueError('Unknown type of data division generated by set_custom_seq_type function')

    def _set_sequence_type(self, seq_type, data_len):
        """
        Set seq_types array that will be used to divide data set to train and validate ones
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
            self.seq_types[:] = np.where(r > 0, DataSetType.dsTrain, DataSetType.dsValidate)
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
            raise ValueError('Unknown type of data division into train and validate sequences')

        self.n_train = 0
        self.n_validate = 0

        if SequenceTypeSet.is_mode1_type(seq_type):
            for i in range(data_len, 0, -1):
                if (data_len-i) % n == 0:
                    self.seq_types[i-1] = DataSetType.dsValidate
                    self.n_validate += 1
                else:
                    self.seq_types[i-1] = DataSetType.dsTrain
                    self.n_train += 1

        if SequenceTypeSet.is_mode2_type(seq_type):
            for i in range(data_len, 0, -1):
                if (data_len-i) % n == 0:
                    self.seq_types[i-1] = DataSetType.dsTrain
                    self.n_train += 1
                else:
                    self.seq_types[i-1] = DataSetType.dsValidate
                    self.n_validate += 1

    def _set_sequence(self, data, y, seq_type):
        """
        fill data according to sequence type, train or validate
        """
        j = 0
        for i in range(0, self.seq_types.shape[0]):
            if self.seq_types[i] == seq_type:
                for k in range(0, data.shape[1]):
                    data[j, k] = self.data_x[i, k]
                y[j] = self.data_y[i]
                # y[j, :] = self.data_y[i, :]
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
        # array specified how the original data will be divided into train and validate data sets
        self.seq_types = None
        self.data_x = None
        self.data_y = None
        self.input_train_x = None
        self.input_validate_x = None
        self.train_y = None
        self.validate_y = None
        self.train_x = None
        self.validate_x = None
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
        # if self.param.n_jobs > 1:
        #     self.pool = Pool(processes=self.param.n_jobs)
        #     manager = Manager()
        #     self.mlst = manager.list()

        while True:

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



        if self.valid:
            self.layer_err.resize((len(self.layers),))
            self.train_layer_err.resize((len(self.layers),))
            for i in range(0, len(self.layers)):
                self.layer_err[i] = self.layers[i].err
                self.train_layer_err[i] = self.layers[i].train_err
            # delete unused layers keeping only error_min_index layers
            del self.layers[error_min_index+1:]
            # to be implemented - delete invalid models

            if not self.keep_partial_models:
                self._delete_unused_models()

        # if self.param.n_jobs > 1:
        #     self.pool = None
        #     manager = None
        #     self.mlst = None

        self._clear_train_data()


    def _set_data(self, data_x, data_y):
        """
        Split train and validate data sets from input data set and target
        """
        data_len = data_x.shape[0]
        self.n_features = data_x.shape[1]
        self._set_sequence_type(self.param.seq_type, data_len)
        self.l_count = self.n_features

        # allocate arrays for train and validate sets
        self.input_train_x = np.empty((self.n_train, self.n_features), dtype=np.double)
        self.train_y = np.empty((self.n_train, self.y_components), dtype=np.double)
        # self.train_y = np.empty(self.n_train, dtype=np.double)
        self.input_validate_x = np.empty((self.n_validate, self.n_features), dtype=np.double)
        self.validate_y = np.empty((self.n_validate, self.y_components), dtype=np.double)
        # self.validate_y = np.empty(self.n_validate, dtype=np.double)

        # set train and validate data sets
        self._set_sequence(self.input_train_x, self.train_y, DataSetType.dsTrain)
        self._set_sequence(self.input_validate_x, self.validate_y, DataSetType.dsValidate)

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
        if self.n_validate == 0:
            raise ValueError('Error: validate data set size is zero')


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
        self.y_components = get_y_components(data_y)
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

        model = self.layers[-1][0]
        output_y = model.transform(layer_data_x)

        return output_y

    def predict_neuron_output(self, input_data_x, layer_idx, neuron_idx):

        if layer_idx >= len(self.layers) or layer_idx < 0:
            raise ValueError('layer index is out of range')
        if neuron_idx >= len(self.layers[layer_idx]) or neuron_idx < 0:
            raise ValueError('neuron index is out of range')

        # check dimensions
        # check validity of the model
        input_data_x, data_len = predict_preprocessing(input_data_x, self.n_features)

        if self.param.normalize:
            input_data_x = np.array(self.scaler.transform(input_data_x), copy=True)
        layer_data_x = None

        prev_layer = None
        # calculate outputs of all layers with indexes up to layer_idx
        for n in range(0, max(1, layer_idx - 1)):
            layer_data_x = self._set_internal_data(prev_layer, input_data_x, layer_data_x)
            prev_layer = self.layers[n]

        # calculate output for the last layer
        # we choose the first (best) model of the last layer as output of multilayered gmdh
        output_y = np.zeros([data_len], dtype=np.double)
        model = self.layers[layer_idx][neuron_idx]
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

    def _get_n_features_names(self, features_set):
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

    def get_unselected_n_features_names(self):
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
            return self._get_n_features_names(unselected_features)

    def get_selected_n_features_names(self):
        """
        Return names of features that was selected as useful for model during fit

        Returns
        -------
        string
        """
        return self._get_n_features_names(self.get_selected_features())


    def plot_layer_error(self):
        """Plot layer error on validate set vs layer index
        """

        fig = plt.figure()
        y = self.layer_err
        x = range(0, y.shape[0])
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y, 'b')
        ax1.set_title('Layer error on validate set')
        plt.xlabel('layer index')
        plt.ylabel('error')
        idx = len(self.layers)-1
        plt.plot(x[idx], y[idx], 'rD')
        plt.show()















