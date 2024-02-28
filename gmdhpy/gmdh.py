# -*- coding: utf-8 -*-
"""
*******************************************************************************
Self-organizing deep learning polynomial neural network for Python
also known as
Multilayered group method of data handling of Machine learning for Python

Refs:
Mueller J.A. Lemke F., Self-organising Data Mining, Berlin (2000)
J. Schmidhuber. Deep Learning in Neural Networks: An Overview. Neural Networks,
Volume 61, January 2015, Pages 85-117
https://en.wikipedia.org/wiki/Group_method_of_data_handling
http://www.gmdh.net/
*******************************************************************************

author: 'Konstantin Kolokolov'

"""
from __future__ import print_function
import numpy as np
import sys
import multiprocessing as mp
import six
import time
import math
import matplotlib.pyplot as plt

from gmdhpy.neuron import RefFunctionType, CriterionType
from gmdhpy.neuron import Layer, LayerCreationError
from gmdhpy.data_preprocessing import train_preprocessing, predict_preprocessing, split_dataset, SequenceTypeSet
from gmdhpy.neuron import fit_layer, FitLayerData
from sklearn.preprocessing import StandardScaler, LabelEncoder
from multiprocessing import Pool
from itertools import islice, chain
from collections import namedtuple


FitData = namedtuple('FitData', ['train_x', 'train_y', 'validate_x', 'validate_y', 'data_x', 'data_y',
                                 'input_train_x', 'input_validate_x', 'input_data_x'])


class BaseSONNParam(object):
    """Parameters of self-organizing deep learning polynomial neural network
    ----------------------------
    admix_features - if set to true the original features will be added to the list of features of each layer
        default value is true

    criterion_type - criterion for selecting best neurons
    the following criteria are possible:
        'validate': the default value,
            neurons are compared on the basis of validate error
        'bias': neurons are compared on the basis of bias error
        'validate_bias': combined criterion, neurons are compared on the basis of bias and validate errors
        'bias_retrain': firstly, neurons are compared on the basis of bias error, then neurons are retrain
            on the total data set (train and validate)
    example of using:
        model = Regressor(criterion_type='bias_retrain')

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

    example of using:
        model = Regressor(seq_type='random')

    max_layer_count - maximum number of layers,
        the default value is infinite (sys.maxsize)

    criterion_minimum_width - minimum number of layers at the right required to evaluate optimal number of layer
        (the optimal neuron) according to the minimum of criteria. For example, if it is found that
         criterion value has minimum at layer with index 10, the algorithm will proceed till the layer
         with index 15
         the default value is 5

    stop_train_epsilon_condition - the threshold to stop train. If the layer relative training error in compare
        with minimum layer error becomes smaller than stop_train_epsilon_condition the train is stopped. Default value is
        0.001

    manual_best_neurons_selection - if this value set to False, the number of best neurons to be
        selected is determined automatically and it is equal to the number of original features.
        Otherwise the number of best neurons to be selected is determined as
        max(original features, min_best_neurons_count) but not more than max_best_neurons_count.
        min_best_neurons_count (default 5) or max_best_neurons_count (default inf) has to be provided.
        For example, if you have N=10 features, the number of all generated neurons will be
        N*(N-1)/2=45, the number of selected best neurons will be 10, but you can increase this number to
        20 by setting manual_min_l_count_value = True and min_best_neurons_count = 20.
        If you have N=100 features, the number of all generated neurons will be
        N*(N-1)/2=4950, by default the number of partial neurons passed to the second layer is equal to the number of
        features = 100. If you want to reduce this number for some smaller number, 50 for example, set
        manual_best_neurons_selection=True and max_best_neurons_count=50.
        Note: if min_best_neurons_count is larger than number of generated neurons of the layer it will be reduced
        to that number
    example of using:
        model = Regressor(manual_best_neurons_selection=True, min_best_neurons_count=20)
        or
        model = Regressor(manual_best_neurons_selection=True, max_best_neurons_count=50)

    ref_function_types - set of reference functions, by default the set contains linear combination of two inputs
        and covariation: y = w0 + w1*x1 + w2*x2 + w3*x1*x2
        you can add other reference functions:
        'linear': y = w0 + w1*x1 + w2*x2
        'linear_cov': y = w0 + w1*x1 + w2*x2 + w3*x1*x2
        'quadratic': full polynom of the 2-nd degree
        'cubic': - full polynom of the 3-rd degree
        examples of using:
         - Regressor(ref_functions='linear')
         - Regressor(ref_functions=('linear_cov', 'quadratic', 'cubic', 'linear'))
         - Regressor(ref_functions=('quadratic', 'linear'))

    normalize - scale and normalize features if set to True. Default value is True

    layer_err_criterion - criterion of layer error calculation: 'top' - the topmost best neuron error is chosen
        as layer error; 'avg' - the layer error is the average error of the selected best neurons
        default value is 'top'

    l2 - regularization value used in neuron fit by Ridge regression (see sklearn linear_neuron.Ridge)
        default value is 0.5

    n_jobs - number of parallel processes(threads) to train model, default 1. Use 'max' to train using
        all available threads.

    """
    def __init__(self):
        self.ref_function_types = set()
        self.admix_features = True
        self.criterion_type = CriterionType.cmpValidate
        self.seq_type = SequenceTypeSet.sqMode1
        self.max_layer_count = sys.maxsize
        self.criterion_minimum_width = 5
        self.stop_train_epsilon_condition = 0.001
        self.manual_best_neurons_selection = False
        self.min_best_neurons_count = 0
        self.max_best_neurons_count = 0
        self.normalize = True
        self.layer_err_criterion = 'top'
        self.l2 = 0.5
        self.n_jobs = 1
        self.keep_partial_neurons = False


class BaseSONN(object):
    """Base class for self-organizing deep learning polynomial neural network
    """
    model_class = None

    def __init__(self, seq_type, ref_functions, criterion_type, feature_names, max_layer_count,
                 admix_features, manual_best_neurons_selection, min_best_neurons_count, max_best_neurons_count,
                 criterion_minimum_width, stop_train_epsilon_condition, normalize, layer_err_criterion, l2,
                 verbose, keep_partial_neurons, n_jobs):
        self.param = BaseSONNParam()                  # parameters
        self.param.seq_type = SequenceTypeSet.get(seq_type)

        if isinstance(ref_functions, RefFunctionType):
            self.param.ref_function_types.add(ref_functions)
        elif not isinstance(ref_functions, six.string_types):
            for ref_function in ref_functions:
                self.param.ref_function_types.add(RefFunctionType.get(ref_function))
        else:
            self.param.ref_function_types.add(RefFunctionType.get(ref_functions))

        self.param.criterion_type = CriterionType.get(criterion_type)

        self.feature_names = feature_names       # name of inputs, used to print model
        if isinstance(self.feature_names, np.ndarray):
            self.feature_names = self.feature_names.tolist()

        self.param.max_layer_count = max_layer_count
        self.param.admix_features = admix_features
        self.param.manual_best_neurons_selection = manual_best_neurons_selection
        self.param.min_best_neurons_count = min_best_neurons_count
        self.param.max_best_neurons_count = max_best_neurons_count
        self.param.criterion_minimum_width = criterion_minimum_width
        self.param.stop_train_epsilon_condition = stop_train_epsilon_condition
        self.param.normalize = normalize
        self.param.layer_err_criterion = layer_err_criterion
        self.param.l2 = l2
        self.keep_partial_neurons = keep_partial_neurons
        self.verbose = verbose # verbose: 0 for no logging to stdout, 1 for logging progress
        self.scaler = None

        if isinstance(n_jobs, six.string_types):
            if n_jobs == 'max':
                self.param.n_jobs = mp.cpu_count()
            else:
                raise ValueError(n_jobs)
        else:
            self.param.n_jobs = max(1, min(mp.cpu_count(), n_jobs))

        self.l_count = 0        # number of best neurons to be selected
        self.layers = []        #: :type: list of Layer
        self.n_features = 0     # number of original features
        self.n_train = 0        # number of train samples
        self.n_validate = 0         # number of validate samples

        self.layer_err = np.array([], dtype=np.double)          # array of layer's errors
        self.train_layer_err = np.array([], dtype=np.double)    # array of layer's train errors
        self.valid = False
        self.loss = None

    def __str__(self):
        return "Self-organizing deep learning polynomial neural network"

    @property
    def refit_required(self):
        return self.param.criterion_type == CriterionType.cmpComb_bias_retrain

    def _select_best_neurons(self, layer):
        """Select l_count the best neurons from the list
        :param layer
        :type layer: Layer
        """

        if self.param.manual_best_neurons_selection:
            layer.l_count = max(layer.l_count, self.param.min_best_neurons_count)
            layer.l_count = min(layer.l_count, self.param.max_best_neurons_count)

        # the number of selected best neurons can't be larger than the
        # total number of neurons in the layer
        layer.l_count = min(layer.l_count, len(layer))

        # check the validity of the neurons
        for n in range(0, len(layer)):
            if not layer[n].valid:
                layer.l_count = min(layer.l_count, n)
                break

        layer.valid = layer.l_count > 0

    def _set_layer_errors(self, layer):
        """Set layer errors
        :param layer
        :type layer: Layer
        """
        # neurons are already sorted, the layer error is the error of the first neuron
        # (the neuron with smallest error according to the specified criterion)

        if self.param.layer_err_criterion == 'top':
            layer.err = layer[0].get_error(self.param.criterion_type)
            layer.train_err = sys.float_info.max
        elif self.param.layer_err_criterion == 'avg':
            layer.err = 0
            layer.train_err = 0
        else:
            raise NotImplementedError

        den = 1.0/float(len(layer))
        for neuron in layer:
            if neuron.valid:
                if self.param.layer_err_criterion == 'avg':
                    layer.err += den*neuron.get_error()
                    layer.train_err += den*neuron.train_err
                elif self.param.layer_err_criterion == 'top':
                    layer.train_err = min(layer.train_err, neuron.train_err)
                else:
                    raise NotImplementedError

    def _new_layer_with_all_neurons(self):
        """Generate new layer with all possible neurons
        """
        layers_count = len(self.layers)
        layer = Layer(self, layers_count)

        if layers_count == 0:
            # the first layer, number of inputs equals to the number of the original features
            n = self.n_features
        else:
            # all other layers: number of inputs equals to the number of selected
            # neurons from the previous layer plus number of the original
            # features if param.admix_features is True
            n = self.layers[-1].l_count
            if self.param.admix_features:
                n += self.n_features

        # number of all possible combination of input pairs is N = (n * (n-1)) / 2
        # add all neurons to the layer
        for u1 in range(0, n):
            for u2 in range(u1 + 1, n):

                # y = w0 + w1*x1 + w2*x2
                if RefFunctionType.rfLinear in self.param.ref_function_types:
                    layer.add_neuron(u1, u2, RefFunctionType.rfLinear, self.model_class, self.loss)

                # y = w0 + w1*x1 + w2*x2 + w3*x1*x2
                if RefFunctionType.rfLinearCov in self.param.ref_function_types:
                    layer.add_neuron(u1, u2, RefFunctionType.rfLinearCov, self.model_class, self.loss)

                # y = full polynom of the 2-nd degree
                if RefFunctionType.rfQuadratic in self.param.ref_function_types:
                    layer.add_neuron(u1, u2, RefFunctionType.rfQuadratic, self.model_class, self.loss)

                # y = full polynom of the 3-rd degree
                if RefFunctionType.rfCubic in self.param.ref_function_types:
                    layer.add_neuron(u1, u2, RefFunctionType.rfCubic, self.model_class, self.loss)

        if len(layer) == 0:
            raise LayerCreationError('Error creating layer. No functions were created', layer.layer_index)

        return layer

    def _refit_layer(self, layer, fit_data, fit_params):
        """Fit neuron on total (original) data set (train and validate sets)
        :param layer
        :type layer: Layer
        :param fit_data
        :type fit_data: FitData
        :param fit_params
        :type fit_params: dict
        """
        for neuron in layer:
            # Train the neuron using all data (train and validate sets)
            neuron.w = neuron.fit_function(fit_data.data_x, fit_data.data_y, fit_params)

    @staticmethod
    def batch(items, n):
        """Split list of items to n batches
        :param items
        :param n
        :type n: int
        :rtype: list
        """
        size = int(math.ceil(len(items) / float(n)))
        it_items = iter(items)
        return list(iter(lambda: tuple(islice(it_items, size)), ()))

    def _create_layer(self, pool, fit_data):
        """Create new layer, calculate neurons weights, select best neurons
        :param pool
        :type pool: Pool
        :param fit_data
        :type fit_data: FitData
        """
        # compute features for the layer to be created for train and validate data sets
        # if the there are no previous layers just copy original features

        if len(self.layers) > 0:
            prev_layer = self.layers[-1]
            train_x = self._set_internal_data(prev_layer, fit_data.input_train_x, fit_data.train_x)
            validate_x = self._set_internal_data(prev_layer, fit_data.input_validate_x, fit_data.validate_x)
            if self.refit_required:
                layer_data_x = self._set_internal_data(prev_layer, fit_data.input_data_x, fit_data.data_x)
            else:
                layer_data_x = None
            new_fit_data = FitData(train_x, fit_data.train_y, validate_x, fit_data.validate_y, layer_data_x,
                                   fit_data.data_y,
                                   fit_data.input_train_x, fit_data.input_validate_x, fit_data.input_data_x)
        else:
            new_fit_data = fit_data

        # create new layer with all possible neurons
        layer = self._new_layer_with_all_neurons()

        fit_params = {'l2': self.param.l2,
                      'layer_index': layer.layer_index,
                      'criterion_type': self.param.criterion_type}

        # calculate neuron coefficients (weights)
        self._fit_layer(layer, pool, new_fit_data, fit_params)

        # sort neurons in ascending error order according to specified criterion
        layer.sort(key=lambda x: x.get_error(self.param.criterion_type))

        # reset neuron indexes
        for n, neuron in enumerate(layer):
            neuron.neuron_index = n

        # select l_count best neurons from the list and check the neurons validity
        self._select_best_neurons(layer)

        # delete unused neurons keeping only l_count best neurons
        del layer[layer.l_count:]

        # calculate and set layer errors
        self._set_layer_errors(layer)

        # if criterion is cmpComb_bias_retrain we need to retrain neuron on total data set
        # before calculation of train and validate errors
        if self.refit_required:
            self._refit_layer(layer, new_fit_data, fit_params)

        # add created layer
        self.layers.append(layer)

        return layer, new_fit_data

    def _set_internal_data(self, layer, data, x):
        """Compute inputs(features) for the layer
        data - original features of algorithm , the dimensionality is (data size) x (number of original features)
        x is the output of selected neurons from the previous layer
        :param layer
        :type layer: Layer
        :param data
        :type data: numpy.ndarray
        :param x
        :type x: numpy.ndarray
        """

        data_m = data.shape[0]
        if layer is None:
            # the first layer, its features are original features of the algorithm
            # just copy them
            out_x = data
        else:
            # the second or higher layer
            # its features are outputs of the previous layer
            # we need to compute them
            out_size = min(len(layer), layer.l_count)
            out_x = np.zeros((data_m, out_size), dtype=np.double)
            for j in range(out_size):
                neuron = layer[j]
                out_x[:, j] = neuron.transfer(x[:, neuron.u1_index], x[:, neuron.u2_index], neuron.w)

            # if parameter admix_features set to true we need to add original features to
            # the current features of the layer
            if self.param.admix_features:
                out_x = np.hstack([out_x, data])

        return out_x

    def _neuron_not_in_use(self, neuron):
        """
        :param neuron
        :type neuron: PolynomNeuron
        :rtype bool
        """
        if neuron.layer_index == len(self.layers)-1:
            return neuron.neuron_index > 0
        else:
            next_layer = self.layers[neuron.layer_index+1]
            return neuron.neuron_index not in next_layer.input_index_set

    def _delete_unused_neuron(self, neuron):
        """Delete unused neuron from layer
        :param neuron
        :type neuron: PolynomNeuron
        """
        if neuron.layer_index < len(self.layers)-1:
            next_layer = self.layers[neuron.layer_index+1]
            for next_layer_neuron in next_layer:
                if next_layer_neuron.u1_index >= neuron.neuron_index:
                    next_layer_neuron.u1_index -= 1
                if next_layer_neuron.u2_index >= neuron.neuron_index:
                    next_layer_neuron.u2_index -= 1
        layer = self.layers[neuron.layer_index]
        layer.l_count -= 1
        layer.delete(neuron.neuron_index)

    def _delete_unused_neurons(self):
        """Delete unused neurons from layers
        """
        layers_count = len(self.layers)
        if layers_count == 0:
            return

        layer = self.layers[layers_count-1]
        for neuron_index, neuron in reversed(list(enumerate(layer))):
            if neuron_index > 0:
                self._delete_unused_neuron(neuron)

        for layer in reversed(self.layers):
            for neuron in reversed(layer):
                if self._neuron_not_in_use(neuron):
                    self._delete_unused_neuron(neuron)

    def _fit(self, fit_data):
        """Fit model
        :param fit_data
        :type fit_data: FitData
        """

        min_error = sys.float_info.max
        error_stopped_decrease = False
        del self.layers[:]
        self.valid = False
        error_min_index = 0
        if self.param.n_jobs > 1:
            pool = Pool(processes=self.param.n_jobs)
        else:
            pool = None

        while True:
            # create layer, calculate all possible neurons and then select the best ones
            # using specified criterion
            t0 = time.time()
            layer, fit_data = self._create_layer(pool, fit_data)
            t1 = time.time()
            total_time = (t1 - t0)
            if self.verbose == 1:
                print("train layer{lnum} in {time:0.2f} sec".format(lnum=layer.layer_index,
                                                                    time=total_time))

            # proceed until stop condition is fulfilled

            if layer.err < min_error:
                # layer error has been decreased, memorize the layer index
                error_min_index = layer.layer_index

            if layer.err > min_error and layer.layer_index > 0 and \
                                    layer.layer_index - error_min_index >= self.param.criterion_minimum_width:
                # layer error stopped decreasing
                error_stopped_decrease = True

            if layer.layer_index > 0 and layer.err < min_error and min_error > 0:
                if (min_error - layer.err) / min_error < self.param.stop_train_epsilon_condition:
                    # layer relative error decrease value is below stop condition
                    error_stopped_decrease = True

            min_error = min(min_error, layer.err)

            # if error does not decrease anymore or number of layers reached the limit
            # or the layer does not have any valid neuron - stop training
            if error_stopped_decrease or not (layer.layer_index < self.param.max_layer_count - 1) or \
                    not layer.valid:
                self.valid = True
                break

        if self.valid:
            self.layer_err.resize((len(self.layers),), refcheck=False)
            self.train_layer_err.resize((len(self.layers),), refcheck=False)
            for i in range(0, len(self.layers)):
                self.layer_err[i] = self.layers[i].err
                self.train_layer_err[i] = self.layers[i].train_err
            # delete unused layers keeping only error_min_index layers
            del self.layers[error_min_index + 1:]
            # to be implemented - delete invalid neurons

            if not self.keep_partial_neurons:
                self._delete_unused_neurons()

    def _pre_fit_check(self, train_y, validate_y):
        """Check internal arrays after split input data
        """
        if self.n_train == 0:
            raise ValueError('Error: train data set size is zero')
        if self.n_validate == 0:
            raise ValueError('Error: validate data set size is zero')

    def _get_features_names_by_index(self, features_set):
        """Return names of features
        """
        if self.feature_names is None:
            return ', '.join(
                ['index=inp_{0} '.format(idx) for idx in features_set])
        else:
            return ', '.join(
                [self.feature_names[idx] for idx in features_set])

    def _preprocess_y(self, train_y, validate_y, data_y):
        return train_y, validate_y, data_y

    # *************************************************************
    #                   public methods
    # *************************************************************
    def fit(self, data_x, data_y, validation_data=None, dataset_split=None,
            verbose=None):
        """Fit self-organizing deep learning polynomial neural network

        :param data_x : numpy array or sparse matrix of shape [n_samples,n_features]
                 training data
        :param data_y : numpy array of shape [n_samples]
                 target values

        :return an instance of self.

        Example of using
        ----------------
        from gmdh import Regressor
        model = Regressor()
        model.fit(data_x, data_y)

        """
        if verbose is not None:
            self.verbose = verbose

        data_x, data_y = train_preprocessing(data_x, data_y, self.feature_names)

        if validation_data is None:
            input_train_x, train_y, input_validate_x, validate_y = split_dataset(
                data_x, data_y, self.param.seq_type)
            input_data_x = data_x
        else:
            input_validate_x, validate_y = train_preprocessing(
                validation_data[0], validation_data[1], self.feature_names)
            input_train_x = data_x
            train_y = data_y
            input_data_x = np.vstack((input_train_x, input_validate_x))
            data_x = input_data_x
            data_y = np.hstack((train_y, validate_y))

        self.n_features = data_x.shape[1]
        self.l_count = self.n_features
        self.n_train = input_train_x.shape[0]
        self.n_validate = input_validate_x.shape[0]

        if self.param.normalize:
            self.scaler = StandardScaler()
            input_train_x = self.scaler.fit_transform(input_train_x)
            input_validate_x = self.scaler.transform(input_validate_x)
            input_data_x = self.scaler.transform(input_data_x)

        train_y, validate_y, data_y = self._preprocess_y(train_y, validate_y, data_y)
        fit_data = FitData(input_train_x, train_y,
                           input_validate_x, validate_y,
                           data_x, data_y,
                           input_train_x, input_validate_x, input_data_x)

        self._pre_fit_check(train_y, validate_y)
        self._fit(fit_data)
        return self

    def _predict(self, input_data_x):
        """Predict using self-organizing deep learning polynomial neural network

        :param input_data_x : numpy array of shape [predicted_n_samples, n_features]
                       samples

        :return numpy array of shape [predicted_n_samples]
        Returns predicted values.

        Example of using:
        from gmdh import Regressor, CriterionType
        model = Regressor()
        model.fit(data_x, data_y)
        predict_y = model.predict(test_x)

        where

        data_x - training data, numpy array of shape [n_samples, n_features]
        data_y - target values, numpy array of shape [n_samples]
        predict_x - samples to be predicted, numpy array of shape [predicted_n_samples, n_features]
        """

        if not self.valid:
            raise ValueError('Model is not fit')

        # check dimensions
        # check validity of the neuron
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
        # we choose the first (best) neuron of the last layer as output of network
        neuron = self.layers[-1][0]
        u1 = layer_data_x[:, neuron.u1_index]
        u2 = layer_data_x[:, neuron.u2_index]
        output_y = neuron.transfer(u1, u2, neuron.w)

        return output_y

    def predict_neuron_output(self, input_data_x, layer_idx, neuron_idx):
        """Return output od specified neuron
        :param input_data_x:
        :param layer_idx: layer index
        :type layer_idx: int
        :param neuron_idx: neuron index within the layer
        :type neuron_idx: int
        :rtype: double
        """

        if layer_idx >= len(self.layers) or layer_idx < 0:
            raise ValueError('layer index is out of range')
        if neuron_idx >= len(self.layers[layer_idx]) or neuron_idx < 0:
            raise ValueError('neuron index is out of range')

        # check dimensions
        # check validity of the neuron
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
        # we choose the first (best) neuron of the last layer as output of network
        neuron = self.layers[layer_idx][neuron_idx]
        u1 = layer_data_x[:, neuron.u1_index]
        u2 = layer_data_x[:, neuron.u2_index]
        output_y = neuron.transfer(u1, u2, neuron.w)

        return output_y

    def get_selected_features_indices(self):
        """Return features that was selected as useful for neuron during fit
        """
        selected_features_set = set()
        for neuron in self.layers[0]:
            selected_features_set.add(neuron.u1_index)
            selected_features_set.add(neuron.u2_index)

        if self.param.admix_features and len(self.layers) > 1:
            for layer in self.layers[1:]:
                for neuron in layer:
                    prev_layer = self.layers[layer.layer_index-1]
                    u1_index = neuron.u1_index - prev_layer.l_count
                    u2_index = neuron.u2_index - prev_layer.l_count
                    if u1_index >= 0:
                        selected_features_set.add(u1_index)
                    if u2_index >= 0:
                        selected_features_set.add(u2_index)
        return list(selected_features_set)

    def get_unselected_features_indices(self):
        """Return features that was not selected as useful for neuron during fit
        """
        return list(set(np.arange(self.n_features).tolist()) -
                    set(self.get_selected_features_indices()))

    def get_unselected_features(self):
        """Return names of features that was not selected as useful for neuron during fit
        """
        unselected_features = self.get_unselected_features_indices()
        if len(unselected_features) == 0:
            return "No unselected features"
        else:
            return self._get_features_names_by_index(unselected_features)

    def get_selected_features(self):
        """Return names of features that was selected as useful for neuron during fit
        """
        return self._get_features_names_by_index(self.get_selected_features_indices())

    def describe(self):
        """Describe the model"""
        s = ['*' * 50,
             'Model',
             '*' * 50,
            'Number of layers: {0}'.format(len(self.layers)),
            'Max possible number of layers: {0}'.format(self.param.max_layer_count),
            'Model selection criterion: {0}'.format(CriterionType.get_name(self.param.criterion_type)),
            'Number of features: {0}'.format(self.n_features),
            'Include features to inputs list for each layer: {0}'.format(self.param.admix_features),
            'Data size: {0}'.format(self.n_train + self.n_validate),
            'Train data size: {0}'.format(self.n_train),
            'Test data size: {0}'.format(self.n_validate),
            'Selected features by index: {0}'.format(self.get_selected_features_indices()),
            'Selected features by name: {0}'.format(self.get_selected_features()),
            'Unselected features by index: {0}'.format(self.get_unselected_features_indices()),
            'Unselected features by name: {0}'.format(self.get_unselected_features()),
        ]
        for layer in self.layers:
            s.append('\n' + layer.describe(self.feature_names, self.layers))
        return '\n'.join(s)

    def describe_layer(self, layer_index):
        """Describe the layer
        :param layer_index
        :type layer_index: int
        :rtype str
        """
        return self.layers[layer_index].describe(self.feature_names, self.layers)

    def describe_neuron(self, layer_index, neuron_index):
        """Describe the neuron
        :param layer_index
        :type layer_index: int
        :param neuron_index
        :type neuron_index: int
        :rtype str
        """
        return self.layers[layer_index][neuron_index].describe(self.feature_names, self.layers)

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

    def _fit_layer(self, layer, pool, fit_data, fit_params):
        """Calculate neuron weights
        """
        job_args = [FitLayerData(neurons,
                                 fit_data.train_x,
                                 fit_data.train_y,
                                 fit_data.validate_x,
                                 fit_data.validate_y,
                                 fit_params)
                    for neurons in self.batch(layer, self.param.n_jobs)]

        if self.param.n_jobs > 1:

            fitted_neurons = pool.map(fit_layer, job_args)
            del layer[:]
            layer.extend(chain(*fitted_neurons))
        else:
            fit_layer(job_args[0])


# **********************************************************************************************************************
#   Regressor class
# **********************************************************************************************************************

class Regressor(BaseSONN):
    """Self-organizing deep learning polynomial neural network
    """
    model_class = 'regression'

    def __init__(self, seq_type=SequenceTypeSet.sqMode1,
                 ref_functions=RefFunctionType.rfLinearCov,
                 criterion_type=CriterionType.cmpValidate, feature_names=None, max_layer_count=50,
                 admix_features=True, manual_best_neurons_selection=False, min_best_neurons_count=5,
                 max_best_neurons_count=10000000, criterion_minimum_width=5,
                 stop_train_epsilon_condition=0.001, normalize=True, layer_err_criterion='top', l2=0.5,
                 verbose=1, keep_partial_neurons=False, n_jobs=1):
        super(self.__class__, self).__init__(seq_type,
                 ref_functions,
                 criterion_type, feature_names, max_layer_count,
                 admix_features, manual_best_neurons_selection, min_best_neurons_count, max_best_neurons_count,
                 criterion_minimum_width, stop_train_epsilon_condition, normalize, layer_err_criterion, l2,
                 verbose, keep_partial_neurons, n_jobs)
        self.loss = 'mse'

    def predict(self, data_x):
        """Predict using self-organizing deep learning polynomial
        neural network

        Parameters
        ----------
        data_x : numpy array of shape [predicted_n_samples, n_features]

        Returns
        -------
        predicted classes : numpy array

        Example of using:
        from gmdh import Regressor
        model = Regressor()
        model.fit(data_x, data_y)
        predict_y = model.predict(test_x)

        where

        data_x - training data, numpy array of shape [n_samples, n_features]
        data_y - target values, numpy array of shape [n_samples]
        test_x - samples to be predicted, numpy array of shape [test_n_samples, n_features]
        """
        return self._predict(data_x)


class Classifier(BaseSONN):
    """Self-organizing deep learning polynomial neural network classifier
    """
    model_class = 'classification'

    def __init__(self, seq_type=SequenceTypeSet.sqMode1,
                 ref_functions=RefFunctionType.rfLinearCov,
                 criterion_type=CriterionType.cmpValidate, feature_names=None, max_layer_count=50,
                 admix_features=True, manual_best_neurons_selection=False, min_best_neurons_count=5,
                 max_best_neurons_count=10000000, criterion_minimum_width=5,
                 stop_train_epsilon_condition=0.001, normalize=True, layer_err_criterion='top', l2=0.5,
                 verbose=1, keep_partial_neurons=False, n_jobs=1):
        super(self.__class__, self).__init__(seq_type,
                 ref_functions,
                 criterion_type, feature_names, max_layer_count,
                 admix_features, manual_best_neurons_selection, min_best_neurons_count, max_best_neurons_count,
                 criterion_minimum_width, stop_train_epsilon_condition, normalize, layer_err_criterion, l2,
                 verbose, keep_partial_neurons, n_jobs)
        self.loss = 'logloss'
        self.le = LabelEncoder()

    def _pre_fit_check(self, train_y, validate_y):
        """Check internal arrays after split input data
        """
        super(self.__class__, self)._pre_fit_check(train_y, validate_y)
        if len(np.unique(train_y)) > 2 or len(np.unique(validate_y)) > 2:
            raise ValueError('Only binary classification is supported')

    def _preprocess_y(self, train_y, validate_y, data_y):
        train_y = self.le.fit_transform(train_y)
        validate_y = self.le.transform(validate_y)
        data_y = self.le.transform(data_y)
        return train_y, validate_y, data_y

    def predict_proba(self, data_x):
        """Predict probabilities of classes using self-organizing deep learning polynomial
        neural network classifier

        Parameters
        ----------
        data_x : numpy array of shape [predicted_n_samples, n_features]

        Returns
        -------
        predicted classes : numpy array

        Example of using:
        from gmdh import Classifier
        model = Classifier()
        model.fit(data_x, data_y)
        predict_y = model.predict_proba(test_x)

        where

        data_x - training data, numpy array of shape [n_samples, n_features]
        data_y - target values, numpy array of shape [n_samples]
        test_x - samples to be predicted, numpy array of shape [test_n_samples, n_features]
        """
        return self._predict(data_x)

    def predict(self, data_x):
        """Predict classes using self-organizing deep learning polynomial
        neural network classifier

        Parameters
        ----------
        data_x : numpy array of shape [predicted_n_samples, n_features]

        Returns
        -------
        predicted classes : numpy array

        Example of using:
        from gmdh import Classifier
        model = Classifier()
        model.fit(data_x, data_y)
        predict_y = model.predict(test_x)

        where

        data_x - training data, numpy array of shape [n_samples, n_features]
        data_y - target values, numpy array of shape [n_samples]
        test_x - samples to be predicted, numpy array of shape [test_n_samples, n_features]
        """
        return self.le.transform(np.argmax(self.predict_proba(data_x)))


#aliases
MultilayerGMDH = Regressor












