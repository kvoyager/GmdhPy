## GmdhPy
GmdhPy is the Python library with implementation of iterational group method of data handling algorithm with polynomial reference functions.
The resulting models are also known self-organizing deep learning polynomial neural network. It is one of the earliest [deep learning methods](http://www.sciencedirect.com/science/article/pii/S0893608014002135).

## Installation

GmdhPy uses the following dependencies:

- numpy
- six
- scikit-learn
- matplotlib (optional, required if you use plotting tools)
- graphviz (optional but recommended if you want to plot model graph, requires installation of binaries from http://www.graphviz.org/ as well)

To install GmdhPy:

```
sudo pip install git+git://github.com/kvoyager/GmdhPy.git
```

To uninstall GmdhPy:

```
sudo pip uninstall gmdhpy
```

## Documentation
### Regression
```py
from gmdhpy.gmdh import Regressor
model = Regressor()
model.fit(data_x, data_y)
predict_y = model.predict(test_x)
```
where
* data_x - train data, numpy array shape [num_samples, num_features]
* data_y - target values, numpy array of shape [num_samples]
* test_x - test data, numpy array of shape [num_test_samples, num_feature]

data_x will be splitted to train and validate inside fitting

or
```py
from gmdhpy.gmdh import Regressor
model = Regressor()
model.fit(train_x, train_y, validation_data=(validate_x, validate_y))
predict_y = model.predict(test_x)
```

where
* train_x - train data, numpy array shape [num_train_samples, num_features]
* train_y - train target values, numpy array of shape [num_train_samples]
* validate_x - validate data, numpy array shape [num_validate_samples, num_features]
* validate_y - validate target values, numpy array of shape [num_vaidate_samples]
* test_x - test data, numpy array of shape [num_test_samples, num_feature]

### Classification
```py
from gmdhpy.gmdh import Classifier
model = Classifier()
model.fit(data_x, data_y)
```
or
```py
model.fit(train_x, train_y, validation_data=(validate_x, validate_y))
```
predict
```py
predicted_scores = model.predict_proba(test_x)
predicted_lables = model.predict(test_x)
```
where
* predicted_scores - predicted probabilities of classes (scores)
* predicted_lables - predcited class labels

Note: only binary classification is supported


### Setting Parameters

##### Parameters description:
*    **admix_features** - if set to true the original features will be added to the list of features of each layer
        default value is true

*   **criterion_type** - criterion for selecting best neurons
    the following criteria are possible:
        *    ***validate***: the default value,
                neurons are compared on the basis of validate error
        *    ***bias***: neurons are compared on the basis of bias error
        *    ***validate_bias***: combined criterion, neurons are compared on the basis of bias and validate errors
        *    ***bias_retrain***: firstly, neurons are compared on the basis of bias error, then neurons are retrain
                on the total data set (train and validate)
    **example of using:**
    ```py
        model = Regressor(criterion_type='bias_retrain')
    ```

*   **seq_type** - method to split data set to train and validate
    *   ***mode1*** -   the default value
                    data set is split to train and validate data sets in the following way:
                    ... train validate train validate train validate ... train validate.
                    The last point is chosen to belong to validate set
    *   ***mode2*** -   data set is split to train and validate data sets in the following way:
                    ... train validate train validate train validate ... validate train.
                    The last point is chosen to belong to train set
    *   ***mode3_1*** - data set is split to train and validate data sets in the following way:
                    ... train validate train train validate train train validate ... train train validate.
                    The last point is chosen to belong to validate set
    *   ***mode4_1*** - data set is split to train and validate data sets in the following way:
                    ... train validate train train train validate ... validate train train train validate.
                    The last point is chosen to belong to validate set
    *   ***mode3_2*** - data set is split to train and validate data sets in the following way:
                    ... train validate validate train validate validate train validate ... validate validate train.
                    The last point is chosen to belong to train set
    *   ***mode4_2*** - data set is split to train and validate data sets in the following way:
                    ... train validate validate validate train validate ... train validate validate validate train.
                    The last point is chosen to belong to train set
    *   ***random*** -  Random split data to train and validate

    **example of using:**
    ```py
        model = Regressor(seq_type='random')
    ```

*   **max_layer_count** - maximum number of layers,
        the default value is infinite (sys.maxsize)

*   **criterion_minimum_width** - minimum number of layers at the right required to evaluate optimal number of layer
        (the optimal neuron) according to the minimum of criteria. For example, if it is found that
         criterion value has minimum at layer with index 10, the algorithm will proceed till the layer
         with index 15
         the default value is 5

*   **stop_train_epsilon_condition** - the threshold to stop train. If the layer relative training error in compare
        with minimum layer error becomes smaller than stop_train_epsilon_condition the train is stopped. Default value is
        0.001

*   **manual_best_neurons_selection** - if this value set to False, the number of best neurons to be
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
        _Note_: if min_best_neurons_count is larger than number of generated neurons of the layer it will be reduced
        to that number
    **example of using:**
    ```py
        model = Regressor(manual_best_neurons_selection=True, min_best_neurons_count=20)
    ```
    or
    ```py
        model = Regressor(manual_best_neurons_selection=True, max_best_neurons_count=50)
    ```

*   **ref_function_types** - set of reference functions, by default the set contains linear combination of two inputs
        and covariation: y = w0 + w1*x1 + w2*x2 + w3*x1*x2
        you can add other reference functions:
    *    **linear**: y = w0 + w1*x1 + w2*x2
    *    **linear_cov**: y = w0 + w1*x1 + w2*x2 + w3*x1*x2
    *    **quadratic**: full polynom of the 2-nd degree
    *    **cubic**: - full polynom of the 3-rd degree
    **examples of using:**
    ```py
    Regressor(ref_functions='linear')
    Regressor(ref_functions=('linear_cov', 'quadratic', 'cubic', 'linear'))
    Regressor(ref_functions=('quadratic', 'linear'))
    ```

*   **normalize** - scale and normalize features if set to True. Default value is True

*   **layer_err_criterion** - criterion of layer error calculation: 'top' - the topmost best neuron error is chosen
        as layer error; 'avg' - the layer error is the average error of the selected best neurons
        default value is 'top'

*   **l2** - regularization value used in neuron fit by Ridge regression (see sklearn linear_neuron.Ridge)
        default value is 0.5

*   **n_jobs** - number of parallel processes(threads) to train model, default 1. Use 'max' to train using
        all available threads.

##### Example of using:

```py
params = {
    'ref_functions': 'linear_cov',
    'criterion_type': 'test_bias',
    'feature_names': feature_names,
    'criterion_minimum_width': 5,
    'admix_features': True,
    'max_layer_count': 50,
    'normalize': True,
    'stop_train_epsilon_condition': 0.0001,
    'layer_err_criterion': 'top',
    'alpha': 0.5,
    'n_jobs': 4
}
model = Regressor(**params)
```

## License
MIT

## References
- Mueller J.A. Lemke F., Self-organising Data Mining, Berlin (2000)
- J. Schmidhuber. Deep Learning in Neural Networks: An Overview. Neural Networks, Volume 61, January 2015, Pages 85-117
- https://en.wikipedia.org/wiki/Group_method_of_data_handling
- http://www.gmdh.net/