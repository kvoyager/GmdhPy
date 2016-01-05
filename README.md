# GmdhPy

Multilayered iterational algorithm of the Group Method of Data Handling for Python


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
pip uninstall gmdhpy
```

## Documentation
### Training
```py
from gmdhpy.gmdh import MultilayerGMDH
gmdh = MultilayerGMDH()
gmdh.fit(data_x, data_y)
```
where
* data\_x - training data, numpy array of shape [n\_samples, n\_features]
* data\_y - target values, numpy array of shape [n\_samples]

### Predict

```py
exam_y = gmdh.predict(exam_x)
```
where
* exam\_x - predicting data, numpy array of shape [exam\_n\_samples, n\_features]

### Setting Parameters

```py
gmdh = MultilayerGMDH(ref_functions=('linear_cov',),
                      criterion_type='test_bias',
                      feature_names=iris.feature_names,
                      criterion_minimum_width=5,
                      admix_features=True,
                      max_layer_count=50,
                      normalize=True,
                      stop_train_epsilon_condition=0.0001,
                      layer_err_criterion='top',
                      alpha=0.5,
                      n_jobs=4)
```
##### Parameters description:
*    **admix\_features** - if set to true the original features will be added to the list of features of each layer
        default value is true

*    **criterion\_type** - criterion for selecting best models. The following criteria are possible:
    *    ***'test'***: the default value,
            models are compared on the basis of test error
    *    ***'bias'***: models are compared on the basis of bias error
    *    ***'test_bias'***: combined criterion, models are compared on the basis of bias and test errors
    *    ***'bias_retrain'***: firstly, models are compared on the basis of bias error, then models are retrain
            on the total data set (train and test)

    **example of using:**

   ```py
        gmdh = MultilayerGMDH(criterion\_type='bias_retrain')
   ```

*    **seq\_type** - method to split data set to train and test
    *    ***'mode1'*** -   the default value
                    data set is split to train and test data sets in the following way:
                    ... train test train test train test ... train test.
                    The last point is chosen to belong to test set
    *    ***'mode2'*** -   data set is split to train and test data sets in the following way:
                    ... train test train test train test ... test train.
                    The last point is chosen to belong to train set
    *    ***'mode3_1'*** - data set is split to train and test data sets in the following way:
                    ... train test train train test train train test ... train train test.
                    The last point is chosen to belong to test set
    *    ***'mode4_1'*** - data set is split to train and test data sets in the following way:
                    ... train test train train train test ... test train train train test.
                    The last point is chosen to belong to test set
    *    ***'mode3_2'*** - data set is split to train and test data sets in the following way:
                    ... train test test train test test train test ... test test train.
                    The last point is chosen to belong to train set
    *    ***'mode4_2'*** - data set is split to train and test data sets in the following way:
                    ... train test test test train test ... train test test test train.
                    The last point is chosen to belong to train set
    *    ***'random'*** -  Random split data to train and test
    *    ***'custom'*** -  custom way to split data to train and test. set_custom\_seq\_type has to be provided.
         Example:
```py
                    def my_set_custom_sequence_type(seq_types):
                        r = np.random.uniform(-1, 1, seq_types.shape)
                        seq_types[:] = np.where(r > 0, DataSetType.dsTrain, DataSetType.dsTest)
                    MultilayerGMDH(seq_type='custom', set_custom_seq_type=my_set_custom_sequence_type)
```

    **example of using:**
```py
        gmdh = MultilayerGMDH(seq_type='random')
```

*    **max\_layer\_count** - maximum number of layers,
        the default value is mostly infinite (sys.maxsize)

*    **criterion\_minimum\_width** - minimum number of layers at the right required to evaluate optimal number of layer
        (the optimal model) according to the minimum of criteria. For example, if it is found that
         criterion value has minimum at layer with index 10, the algorithm will proceed till the layer
         with index 15
         the default value is 5

*    **stop\_train\_epsilon\_condition** - the threshold to stop train. If the layer relative training error in compare
        with minimum layer error becomes smaller than stop\_train\_epsilon_condition the train is stopped. Default value is
        0.001

*    **manual\_best\_models\_selection** - if this value set to False, the number of best models to be
        selected is determined automatically and it is equal number of original features.
        Otherwise the number of best models to be selected is determined as
        max(original features, min\_best\_models\_count). The value min\_best\_models\_count has to be provided.
        For example, if you have N=10 features, the number of all generated models will be at least
        N*(N-1)/2=45, the number of selected best models will be 10, but you can increase this number to
        20 by setting manual\_min\_l\_count\_value = True and min\_best\_models\_count = 20.
        Note: if min\_best\_models\_count is larger than number of generated models of the layer it will be reduced
        to that number

    **example of using:**
    ```py
        gmdh = MultilayerGMDH(manual_best_models_selection=True, min_best_models_count=20)
    ```

*    **ref\_function\_types** - set of reference functions, by default the set contains linear combination of two inputs
        and covariation: y = w0 + w1\*x1 + w2\*x2 + w3\*x1\*x2
        you can add other reference functions:
    *    'linear': y = w0 + w1\*x1 + w2\*x2
    *    'linear_cov': y = w0 + w1\*x1 + w2\*x2 + w3\*x1\*x2
    *    'quadratic': full polynom of the 2-nd degree
    *    'cubic': - full polynom of the 3-rd degree
     
   **examples of using:**
         ```py
         MultilayerGMDH(ref_functions='linear')
         MultilayerGMDH(ref_functions=('linear_cov', 'quadratic', 'cubic', 'linear'))
         MultilayerGMDH(ref_functions=('quadratic', 'linear'))
         ```

*    **normalize** - scale and normalizefeatures if set to True. Default value is True

*    **layer\_err\_criterion** - criterion of layer error calculation: 'top' - the topmost best model error is chosen
        as layer error; 'avg' - the layer error is the average error of the selected best models
        default value is 'top'

*    **alpha** - alpha value used in model train by Ridge regression (see scikit-learn linear_model.Ridge)
        default value is 0.5

*    **print\_debug** - print debug information while training, default = true

*    **n\_jobs** - number of parallel processes(threads) to train model, default 1. Use 'max' to train using
        all available threads.


## License
MIT

## References
- https://en.wikipedia.org/wiki/Group_method_of_data_handling
- http://www.gmdh.net/