GmdhPy
=====
Multilayered iterational algorithm of the Group Method of Data Handling for Python

Documentation
=============
Training
--------
```py
from gmdh import MultilayerGMDH
gmdh = MultilayerGMDH()
gmdh.fit(data_x, data_y)
```
where
* data_x - training data, numpy array shape [n_samples, n_features]
* data_y - target values, numpy array of shape [n_samples]

Predict
-------
```py
exam_y = gmdh.predict(exam_x)
```
where
* exam_x - predicting data, numpy array of shape [exam_n_samples, n_features]

Setting Parameters
------------------
```py
from gmdh import MultilayerGMDH, CriterionType
gmdh = MultilayerGMDH()
gmdh.param.criterion_type = CriterionType.cmpComb_bias_retrain
gmdh.fit(data_x, data_y)
```
##### Parameters description:
* **admix_features** - if set to true the original features will be added to the list of features of each layer
        default value is true

* **criterion_type** - criterion for selecting best models
    the following criteria are possible:
    * ***cmpTest***: the default value,
            models are compared on the basis of test error
    *   ***cmpBias***: models are compared on the basis of bias error
    *    ***cmpComb_train_bias***: combined criterion, models are compared on the basis of bias and train errors
    *    ***cmpComb_test_bias***: combined criterion, models are compared on the basis of bias and test errors
    *    ***cmpComb_bias_retrain***: firstly, models are compared on the basis of bias error, then models are retrain
            on the total data set (train and test)
    * **example of using**:
    ```py
        from gmdh import MultilayerGMDH, CriterionType
        gmdh = MultilayerGMDH()
        gmdh.param.criterion_type = CriterionType.cmpComb_bias_retrain
    ```

* **seq_type** - method to split data set to train and test
    * ***sqMode1*** - 	the default value
                    data set is split to train and test data sets in the following way:
                    ... train test train test train test ... train test.
                    The last point is chosen to belong to test set
    * ***sqMode2*** - 	data set is split to train and test data sets in the following way:
                    ... train test train test train test ... test train.
                    The last point is chosen to belong to train set
    * ***sqMode3_1*** - data set is split to train and test data sets in the following way:
                    ... train test train train test train train test ... train train test.
                    The last point is chosen to belong to test set
    * ***sqMode4_1*** - data set is split to train and test data sets in the following way:
                    ... train test train train train test ... test train train train test.
                    The last point is chosen to belong to test set
    * ***sqMode3_2*** - data set is split to train and test data sets in the following way:
                    ... train test test train test test train test ... test test train.
                    The last point is chosen to belong to train set
    * ***sqMode4_2*** - data set is split to train and test data sets in the following way:
                    ... train test test test train test ... train test test test train.
                    The last point is chosen to belong to train set
    * ***sqRandom*** -  Random split data to train and test
    * ***sqCustom*** -  custom way to split data to train and test
                    set_custom_seq_type has to be provided
                    Example:
    ```py
    def my_set_custom_sequence_type(seq_types):
        r = np.random.uniform(-1, 1, seq_types.shape)
        seq_types[:] = np.where(r > 0, DataSetType.dsTrain, DataSetType.dsTest)
    gmdh.param.seq_type = SequenceTypeSet.sqCustom
    gmdh.param.set_custom_seq_type = my_set_custom_sequence_type
    ```
    * **example of using**:
    ```py
        from gmdh import MultilayerGMDH, SequenceTypeSet
        gmdh = MultilayerGMDH()
        gmdh.param.seq_type = SequenceTypeSet.sqMode2
    ```

* **max_layer_count** - maximum number of layers,
        the default value is mostly infinite (sys.maxsize)

* **criterion_minimum_width** - minimum number of layers at the right required to evaluate optimal number of layer
        (the optimal model) according to the minimum of criteria. For example, if it is found that
         criterion value has minimum at layer with index 10, the algorithm will proceed till the layer
         with index 15
         the default value is 5

* **manual_min_l_count_value** - if this value set to False, the number of best models to be
        selected is determined automatically and it is equal number of original features.
        Otherwise the number of best models to be selected is determined as
        max(original features, min_l_count). min_l_count has to be provided
        For example, if you have N=10 features, the number of all generated models will be at least
        N*(N-1)/2=45, the number of selected best models will be 10, but you increase this number to
        20 by setting manual_min_l_count_value = True and min_l_count = 20
        Note: if min_l_count is larger than number of generated models of the layer it will be reduced
        to that number
    * **example of using**:
    ```py
        from gmdh import MultilayerGMDH
        gmdh = MultilayerGMDH()
        gmdh.param.manual_min_l_count_value = True
        gmdh.param.min_l_count = 20
    ```

* **ref_function_types** - set of reference functions, by default the set contains polynom
        of the second degree: y = w0 + w1*x1 + w2*x2 + w3*x1*x2
        you can add other reference functions in the following way:
    * param.ref_function_types.add(RefFunctionType.rfLinear) - y = w0 + w1*x1 + w2*x2
    * param.ref_function_types.add(RefFunctionType.rfQuadratic) - full polynom of the 2-nd degree
    * param.ref_function_types.add(RefFunctionType.rfCubic) - full polynom of the 3-rd degree
    

# License
MIT

# References
- https://en.wikipedia.org/wiki/Group_method_of_data_handling
- http://www.gmdh.net/