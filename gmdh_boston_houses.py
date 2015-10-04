__author__ = 'Konstantin Kolokolov'

from sklearn.datasets import load_boston

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from gmdh import MultilayerGMDH, RefFunctionType, SequenceTypeSet, CriterionType
from plot_gmdh import PlotGMDH

boston = load_boston()

n_samples = boston.data.shape[0]

train_data_is_the_first_half = False
if train_data_is_the_first_half:
    train_x = boston.data[:n_samples // 2]
    train_y = boston.target[:n_samples // 2]
    exam_x = boston.data[n_samples // 2:]
    exam_y = boston.target[n_samples // 2:]
else:
    train_x = boston.data[n_samples // 2:]
    train_y = boston.target[n_samples // 2:]
    exam_x = boston.data[:n_samples // 2]
    exam_y = boston.target[:n_samples // 2]

gmdh = MultilayerGMDH()



#gmdh.param.seq_type = SequenceTypeSet.sqRandom
#gmdh.param.seq_type = SequenceTypeSet.sqMode2
# gmdh.param.ref_function_types.add(RefFunctionType.rfLinear)
# gmdh.param.ref_function_types.add(RefFunctionType.rfQuadratic)
# gmdh.param.criterion_type = CriterionType.cmpBias
gmdh.param.criterion_type = CriterionType.cmpComb_test_bias
# gmdh.param.criterion_type = CriterionType.cmpComb_bias_retrain
gmdh.feature_names = boston.feature_names
#gmdh.feature_names.append('random feature')
gmdh.param.max_layer_count = 50
# gmdh.param.admix_features = False
gmdh.param.manual_min_l_count_value = False
gmdh.param.min_l_count = 6
gmdh.fit(train_x, train_y)


# Now predict the value of the second half:
# predict with GMDH
y_pred = gmdh.predict(exam_x)
mse = metrics.mean_squared_error(exam_y, y_pred)
mae = metrics.mean_absolute_error(exam_y, y_pred)

print("mean squared error on test set: {mse:.2f}".format(mse=mse))
print("mean absolute error on test set: {mae:.2f}".format(mae=mae))

PlotGMDH(gmdh, filename='img/boston_house_model', plot_model_name=True, view=True)
gmdh.plot_layer_error()
