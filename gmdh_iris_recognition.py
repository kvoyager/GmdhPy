__author__ = 'Konstantin Kolokolov'

import pylab as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix
from gmdh import MultilayerGMDH, RefFunctionType, SequenceTypeSet, CriterionType
from plot_gmdh import PlotGMDH

iris = datasets.load_iris()
print(iris.data.shape)


def iris_class(value):
    if value > 1.5:
        return 2
    elif value <= 1.5 and value >= 0.5:
        return 1
    return 0
viris_class = np.vectorize(iris_class, otypes=[np.int])

n_samples = iris.data.shape[0]

data = np.empty_like(iris.data)
target = np.empty_like(iris.target)
j = 0
n = n_samples // 3
for i in range(0, n):
    data[j] = iris.data[i]
    data[j+1] = iris.data[i+n]
    data[j+2] = iris.data[i+2*n]
    target[j] = iris.target[i]
    target[j+1] = iris.target[i+n]
    target[j+2] = iris.target[i+2*n]
    j += 3

# add random data to features
#eps = np.random.uniform(5, 15, (n_samples, 1))
#data = np.append(data, eps, axis=1)

train_data_is_the_first_half = False
if train_data_is_the_first_half:
    train_x = data[:n_samples // 2]
    train_y = target[:n_samples // 2]
    exam_x = data[n_samples // 2:]
    exam_y = target[n_samples // 2:]
else:
    train_x = data[n_samples // 2:]
    train_y = target[n_samples // 2:]
    exam_x = data[:n_samples // 2]
    exam_y = target[:n_samples // 2]

svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(train_x, train_y)

gmdh = MultilayerGMDH()
#gmdh.param.seq_type = SequenceTypeSet.sqRandom
#gmdh.param.seq_type = SequenceTypeSet.sqMode2
#gmdh.param.ref_function_types.add(RefFunctionType.rfLinear)
#gmdh.param.ref_function_types.add(RefFunctionType.rfCubic)
#gmdh.param.criterion_type = CriterionType.cmpBias
#gmdh.param.criterion_type = CriterionType.cmpComb_test_bias
#gmdh.param.criterion_type = CriterionType.cmpComb_bias_retrain
gmdh.feature_names = iris.feature_names
#gmdh.feature_names.append('random feature')
gmdh.param.max_layer_count = 30
#gmdh.param.admix_features = False
gmdh.param.manual_min_l_count_value = True
gmdh.param.min_l_count = 6
gmdh.fit(train_x, train_y)

# Now predict the value of the second half:
# predict with GMDH
pred_y_row = gmdh.predict(exam_x)

# predict with SVM
#pred_y_row = svm_clf.predict(exam_x)

pred_y = viris_class(pred_y_row)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

fig = plt.figure()

# Compute confusion matrix
cm = confusion_matrix(exam_y, pred_y)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
ax1 = fig.add_subplot(121)
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
ax2 = fig.add_subplot(122)
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

#print(gmdh)
PlotGMDH(gmdh, filename='img/iris_model', plot_model_name=True, view=True)
# gmdh.plot_layer_error()
plt.show()

