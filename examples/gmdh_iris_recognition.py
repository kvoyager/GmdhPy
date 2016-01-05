__author__ = 'Konstantin Kolokolov'

import pylab as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix
from gmdhpy.gmdh import MultilayerGMDH
from gmdhpy.plot_gmdh import PlotGMDH


def iris_class(value):
    if value > 1.5:
        return 2
    elif value <= 1.5 and value >= 0.5:
        return 1
    else:
        return 0


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


if __name__ == '__main__':

    iris = datasets.load_iris()

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

    train_data_is_the_first_half = False
    n = n_samples // 2
    if train_data_is_the_first_half:
        train_x = data[:n]
        train_y = target[:n]
        exam_x = data[n:]
        exam_y = target[n:]
    else:
        train_x = data[n:]
        train_y = target[n:]
        exam_x = data[:n]
        exam_y = target[:n]

    svm_clf = svm.SVC(kernel='linear')
    svm_clf.fit(train_x, train_y)

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
    gmdh.fit(train_x, train_y)

    # Now predict the value of the second half:
    # predict with GMDH
    pred_y_row = gmdh.predict(exam_x)

    # predict with SVM
    #pred_y_row = svm_clf.predict(exam_x)

    pred_y = viris_class(pred_y_row)

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

    PlotGMDH(gmdh, filename='iris_model', plot_model_name=True, view=True)
    gmdh.plot_layer_error()
    plt.show()
