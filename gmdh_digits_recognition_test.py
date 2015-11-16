__author__ = 'Konstantin Kolokolov'

import pylab as plt

from sklearn.datasets import load_digits

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from gmdh import MultilayerGMDH, RefFunctionType, SequenceTypeSet, CriterionType

digits = load_digits()
print(digits.data.shape)

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 3 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# pylab.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

gmdh = MultilayerGMDH()
mdh = MultilayerGMDH(ref_functions=(RefFunctionType.rfLinearPerm,),
                          criterion_type=CriterionType.cmpComb_test_bias,
                          admix_features=False,
                          criterion_minimum_width=5, max_layer_count=100, n_jobs=1)
gmdh.fit(data[:n_samples / 2], digits.target[:n_samples / 2])


#     # Now predict the value of the second half:
#     # predict with GMDH
#     y_pred = gmdh.predict(exam_x)
#     mse = metrics.mean_squared_error(exam_y, y_pred)
#     mae = metrics.mean_absolute_error(exam_y, y_pred)
#
# # Now predict the value of the digit on the second half:
# expected = digits.target[n_samples / 2:]
# predicted = classifier.predict(data[n_samples / 2:])
#
# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#
# images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
# for index, (image, prediction) in enumerate(images_and_predictions[:4]):
#     plt.subplot(2, 4, index + 5)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Prediction: %i' % prediction)
#
#
#
# plt.show()