# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 10 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# pylab.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(13, 4))
for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(1, 10, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)

print('There are {0} images in the dataset'.format(n_samples))
print(digits.images[0].shape)


"""
Aufgabe
Für die SVM wird ein Eingangsvektor, keine Matrix benötigt. Dafür gibt es den Befehl reshape(). Mehr unter http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.reshape.html#numpy.reshape

Die SVM muss initialisiert werden. Der Parameter C muss bestimmt werden. Benutzt metrics.classification_report und metrics.confusion_matrix um die Güte der Klassifikation zu beurteilen.

Damit man sich von der Prädiktion überzeugen kann, plottet ähnlich wie oben einige Zahlen mit ihrem wahren Label und eurer Prädiktion
"""

import numpy as np
a = np.zeros((8,8))
print(a)
print(a.shape)
print(a.reshape((64, )))