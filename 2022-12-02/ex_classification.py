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
print(digits.images.shape)

"""
Aufgabe
Für die SVM wird ein Eingangsvektor, keine Matrix benötigt. Dafür gibt es den Befehl reshape(). Mehr unter http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.reshape.html#numpy.reshape
Die SVM muss initialisiert werden. Der Parameter C muss bestimmt werden. Benutzt metrics.classification_report und metrics.confusion_matrix um die Güte der Klassifikation zu beurteilen.
Damit man sich von der Prädiktion überzeugen kann, plottet ähnlich wie oben einige Zahlen mit ihrem wahren Label und eurer Prädiktion
"""

# Dreidimensionales Array in zweidimensionales reshapen
# (1712, 8, 8) -> (1712, 64)
# 8x8 Array an Informationen wird in 1x64 Array geschrieben
reshaped_data = digits.images.reshape((n_samples, 64))  # 8x8 -> 64

from sklearn.model_selection import train_test_split

# Daten werden in einen Train- und einen Test-Datensatz aufgeteilt.
# Der Test-Datensatz stellt 25 Prozent der Originaldaten dar.
reshaped_data_train, reshaped_data_test, target_data_train, target_data_test = train_test_split(reshaped_data,
                                                                                                digits.target,
                                                                                                test_size=0.25,
                                                                                                random_state=0)

classifier = svm.SVC()
# Einlernen der Daten
classifier.fit(reshaped_data_train, target_data_train)

# Ausführen der Test-Daten
y = classifier.predict(reshaped_data_test)

print('Classification Report:')
print(metrics.classification_report(target_data_test, y))

# Die Confusion Matrix zeigt das von dem Modell erkannte Ergebnis in zusammenhang zu dem Input.
# Wenn z.B. eine 6 als 8 erkannt wurde, erhöht sie die Zahl an der Stelle [6,8] um eins
# Wenn eine 7 korrekt als eine 7 erkannt wird, wir die Stelle [7,7] um eins erhöht.
print('Confusion Matrix:')
print(metrics.confusion_matrix(target_data_test, y))


# %%
