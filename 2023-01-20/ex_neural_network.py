from sklearn.neural_network import MLPClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
n_samples = len(digits.images)
reshaped_data = digits.images.reshape((n_samples, 64))  # 8x8 -> 64

reshaped_data_train, reshaped_data_test, target_data_train, target_data_test = train_test_split(reshaped_data, digits.target, test_size=0.25, random_state=0)
neuronen_schicht_1 = 10
neuronen_schicht_2 = 20

clf = MLPClassifier(#random_state=1,
                    max_iter = 1000, activation='tanh', hidden_layer_sizes=[neuronen_schicht_1, neuronen_schicht_2]).fit(reshaped_data_train, target_data_train)

result = clf.predict_proba(reshaped_data_test)
for i, result in enumerate(result):
    print('Datenpunkt ' + str(i) + ': ')
    print('   Wahrscheinlichkeit Klasse 1: ' + str(result[0]))
    print('   Wahrscheinlichkeit Klasse 2: ' + str(result[1]))
