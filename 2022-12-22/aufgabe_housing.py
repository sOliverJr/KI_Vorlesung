import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

housing = pandas.read_csv('housing.csv')

# Zellen: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']
X = housing[['Longitude', 'Latitude', 'HouseAge', 'MedInc']]
X2 = housing[['Longitude', 'Latitude', 'HouseAge']]

plt.figure()
plt.title('Scatterplot')
seaborn.scatterplot(data=X, x='Longitude', y='Latitude', hue="HouseAge")
# plt.show()



#### K-Means
plt.figure()
ax = plt.axes()
plt.title('K-Means')

# Erstellen KMeans objekt
k_means = KMeans(init='k-means++', n_clusters=4)

# Geben KMeans die Daten (X)
X = X.to_numpy()
k_means.fit(X)

# Speichern die Klassen-Variablen in eigenen Variablen
k_means_labels = k_means.labels_                            # Namen sind Integer 0,1,...,n
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = numpy.unique(k_means_labels)

# Wenn wir Sachen Gruppen zuordnen wollen
# Definiere die Farben für die Cluster
colors = ['#4EACC5', '#FF9C34']

# Mappen die Zentren zu Farben in Tupel
for k, col in zip(range(2), colors):
    mask = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[mask, 0], X[mask, 1], '.w', markerfacecolor=col)
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
# plt.show()


#### DBSCAN
plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('House Age')

scaler = preprocessing.MinMaxScaler()
# Normalisieren alle Daten -> [0,1] damit epsilon sowohl die X- also auch Y-Axe erfasst
normalized_dataset = scaler.fit_transform(X2)

# Ausreißer finden :)
db = DBSCAN(eps=0.05, min_samples=10).fit(normalized_dataset)

# Normalisieren alle Daten zurück damit die Daten auch stimmen
X2 = scaler.inverse_transform(normalized_dataset)

for l in numpy.unique(db.labels_):
    plt.plot(X2[db.labels_ == l, 0], X2[db.labels_ == l, 1], X2[db.labels_ == l, 2], 'o', label='anomaly' if l == -1 else l)

plt.legend()
# plt.show()


#### GaussianMixture

scaler = MinMaxScaler().fit(X2)
XX = scaler.transform(X2)

forms = ['diag', 'spherical', 'full', 'tied']
for form in forms:
    cls = GaussianMixture(n_components=3, covariance_type=form).fit(XX)
    y = cls.predict(XX)
    plt.figure(figsize=(12, 6))
    for i in numpy.unique(y):
        mask = y == i
        plt.plot(XX[mask, 0], XX[mask, 1], 'o', label=i)
    plt.title('GaussianMixture - ' + form)
    plt.legend()
plt.show()
