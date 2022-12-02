"""
Old Faithful

Besonderheiten des Geysirs
Nach Wikipedia: Es zeigte sich, dass seine Ausbruchshöhen und Intervalle am allerwenigsten von allen beobachteten Geysiren variieren. Seit damals hat er nie einen periodischen Ausbruch verpasst. Der nächste Ausbruch wird anhand einer Formel von Dr. George D. Marler abgeschätzt, in die das letzte Intervall, die letzte Ausbruchsdauer, die Ausbruchszeit und einige andere Parameter einfließen. Diese Zeit wird im Besucherzentrum (Visitor Center) bekannt gegeben.

Informationen zum Datensatz: https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/faithful.html
"""
import matplotlib.pyplot as plt
import pydataset
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN

data = pydataset.data('faithful')
X = np.array(data)
print(X.shape)

# Erstellen KMeans objekt
k_means = KMeans(init='k-means++', n_clusters=2)
# Geben KMeans die Daten (X)
k_means.fit(X)
# Speichern die Klassen-Variablen in eigenen Variablen
k_means_labels = k_means.labels_                            # Namen sind Integer 0,1,...,n
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

# Erstelle die Grafik
plt.figure(figsize=(12, 6))
plt.plot(data['eruptions'], data['waiting'], 'ob')
plt.title('Old Faithful')
plt.xlabel('eruption time in min')
plt.ylabel('waiting time in min')
ax = plt.axes()

# # Aufgabe 1
# # Wenn wir Sachen Gruppen zuordnen wollen
# # Definiere die Farben für die Cluster
# colors = ['#4EACC5', '#FF9C34']
#
# # Mappen die Zentren zu Farben in Tupel
# for k, col in zip(range(2), colors):
#     print(f'k={k}, col={col}')
#     mask = k_means_labels == k
#     cluster_center = k_means_cluster_centers[k]
#     ax.plot(X[mask, 0], X[mask, 1], '.w', markerfacecolor=col)
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=6)


# Aufgabe 2
scaler = preprocessing.MinMaxScaler()
# Normalisieren alle Daten -> [0,1] damit epsilon sowohl die X- also auch Y-Axe erfasst
normalized_dataset = scaler.fit_transform(X)

# Ausreißer finden :)
db = DBSCAN(eps=0.08, min_samples=10).fit(normalized_dataset)

# Normalisieren alle Daten zurück damit die Daten auch stimmen
X = scaler.inverse_transform(normalized_dataset)

for l in np.unique(db.labels_):
    plt.plot(X[db.labels_==l, 0], X[db.labels_==l, 1], 'o', label='anomaly' if l == -1 else l)

plt.legend()
plt.show()
"""
Aufgaben
1) Finde Cluster
2) Finde Ausreißer
"""
#%%

#%%
