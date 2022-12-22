"""
Old Faithful

Besonderheiten des Geysirs
Nach Wikipedia: Es zeigte sich, dass seine Ausbruchshöhen und Intervalle am allerwenigsten von allen beobachteten Geysiren variieren. Seit damals hat er nie einen periodischen Ausbruch verpasst. Der nächste Ausbruch wird anhand einer Formel von Dr. George D. Marler abgeschätzt, in die das letzte Intervall, die letzte Ausbruchsdauer, die Ausbruchszeit und einige andere Parameter einfließen. Diese Zeit wird im Besucherzentrum (Visitor Center) bekannt gegeben.

Informationen zum Datensatz: https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/faithful.html
"""
import matplotlib.pyplot as plt
import pydataset
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
import matplotlib as mpl

from sklearn.preprocessing import MinMaxScaler

colors = ["navy", "turquoise"] #, "darkorange"]

def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")

data = pydataset.data('faithful')
X = np.array(data)
print(X.shape)

plt.figure(figsize=(12, 6))
plt.plot(data['eruptions'], data['waiting'], 'ob')
plt.title('Old Faithful - KMEANS')
plt.xlabel('eruption time in min')
plt.ylabel('waiting time in min')


plt.figure(figsize=(12, 6))

scaler = MinMaxScaler().fit(X)
XX = scaler.transform(X)
n_clusters = 3
cls = KMeans(n_clusters=n_clusters).fit(XX)
ax = plt.axes()
for k, col in zip(range(n_clusters), ['b', 'r', 'y']):
    mask = cls.labels_ == k
    cluster_center = cls.cluster_centers_[k]
    ax.plot(X[mask, 0], X[mask, 1], 'ow', markerfacecolor=col)
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)

scaler = MinMaxScaler().fit(X)
XX = scaler.transform(X)
cls = DBSCAN(eps=1, min_samples=10).fit(XX)
y = cls.labels_
plt.figure(figsize=(12, 6))
for i in np.unique(y):
    mask = (y == i)
    plt.plot(X[mask, 0], X[mask, 1], 'o', label=i)
plt.title('Old Faithful - DBSCAN')
plt.xlabel('eruption time in min')
plt.ylabel('waiting time in min')
plt.legend()

scaler = MinMaxScaler().fit(X)
XX = scaler.transform(X)

forms = ['diag', 'spherical', 'full', 'tied']
for form in forms:
    cls = GaussianMixture(n_components=3, covariance_type=form).fit(XX)
    y = cls.predict(XX)
    plt.figure(figsize=(12, 6))
    for i in np.unique(y):
        mask = y == i
        plt.plot(XX[mask, 0], XX[mask, 1], 'o', label=i)
    plt.title('Old Faithful - GMM - ' + form)
    plt.xlabel('eruption time in min')
    plt.ylabel('waiting time in min')
    plt.legend()
    make_ellipses(cls, plt.gca())


clust = OPTICS(min_samples=30, min_cluster_size=0.05, xi=0.1)

# Run the fit
clust.fit(XX)
space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]
# Reachability plot
colors = ["g.", "r.", "b.", "y.", "c."]
plt.figure()
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    plt.plot(Xk, Rk, color, alpha=0.3)
plt.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
plt.gca().set_ylabel("Reachability (epsilon distance)")
plt.gca().set_title("Reachability Plot")

# OPTICS
colors = ["g.", "r.", "b.", "y.", "c."]
plt.figure()
for klass, color in zip(range(0, 5), colors):
    Xk = X[clust.labels_ == klass]
    plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
plt.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], "k+", alpha=0.1)
plt.gca().set_title("Automatic Clustering\nOPTICS")

plt.show()
"""
Aufgaben
1) Finde Cluster
2) Finde Ausreißer
"""