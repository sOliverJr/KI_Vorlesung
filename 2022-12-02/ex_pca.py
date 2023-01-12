"""Best Place to live in North America
Informationen über die Daten (Places Rated Almanac, by Richard Boyer and David Savageau): https://cran.r-project.org/web/packages/tourr/tourr.pdf
"""
import numpy as np
data = np.loadtxt('data/places.csv', delimiter=',', skiprows=1)

# print(data[:1, 1:10])

# Eigenleistung
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(data[:, 1:10])

print(pca.explained_variance_ratio_)

from itertools import accumulate
print([i for i in accumulate(pca.explained_variance_ratio_)])


# Gibt die Komponenten (alle) von den ersten 4 ersten und besten Vektoren zurück
print(pca.components_[:4, :])

"""
Aufgaben
1) Welche Hauptkomponenten werden benötigt? Was sagen diese aus?
Hauptkomponent = vektor
97% reichen in dem Fall, weshalb die ersten 4 Hauptkomponenten benutzt werden

2) Mit welchen Variablen kann der Datensatz am einfachsten erklärt werden?
recreat, healthcare, educ, casenum

# 3) Welche Städte haben ungewöhnliche Daten? (z.B. mit matplotlib)
"""