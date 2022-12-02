"""Best Place to live in North America
Informationen über die Daten (Places Rated Almanac, by Richard Boyer and David Savageau): https://cran.r-project.org/web/packages/tourr/tourr.pdf
"""
import numpy as np
data = np.loadtxt('data/places.csv', delimiter=',', skiprows=1)

print(data[:, 1:10])

"""
Aufgaben
1) Welche Hauptkomponenten werden benötigt? Was sagen diese aus?
2) Mit welchen Variablen kann der Datensatz am einfachsten erklärt werden?
3) Welche Städte haben ungewöhnliche Daten? (z.B. mit matplotlib)
"""