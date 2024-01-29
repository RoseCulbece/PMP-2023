import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom
'''
Se genereaza folosind distributia geometrica valori aleatoare pentru variabilele X si Y.
Se calculeaza numarul de cazuri in care X>Y**2 si se calculeaza probabilitatea ca fiind numarul de cazuri in care X>Y**2 impartit la numarul totoal.
'''
def monte_corlo_prb(param_X, param_Y, N):

    X_values = geom.rvs(param_X, size=N)
    Y_values = geom.rvs(param_Y, size=N)

    count = np.sum(X_values > Y_values**2)
    probabilitate = count / N

    return probabilitate, X_values, Y_values

'''
Se definesc datele problemei cum sunt in enunt.
'''
param_X = 0.3
param_Y = 0.5
N = 10000
k = 30

#Subpunctul a:

'''
Se calculeaza probabilitatea(P(X > Y**2)) folosind functia descrisa mai sus si valorile X, Y si se afiseaza
'''
probabilitate, X_values, Y_values = monte_corlo_prb(param_X, param_Y, N)
print("Monte Carlo P(X > Y**2):", probabilitate)
plt.scatter(X_values, Y_values, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Distributia valorilor generate pentru X si Y')
plt.show()

#Subpunctul b:

results = []

'''
Se executa k simulari Monte Carlo si adaugam in vectorul results probabilitatea optinuta
'''
for i in range(k):
    probabilitate, a, b = monte_corlo_prb(param_X, param_Y, N)
    results.append(probabilitate)

'''
Se calculeaza media si deviatia standard ale probabilitailor estimate din lista results folosind functiile mean si std si se afiseaza
'''
mean_estimate = np.mean(results)
std_dev_estimate = np.std(results)
print(f'Media estimata pentru P(X > Y**2) peste {k}: {mean_estimate}')
print(f'Deviatia standard estimata: {std_dev_estimate}')
