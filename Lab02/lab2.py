#ex 1

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

lambda1 = 4  
lambda2 = 6
gen = 10000

x = stats.expon(0, 1/lambda1).rvs(size=gen)
y = stats.expon(0, 1/lambda2).rvs(size=gen)
X = x*y

media_X = np.mean(X)
deviatia_standard_X = np.std(X)

print("Media lui X:", media_X)
print("Deviatia standard a lui X:", deviatia_standard_X)

plt.hist(X, bins=50, density=True, alpha=0.6, color='b', label='Distribuția lui X')
plt.xlabel('Timp')
plt.ylabel('Densitate')
plt.title('Distributia timpului de servire pentru clienti')
plt.show()
