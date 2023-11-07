import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

fig, axes = plt.subplots(len(Y_values), len(theta_values), figsize=(12, 8))

for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        with pm.Model() as model:
            n = pm.Poisson('n', mu=10)
            pm.Binomial('obs', n=n, p=theta, observed=Y)
            trace = pm.sample(2000, tune=2000, cores=1, chains=1)
        az.plot_posterior(trace, ax=axes[i, j])
        axes[i, j].set_title(f"Y={Y}, theta={theta}")

plt.tight_layout()
plt.show()
