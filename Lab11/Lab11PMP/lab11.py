import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

clusters = 3
n_cluster = [200, 150, 150]
means = [5, 0, 3]
std_devs = [2, 2, 2]
mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
az.plot_kde(np.array(mix))
plt.show()

def funct(nr_comp):
    with pm.Model() as model:
      p1 = pm.Dirichlet('p1', a=np.ones(nr_comp))
      means = pm.Normal('means', mu=np.array(mix).mean(), sigma=10, shape=nr_comp)
      sd = pm.HalfNormal('sigma3', sigma=10)
      y = pm.NormalMixture('y', w=p1, mu=means, sigma=sd, observed=np.array(mix))
      idata_mg = pm.sample(random_seed=123, return_inferencedata=True, cores=1, idata_kwargs={'log_likelihood': True})
    return idata_mg

loo = az.compare({'model1': funct(2), 'model2': funct(3), 'model3': funct(4)}, method='stacking', ic='loo', scale='deviance')
print(loo)

waic = az.compare({'model1': funct(2), 'model2': funct(3), 'model3': funct(4)}, method='stacking', ic='waic', scale='deviance')
print(waic)

'''
Pe baza rezultatelor, modelul cu 3 componente pare să fie cel mai bun, deoarece are cel mai bun scor, modelele 2 si 4 fiind aproape la fel de bune.
'''
