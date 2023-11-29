import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import numpy as np

#a, b
df = pd.read_csv('Prices.csv')

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=1)
    beta2 = pm.Normal('beta2', mu=0, sigma=1)
    eps = pm.HalfCauchy('eps', 5)

    niu = pm.Deterministic('niu', alpha + beta1 * df['Speed'] + beta2 * np.log(df['HardDrive']))
    y_pred = pm.Normal('price_pred', mu=niu, sigma=eps, observed=df['Price'])

    data = pm.sample(1000, tune=1000, return_inferencedata=True)

    az.plot_trace(data, var_names=['alpha', 'beta1', 'beta2', 'eps'])
    plt.show()

az.plot_posterior(data, var_names='beta1', hdi_prob=0.95)
plt.show()
az.plot_posterior(data, var_names='beta2', hdi_prob=0.95)
plt.show()
'''
Cu cat frecventa procesorului este mai mare cu atat pretul este mai mare ceea ce semnifica 
faptul ca frecventa procesorului e un predictor util al pretului de vanzare?
In schimb, intervalul pentru marimea hard diskului este in apropierea lui zero, lucru care arata ca
nu va influenta pretul de vanzare. Astfel, marimea hard diskului nu e un predictor
'''
n = np.where((df['Speed'] == 33) & ((df['HardDrive']) == 540))[0]
price = data.posterior['niu'][:, n]
hdi_price = az.hdi(price, hdi_prob=0.90)
print("Prețul de vânzare:", hdi_price)
