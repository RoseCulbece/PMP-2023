import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

#a
'''
Incarca setul de date din Titanic.csv in variabila date
'''
file_path = 'Titanic.csv'
data = pd.read_csv(file_path)

'''
Gestioneaza valorile lipsa eliminand randurile goale si facand media varstelor unde este NULL
'''
data['Age'].fillna(data['Age'].median(), inplace=True)
data.dropna(subset=['Pclass'], inplace=True)
data['Pclass'] = data['Pclass'].astype('category')

#b
with pm.Model() as model:
    # Parametrii modelului
    intercept = pm.Normal('Intercept', mu=0, sigma=10)
    beta_pclass = pm.Normal('beta_pclass', mu=0, sigma=10)
    beta_age = pm.Normal('beta_age', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Modelul liniar
    survival_rate = intercept + beta_pclass * data['Pclass'].cat.codes + beta_age * data['Age']

    # Definirea variabilei dependente
    likelihood = pm.Bernoulli('Survived', pm.math.sigmoid(survival_rate), observed=data['Survived'])

    # Antrenarea modelului
    trace = pm.sample(1000, return_inferencedata=True)

#d
age = 30
pclass = 2 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
'''
Calculam rata de supravietuire si probabilitatea pentru pasagerul specific.

'''
survival_rate_sample = trace.posterior['Intercept'] + trace.posterior['beta_pclass'] * (pclass - 1) + trace.posterior['beta_age'] * age
survival_prob_sample = sigmoid(survival_rate_sample.values)

# Calculul intervalului HDI
hdi_90 = az.hdi(survival_prob_sample, hdi_prob=0.9)

# Afișarea graficului
plt.hist(survival_prob_sample.ravel(), bins=30, color='skyblue', alpha=0.7)

if hdi_90.ndim > 1:
    hdi_90_lower = hdi_90[0].mean()
    hdi_90_upper = hdi_90[1].mean()
else:
    hdi_90_lower, hdi_90_upper = hdi_90

plt.axvline(hdi_90_lower, color='k', linestyle='dashed', linewidth=1)
plt.axvline(hdi_90_upper, color='k', linestyle='dashed', linewidth=1)
plt.title('Distributia Probabilitatii de Supravietuire pentru Pasager de 30 Ani din Clasa a 2-a')
plt.xlabel('Probabilitatea de Supravietuire')
plt.ylabel('Frecventa')
plt.show()