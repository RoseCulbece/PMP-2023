import pymc as pm
import pandas as pd

df = pd.read_csv('Admission.csv')

with pm.Model() as logistic_model:
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    pi = pm.Deterministic('pi', pm.math.sigmoid(beta0 + beta1 * df['GRE'] + beta2 * df['GPA']))
    admission = pm.Bernoulli('admission', p=pi, observed=df['Admission'])
    trace = pm.sample(2000, tune=1000, cores=2)

pm.summary(trace).round(2)