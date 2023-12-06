import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

df = pd.read_csv('Admission.csv')

with pm.Model() as logistic_model:
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    pi = pm.Deterministic('pi', pm.math.sigmoid(beta0 + beta1 * df['GRE'] + beta2 * df['GPA']))
    admission = pm.Bernoulli('admission', p=pi, observed=df['Admission'])
    trace = pm.sample(2000, tune=1000, cores=2)
print(pm.summary(trace).round(2))

plt.figure()
plt.scatter(df['GRE'], df['GPA'], c=df['Admission'], cmap='viridis', alpha=0.7)
plt.xlabel('GRE Scores')
plt.ylabel('GPA Scores')
plt.show()

d1 = {'GRE': (550 - df['GRE'].mean()) / df['GRE'].std(), 'GPA': (3.5 - df['GPA'].mean()) / df['GPA'].std()}
with logistic_model:
    pred_post1 = pm.sample_posterior_predictive(trace, var_names=['admission'], extend_inferencedata=d1)
hdi_1 = pm.hpd(pred_post1['admission'], hdi_prob=0.9)

d2 = {'GRE': (500 - df['GRE'].mean()) / df['GRE'].std(), 'GPA': (3.2 - df['GPA'].mean()) / df['GPA'].std()}
with logistic_model:
    pred_post2 = pm.sample_posterior_predictive(trace, samples=2000, var_names=['admission'], obs=d2)
hdi_2 = pm.hpd(pred_post2['admission'], hdi_prob=0.9)

print("GRE=550, GPA=3.5:", hdi_1)
print("GRE=500, GPA=3.2:", hdi_2)
