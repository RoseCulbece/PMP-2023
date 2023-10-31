import pymc as pm
import pandas as pd

data = pd.read_csv('trafic.csv')

minute = data['minut']
nr_masini = data['nr. masini']
ore_modif = [7, 16, 8, 19]

with pm.Model() as model:
    lam = pm.Exponential('lambda', 1.2)
    traffic = pm.Poisson('traffic', mu=lam, observed=nr_masini)
    d_lambd = pm.Normal("d_lambd", mu=0, sigma=1, shape=len(ore_modif))
with model:
    trace = pm.sample(2000, tune=1000, cores=2)

pm.plot_trace(trace)