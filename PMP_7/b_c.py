import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

with pm.Model() as model:
    data = pd.read_csv('auto-mpg.csv')
    data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
    data.dropna(subset=['horsepower'], inplace=True)
    horsepower_data = data['horsepower'].values.astype(float)
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + beta * data['horsepower']
    mpg = pm.Normal('mpg', mu=mu, sigma=sigma, observed=data['mpg'])

if __name__ == '__main__':
    with model:
        trace = pm.sample(1000, tune=1000)
    beta_mean = trace.posterior['beta'].mean().values
    alpha_mean = trace.posterior['alpha'].mean().values
    print(f"Dreapta de regresie: mpg = {alpha_mean} + {beta_mean} * horsepower")

    print("Distributia")
    az.summary(trace).round(2)
    az.plot_trace(trace)
    plt.show()
    