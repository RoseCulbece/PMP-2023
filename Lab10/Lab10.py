import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

def read_data(file_name):
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt(file_name)
    x = dummy_data[:, 0]
    y = dummy_data[:, 1]
    return x, y

def generate_data(x, order, sd):
    beta = np.random.normal(0, sd, order + 1)
    y_new = np.polyval(beta[::-1], x) + np.random.normal(0, 1, len(x))
    return y_new

def plot_data_and_model(x, y, order, title):
    coef = np.polyfit(x, y, order)
    p = np.poly1d(coef)
    plt.scatter(x, y, label='Date cu zgomot')
    plt.plot(x, p(x), color='red', label=f'Model Polinomial Ordin {order}')
    plt.title(title)
    plt.legend()
    plt.show()

file_name = '/content/dummy.csv'

x, y = read_data(file_name)

y_new = generate_data(x, order=5, sd=10)
plot_data_and_model(x, y_new, order=5, title="sd=10")

y_new = generate_data(x, order=5, sd=100)
plot_data_and_model(x, y_new, order=5, title="sd=100")

sd_array = np.array([10, 0.1, 0.1, 0.1, 0.1, 0.1])
y_new = generate_data(x, order=5, sd=sd_array)
plot_data_and_model(x, y_new, order=5, title="")

x_extended, y_extended = x[:500], y[:500]
y_new = generate_data(x_extended, order=5, sd=10)
plot_data_and_model(x_extended, y_new, order=5, title="500 de Puncte")
