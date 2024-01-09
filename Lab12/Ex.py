import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

grid = np.linspace(0, 1, 50)
prior_left_skewed = 1 - grid

heads = 6
tails = 9
likelihood = stats.binom.pmf(heads, heads + tails, grid)

posterior_custom = likelihood * prior_left_skewed
posterior_custom /= posterior_custom.sum()

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.plot(grid, posterior_custom, 'o-')
plt.title('Custom Prior: Left-Skewed')
plt.yticks([])
plt.xlabel('θ')

N_values = [100, 1000, 10000]
means = []
deviations = []

for N in N_values:
    errors = []
    for _ in range(1000):
        x, y = np.random.uniform(-1, 1, size=(2, N))
        inside = (x ** 2 + y ** 2) <= 1
        pi = inside.sum() * 4 / N
        error = abs((pi - np.pi) / pi) * 100
        errors.append(error)

    mean = sum(errors) / 1000
    a = [(e - mean) * (e - mean) for e in errors]
    deviation = sum(a) / 1000
    means.append(mean)
    deviations.append(deviation)

plt.subplot(132)
plt.errorbar(N_values, means, yerr=deviations, fmt='o-', label='Error vs. N')
plt.xlabel('N')
plt.ylabel('Error')
plt.legend()

trace = np.zeros(10000)
old_x = 0.5
delta = np.random.normal(0, 0.5, 10000)

for i in range(10000):
    new_x = old_x + delta[i]
    if new_x < 0:
        new_x = 0
    elif new_x > 1:
        new_x = 1
    acceptance = stats.beta.pdf(new_x, 2, 2) / stats.beta.pdf(old_x, 2, 2)
    if acceptance >= np.random.random():
        trace[i] = new_x
        old_x = new_x
    else:
        trace[i] = old_x

x = np.linspace(0.01, .99, 100)
y = stats.beta.pdf(x, 2, 2)

plt.subplot(133)
plt.xlim(0, 1)
plt.plot(x, y, 'C1-', lw=3, label='Distribuție reală (beta-binomial)')
plt.hist(trace, bins=25, density=True, label='Distribuție estimată')
plt.xlabel('x')
plt.ylabel('pdf(x)')
plt.yticks([])
plt.legend()

plt.tight_layout()
plt.show()
