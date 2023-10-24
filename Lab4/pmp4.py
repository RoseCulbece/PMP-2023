import numpy as np

timp_plasare = 2 
deviatie_plasare = 0.5  
probabilitate_dorita = 0.95
timp_maxim = 15 

def simuleaza(alpha, timp_maxim):
    probabilitate = 0
    for _ in range(10000):
        timp_pregatire = np.random.exponential(scale=alpha)
        timp_plasare_final = np.random.normal(loc=timp_plasare, scale=deviatie_plasare)
        timp_total = timp_pregatire + timp_plasare_final
        if timp_total <= timp_maxim:
            probabilitate += 1
    
    probabilitate /= 10000
    return probabilitate

alpha_min = 0.01
alpha_max = 10.0
optim = None
optim_2 = 0

for i in range(10000):
    while alpha_max - alpha_min > 0.01:
        alpha_test = (alpha_min + alpha_max) / 2
        probabilitate_servire = simuleaza(alpha_test, timp_maxim)
        
        if probabilitate_servire < probabilitate_dorita:
            alpha_max = alpha_test
        else:
            alpha_min = alpha_test
            optim = alpha_test
    optim_2 += optim + np.random.normal(loc=timp_plasare, scale=deviatie_plasare)

print("α maxim: ", optim)
print("timpul mediu de aşteptare: ", optim_2/10000)
