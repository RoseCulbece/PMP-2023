import random
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def masluita():
    return 1 if random.random() < 2/3 else 0

def normala():
    return random.choice([0, 1])

#Functie pentru simularea jocului
def simulare_multipla(numar_jocuri):
    castiguri_j0 = 0
    castiguri_j1 = 0

    for i in range(numar_jocuri):
        c1 = masluita()
        c2 = normala()
        if c1 >= c2:
            castigator = c1
        else:
            castigator = c2 
        if castigator == 0:
            castiguri_j0 += 1
        else:
            castiguri_j1 += 1

    procentaj_j0 = (castiguri_j0 / numar_jocuri) * 100
    procentaj_j1 = (castiguri_j1 / numar_jocuri) * 100

    return procentaj_j0, procentaj_j1

# Simularea a 10.000 de jocuri
rezultate = simulare_multipla(10000)
print("J0:", rezultate[0])
print("J1:", rezultate[1])

#Reteaua bayesiana
model = BayesianModel([('J0_moneda', 'Castigator'), ('J1_moneda', 'Castigator')])

# Se genereaza datele si se retine de cate ori castiga j1 si j0 si cine castiga
date_antrenament = []
for _ in range(10000):
    j0_moneda = aruncare_moneda_masluita()
    j1_moneda = aruncare_moneda_normala()
    castigator = max(j0_moneda, j1_moneda)
    date_antrenament.append({'J0_moneda': j0_moneda, 'J1_moneda': j1_moneda, 'Castigator': castigator})

model.fit(date_antrenament, estimator=MaximumLikelihoodEstimator)

# Se calculeaza probabilitatile luand pe fiecare in parte 
inference = VariableElimination(model)
prob_castig_j0 = inference.query(variables=['Castigator'], evidence={'J0_moneda': 1, 'J1_moneda': 0}).values[0]
prob_castig_j1 = inference.query(variables=['Castigator'], evidence={'J0_moneda': 0, 'J1_moneda': 1}).values[0]
print("J0:", prob_castig_j0)
print("J1:", prob_castig_j1)


