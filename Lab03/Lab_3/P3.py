from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

model = BayesianNetwork([('C', 'I'), ('C', 'A'), ('I', 'A')])

cpd_cutremur = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]])
cpd_incendiu = TabularCPD(variable='I', variable_card=2, values=[[0.99, 0.03], [0.01, 0.97]], evidence=['C'], evidence_card=[2])
cpd_alarma = TabularCPD(variable='A', variable_card=2, values=[[0.99, 0.02, 0.01, 0.98],
                                                                    [0.01, 0.98, 0.99, 0.02]], 
                                                                    evidence=['C', 'I'], evidence_card=[2, 2])

model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarma)

assert model.check_model()

infer = VariableElimination(model)

result = infer.query(variables=['I'], evidence={'A': 0})
print("Probabilitate incendiu fara declansarea alarmei:", result.values[1], result)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()
