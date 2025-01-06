import os
from fokg_mini_project.kg import KG
from fokg_mini_project.trans_e import TransEForward
from fokg_mini_project.distmult import DistMultForward
from fokg_mini_project.complex import ComplexForward
from fokg_mini_project.kge import KGE
from fokg_mini_project.loss_functions import MarginLoss, LogExpLoss
import torch

# Define hyperparameter lists
from itertools import product
# Load the knowledge graph
kg = KG(kb_path='data/family.nt')

model = KGE(kg, forward=ComplexForward(), loss_func=LogExpLoss(), embedding_dim=128, lr=0.001)

model.train(num_epochs=100)

correct_count = 0
for h, r, t in kg.triples:
    preds = model.predict_tail(kg.entity_idx[h],kg.relation_idx[r])
    print(f"Predicted: {kg.entities[preds[0][0]]}, Ground Truth: {kg.entities[kg.entity_idx[t]]}")  
    if kg.entities[preds[0][0]] == kg.entities[kg.entity_idx[t]] or kg.entities[preds[1][0]] == kg.entities[kg.entity_idx[t]]:
        correct_count += 1

print(f"Accuracy: {correct_count/len(kg.triples_idx)}")


