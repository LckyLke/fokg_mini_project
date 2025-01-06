from fokg_mini_project.kg import KG
from fokg_mini_project.trans_e import TransEForward
from fokg_mini_project.distmult import DistMultForward
from fokg_mini_project.complex import ComplexForward
from fokg_mini_project.kge import KGE
from fokg_mini_project.loss_functions import MarginLoss, LogExpLoss
import torch


kg = KG(kb_path='data/reference-kg.nt')

kg_test = KG(train_path="data/fokg-sw-train-2024.nt")

for key in kg_test.truth:
	print(key, kg_test.truth[key])

model = KGE(kg, forward=TransEForward(), loss_func=MarginLoss(), embedding_dim=32, lr=0.001)
model.train(num_epochs=20)

correct_count = 0
for h, r, t in kg_test.triples:
	model_pred = model.score_triple(kg.entity_idx[h], kg.relation_idx[r], kg.entity_idx[t])
	ground_truth= kg_test.truth_str[(h, r, t)]
	if model_pred < 0 and ground_truth == 0 or model_pred > 0 and ground_truth == 1:
		correct_count += 1
	
print(f"Accuracy: {correct_count/len(kg_test.triples_idx)}")