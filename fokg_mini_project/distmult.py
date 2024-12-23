import torch
from fokg_mini_project.kge import ForwardFunction, LossFunction	

class DistMultForward(ForwardFunction):
	def __call__(self, emb_ent, emb_rel, e1_idx, rel_idx, e2_idx, *args, **kwargs):
		emb_head = emb_ent(e1_idx)
		emb_rel = emb_rel(rel_idx)
		emb_tail = emb_ent(e2_idx)

		scores = torch.sum(emb_head * emb_rel * emb_tail, dim=1)
		return scores
	
