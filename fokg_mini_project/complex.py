import torch
from fokg_mini_project.kge import ForwardFunction, LossFunction

class ComplexForward(ForwardFunction):
	def __call__(self, emb_ent, emb_rel, e1_idx, rel_idx, e2_idx, *args, **kwargs):
		emb_head = emb_ent(e1_idx)
		emb_rel = emb_rel(rel_idx)
		emb_tail = emb_ent(e2_idx)

		emb_head_real, emb_head_img = torch.chunk(emb_head, 2, dim=1)
		emb_rel_real, emb_rel_img = torch.chunk(emb_rel, 2, dim=1)
		emb_tail_real, emb_tail_img = torch.chunk(emb_tail, 2, dim=1)

		scores = torch.sum(emb_head_real * emb_tail_real * emb_rel_real + emb_head_real * emb_tail_img * emb_rel_img + emb_head_img * emb_tail_real * emb_rel_img - emb_head_img * emb_tail_img * emb_rel_real, dim=1)
		return scores
	
