import torch
from fokg_mini_project.kge import ForwardFunction, LossFunction

class TransEForward(ForwardFunction):
    def __init__(self, distance_norm=2):
        self.distance_norm = distance_norm

    def __call__(self,emb_ent, emb_rel, e1_idx, rel_idx, e2_idx, *args, **kwargs):
        emb_head = emb_ent(e1_idx)
        emb_rel = emb_rel(rel_idx)
        emb_tail = emb_ent(e2_idx)

        emb_head = torch.nn.functional.normalize(emb_head, p=self.distance_norm, dim=1)
        emb_tail = torch.nn.functional.normalize(emb_tail, p=self.distance_norm, dim=1)

        emb_sum = emb_head + emb_rel - emb_tail
        distance = torch.norm(emb_sum, p=self.distance_norm, dim=1)

        return distance

 



