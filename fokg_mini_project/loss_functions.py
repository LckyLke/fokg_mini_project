import torch
from fokg_mini_project.kge import LossFunction

class MarginLoss(LossFunction):
    def __init__(self, margin=1.0):
        self.margin = torch.nn.Parameter(torch.tensor(margin), requires_grad=False)
        

    def __call__(self, pos_scores, neg_scores, *args, **kwargs):
        return torch.nn.functional.relu(self.margin + pos_scores - neg_scores).sum()