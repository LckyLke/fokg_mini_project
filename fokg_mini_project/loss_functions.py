import torch
from fokg_mini_project.kge import LossFunction

class MarginLoss(LossFunction):
    def __init__(self, margin=1.0):
        self.margin = torch.nn.Parameter(torch.tensor(margin), requires_grad=False)
        

    def __call__(self, pos_scores, neg_scores, *args, **kwargs):
        return torch.nn.functional.relu(self.margin + pos_scores - neg_scores).sum()


class LogExpLoss(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, scores, labels, *args, **kwargs):
        # Rescale labels to {-1, 1}
        rescaled_labels = 2 * labels - 1
        # Compute the log-exp loss
        loss = torch.log1p(torch.exp(-rescaled_labels * scores))
        return loss.mean()  # Average the loss over the batch