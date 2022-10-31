import torch


class FocalLoss(torch.nn.Module):
    """
    Focal Loss Implementation from the source:
    https://www.tutorialexample.com/implement-focal-loss-for-multi-label-classification-in-pytorch-pytorch-tutorial/
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions, targets):
        ce_loss = torch.nn.functional.cross_entropy(predictions, targets, reduction=self.reduction)
        loss = self.alpha * ((1 - torch.exp(-ce_loss)) ** self.gamma) * ce_loss
        return loss
