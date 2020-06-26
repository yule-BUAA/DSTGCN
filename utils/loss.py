import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, truth, predict):
        """
        Args:
            truth: tensor, (batch_size, predict_dim)
            predict: tenor, (batch_size, predict_dim)
        Returns:
            output: loss value
        """
        loss = self.mse_loss(predict, truth)
        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, truth, predict):
        """
        Args:
            truth: tensor, (N)
            predict: tenor, (N)
        Returns:
            output: loss value
        """
        loss = self.bce_loss(predict, truth)
        return loss
