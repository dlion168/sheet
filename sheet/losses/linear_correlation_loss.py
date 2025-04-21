import torch
import torch.nn as nn


class LinearCorrelationLoss(nn.Module):
    """
    Linear Correlation Loss (LCC Loss)
    Encourages high linear correlation between predicted scores and ground-truth scores.

    Implements:
        L_linear = -1 * Cov(s, ŝ) / (σ_s * σ_ŝ + ε)
    """

    def __init__(self, eps=1e-6):
        super(LinearCorrelationLoss, self).__init__()
        self.eps = eps

    def forward(self, pred_score, gt_score, lens=None, device=None):
        """
        Args:
            pred_score: Tensor of shape [B]
            gt_score: Tensor of shape [B]
        """
        if pred_score.dim() > 1:
            pred_score = pred_score.view(-1)
        if gt_score.dim() > 1:
            gt_score = gt_score.view(-1)

        # mean
        pred_mean = torch.mean(pred_score)
        gt_mean = torch.mean(gt_score)

        # covariance using sample formula: (1 / (N - 1)) * sum((x - x̄) * (y - ȳ))
        batch_size = pred_score.size(0)
        cov = torch.sum((pred_score - pred_mean) * (gt_score - gt_mean)) / (batch_size - 1)

        # standard deviations
        std_pred = torch.std(pred_score, unbiased=True)
        std_gt = torch.std(gt_score, unbiased=True)

        # linear correlation loss
        lcc = cov / (std_pred * std_gt + self.eps)
        loss = -1 * lcc

        return loss
