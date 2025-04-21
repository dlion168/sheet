import torch
import torch.nn as nn


class RelativeRankingLoss(nn.Module):
    """
    Relative Ranking Loss as defined in:
    Golestaneh et al., WACV 2022, Section 3.4

    This loss encourages the relative ranking among the extreme cases
    within a batch based on predicted and subjective quality scores.
    """

    def __init__(self):
        super(RelativeRankingLoss, self).__init__()

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_score: Tensor of shape (B,) - predicted quality scores
            gt_score: Tensor of shape (B,) - ground truth subjective scores
        Returns:
            loss: scalar
        """
        assert pred_score.shape == gt_score.shape, "Shape mismatch"

        # Get indices for sorting the ground truth scores
        sorted_gt, indices = torch.sort(gt_score, descending=True)

        # Get extreme indices
        idx_qmax = indices[0]
        idx_qmax_2 = indices[1]
        idx_qmin = indices[-1]
        idx_qmin_2 = indices[-2]

        # Fetch corresponding predicted scores
        qmax = pred_score[idx_qmax]
        qmax_2 = pred_score[idx_qmax_2]
        qmin = pred_score[idx_qmin]
        qmin_2 = pred_score[idx_qmin_2]

        # Fetch corresponding ground-truth scores for margin calculation
        s_qmax = gt_score[idx_qmax]
        s_qmax_2 = gt_score[idx_qmax_2]
        s_qmin = gt_score[idx_qmin]
        s_qmin_2 = gt_score[idx_qmin_2]

        # Adaptive margins from paper
        margin1 = s_qmax_2 - s_qmin
        margin2 = s_qmax - s_qmin_2

        # d(x, y) = |x - y|
        dist_qmax_qmax2 = torch.abs(qmax - qmax_2)
        dist_qmax_qmin = torch.abs(qmax - qmin)
        dist_qmin_qmin2 = torch.abs(qmin - qmin_2)

        # Triplet loss components
        triplet1 = torch.clamp(dist_qmax_qmax2 - dist_qmax_qmin + margin1, min=0.0)
        triplet2 = torch.clamp(dist_qmin_qmin2 - dist_qmax_qmin + margin2, min=0.0)

        # Total loss
        loss = triplet1 + triplet2

        return loss
