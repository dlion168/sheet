import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSumLayer(nn.Module):
    def __init__(self, num_layers, normalize=False):
        super(WeightedSumLayer, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.normalize = normalize
        
    def forward(self, all_hs, ):
        """
        all_hs: List of tensors of shape [batch_size, time_steps, hidden_dim]
        Returns: Tensor of shape [batch_size, time_steps, hidden_dim]
        """     
        
        if len(all_hs) == 1:
            return all_hs[0]
        
        stacked_hs = torch.stack(all_hs, dim=1)  # [batch_size, num_layers, time_steps, hidden_dim]
        
        if self.normalize:
            stacked_hs = F.layer_norm(stacked_hs, (stacked_hs.shape[-1],))
        
        norm_weights = F.softmax(self.weights, dim=-1)
        norm_weights = norm_weights.view(1, -1, 1, 1)
        
        weighted_hs = (stacked_hs * norm_weights).sum(dim=1)
        
        return weighted_hs