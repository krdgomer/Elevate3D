import torch
import torch.nn as nn

class ElevationLoss(nn.Module):
    def __init__(self, base_weight=1.0, critical_range_weight=3.0, critical_range=(0, 100)):
        """
        Custom loss function for DSM generation that applies higher weights to errors
        in specified elevation ranges.
        
        Args:
            base_weight (float): Base weight for all elevation values
            critical_range_weight (float): Weight multiplier for the critical range
            critical_range (tuple): (min, max) values of the critical elevation range
        """
        super().__init__()
        self.base_weight = base_weight
        self.critical_range_weight = critical_range_weight
        self.critical_min = critical_range[0]
        self.critical_max = critical_range[1]
        
    def get_weights(self, target):
        """
        Generate weights based on elevation values.
        Returns a tensor of same shape as target with weights.
        """
        weights = torch.ones_like(target) * self.base_weight
        critical_mask = (target >= self.critical_min) & (target <= self.critical_max)
        weights[critical_mask] *= self.critical_range_weight
        return weights
        
    def forward(self, pred, target):
        """
        Compute weighted L1 loss.
        
        Args:
            pred (torch.Tensor): Predicted DSM values
            target (torch.Tensor): Ground truth DSM values
            
        Returns:
            torch.Tensor: Weighted L1 loss
        """
        weights = self.get_weights(target)
        pixel_losses = torch.abs(pred - target) * weights
        return pixel_losses.mean()