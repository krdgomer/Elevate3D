import torch
import torch.nn as nn
from torchvision import models


class ElevationLoss(nn.Module):
    def __init__(
        self,
        base_weight=1.0,
        critical_range_weight=2.0,
        critical_range=(144, 200),
        perceptual_weight=0.1,
    ):
        """
        Custom loss function combining weighted L1 loss with perceptual loss.

        Args:
            base_weight (float): Base weight for all elevation values.
            critical_range_weight (float): Weight multiplier for the critical range.
            critical_range (tuple): (min, max) values of the critical elevation range.
            perceptual_weight (float): Scaling factor for perceptual loss.
        """
        super().__init__()
        self.base_weight = base_weight
        self.critical_range_weight = critical_range_weight
        self.critical_min = critical_range[0]
        self.critical_max = critical_range[1]
        self.perceptual_weight = perceptual_weight

        # Load VGG model for perceptual loss
        vgg = models.vgg19(pretrained=True).features[:16].eval()  # Use first 16 layers
        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights
        self.vgg = vgg
        self.l1_loss = nn.L1Loss()  # For perceptual loss

    def get_weights(self, target):
        """
        Generate weights based on elevation values.
        Returns a tensor of same shape as target with weights.
        """
        weights = torch.ones_like(target) * self.base_weight
        critical_mask = (target >= self.critical_min) & (target <= self.critical_max)
        weights[critical_mask] *= self.critical_range_weight
        return weights

    def perceptual_loss(self, pred, target):
        """
        Compute perceptual loss using VGG feature maps.
        Args:
            pred (torch.Tensor): Predicted DSM values.
            target (torch.Tensor): Ground truth DSM values.
        Returns:
            torch.Tensor: Perceptual loss value.
        """
        pred_features = self.vgg(
            pred.repeat(1, 3, 1, 1)
        )  # Repeat channels to match VGG input
        target_features = self.vgg(target.repeat(1, 3, 1, 1))
        return self.l1_loss(pred_features, target_features)

    def forward(self, pred, target):
        """
        Compute combined weighted L1 loss and perceptual loss.

        Args:
            pred (torch.Tensor): Predicted DSM values.
            target (torch.Tensor): Ground truth DSM values.
        Returns:
            torch.Tensor: Total loss (Elevation Loss + Perceptual Loss).
        """
        weights = self.get_weights(target)
        pixel_losses = torch.abs(pred - target) * weights
        elevation_loss = pixel_losses.mean()

        # Compute perceptual loss
        perceptual_loss_value = self.perceptual_loss(pred, target)

        # Combine both losses
        total_loss = elevation_loss + self.perceptual_weight * perceptual_loss_value
        return total_loss
