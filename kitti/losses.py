"""Loss functions for KITTI training and evaluation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularDirectionLoss(nn.Module):
    """Angular negative log-likelihood used for direction regression."""

    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred_mu_vec: torch.Tensor, pred_kappa: torch.Tensor, gt_vec: torch.Tensor) -> torch.Tensor:
        pred_direction = F.normalize(pred_mu_vec, p=2, dim=1)
        gt_direction = F.normalize(gt_vec, p=2, dim=1)
        dot = torch.cosine_similarity(pred_direction, gt_direction, dim=1)
        dot = torch.clamp(dot, -1.0 + self.epsilon, 1.0 - self.epsilon)
        angular_error = torch.acos(dot)

        kappa = torch.clamp(pred_kappa.squeeze(-1), min=1e-6)
        loss = -torch.log(kappa.square() + 1e-6) + kappa * angular_error + torch.log1p(torch.exp(-kappa * math.pi))
        return loss.mean()


class OrientationLoss(nn.Module):
    """Cosine loss for the optional orientation branch."""

    def forward(self, pred_orientation: torch.Tensor, gt_theta: torch.Tensor, rotation_range: float = 10.0) -> torch.Tensor:
        gt_theta_rad = gt_theta.view(-1, 1) * rotation_range * math.pi / 180.0
        gt_orientation = torch.cat([torch.cos(gt_theta_rad), torch.sin(gt_theta_rad)], dim=1)
        cos_sim = F.cosine_similarity(pred_orientation, gt_orientation, dim=1)
        return (1.0 - cos_sim).mean()

    @staticmethod
    def compute_angular_error(pred_orientation: torch.Tensor, gt_theta: torch.Tensor, rotation_range: float = 10.0) -> torch.Tensor:
        pred_angle = torch.atan2(pred_orientation[:, 1], pred_orientation[:, 0])
        gt_theta_rad = gt_theta.squeeze(-1) * rotation_range * math.pi / 180.0
        diff = torch.abs(pred_angle - gt_theta_rad)
        diff = torch.minimum(diff, 2 * math.pi - diff)
        return diff * 180.0 / math.pi
