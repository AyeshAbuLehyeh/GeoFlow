"""Reusable training and evaluation helpers for KITTI GeoFlow experiments."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .losses import AngularDirectionLoss, OrientationLoss
except ImportError:  # pragma: no cover - standalone execution fallback
    from losses import AngularDirectionLoss, OrientationLoss


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def build_parameter_groups(model: torch.nn.Module):
    backbone_params = []
    other_params = []
    for name, param in unwrap_model(model).named_parameters():
        if any(key in name.lower() for key in ["extractor", "backbone", "efficientnet"]):
            backbone_params.append(param)
        else:
            other_params.append(param)
    return backbone_params, other_params


def create_optimizer(model: torch.nn.Module, lr: float, backbone_lr: float, weight_decay: float):
    backbone_params, other_params = build_parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


def move_batch_to_device(batch, device):
    sat_map, camera_k, grd_img, x_shift, y_shift, theta, file_name = batch
    return (
        sat_map.to(device, non_blocking=True),
        camera_k.to(device, non_blocking=True),
        grd_img.to(device, non_blocking=True),
        x_shift.to(device, non_blocking=True),
        y_shift.to(device, non_blocking=True),
        theta.to(device, non_blocking=True),
        file_name,
    )


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None, epoch: Optional[int] = None, best_metric: Optional[float] = None, args: Optional[Any] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {"model": unwrap_model(model).state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if best_metric is not None:
        payload["best_metric"] = best_metric
    if args is not None:
        if is_dataclass(args):
            payload["args"] = asdict(args)
        elif hasattr(args, "__dict__"):
            payload["args"] = dict(vars(args))
        else:
            payload["args"] = args
    torch.save(payload, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    unwrap_model(model).load_state_dict(state_dict, strict=True)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and isinstance(checkpoint, dict) and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def train_one_epoch(model: torch.nn.Module, dataloader, optimizer, device, lambda_r: float, max_grad_norm: float, direction_loss_fn: Optional[AngularDirectionLoss] = None, orientation_loss_fn: Optional[OrientationLoss] = None, rotation_range: float = 10.0):
    model.train()
    direction_loss_fn = direction_loss_fn or AngularDirectionLoss()
    loss_r_fn = torch.nn.GaussianNLLLoss()

    total_losses = 0.0
    theta_losses = 0.0
    r_losses = 0.0
    orient_losses = 0.0

    for sat_map, camera_k, grd_img, x_shift, y_shift, theta, file_name in dataloader:
        sat_img = sat_map.to(device, non_blocking=True)
        grd_img = grd_img.to(device, non_blocking=True)
        gt_correction = torch.cat((x_shift, y_shift), dim=1).to(device, non_blocking=True)
        init_coord = torch.rand_like(gt_correction, device=device) * 2 - 1

        if getattr(model, "predict_orientation", False):
            pred_mu_r, pred_var, pred_mu_vec, pred_kappa, pred_orientation = model(sat_img, grd_img, coord=init_coord)
        else:
            pred_mu_r, pred_var, pred_mu_vec, pred_kappa = model(sat_img, grd_img, coord=init_coord)

        gt_flow = gt_correction - init_coord
        gt_r = torch.norm(gt_flow, p=2, dim=1)
        gt_direction = F.normalize(gt_flow, p=2, dim=1)

        loss_theta = direction_loss_fn(pred_mu_vec, pred_kappa, gt_direction)
        gt_log_r = torch.log(gt_r + 1e-6)
        loss_r = loss_r_fn(input=pred_mu_r.squeeze(-1), target=gt_log_r, var=pred_var.squeeze(-1))
        total_loss = loss_theta + (lambda_r * loss_r)

        if getattr(model, "predict_orientation", False):
            if orientation_loss_fn is None:
                orientation_loss_fn = OrientationLoss()
            loss_orientation = orientation_loss_fn(pred_orientation, theta, rotation_range=rotation_range)
            total_loss = total_loss + loss_orientation
            orient_losses += loss_orientation.item()

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_losses += total_loss.item()
        theta_losses += loss_theta.item()
        r_losses += loss_r.item()

    num_batches = max(1, len(dataloader))
    return {
        "total": total_losses / num_batches,
        "theta": theta_losses / num_batches,
        "r": r_losses / num_batches,
        "orientation": orient_losses / num_batches if getattr(model, "predict_orientation", False) else 0.0,
    }


@torch.inference_mode()
def evaluate_localization(model: torch.nn.Module, loader, device: str, num_iterations: int = 5, num_random_starts: int = 10, step_fraction: float = 0.5):
    model.eval()

    dataset = loader.dataset
    shift_range_pixels_lat = dataset.shift_range_pixels_lat
    shift_range_pixels_lon = dataset.shift_range_pixels_lon
    meter_per_pixel = dataset.meter_per_pixel
    rotation_range_rad = np.deg2rad(dataset.rotation_range)

    all_errors_total_m = []
    all_errors_lateral_m = []
    all_errors_longitudinal_m = []

    center_x, center_y = 512 / 2, 512 / 2

    for batch in loader:
        sat_map, camera_k, grd_img, x_shift, y_shift, theta, file_name = batch
        sat_img = sat_map.to(device, non_blocking=True)
        grd_img = grd_img.to(device, non_blocking=True)
        gt_location_norm = torch.cat((x_shift, y_shift), dim=1).to(device, non_blocking=True)

        sat_batch = sat_img.repeat_interleave(num_random_starts, dim=0)
        grd_batch = grd_img.repeat_interleave(num_random_starts, dim=0)
        current_coords = torch.rand(sat_batch.shape[0], 2, device=device) * 2 - 1

        for _ in range(num_iterations):
            pred_mu_r, _, pred_mu_vec, _ = unwrap_model(model)(sat_batch, grd_batch, coord=current_coords)
            pred_direction = F.normalize(pred_mu_vec, p=2, dim=1)
            pred_distance = torch.exp(pred_mu_r)
            current_coords = current_coords + pred_direction * pred_distance * step_fraction

        final_candidates = current_coords.view(sat_img.shape[0], num_random_starts, 2)
        final_prediction_norm = torch.mean(final_candidates, dim=1)

        pred_pix_x = center_x + (final_prediction_norm[:, 0] * shift_range_pixels_lon)
        pred_pix_y = center_y + (final_prediction_norm[:, 1] * shift_range_pixels_lat)
        gt_pix_x = center_x + (gt_location_norm[:, 0] * shift_range_pixels_lon)
        gt_pix_y = center_y + (gt_location_norm[:, 1] * shift_range_pixels_lat)

        error_pix_x = pred_pix_x - gt_pix_x
        error_pix_y = pred_pix_y - gt_pix_y

        gt_yaw_rad_tensor = (theta.to(device, non_blocking=True) * rotation_range_rad).squeeze(-1)
        cos_yaw = torch.cos(-gt_yaw_rad_tensor)
        sin_yaw = torch.sin(-gt_yaw_rad_tensor)
        error_longitudinal_pix = error_pix_x * cos_yaw - error_pix_y * sin_yaw
        error_lateral_pix = error_pix_x * sin_yaw + error_pix_y * cos_yaw

        final_error_total_m = torch.sqrt(error_pix_x ** 2 + error_pix_y ** 2) * meter_per_pixel
        final_error_longitudinal_m = torch.abs(error_longitudinal_pix) * meter_per_pixel
        final_error_lateral_m = torch.abs(error_lateral_pix) * meter_per_pixel

        all_errors_total_m.extend(final_error_total_m.cpu().numpy())
        all_errors_longitudinal_m.extend(final_error_longitudinal_m.cpu().numpy())
        all_errors_lateral_m.extend(final_error_lateral_m.cpu().numpy())

    all_errors_total_m = np.asarray(all_errors_total_m)
    all_errors_lateral_m = np.asarray(all_errors_lateral_m)
    all_errors_longitudinal_m = np.asarray(all_errors_longitudinal_m)

    return {
        "mean_error": float(np.mean(all_errors_total_m)),
        "median_error": float(np.median(all_errors_total_m)),
        "r1": float(np.mean(all_errors_total_m <= 1.0) * 100),
        "r5": float(np.mean(all_errors_total_m <= 5.0) * 100),
        "lat_r1": float(np.mean(all_errors_lateral_m <= 1.0) * 100),
        "lat_r5": float(np.mean(all_errors_lateral_m <= 5.0) * 100),
        "lon_r1": float(np.mean(all_errors_longitudinal_m <= 1.0) * 100),
        "lon_r5": float(np.mean(all_errors_longitudinal_m <= 5.0) * 100),
    }
