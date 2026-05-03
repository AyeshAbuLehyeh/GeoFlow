from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .data import load_test1_data, load_test2_data
    from .engine import load_checkpoint, write_json
    from .losses import OrientationLoss
    from .models import GeoFlowKITTIOrientation
except ImportError:  # pragma: no cover - standalone execution fallback
    from data import load_test1_data, load_test2_data
    from engine import load_checkpoint, write_json
    from losses import OrientationLoss
    from models import GeoFlowKITTIOrientation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GeoFlow on KITTI with orientation prediction")
    parser.add_argument("--checkpoint-path", required=True, type=str)
    parser.add_argument("--backbone", default="efficientnet_b0", type=str)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--sat-size", default=512, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--rotation-range", default=10.0, type=float)
    parser.add_argument("--shift-range-lat", default=20.0, type=float)
    parser.add_argument("--shift-range-lon", default=20.0, type=float)
    parser.add_argument("--output-dir", default="inference_outputs", type=str)
    return parser.parse_args()


def evaluate_with_orientation(model, test_loader, device, rotation_range=10.0, shift_range_lat=20.0, shift_range_lon=20.0):
    model.eval()

    lateral_errors = []
    longitudinal_errors = []
    distance_errors = []
    orientation_errors = []
    orientation_loss_fn = OrientationLoss()

    shift_range_lat_t = torch.tensor(shift_range_lat, device=device)
    shift_range_lon_t = torch.tensor(shift_range_lon, device=device)
    rotation_range_t = torch.tensor(rotation_range, device=device)

    with torch.inference_mode():
        for sat_map, camera_k, grd_img, x_shift, y_shift, theta_normalized, file_name in test_loader:
            sat_img = sat_map.to(device, non_blocking=True)
            grd_img = grd_img.to(device, non_blocking=True)
            theta_normalized = theta_normalized.to(device, non_blocking=True)

            theta_true = theta_normalized * rotation_range_t
            gt_correction_normalized = torch.cat((x_shift, y_shift), dim=1).to(device, non_blocking=True)
            init_coord = torch.zeros_like(gt_correction_normalized, device=device)

            pred_mu_r, pred_var, pred_mu_vec, pred_kappa, pred_orientation = model(sat_img, grd_img, coord=init_coord)

            pred_r = torch.exp(pred_mu_r)
            pred_direction = F.normalize(pred_mu_vec, p=2, dim=1)
            pred_correction_normalized = pred_r * pred_direction

            gt_correction_meters = torch.stack(
                [gt_correction_normalized[:, 0] * shift_range_lat_t, gt_correction_normalized[:, 1] * shift_range_lon_t],
                dim=1,
            )
            pred_correction_meters = torch.stack(
                [pred_correction_normalized[:, 0] * shift_range_lat_t, pred_correction_normalized[:, 1] * shift_range_lon_t],
                dim=1,
            )

            error_vector_global_meters = pred_correction_meters - gt_correction_meters
            distance_error = torch.norm(error_vector_global_meters, p=2, dim=1)

            theta_rad = (theta_true * (torch.pi / 180.0)).reshape(-1)
            err_x = error_vector_global_meters[:, 0]
            err_y = error_vector_global_meters[:, 1]
            cos_n_theta = torch.cos(-theta_rad)
            sin_n_theta = torch.sin(-theta_rad)
            longitudinal_error_local = err_x * sin_n_theta + err_y * cos_n_theta
            lateral_error_local = err_x * cos_n_theta - err_y * sin_n_theta

            lateral_errors.extend(torch.abs(lateral_error_local).cpu().numpy().tolist())
            longitudinal_errors.extend(torch.abs(longitudinal_error_local).cpu().numpy().tolist())
            distance_errors.extend(distance_error.cpu().numpy().tolist())
            orientation_errors.extend(
                orientation_loss_fn.compute_angular_error(pred_orientation, theta_normalized, rotation_range).cpu().numpy().tolist()
            )

    lateral_errors_arr = np.asarray(lateral_errors)
    longitudinal_errors_arr = np.asarray(longitudinal_errors)
    distance_errors_arr = np.asarray(distance_errors)
    orientation_array = np.asarray(orientation_errors)

    return {
        "lateral_mean": float(np.mean(lateral_errors_arr)),
        "lateral_median": float(np.median(lateral_errors_arr)),
        "longitudinal_mean": float(np.mean(longitudinal_errors_arr)),
        "longitudinal_median": float(np.median(longitudinal_errors_arr)),
        "distance_mean": float(np.mean(distance_errors_arr)),
        "distance_median": float(np.median(distance_errors_arr)),
        "orientation_mean": float(np.mean(orientation_array)),
        "orientation_median": float(np.median(orientation_array)),
        "orientation_std": float(np.std(orientation_array)),
        "orientation_success_1deg": float(np.mean(orientation_array < 1.0) * 100),
        "orientation_success_2deg": float(np.mean(orientation_array < 2.0) * 100),
        "orientation_success_5deg": float(np.mean(orientation_array < 5.0) * 100),
    }


def _print_results(title: str, results: dict):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Distance Error:     {results['distance_median']:.3f} m (median)")
    print(f"Distance Error:     {results['distance_mean']:.3f} m (mean)")
    print(f"Lateral Error:      {results['lateral_median']:.3f} m (median)")
    print(f"Longitudinal Error: {results['longitudinal_median']:.3f} m (median)")
    print(f"Orientation Error:  {results['orientation_median']:.2f}° (median)")
    print(f"Orientation Mean:   {results['orientation_mean']:.2f}°")
    print(f"Orientation Std:    {results['orientation_std']:.2f}°")
    print(f"R@1 (<1°):          {results['orientation_success_1deg']:.1f}%")
    print(f"R@5 (<5°):          {results['orientation_success_5deg']:.1f}%")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test1_loader = load_test1_data(batch_size=args.batch_size, shift_range_lat=args.shift_range_lat, shift_range_lon=args.shift_range_lon, rotation_range=args.rotation_range)
    test2_loader = load_test2_data(batch_size=args.batch_size, shift_range_lat=args.shift_range_lat, shift_range_lon=args.shift_range_lon, rotation_range=args.rotation_range)

    model = GeoFlowKITTIOrientation(d_model=args.d_model, backbone=args.backbone, sat_size=args.sat_size).to(device)
    load_checkpoint(args.checkpoint_path, model, device=device)

    results_test1 = evaluate_with_orientation(model, test1_loader, device, args.rotation_range, args.shift_range_lat, args.shift_range_lon)
    results_test2 = evaluate_with_orientation(model, test2_loader, device, args.rotation_range, args.shift_range_lat, args.shift_range_lon)

    _print_results("EVALUATING ON TEST1 (Same Area)", results_test1)
    _print_results("EVALUATING ON TEST2 (Cross Area)", results_test2)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Metric':<34} {'Test1':<14} {'Test2':<14}")
    print("-" * 70)
    print(f"{'Distance Error (m) - Median':<34} {results_test1['distance_median']:<14.2f} {results_test2['distance_median']:<14.2f}")
    print(f"{'Orientation Error (°) - Median':<34} {results_test1['orientation_median']:<14.2f} {results_test2['orientation_median']:<14.2f}")
    print(f"{'R@1 (Orient < 1°) %':<34} {results_test1['orientation_success_1deg']:<14.1f} {results_test2['orientation_success_1deg']:<14.1f}")
    print(f"{'R@5 (Orient < 5°) %':<34} {results_test1['orientation_success_5deg']:<14.1f} {results_test2['orientation_success_5deg']:<14.1f}")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    report_path = os.path.join(args.output_dir, f"orientation_results_{timestamp}.json")
    write_json(report_path, {"args": vars(args), "test1": results_test1, "test2": results_test2})
    print(f"Saved report to {report_path}")


if __name__ == '__main__':
    main()