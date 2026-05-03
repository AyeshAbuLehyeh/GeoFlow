from __future__ import annotations

import argparse
import json
import os
import time

import torch

try:
    from .data import load_test1_data, load_test2_data
    from .engine import evaluate_localization, load_checkpoint, write_json
    from .models import GeoFlowKITTI
except ImportError:  # pragma: no cover - standalone execution fallback
    from data import load_test1_data, load_test2_data
    from engine import evaluate_localization, load_checkpoint, write_json
    from models import GeoFlowKITTI


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GeoFlow on KITTI")
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--test-set", default="test2", choices=["test1", "test2"])
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-iterations", default=5, type=int)
    parser.add_argument("--num-random-starts", default=10, type=int)
    parser.add_argument("--backbone", default="efficientnet_b0", type=str)
    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--sat-size", default=512, type=int)
    parser.add_argument("--output-dir", default="inference_outputs", type=str)
    return parser.parse_args()


def _format_results(results: dict) -> list[str]:
    return [
        "----- FINAL AGGREGATE RESULTS -----",
        f"Metric       | Mean       | Median     | R@1m    | R@5m",
        f"Localization  | {results['mean_error']:<10.3f} | {results['median_error']:<10.3f} | {results['r1']:<7.2f} | {results['r5']:<7.2f}",
        f"Lateral (%)   | N/A        | N/A        | {results['lat_r1']:<7.2f} | {results['lat_r5']:<7.2f}",
        f"Longitudinal (%) | N/A     | N/A        | {results['lon_r1']:<7.2f} | {results['lon_r5']:<7.2f}",
    ]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.test_set == "test1":
        loader = load_test1_data(batch_size=args.batch_size)
    else:
        loader = load_test2_data(batch_size=args.batch_size)

    model = GeoFlowKITTI(d_model=args.d_model, backbone=args.backbone, sat_size=args.sat_size).to(device)
    load_checkpoint(args.model_path, model, device=device)
    model.eval()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    results = evaluate_localization(
        model=model,
        loader=loader,
        device=device,
        num_iterations=args.num_iterations,
        num_random_starts=args.num_random_starts,
    )

    report_path = os.path.join(args.output_dir, f"inference_results_{timestamp}.txt")
    json_path = os.path.join(args.output_dir, f"inference_results_{timestamp}.json")

    final_lines = _format_results(results)
    with open(report_path, "w", encoding="utf-8") as log_file:
        for line in final_lines:
            print(line)
            log_file.write(line + "\n")

    write_json(json_path, {"args": vars(args), "results": results})
    print(f"Saved evaluation report to {report_path}")
    print(f"Saved evaluation JSON to {json_path}")


if __name__ == "__main__":
    main()
