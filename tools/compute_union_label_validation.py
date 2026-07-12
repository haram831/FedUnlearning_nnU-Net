#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk


def read_segmentation(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def compute_binary_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    pred_fg = pred.astype(bool)
    ref_fg = ref.astype(bool)

    tp = int(np.logical_and(pred_fg, ref_fg).sum())
    fp = int(np.logical_and(pred_fg, ~ref_fg).sum())
    fn = int(np.logical_and(~pred_fg, ref_fg).sum())
    tn = int(np.logical_and(~pred_fg, ~ref_fg).sum())
    n_pred = tp + fp
    n_ref = tp + fn

    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn

    return {
        "Dice": 1.0 if dice_denominator == 0 else 2 * tp / dice_denominator,
        "FN": fn,
        "FP": fp,
        "IoU": 1.0 if iou_denominator == 0 else tp / iou_denominator,
        "TN": tn,
        "TP": tp,
        "n_pred": n_pred,
        "n_ref": n_ref,
    }


def mean_metrics(metrics_per_case: List[Dict[str, float]]) -> Dict[str, float]:
    keys = metrics_per_case[0].keys()
    return {
        key: float(np.mean([case_metrics[key] for case_metrics in metrics_per_case]))
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate existing segmentations after merging multiple labels into one foreground label."
    )
    parser.add_argument(
        "--prediction_dir",
        type=Path,
        required=True,
        help="Directory containing validation predictions.",
    )
    parser.add_argument(
        "--reference_dir",
        type=Path,
        required=True,
        help="Directory containing reference segmentations.",
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        required=True,
        help="Labels to merge into the binary foreground.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON summary path.",
    )
    args = parser.parse_args()

    labels = set(args.labels)
    case_summaries = []
    metrics = []

    for prediction_file in sorted(args.prediction_dir.glob("*.nii.gz")):
        reference_file = args.reference_dir / prediction_file.name
        if not reference_file.is_file():
            raise FileNotFoundError(f"Missing reference for {prediction_file.name}: {reference_file}")

        pred = read_segmentation(prediction_file)
        ref = read_segmentation(reference_file)
        if pred.shape != ref.shape:
            raise ValueError(
                f"Shape mismatch for {prediction_file.name}: prediction {pred.shape}, reference {ref.shape}"
            )

        pred_binary = np.isin(pred, list(labels))
        ref_binary = np.isin(ref, list(labels))
        case_metrics = compute_binary_metrics(pred_binary, ref_binary)
        metrics.append(case_metrics)
        case_summaries.append(
            {
                "metrics": {"1": case_metrics},
                "prediction_file": str(prediction_file),
                "reference_file": str(reference_file),
            }
        )

    if not metrics:
        raise RuntimeError(f"No .nii.gz prediction files found in {args.prediction_dir}")

    foreground_mean = mean_metrics(metrics)
    summary = {
        "foreground_mean": foreground_mean,
        "mean": {"1": foreground_mean},
        "metric_per_case": case_summaries,
        "merged_labels": sorted(labels),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(summary, f, indent=4)
    print(f"Wrote {args.output}")
    print(f"Dice: {foreground_mean['Dice']:.6f}")
    print(f"IoU: {foreground_mean['IoU']:.6f}")


if __name__ == "__main__":
    main()
