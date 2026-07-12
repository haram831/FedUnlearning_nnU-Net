#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def read_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def normalize_slice(image_slice: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image_slice, (1, 99))
    if high <= low:
        return np.zeros_like(image_slice, dtype=np.float32)
    clipped = np.clip(image_slice, low, high)
    return (clipped - low) / (high - low)


def pick_slices(mask: np.ndarray, count: int) -> List[int]:
    foreground_by_slice = mask.reshape(mask.shape[0], -1).sum(axis=1)
    foreground_slices = np.where(foreground_by_slice > 0)[0]
    if len(foreground_slices) == 0:
        return [mask.shape[0] // 2]
    if len(foreground_slices) <= count:
        return [int(i) for i in foreground_slices]
    positions = np.linspace(0, len(foreground_slices) - 1, count)
    return [int(foreground_slices[int(round(pos))]) for pos in positions]


def add_overlay(ax, base: np.ndarray, gt: np.ndarray, pred: np.ndarray, title: str) -> None:
    ax.imshow(normalize_slice(base), cmap="gray")

    gt_only = np.logical_and(gt, ~pred)
    pred_only = np.logical_and(pred, ~gt)
    overlap = np.logical_and(gt, pred)

    overlay = np.zeros((*gt.shape, 4), dtype=np.float32)
    overlay[gt_only] = [0.0, 0.85, 0.2, 0.55]
    overlay[pred_only] = [1.0, 0.15, 0.05, 0.55]
    overlay[overlap] = [1.0, 0.9, 0.0, 0.55]
    ax.imshow(overlay)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create GT/prediction overlay PNGs for validation segmentations.")
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--prediction_dir", type=Path, required=True)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--labels", type=int, nargs="+", default=[1])
    parser.add_argument("--slices", type=int, default=5)
    parser.add_argument("--case", type=str, default=None, help="Optional case id, for example prostate_00.")
    args = parser.parse_args()

    labels = list(args.labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = sorted(args.prediction_dir.glob("*.nii.gz"))
    if args.case is not None:
        prediction_files = [path for path in prediction_files if path.name == f"{args.case}.nii.gz"]
    if not prediction_files:
        raise RuntimeError("No prediction files matched the requested inputs.")

    for prediction_file in prediction_files:
        case_id = prediction_file.name.removesuffix(".nii.gz")
        image_file = args.image_dir / f"{case_id}_0000.nii.gz"
        reference_file = args.reference_dir / prediction_file.name
        if not image_file.is_file():
            raise FileNotFoundError(f"Missing image file: {image_file}")
        if not reference_file.is_file():
            raise FileNotFoundError(f"Missing reference file: {reference_file}")

        image = read_array(image_file)
        pred = np.isin(read_array(prediction_file), labels)
        ref = np.isin(read_array(reference_file), labels)
        combined = np.logical_or(pred, ref)
        selected_slices = pick_slices(combined, args.slices)

        fig, axes = plt.subplots(1, len(selected_slices), figsize=(3.2 * len(selected_slices), 3.6))
        axes = np.atleast_1d(axes)
        for ax, z in zip(axes, selected_slices):
            add_overlay(ax, image[z], ref[z], pred[z], f"{case_id} z={z}")

        fig.suptitle(
            "Yellow: overlap, Green: GT only, Red: prediction only",
            fontsize=11,
        )
        fig.tight_layout()
        output_file = args.output_dir / f"{case_id}_overlay_labels_{'_'.join(map(str, labels))}.png"
        fig.savefig(output_file, dpi=160)
        plt.close(fig)
        print(output_file)


if __name__ == "__main__":
    main()
