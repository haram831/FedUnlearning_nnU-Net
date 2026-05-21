import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def make_case(rng: np.random.Generator, case_idx: int, center_shift: int) -> tuple[np.ndarray, np.ndarray]:
    shape = (24, 24, 24)
    z, y, x = np.indices(shape)
    center = np.array(shape) // 2 + np.array([center_shift, 0, -center_shift])
    radius = 5 + (case_idx % 2)
    lesion = (
        (z - center[0]) ** 2
        + (y - center[1]) ** 2
        + (x - center[2]) ** 2
    ) <= radius**2

    image = rng.normal(loc=0.0, scale=0.05, size=shape).astype(np.float32)
    image += lesion.astype(np.float32) * 1.0
    image += (z / shape[0]).astype(np.float32) * 0.15

    label = lesion.astype(np.uint8)
    return image, label


def write_dataset(root: Path, dataset_id: int, cases: int) -> None:
    dataset_name = f"Dataset{dataset_id:03d}_SyntheticFederated"
    dataset_dir = root / "nnUNet_raw" / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(dataset_id)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    center_shift = dataset_id - 302

    for case_idx in range(cases):
        image, label = make_case(rng, case_idx, center_shift)
        case_name = f"synthetic_{dataset_id}_{case_idx:03d}"
        nib.save(
            nib.Nifti1Image(image, affine),
            images_tr / f"{case_name}_0000.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(label, affine),
            labels_tr / f"{case_name}.nii.gz",
        )

    dataset_json = {
        "channel_names": {"0": "synthetic_mri"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": cases,
        "file_ending": ".nii.gz",
    }
    with (dataset_dir / "dataset.json").open("w") as f:
        json.dump(dataset_json, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("testdata"))
    parser.add_argument("--dataset-ids", type=int, nargs="+", default=[301, 302, 303])
    parser.add_argument("--cases", type=int, default=5)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "nnUNet_preprocessed").mkdir(exist_ok=True)
    (args.root / "nnUNet_results").mkdir(exist_ok=True)

    for dataset_id in args.dataset_ids:
        write_dataset(args.root, dataset_id, args.cases)


if __name__ == "__main__":
    main()
