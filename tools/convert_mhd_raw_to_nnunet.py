import argparse
import json
import shutil
from pathlib import Path

import SimpleITK as sitk


def find_label_for_image(image_path: Path, label_suffix: str, label_dir: Path | None) -> Path:
    search_dir = label_dir if label_dir is not None else image_path.parent
    label_path = search_dir / f"{image_path.stem}{label_suffix}{image_path.suffix}"
    if not label_path.is_file():
        raise FileNotFoundError(
            f"Could not find label for {image_path.name}. Expected {label_path.name}"
        )
    return label_path


def ensure_metaimage_pair(mhd_path: Path) -> None:
    if mhd_path.suffix.lower() != ".mhd":
        return
    raw_path = mhd_path.with_suffix(".raw")
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"{mhd_path} exists but matching {raw_path.name} is missing"
        )


def write_nifti_from_mhd(input_path: Path, output_path: Path, is_label: bool) -> None:
    image = sitk.ReadImage(str(input_path))
    if is_label:
        image = sitk.Cast(image, sitk.sitkUInt8)
    sitk.WriteImage(image, str(output_path), useCompression=True)


def build_dataset_json(
    output_dir: Path,
    dataset_name: str,
    num_training: int,
    channel_name: str,
    label_name: str,
) -> None:
    dataset_json = {
        "name": dataset_name,
        "channel_names": {"0": channel_name},
        "labels": {"background": 0, label_name: 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    with (output_dir / "dataset.json").open("w") as f:
        json.dump(dataset_json, f, indent=2)


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    label_dir: Path | None,
    image_glob: str,
    label_suffix: str,
    case_prefix: str,
    channel_name: str,
    label_name: str,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    images_tr = output_dir / "imagesTr"
    labels_tr = output_dir / "labelsTr"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)

    all_mhd_files = sorted(input_dir.rglob(image_glob))
    image_files = [
        path
        for path in all_mhd_files
        if path.is_file() and not path.stem.endswith(label_suffix)
    ]
    if not image_files:
        raise RuntimeError(f"No image files found in {input_dir} with glob {image_glob}")

    for case_idx, image_path in enumerate(image_files):
        label_path = find_label_for_image(image_path, label_suffix, label_dir)
        ensure_metaimage_pair(image_path)
        ensure_metaimage_pair(label_path)

        case_id = f"{case_prefix}_{case_idx:03d}"
        write_nifti_from_mhd(image_path, images_tr / f"{case_id}_0000.nii.gz", False)
        write_nifti_from_mhd(label_path, labels_tr / f"{case_id}.nii.gz", True)
        print(f"Converted {image_path.name} -> {case_id}")

    build_dataset_json(
        output_dir=output_dir,
        dataset_name=output_dir.name,
        num_training=len(image_files),
        channel_name=channel_name,
        label_name=label_name,
    )
    print(f"Done. Wrote {len(image_files)} cases to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SimpleITK-readable image-label pairs to nnU-Net .nii.gz format."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing images, for example .mhd/.raw or .nrrd files.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=None,
        help="Optional folder containing labels. Defaults to --input-dir/file parent.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Target nnU-Net dataset folder, for example $nnUNet_raw/Dataset301_PROMISE12Client1.",
    )
    parser.add_argument(
        "--image-glob",
        default="*.mhd",
        help="Glob used to find input image files recursively, for example '*.mhd' or '*.nrrd'.",
    )
    parser.add_argument(
        "--label-suffix",
        default="_segmentation",
        help="Label suffix before .mhd. PROMISE12 default: Case00_segmentation.mhd.",
    )
    parser.add_argument(
        "--case-prefix",
        default="promise",
        help="Prefix for generated nnU-Net case ids.",
    )
    parser.add_argument(
        "--channel-name",
        default="MRI",
        help="Name of image channel 0 in dataset.json.",
    )
    parser.add_argument(
        "--label-name",
        default="prostate",
        help="Name of label value 1 in dataset.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory before conversion if it already exists.",
    )
    args = parser.parse_args()

    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        label_dir=args.label_dir,
        image_glob=args.image_glob,
        label_suffix=args.label_suffix,
        case_prefix=args.case_prefix,
        channel_name=args.channel_name,
        label_name=args.label_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
