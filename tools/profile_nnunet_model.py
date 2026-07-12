#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
NNUNET_ROOT = REPO_ROOT / "nnUNet"


def _prepend_repo_import_paths() -> None:
    for path in (REPO_ROOT, NNUNET_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_shape(value: str) -> tuple[int, ...]:
    parts = [
        item
        for item in value.lower().replace("x", ",").replace(" ", ",").split(",")
        if item
    ]
    if not parts:
        raise argparse.ArgumentTypeError("shape must contain at least one integer")
    try:
        shape = tuple(int(item) for item in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "shape must be comma-separated or x-separated integers"
        ) from e
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("all shape dimensions must be positive")
    return shape


def _format_count(value: int) -> str:
    units = ("", "K", "M", "G", "T", "P")
    scaled = float(value)
    for unit in units:
        if abs(scaled) < 1000.0 or unit == units[-1]:
            return f"{scaled:.3f}{unit}"
        scaled /= 1000.0
    return str(value)


def _set_nnunet_env(args: argparse.Namespace) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path(tempfile.gettempdir()) / "matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    env_paths = {
        "nnUNet_preprocessed": args.preprocessed_dir,
        "nnUNet_results": args.results_dir,
        "nnUNet_raw": args.raw_dir,
    }
    for key, value in env_paths.items():
        if value is not None:
            os.environ[key] = str(Path(value).expanduser().resolve())

    for key in ("nnUNet_raw", "nnUNet_results"):
        if key not in os.environ:
            os.environ[key] = str(Path(tempfile.gettempdir()) / f"{key}_unused")


def _parse_model_folder_name(folder: Path) -> tuple[Optional[str], Optional[str]]:
    parts = folder.name.split("__")
    if len(parts) != 3:
        return None, None
    trainer_name, _plans_identifier, configuration = parts
    return trainer_name, configuration


def _resolve_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], str, str, str]:
    if args.model_folder is not None:
        model_folder = Path(args.model_folder).expanduser().resolve()
        plans_file = model_folder / "plans.json"
        dataset_json_file = model_folder / "dataset.json"
        if not plans_file.is_file():
            raise SystemExit(f"plans.json not found in model folder: {model_folder}")
        if not dataset_json_file.is_file():
            raise SystemExit(f"dataset.json not found in model folder: {model_folder}")

        parsed_trainer, parsed_configuration = _parse_model_folder_name(model_folder)
        trainer_name = args.trainer or parsed_trainer or "nnUNetTrainer"
        configuration = args.config or args.configuration or parsed_configuration
        if configuration is None:
            raise SystemExit(
                "Could not infer configuration from --model-folder. "
                "Pass it as positional configuration or with --config."
            )
        source = str(model_folder)
        return (
            _read_json(plans_file),
            _read_json(dataset_json_file),
            configuration,
            trainer_name,
            source,
        )

    if args.plans_file is not None:
        if args.dataset_json is None:
            raise SystemExit("--dataset-json is required when --plans-file is used")
        configuration = args.config or args.configuration
        if configuration is None:
            raise SystemExit(
                "configuration is required when --plans-file is used. "
                "Pass it as positional configuration or with --config."
            )
        trainer_name = args.trainer or "nnUNetTrainer"
        plans_file = Path(args.plans_file).expanduser().resolve()
        dataset_json_file = Path(args.dataset_json).expanduser().resolve()
        if not plans_file.is_file():
            raise SystemExit(f"plans file not found: {plans_file}")
        if not dataset_json_file.is_file():
            raise SystemExit(f"dataset json not found: {dataset_json_file}")
        source = str(plans_file)
        return (
            _read_json(plans_file),
            _read_json(dataset_json_file),
            configuration,
            trainer_name,
            source,
        )

    if args.dataset_name_or_id is None or args.configuration is None:
        raise SystemExit(
            "dataset_name_or_id and configuration are required unless "
            "--model-folder or --plans-file is used"
        )

    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    if nnUNet_preprocessed is None:
        raise SystemExit(
            "nnUNet_preprocessed is not set. Export it or pass --preprocessed-dir."
        )

    dataset_name = maybe_convert_to_dataset_name(args.dataset_name_or_id)
    preprocessed_base = Path(nnUNet_preprocessed) / dataset_name
    plans_file = preprocessed_base / f"{args.plans_identifier}.json"
    dataset_json_file = preprocessed_base / "dataset.json"
    if not plans_file.is_file():
        raise SystemExit(f"plans file not found: {plans_file}")
    if not dataset_json_file.is_file():
        raise SystemExit(f"dataset json not found: {dataset_json_file}")

    trainer_name = args.trainer or "nnUNetTrainer"
    source = str(preprocessed_base)
    return (
        _read_json(plans_file),
        _read_json(dataset_json_file),
        args.configuration,
        trainer_name,
        source,
    )


def _resolve_trainer_class(trainer_name: str) -> type:
    if "." in trainer_name:
        module_name, class_name = trainer_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    import nnunetv2
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    trainer_class = recursive_find_python_class(
        str(Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )
    if trainer_class is None:
        raise SystemExit(
            f"Could not find trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer"
        )
    return trainer_class


def _resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def _build_model(
    plans: dict[str, Any],
    dataset_json: dict[str, Any],
    configuration: str,
    trainer_name: str,
    enable_deep_supervision: bool,
):
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_input_channels = determine_num_input_channels(
        plans_manager,
        configuration_manager,
        dataset_json,
    )
    trainer_class = _resolve_trainer_class(trainer_name)
    model = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        label_manager.num_segmentation_heads,
        enable_deep_supervision=enable_deep_supervision,
    )
    metadata = {
        "architecture": configuration_manager.network_arch_class_name,
        "patch_size": tuple(int(i) for i in configuration_manager.patch_size),
        "num_input_channels": int(num_input_channels),
        "num_output_channels": int(label_manager.num_segmentation_heads),
    }
    return model, metadata


def _profile(args: argparse.Namespace) -> None:
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
    except ImportError as e:
        raise SystemExit(
            "fvcore is required for FLOP counting. Install it with: pip install fvcore"
        ) from e

    import torch

    _prepend_repo_import_paths()
    _set_nnunet_env(args)

    plans, dataset_json, configuration, trainer_name, source = _resolve_inputs(args)
    model, metadata = _build_model(
        plans,
        dataset_json,
        configuration,
        trainer_name,
        enable_deep_supervision=args.deep_supervision,
    )

    if args.input_shape is None:
        input_shape = (
            1,
            metadata["num_input_channels"],
            *metadata["patch_size"],
        )
    else:
        input_shape = args.input_shape
        expected_dims = 2 + len(metadata["patch_size"])
        if len(input_shape) != expected_dims:
            raise SystemExit(
                f"--input-shape must have {expected_dims} dimensions for this configuration "
                f"(N,C,{','.join(['spatial'] * len(metadata['patch_size']))})"
            )

    device = _resolve_device(args.device)
    if device.type == "cuda":
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    model = model.to(device)
    model.eval()
    sample = torch.randn(input_shape, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        flops = FlopCountAnalysis(model, sample)
        total_flops = int(flops.total())

    print("Model source:", source)
    print("Trainer:", trainer_name)
    print("Configuration:", configuration)
    print("Architecture:", metadata["architecture"])
    print("Input shape:", tuple(input_shape))
    print("Deep supervision:", args.deep_supervision)
    print(f"FLOPs: {total_flops / 1e9:.3f} GFLOPs")
    print(f"FLOPs raw: {total_flops} ({_format_count(total_flops)})")
    print(f"Parameters: {total_params} ({_format_count(total_params)})")
    print(f"Trainable parameters: {trainable_params} ({_format_count(trainable_params)})")
    print()
    print(parameter_count_table(model, max_depth=args.param_table_depth))

    unsupported_ops = flops.unsupported_ops()
    if unsupported_ops:
        print()
        print("Unsupported ops not included in FLOPs:")
        for op_name, count in sorted(unsupported_ops.items()):
            print(f"  {op_name}: {count}")

    if args.by_operator:
        print()
        print("FLOPs by operator:")
        for op_name, count in sorted(flops.by_operator().items()):
            print(f"  {op_name}: {count} ({_format_count(int(count))})")

    if args.by_module:
        print()
        print("FLOPs by module:")
        for module_name, count in sorted(flops.by_module().items()):
            label = module_name or "<root>"
            print(f"  {label}: {count} ({_format_count(int(count))})")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile FLOPs and parameters for an nnU-Net model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_name_or_id",
        nargs="?",
        help="Dataset id/name, for example 301 or Dataset301_Decathlon.",
    )
    parser.add_argument(
        "configuration",
        nargs="?",
        help="nnU-Net configuration, for example 2d or 3d_fullres.",
    )
    parser.add_argument(
        "-tr",
        "--trainer",
        default=None,
        help=(
            "Trainer class name, or dotted import path. If omitted, model-folder "
            "mode infers it from the folder name; dataset mode uses nnUNetTrainer."
        ),
    )
    parser.add_argument(
        "-p",
        "--plans-identifier",
        default="nnUNetPlans",
        help="Plans identifier used in dataset mode.",
    )
    parser.add_argument(
        "--model-folder",
        help=(
            "Path to a trained model folder containing plans.json and dataset.json, "
            "for example $nnUNet_results/DatasetXXX/nnUNetTrainer__nnUNetPlans__3d_fullres."
        ),
    )
    parser.add_argument("--plans-file", help="Path to a plans JSON file.")
    parser.add_argument("--dataset-json", help="Path to a dataset.json file.")
    parser.add_argument(
        "--config",
        help="Configuration name when using --model-folder or --plans-file.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        help="Sets nnUNet_preprocessed before importing nnU-Net.",
    )
    parser.add_argument("--results-dir", help="Sets nnUNet_results before importing nnU-Net.")
    parser.add_argument("--raw-dir", help="Sets nnUNet_raw before importing nnU-Net.")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--input-shape",
        type=_parse_shape,
        default=None,
        help=(
            "Full input tensor shape N,C,... . If omitted, uses batch size 1, "
            "detected input channels, and the patch_size from plans."
        ),
    )
    parser.add_argument(
        "--deep-supervision",
        action="store_true",
        help=(
            "Measure the training forward with deep supervision enabled. "
            "The default matches nnU-Net inference, where it is disabled."
        ),
    )
    parser.add_argument(
        "--param-table-depth",
        type=int,
        default=3,
        help="Depth passed to fvcore.parameter_count_table.",
    )
    parser.add_argument(
        "--by-operator",
        action="store_true",
        help="Print FLOPs grouped by operator.",
    )
    parser.add_argument(
        "--by-module",
        action="store_true",
        help="Print FLOPs grouped by module.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _profile(args)


if __name__ == "__main__":
    main()
