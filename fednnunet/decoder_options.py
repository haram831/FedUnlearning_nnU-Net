from __future__ import annotations

import copy
import os
import re
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Iterable, List, Optional

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    maybe_mkdir_p,
    save_json,
)


DECODER_ARCH_NNUNET = "nnunet"
DECODER_ARCH_EFFIDEC3D_UXNET = "effidec3d_uxnet"
UNLEARN_DECODER_SWITCH_SAME = "same"
EFFIDEC3D_UXNET_TRAINER = "nnUNetTrainerEffiDec3DUXNet"
EFFIDEC3D_UXNET_CLASS = (
    "nnunetv2.network_architecture.effidec3d.uxnet.EffiDec3DUXNet"
)


def add_decoder_arguments(parser: ArgumentParser, *, include_unlearn_switch: bool = True) -> None:
    parser.add_argument(
        "--decoder_arch",
        choices=(DECODER_ARCH_NNUNET, DECODER_ARCH_EFFIDEC3D_UXNET),
        default=DECODER_ARCH_NNUNET,
        help="Model/decoder architecture to use. Default: nnunet.",
    )
    parser.add_argument(
        "--effidec_channels",
        type=int,
        nargs=4,
        default=[48, 96, 192, 384],
        metavar=("C1", "C2", "C3", "C4"),
        help="3DUXNET_EffiDec3D encoder channels. Default: 48 96 192 384.",
    )
    parser.add_argument(
        "--effidec_n_decoder_channels",
        type=int,
        default=48,
        help="3DUXNET_EffiDec3D decoder channel width. Default: 48.",
    )
    parser.add_argument(
        "--effidec_resolution_factor",
        type=int,
        choices=(1, 2, 4, 8, 16),
        default=2,
        help="EffiDec3D resolution factor. Default: 2.",
    )
    parser.add_argument(
        "--effidec_skip_aggregation",
        choices=("addition", "concatenation"),
        default="addition",
        help="EffiDec3D skip aggregation mode. Default: addition.",
    )
    if include_unlearn_switch:
        parser.add_argument(
            "--unlearn_decoder_switch",
            choices=(UNLEARN_DECODER_SWITCH_SAME, DECODER_ARCH_EFFIDEC3D_UXNET),
            default=UNLEARN_DECODER_SWITCH_SAME,
            help=(
                "Experimental unlearn-only decoder switch. Use effidec3d_uxnet to "
                "force Level 2 compatible transfer into an EffiDec3D model."
            ),
        )


def decoder_options_to_cli_args(args: Namespace, *, include_unlearn_switch: bool = True) -> str:
    values = [
        f"--decoder_arch {getattr(args, 'decoder_arch', DECODER_ARCH_NNUNET)}",
        "--effidec_channels "
        + " ".join(str(i) for i in getattr(args, "effidec_channels", [48, 96, 192, 384])),
        f"--effidec_n_decoder_channels {getattr(args, 'effidec_n_decoder_channels', 48)}",
        f"--effidec_resolution_factor {getattr(args, 'effidec_resolution_factor', 2)}",
        f"--effidec_skip_aggregation {getattr(args, 'effidec_skip_aggregation', 'addition')}",
    ]
    if include_unlearn_switch:
        values.append(
            f"--unlearn_decoder_switch "
            f"{getattr(args, 'unlearn_decoder_switch', UNLEARN_DECODER_SWITCH_SAME)}"
        )
    return " ".join(values)


def _sanitize_identifier_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_") or "value"


def build_effidec3d_plans_identifier(base_identifier: str, args: Namespace) -> str:
    channels = "x".join(str(i) for i in get_effidec3d_channels(args))
    skip = _sanitize_identifier_part(getattr(args, "effidec_skip_aggregation", "addition"))
    return (
        f"{base_identifier}_effidec3d_uxnet"
        f"_c{channels}"
        f"_dc{int(getattr(args, 'effidec_n_decoder_channels', 48))}"
        f"_rf{int(getattr(args, 'effidec_resolution_factor', 2))}"
        f"_{skip}"
    )


def is_effidec3d_plans_identifier(plans_identifier: Optional[str]) -> bool:
    return bool(plans_identifier and "_effidec3d_uxnet_" in plans_identifier)


def get_effidec3d_channels(args: Namespace) -> List[int]:
    channels = list(getattr(args, "effidec_channels", [48, 96, 192, 384]))
    if len(channels) != 4:
        raise ValueError("--effidec_channels must contain exactly four integers")
    return [int(i) for i in channels]


def get_effidec3d_arch_kwargs(args: Namespace) -> Dict[str, Any]:
    return {
        "depths": [2, 2, 2, 2],
        "feat_size": get_effidec3d_channels(args),
        "n_decoder_channels": int(getattr(args, "effidec_n_decoder_channels", 48)),
        "resolution_factor": int(getattr(args, "effidec_resolution_factor", 2)),
        "skip_aggregation": getattr(args, "effidec_skip_aggregation", "addition"),
        "spatial_dims": 3,
        "upsample_logits": True,
    }


def should_use_effidec3d_for_training(args: Namespace) -> bool:
    return getattr(args, "decoder_arch", DECODER_ARCH_NNUNET) == DECODER_ARCH_EFFIDEC3D_UXNET


def should_switch_unlearn_to_effidec3d(args: Namespace) -> bool:
    return (
        getattr(args, "decoder_arch", DECODER_ARCH_NNUNET) == DECODER_ARCH_NNUNET
        and getattr(args, "unlearn_decoder_switch", UNLEARN_DECODER_SWITCH_SAME)
        == DECODER_ARCH_EFFIDEC3D_UXNET
    )


def normalize_training_decoder_args(args: Namespace) -> None:
    if not should_use_effidec3d_for_training(args):
        return
    if hasattr(args, "p"):
        args.base_plans_identifier = args.p
        if not is_effidec3d_plans_identifier(args.p):
            args.p = build_effidec3d_plans_identifier(args.p, args)
    if hasattr(args, "tr"):
        args.base_trainer = args.tr
        args.tr = EFFIDEC3D_UXNET_TRAINER


def _copy_metadata_arch_fields(source_arch_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    metadata_keys = (
        "n_stages",
        "features_per_stage",
        "kernel_sizes",
        "strides",
        "n_conv_per_stage",
        "n_blocks_per_stage",
        "n_conv_per_stage_decoder",
    )
    return {
        key: copy.deepcopy(source_arch_kwargs[key])
        for key in metadata_keys
        if key in source_arch_kwargs
    }


def apply_effidec3d_to_plans(
    plans: Dict[str, Any],
    args: Namespace,
    *,
    plans_identifier: Optional[str] = None,
    configurations: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    updated = copy.deepcopy(plans)
    if plans_identifier is not None:
        updated["plans_name"] = plans_identifier

    selected_configurations = set(configurations) if configurations is not None else None
    for config_name, configuration in updated.get("configurations", {}).items():
        if selected_configurations is not None and config_name not in selected_configurations:
            continue
        patch_size = configuration.get("patch_size", [])
        if len(patch_size) != 3:
            continue

        source_arch_kwargs = configuration.get("architecture", {}).get("arch_kwargs", {})
        arch_kwargs = get_effidec3d_arch_kwargs(args)
        arch_kwargs.update(_copy_metadata_arch_fields(source_arch_kwargs))
        configuration["architecture"] = {
            "network_class_name": EFFIDEC3D_UXNET_CLASS,
            "arch_kwargs": arch_kwargs,
            "_kw_requires_import": [],
        }

    return updated


def get_plan_path(dataset_name_or_id: Any, plans_identifier: str) -> str:
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    return join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")


def ensure_effidec3d_plans_for_dataset(
    dataset_name_or_id: Any,
    base_plans_identifier: str,
    args: Namespace,
    *,
    output_plans_identifier: Optional[str] = None,
) -> str:
    output_plans_identifier = output_plans_identifier or build_effidec3d_plans_identifier(
        base_plans_identifier,
        args,
    )
    output_path = get_plan_path(dataset_name_or_id, output_plans_identifier)
    if is_effidec3d_plans_identifier(base_plans_identifier) and output_plans_identifier == base_plans_identifier:
        return output_plans_identifier

    base_path = get_plan_path(dataset_name_or_id, base_plans_identifier)
    base_plan = load_json(base_path)
    effidec_plan = apply_effidec3d_to_plans(
        base_plan,
        args,
        plans_identifier=output_plans_identifier,
    )
    maybe_mkdir_p(os.path.dirname(output_path))
    save_json(effidec_plan, output_path, sort_keys=False)
    return output_plans_identifier


def effective_training_plans_identifier(
    base_plans_identifier: str,
    args: Namespace,
) -> str:
    if should_use_effidec3d_for_training(args):
        if is_effidec3d_plans_identifier(base_plans_identifier):
            return base_plans_identifier
        return build_effidec3d_plans_identifier(base_plans_identifier, args)
    return base_plans_identifier


def effective_training_trainer(base_trainer: str, args: Namespace) -> str:
    if should_use_effidec3d_for_training(args):
        return EFFIDEC3D_UXNET_TRAINER
    return base_trainer


def decoder_metadata(args: Namespace) -> Dict[str, Any]:
    return {
        "decoder_arch": getattr(args, "decoder_arch", DECODER_ARCH_NNUNET),
        "unlearn_decoder_switch": getattr(
            args,
            "unlearn_decoder_switch",
            UNLEARN_DECODER_SWITCH_SAME,
        ),
        "effidec3d_uxnet": get_effidec3d_arch_kwargs(args),
    }
