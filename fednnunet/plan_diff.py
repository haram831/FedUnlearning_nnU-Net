from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional, Tuple

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    maybe_mkdir_p,
    save_json,
)


ARCHITECTURE_FIELDS = (
    ("architecture.network_class_name", ("architecture", "network_class_name")),
    ("architecture.arch_kwargs.n_stages", ("architecture", "arch_kwargs", "n_stages")),
    ("architecture.arch_kwargs.strides", ("architecture", "arch_kwargs", "strides")),
    ("architecture.arch_kwargs.kernel_sizes", ("architecture", "arch_kwargs", "kernel_sizes")),
    (
        "architecture.arch_kwargs.n_conv_per_stage",
        ("architecture", "arch_kwargs", "n_conv_per_stage"),
    ),
    (
        "architecture.arch_kwargs.n_conv_per_stage_decoder",
        ("architecture", "arch_kwargs", "n_conv_per_stage_decoder"),
    ),
    ("patch_size", ("patch_size",)),
)

PREPROCESSING_FIELDS = (
    ("spacing", ("spacing",)),
    ("median_image_size_in_voxels", ("median_image_size_in_voxels",)),
    ("batch_size", ("batch_size",)),
    ("preprocessor_name", ("preprocessor_name",)),
    ("normalization_schemes", ("normalization_schemes",)),
    ("use_mask_for_norm", ("use_mask_for_norm",)),
)


def get_plan_path(dataset_id: int, plans_identifier: str) -> str:
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

    dataset_name = convert_id_to_dataset_name(dataset_id)
    return join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")


def load_plan(dataset_id: int, plans_identifier: str) -> Dict[str, Any]:
    return load_json(get_plan_path(dataset_id, plans_identifier))


def _get_nested(data: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _changed_fields(
    original: Dict[str, Any],
    minus: Dict[str, Any],
    fields: Tuple[Tuple[str, Tuple[str, ...]], ...],
) -> Dict[str, Dict[str, Any]]:
    changes = {}
    for name, path in fields:
        original_value = _get_nested(original, path)
        minus_value = _get_nested(minus, path)
        if original_value != minus_value:
            changes[name] = {
                "original": original_value,
                "minus_target": minus_value,
            }
    return changes


def _top_level_preprocessing_changes(
    original: Dict[str, Any],
    minus: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    changes = {}
    key = "foreground_intensity_properties_per_channel"
    if original.get(key) != minus.get(key):
        changes[key] = {
            "original": original.get(key),
            "minus_target": minus.get(key),
        }
    return changes


def generate_plan_from_fingerprint(
    dataset_id: int,
    fingerprint: Dict[str, Any],
    plans_identifier: str,
    planner_class_name: str = "ExperimentPlanner",
    preprocessor_name: str = "DefaultPreprocessor",
    gpu_memory_target_in_gb: Optional[float] = None,
) -> Dict[str, Any]:
    import nnunetv2
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import (
        ExperimentPlanner,
    )
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    planner_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "experiment_planning"),
        planner_class_name,
        current_module="nnunetv2.experiment_planning",
    )
    if planner_class is None:
        raise RuntimeError(f"Could not find experiment planner class {planner_class_name}")

    kwargs = {}
    if gpu_memory_target_in_gb is not None:
        kwargs["gpu_memory_target_in_gb"] = float(gpu_memory_target_in_gb)

    planner: ExperimentPlanner = planner_class(
        dataset_id,
        preprocessor_name=preprocessor_name,
        plans_name=plans_identifier,
        **kwargs,
    )
    planner.dataset_fingerprint = fingerprint
    return planner.plan_experiment()


def compute_plan_diff(
    original_plan: Dict[str, Any],
    minus_plan: Dict[str, Any],
    target_client: str,
    planning_dataset_id: int,
    plans_identifier: str,
    minus_plans_identifier: str,
) -> Dict[str, Any]:
    original_configurations = original_plan.get("configurations", {})
    minus_configurations = minus_plan.get("configurations", {})
    original_config_names = set(original_configurations)
    minus_config_names = set(minus_configurations)
    shared_configurations = sorted(original_config_names & minus_config_names)
    added_configurations = sorted(minus_config_names - original_config_names)
    removed_configurations = sorted(original_config_names - minus_config_names)

    configuration_diffs = {}
    architecture_changed = bool(added_configurations or removed_configurations)
    preprocessing_changed = False

    for configuration in shared_configurations:
        original_config = original_configurations[configuration]
        minus_config = minus_configurations[configuration]
        architecture_changes = _changed_fields(
            original_config,
            minus_config,
            ARCHITECTURE_FIELDS,
        )
        preprocessing_changes = _changed_fields(
            original_config,
            minus_config,
            PREPROCESSING_FIELDS,
        )
        if architecture_changes:
            architecture_changed = True
        if preprocessing_changes:
            preprocessing_changed = True
        configuration_diffs[configuration] = {
            "architecture_changes": architecture_changes,
            "preprocessing_changes": preprocessing_changes,
            "metadata_changes": {},
        }

    top_level_preprocessing_changes = _top_level_preprocessing_changes(
        original_plan,
        minus_plan,
    )
    if top_level_preprocessing_changes:
        preprocessing_changed = True

    return {
        "target_client": str(target_client),
        "planning_dataset_id": int(planning_dataset_id),
        "plans_identifier": plans_identifier,
        "minus_plans_identifier": minus_plans_identifier,
        "architecture_changed": bool(architecture_changed),
        "preprocessing_changed": bool(preprocessing_changed and not architecture_changed),
        "normalization_changed": any(
            "normalization_schemes" in diff["preprocessing_changes"]
            or "use_mask_for_norm" in diff["preprocessing_changes"]
            for diff in configuration_diffs.values()
        ),
        "plan_level_unlearning_required": bool(architecture_changed or preprocessing_changed),
        "weight_reuse_safe": not bool(architecture_changed),
        "preprocessing_regeneration_required": bool(architecture_changed or preprocessing_changed),
        "added_configurations": added_configurations,
        "removed_configurations": removed_configurations,
        "shared_configurations": shared_configurations,
        "configurations": configuration_diffs,
        "top_level_preprocessing_changes": top_level_preprocessing_changes,
    }


def save_plan_diff_artifacts(
    artifact_dir: str,
    original_plan: Dict[str, Any],
    minus_plan: Dict[str, Any],
    plan_diff: Dict[str, Any],
) -> Dict[str, str]:
    maybe_mkdir_p(artifact_dir)
    paths = {
        "plan_diff": os.path.join(artifact_dir, "plan_diff.json"),
        "p_all": os.path.join(artifact_dir, "P_all.json"),
        "p_minus_target": os.path.join(artifact_dir, "P_minus_target.json"),
    }
    save_json(plan_diff, paths["plan_diff"], sort_keys=False)
    save_json(original_plan, paths["p_all"], sort_keys=False)
    save_json(minus_plan, paths["p_minus_target"], sort_keys=False)
    return paths


def copy_generated_minus_plan(
    dataset_id: int,
    minus_plans_identifier: str,
    destination_dir: str,
) -> Optional[str]:
    source_path = get_plan_path(dataset_id, minus_plans_identifier)
    if not os.path.isfile(source_path):
        return None
    maybe_mkdir_p(destination_dir)
    destination_path = os.path.join(destination_dir, os.path.basename(source_path))
    shutil.copy(source_path, destination_path)
    return destination_path
