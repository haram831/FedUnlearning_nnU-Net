from __future__ import annotations

import copy
import os
import shutil
from typing import Any, Dict, Optional, Tuple

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    maybe_mkdir_p,
    save_json,
)


ARCHITECTURE_FIELDS = (
    ("architecture.network_class_name", ("architecture", "network_class_name")),
    ("architecture.arch_kwargs.n_stages", ("architecture", "arch_kwargs", "n_stages")),
    ("architecture.arch_kwargs.features_per_stage", ("architecture", "arch_kwargs", "features_per_stage")),
    ("architecture.arch_kwargs.conv_op", ("architecture", "arch_kwargs", "conv_op")),
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

PREPROCESSING_CRITICAL_FIELDS = (
    "spacing",
    "normalization_schemes",
    "use_mask_for_norm",
    "preprocessor_name",
)

PLAN_DISTANCE_WEIGHTS = {
    "spacing": 0.15,
    "patch_size": 0.15,
    "batch_size": 0.08,
    "median_image_size": 0.10,
    "pooling": 0.17,
    "conv_kernel": 0.12,
    "network_depth": 0.10,
    "normalization": 0.05,
    "preprocessor": 0.03,
    "configuration": 0.05,
}


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


def _set_nested(data: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    current: Any = data
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = copy.deepcopy(value)


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


def create_architecture_preserving_level1_plan(
    original_plan: Dict[str, Any],
    minus_plan: Dict[str, Any],
) -> Dict[str, Any]:
    level1_plan = copy.deepcopy(minus_plan)
    original_configurations = original_plan.get("configurations", {})
    level1_configurations = level1_plan.get("configurations", {})

    for configuration, original_config in original_configurations.items():
        if configuration not in level1_configurations:
            level1_configurations[configuration] = copy.deepcopy(original_config)
            continue
        for _, path in ARCHITECTURE_FIELDS:
            original_value = _get_nested(original_config, path)
            if original_value is not None:
                _set_nested(level1_configurations[configuration], path, original_value)

    level1_plan["configurations"] = {
        configuration: level1_configurations[configuration]
        for configuration in original_configurations
        if configuration in level1_configurations
    }
    return level1_plan


def get_preprocessing_critical_changes(plan_diff: Dict[str, Any]) -> Dict[str, Any]:
    critical_changes = {}
    for configuration, diff in plan_diff.get("configurations", {}).items():
        preprocessing_changes = diff.get("preprocessing_changes", {})
        matched = {
            field: change
            for field, change in preprocessing_changes.items()
            if field.split(".")[-1] in PREPROCESSING_CRITICAL_FIELDS
        }
        if matched:
            critical_changes[configuration] = matched
    return critical_changes


def _as_float_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None


def _normalized_l2_value(original: Any, minus: Any, eps: float = 1e-8) -> float:
    original_array = _as_float_array(original)
    minus_array = _as_float_array(minus)
    if original_array is None or minus_array is None:
        return 0.0 if original == minus else 1.0
    if original_array.shape != minus_array.shape:
        return 1.0
    raw_distance = float(np.linalg.norm(original_array - minus_array))
    scale = max(float(np.linalg.norm(original_array)), float(np.linalg.norm(minus_array)), eps)
    return raw_distance / scale


def _categorical_distance(original: Any, minus: Any) -> float:
    return 0.0 if original == minus else 1.0


def _network_depth_value(configuration: Dict[str, Any]) -> Any:
    return [
        _get_nested(configuration, ("architecture", "arch_kwargs", "n_stages")),
        _get_nested(configuration, ("architecture", "arch_kwargs", "n_conv_per_stage")),
        _get_nested(configuration, ("architecture", "arch_kwargs", "n_conv_per_stage_decoder")),
    ]


def _configuration_plan_distances(
    original_config: Dict[str, Any],
    minus_config: Dict[str, Any],
) -> Dict[str, float]:
    return {
        "spacing": _normalized_l2_value(original_config.get("spacing"), minus_config.get("spacing")),
        "patch_size": _normalized_l2_value(original_config.get("patch_size"), minus_config.get("patch_size")),
        "batch_size": _normalized_l2_value(original_config.get("batch_size"), minus_config.get("batch_size")),
        "median_image_size": _normalized_l2_value(
            original_config.get("median_image_size_in_voxels"),
            minus_config.get("median_image_size_in_voxels"),
        ),
        "pooling": _normalized_l2_value(
            _get_nested(original_config, ("architecture", "arch_kwargs", "strides")),
            _get_nested(minus_config, ("architecture", "arch_kwargs", "strides")),
        ),
        "conv_kernel": _normalized_l2_value(
            _get_nested(original_config, ("architecture", "arch_kwargs", "kernel_sizes")),
            _get_nested(minus_config, ("architecture", "arch_kwargs", "kernel_sizes")),
        ),
        "network_depth": _normalized_l2_value(
            _network_depth_value(original_config),
            _network_depth_value(minus_config),
        ),
        "normalization": max(
            _categorical_distance(
                original_config.get("normalization_schemes"),
                minus_config.get("normalization_schemes"),
            ),
            _categorical_distance(
                original_config.get("use_mask_for_norm"),
                minus_config.get("use_mask_for_norm"),
            ),
        ),
        "preprocessor": _categorical_distance(
            original_config.get("preprocessor_name"),
            minus_config.get("preprocessor_name"),
        ),
    }


def _normalize_plan_distance_weights(weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    merged = dict(PLAN_DISTANCE_WEIGHTS)
    if weights:
        merged.update({key: float(value) for key, value in weights.items()})
    total = sum(max(float(value), 0.0) for value in merged.values())
    if total <= 0:
        raise ValueError("At least one plan distance weight must be positive")
    return {key: max(float(value), 0.0) / total for key, value in merged.items()}


def _compute_plan_distance(
    shared_configurations: Any,
    added_configurations: Any,
    removed_configurations: Any,
    original_configurations: Dict[str, Any],
    minus_configurations: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    normalized_weights = _normalize_plan_distance_weights(weights)
    component_values = {component: [] for component in normalized_weights}
    configuration_details = {}
    component_values["configuration"].append(
        1.0 if added_configurations or removed_configurations else 0.0
    )

    for configuration in shared_configurations:
        distances = _configuration_plan_distances(
            original_configurations[configuration],
            minus_configurations[configuration],
        )
        configuration_details[configuration] = distances
        for component, value in distances.items():
            component_values[component].append(float(value))

    component_distances = {
        component: float(np.mean(values)) if values else 0.0
        for component, values in component_values.items()
    }
    contributions = {
        component: normalized_weights[component] * distance
        for component, distance in component_distances.items()
    }
    top_contributors = [
        {
            "component": component,
            "value": component_distances[component],
            "weighted_value": contribution,
        }
        for component, contribution in sorted(
            contributions.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return {
        "plan_distance": float(sum(contributions.values())),
        "plan_distance_weights": normalized_weights,
        "plan_distance_components": component_distances,
        "plan_distance_contributions": contributions,
        "plan_top_contributors": top_contributors,
        "configuration_plan_distance_components": configuration_details,
    }


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
    preprocessing_changed_any = False

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
            preprocessing_changed_any = True
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
        preprocessing_changed_any = True
    plan_distance = _compute_plan_distance(
        shared_configurations,
        added_configurations,
        removed_configurations,
        original_configurations,
        minus_configurations,
    )

    result = {
        "target_client": str(target_client),
        "planning_dataset_id": int(planning_dataset_id),
        "plans_identifier": plans_identifier,
        "minus_plans_identifier": minus_plans_identifier,
        "architecture_changed": bool(architecture_changed),
        "preprocessing_changed": bool(preprocessing_changed and not architecture_changed),
        "preprocessing_changed_any": bool(preprocessing_changed_any),
        "preprocessing_only_changed": bool(preprocessing_changed and not architecture_changed),
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
    result.update(plan_distance)
    return result


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
