from __future__ import annotations

import os
from typing import Any, Dict

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, save_json


DEFAULT_TAU_FP_LOW = 0.05
DEFAULT_TAU_PLAN_LOW = 0.05
DEFAULT_TAU_PLAN_HIGH = 0.25


POLICY_LEVELS = {
    0: {
        "label": "fingerprint-plan-stable",
        "actions": [
            "keep_existing_plan",
            "keep_existing_preprocessing",
            "run_weight_level_federaser",
        ],
    },
    1: {
        "label": "preprocessing-aware",
        "actions": [
            "save_minus_global_fingerprint",
            "keep_architecture",
            "refresh_preprocessing_statistics",
            "repreprocess_retained_clients_or_run_correction_round",
            "run_weight_level_federaser",
        ],
    },
    2: {
        "label": "planning-aware",
        "actions": [
            "save_minus_global_fingerprint",
            "use_minus_target_plan",
            "replan",
            "repreprocess_retained_clients",
            "transfer_compatible_weights",
            "run_retained_retraining_or_correction",
        ],
    },
}


def decide_unlearning_policy(
    fingerprint_diff: Dict[str, Any],
    plan_diff: Dict[str, Any],
    tau_fp_low: float = DEFAULT_TAU_FP_LOW,
    tau_plan_low: float = DEFAULT_TAU_PLAN_LOW,
    tau_plan_high: float = DEFAULT_TAU_PLAN_HIGH,
) -> Dict[str, Any]:
    if tau_fp_low < 0 or tau_plan_low < 0 or tau_plan_high < 0:
        raise ValueError("Unlearning policy thresholds must be non-negative")
    if tau_plan_low > tau_plan_high:
        raise ValueError("tau_plan_low must be <= tau_plan_high")

    d_fp = float(fingerprint_diff.get("overall_distance", 0.0))
    d_plan = float(plan_diff.get("plan_distance", 0.0))
    architecture_changed = bool(plan_diff.get("architecture_changed", False))
    preprocessing_changed = bool(
        plan_diff.get("preprocessing_changed_any")
        or plan_diff.get("preprocessing_changed")
    )

    if architecture_changed:
        level = 2
        reason = "Plan architecture changed; compatible weight reuse is not guaranteed."
    elif d_fp < tau_fp_low and d_plan < tau_plan_low and not preprocessing_changed:
        level = 0
        reason = "Fingerprint and plan distances are below low thresholds and preprocessing is stable."
    elif d_plan < tau_plan_high:
        level = 1
        reason = "Architecture is stable, but fingerprint or preprocessing/plan changes require preprocessing-aware handling."
    else:
        level = 2
        reason = "Plan distance exceeds high threshold even without an explicit architecture hard-gate change."

    level_spec = POLICY_LEVELS[level]
    return {
        "level": level,
        "label": level_spec["label"],
        "reason": reason,
        "actions": level_spec["actions"],
        "metrics": {
            "fingerprint_distance": d_fp,
            "plan_distance": d_plan,
            "architecture_changed": architecture_changed,
            "preprocessing_changed": preprocessing_changed,
        },
        "thresholds": {
            "tau_fp_low": float(tau_fp_low),
            "tau_plan_low": float(tau_plan_low),
            "tau_plan_high": float(tau_plan_high),
        },
        "inputs": {
            "fingerprint_decision_band": fingerprint_diff.get("decision_band"),
            "plan_top_contributors": plan_diff.get("plan_top_contributors", []),
            "fingerprint_top_contributors": fingerprint_diff.get("top_contributors", []),
        },
    }


def save_unlearning_policy_artifact(
    artifact_dir: str,
    policy: Dict[str, Any],
) -> str:
    maybe_mkdir_p(artifact_dir)
    path = os.path.join(artifact_dir, "unlearning_policy.json")
    save_json(policy, path, sort_keys=False)
    return path
