from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch


RUNS_DIRECTORY_NAME = "fednnunet_runs"
ARTIFACTS_DIRECTORY_NAME = "artifacts"
RESULTS_DIRECTORY_NAME = "nnUNet_results"
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_STATE_NAME = "run_state.json"
RESUME_CHECKPOINT_NAME = "global_checkpoint.pt"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def make_run_id(task: str, configuration: str, fold: int | str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{task}_{configuration}_fold_{fold}"


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return run_id


def get_runs_root(runs_root: Optional[str] = None) -> str:
    if runs_root:
        return os.path.abspath(runs_root)
    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed")
    if not nnunet_preprocessed:
        raise RuntimeError(
            "nnUNet_preprocessed must be set when --runs_root is not provided"
        )
    return os.path.join(os.path.abspath(nnunet_preprocessed), RUNS_DIRECTORY_NAME)


def get_run_dir(run_id: str, runs_root: Optional[str] = None) -> str:
    return os.path.join(get_runs_root(runs_root), validate_run_id(run_id))


def get_artifact_dir(run_dir: str) -> str:
    return os.path.join(os.path.abspath(run_dir), ARTIFACTS_DIRECTORY_NAME)


def get_results_dir(run_dir: str) -> str:
    return os.path.join(os.path.abspath(run_dir), RESULTS_DIRECTORY_NAME)


def atomic_save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_json_file(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def write_run_manifest(run_dir: str, manifest: Dict[str, Any]) -> str:
    path = os.path.join(run_dir, RUN_MANIFEST_NAME)
    atomic_save_json(manifest, path)
    return path


def load_run_manifest(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, RUN_MANIFEST_NAME)
    if not os.path.isfile(path):
        raise RuntimeError(f"Run manifest not found: {path}")
    return load_json_file(path)


def validate_resume_manifest(
    manifest: Dict[str, Any], expected: Dict[str, Any]
) -> None:
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = manifest.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: saved={actual_value!r}, requested={expected_value!r}")
    if mismatches:
        raise ValueError(
            "Resume configuration does not match the saved run:\n  "
            + "\n  ".join(mismatches)
        )


def _atomic_torch_save(value: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def get_resume_checkpoint_path(artifact_dir: str) -> str:
    return os.path.join(artifact_dir, "resume", RESUME_CHECKPOINT_NAME)


def save_resume_global_checkpoint(
    artifact_dir: str,
    global_round: int,
    total_rounds: int,
    state_dict: Dict[str, Any],
) -> str:
    path = get_resume_checkpoint_path(artifact_dir)
    _atomic_torch_save(
        {
            "format_version": 1,
            "global_round": int(global_round),
            "total_rounds": int(total_rounds),
            "state_dict": state_dict,
        },
        path,
    )
    atomic_save_json(
        {
            "global_round": int(global_round),
            "total_rounds": int(total_rounds),
            "checkpoint": path,
        },
        os.path.join(artifact_dir, RUN_STATE_NAME),
    )
    return path


def load_resume_global_checkpoint(
    artifact_dir: str,
) -> Tuple[int, int, Dict[str, Any]]:
    path = get_resume_checkpoint_path(artifact_dir)
    if not os.path.isfile(path):
        raise RuntimeError(f"Resume checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"global_round", "total_rounds", "state_dict"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError(f"Invalid resume checkpoint format: {path}")
    return (
        int(payload["global_round"]),
        int(payload["total_rounds"]),
        payload["state_dict"],
    )


def get_client_checkpoint_dir(artifact_dir: str, client_id: str) -> str:
    return os.path.join(artifact_dir, "client_checkpoints", f"client_{client_id}")


def get_client_checkpoint_paths(
    artifact_dir: str,
    client_id: str,
    pending: bool,
    global_round: Optional[int] = None,
) -> Tuple[str, str]:
    directory = get_client_checkpoint_dir(artifact_dir, client_id)
    if pending:
        stem = "checkpoint_pending"
    else:
        if global_round is None:
            raise ValueError("global_round is required for a committed checkpoint")
        stem = f"checkpoint_round_{int(global_round):04d}"
    return os.path.join(directory, f"{stem}.pth"), os.path.join(
        directory, f"{stem}.json"
    )


def save_pending_client_checkpoint(
    trainer: Any,
    artifact_dir: str,
    client_id: str,
    global_round: int,
) -> str:
    checkpoint_path, metadata_path = get_client_checkpoint_paths(
        artifact_dir, client_id, pending=True
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    temporary_path = f"{checkpoint_path}.tmp-{os.getpid()}"
    # nnU-Net's save_checkpoint stores current_epoch + 1. A federated round has
    # already incremented current_epoch by the time this helper is called.
    trainer.current_epoch -= 1
    try:
        trainer.save_checkpoint(temporary_path)
    finally:
        trainer.current_epoch += 1
    if not os.path.isfile(temporary_path):
        raise RuntimeError(
            "Client resume checkpoint was not written. Checkpointing must be enabled."
        )
    os.replace(temporary_path, checkpoint_path)
    atomic_save_json(
        {
            "client_id": str(client_id),
            "global_round": int(global_round),
            "current_epoch": int(trainer.current_epoch),
            "checkpoint": checkpoint_path,
        },
        metadata_path,
    )
    return checkpoint_path


def commit_pending_client_checkpoint(
    artifact_dir: str, client_id: str, global_round: int
) -> bool:
    pending_checkpoint, pending_metadata = get_client_checkpoint_paths(
        artifact_dir, client_id, pending=True
    )
    if not os.path.isfile(pending_checkpoint) or not os.path.isfile(pending_metadata):
        return False
    metadata = load_json_file(pending_metadata)
    if int(metadata.get("global_round", -1)) != int(global_round):
        return False
    committed_checkpoint, committed_metadata = get_client_checkpoint_paths(
        artifact_dir,
        client_id,
        pending=False,
        global_round=global_round,
    )
    if not os.path.exists(committed_checkpoint):
        os.link(pending_checkpoint, committed_checkpoint)
    metadata["checkpoint"] = committed_checkpoint
    atomic_save_json(metadata, committed_metadata)
    os.remove(pending_checkpoint)
    os.remove(pending_metadata)
    directory = get_client_checkpoint_dir(artifact_dir, client_id)
    committed_stem = f"checkpoint_round_{int(global_round):04d}"
    for entry in os.listdir(directory):
        if not entry.startswith("checkpoint_round_") or entry.startswith(committed_stem):
            continue
        path = os.path.join(directory, entry)
        if os.path.isfile(path):
            os.remove(path)
    return True


def resolve_client_resume_checkpoint(
    artifact_dir: str, client_id: str, resume_round: int
) -> Optional[Tuple[str, Dict[str, Any]]]:
    candidates = []
    directory = get_client_checkpoint_dir(artifact_dir, client_id)
    metadata_paths = []
    if os.path.isdir(directory):
        metadata_paths.extend(
            os.path.join(directory, entry)
            for entry in os.listdir(directory)
            if entry.startswith("checkpoint_round_") and entry.endswith(".json")
        )
    _, pending_metadata = get_client_checkpoint_paths(
        artifact_dir, client_id, pending=True
    )
    metadata_paths.append(pending_metadata)
    for metadata_path in metadata_paths:
        if not os.path.isfile(metadata_path):
            continue
        metadata = load_json_file(metadata_path)
        checkpoint_path = metadata.get("checkpoint")
        if not checkpoint_path:
            continue
        if not os.path.isfile(checkpoint_path) or not os.path.isfile(metadata_path):
            continue
        checkpoint_round = int(metadata.get("global_round", -1))
        if checkpoint_round <= int(resume_round):
            candidates.append((checkpoint_round, checkpoint_path, metadata))
    if not candidates:
        return None
    _, checkpoint_path, metadata = max(candidates, key=lambda item: item[0])
    return checkpoint_path, metadata
