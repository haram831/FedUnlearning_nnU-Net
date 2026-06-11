from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p, save_json


@dataclass
class ClientUpdateRecord:
    client_id: str
    update: Dict[str, torch.Tensor]
    num_examples: int
    update_path: str


@dataclass
class RetainedRound:
    round_id: int
    client_updates: Dict[str, ClientUpdateRecord]


def state_dict_l2_norm(update: Dict[str, torch.Tensor]) -> torch.Tensor:
    tensors = [
        value.detach().float().reshape(-1)
        for value in update.values()
        if torch.is_tensor(value)
    ]
    if not tensors:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.cat(tensors), ord=2)


def zeros_like_state_dict(reference: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: torch.zeros_like(value.detach().cpu())
        for key, value in reference.items()
        if torch.is_tensor(value)
    }


def subtract_state_dicts(
    minuend: Dict[str, torch.Tensor],
    subtrahend: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    update = {}
    for key, value in minuend.items():
        base = subtrahend.get(key)
        if torch.is_tensor(value) and torch.is_tensor(base) and value.shape == base.shape:
            update[key] = value.detach().cpu() - base.detach().cpu()
    return update


def calibrate_update(
    retained_update: Dict[str, torch.Tensor],
    current_update: Dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    retained_norm = state_dict_l2_norm(retained_update)
    current_norm = state_dict_l2_norm(current_update)
    compatible_current = {
        key: value
        for key, value in current_update.items()
        if key in retained_update
        and torch.is_tensor(value)
        and torch.is_tensor(retained_update[key])
        and value.shape == retained_update[key].shape
    }
    if current_norm < eps:
        return zeros_like_state_dict(compatible_current)
    scale = retained_norm / (current_norm + eps)
    return {
        key: value.detach().cpu() * scale
        for key, value in compatible_current.items()
    }


def aggregate_updates(
    updates_by_client: Dict[str, Dict[str, torch.Tensor]],
    num_examples_by_client: Dict[str, int],
    aggregation: str = "weighted",
) -> Dict[str, torch.Tensor]:
    if not updates_by_client:
        raise ValueError("Cannot aggregate an empty update set")

    first_update = next(iter(updates_by_client.values()))
    aggregated = zeros_like_state_dict(first_update)
    if aggregation == "uniform":
        weights = {client_id: 1.0 for client_id in updates_by_client}
    elif aggregation == "weighted":
        total = sum(max(int(num_examples_by_client.get(client_id, 0)), 0) for client_id in updates_by_client)
        if total <= 0:
            weights = {client_id: 1.0 for client_id in updates_by_client}
        else:
            weights = {
                client_id: max(int(num_examples_by_client.get(client_id, 0)), 0) / total
                for client_id in updates_by_client
            }
    else:
        raise ValueError(f"Unknown aggregation mode: {aggregation}")

    if aggregation == "uniform":
        scale = 1.0 / len(updates_by_client)
        weights = {client_id: scale for client_id in updates_by_client}

    for client_id, update in updates_by_client.items():
        alpha = float(weights[client_id])
        for key, value in update.items():
            if key in aggregated and value.shape == aggregated[key].shape:
                aggregated[key] += value.detach().cpu() * alpha
    return aggregated


def apply_update(
    state_dict: Dict[str, torch.Tensor],
    update: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    updated = {}
    for key, value in state_dict.items():
        if key in update and torch.is_tensor(value) and torch.is_tensor(update[key]) and value.shape == update[key].shape:
            updated[key] = value.detach().cpu() + update[key].detach().cpu().to(dtype=value.dtype)
        else:
            updated[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return updated


def load_global_checkpoint(artifact_dir: str, round_id: int) -> Dict[str, torch.Tensor]:
    checkpoint_path = os.path.join(
        artifact_dir,
        "global_checkpoints",
        f"round_{round_id:04d}",
        "global_checkpoint.pt",
    )
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(f"Global checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def list_retained_rounds(artifact_dir: str) -> List[int]:
    updates_root = os.path.join(artifact_dir, "client_updates")
    if not os.path.isdir(updates_root):
        raise RuntimeError(f"Client update directory not found: {updates_root}")
    rounds = []
    for entry in os.listdir(updates_root):
        if entry.startswith("round_") and os.path.isdir(os.path.join(updates_root, entry)):
            rounds.append(int(entry[len("round_") :]))
    return sorted(rounds)


def load_retained_round(artifact_dir: str, round_id: int) -> RetainedRound:
    round_dir = os.path.join(artifact_dir, "client_updates", f"round_{round_id:04d}")
    if not os.path.isdir(round_dir):
        raise RuntimeError(f"Retained round directory not found: {round_dir}")

    client_updates = {}
    for entry in sorted(os.listdir(round_dir)):
        client_dir = os.path.join(round_dir, entry)
        metadata_path = os.path.join(client_dir, "metadata.json")
        if not entry.startswith("client_") or not os.path.isfile(metadata_path):
            continue
        metadata = load_json(metadata_path)
        client_id = str(metadata["client_id"])
        update_path = metadata["client_update"]
        client_updates[client_id] = ClientUpdateRecord(
            client_id=client_id,
            update=torch.load(update_path, map_location="cpu", weights_only=False),
            num_examples=int(metadata.get("num_examples", 0)),
            update_path=update_path,
        )
    if not client_updates:
        raise RuntimeError(f"No client updates found for retained round {round_id}")
    return RetainedRound(round_id=round_id, client_updates=client_updates)


def get_unlearning_run_dir(artifact_dir: str, target_client: str) -> str:
    return os.path.join(artifact_dir, "unlearning", f"target_client_{target_client}")


def save_unlearned_checkpoint(
    run_dir: str,
    round_id: int,
    state_dict: Dict[str, torch.Tensor],
) -> str:
    checkpoint_dir = os.path.join(run_dir, "checkpoints", f"round_{round_id:04d}")
    maybe_mkdir_p(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "global_checkpoint.pt")
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


def save_final_unlearned_model(
    run_dir: str,
    state_dict: Dict[str, torch.Tensor],
) -> str:
    maybe_mkdir_p(run_dir)
    path = os.path.join(run_dir, "final_unlearned_model.pt")
    torch.save(state_dict, path)
    return path


def save_unlearning_report(
    run_dir: str,
    report: Dict[str, Any],
) -> str:
    maybe_mkdir_p(run_dir)
    path = os.path.join(run_dir, "unlearning_report.json")
    save_json(report, path, sort_keys=False)
    return path


def start_timer() -> float:
    return time.time()


def elapsed_seconds(start_time: float) -> float:
    return time.time() - start_time
