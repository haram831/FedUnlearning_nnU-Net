import json
import os
from io import BytesIO
from logging import INFO, WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p, save_json
from flwr.common import EvaluateIns, FitIns, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fednnunet.fingerprint_diff import (
    build_fingerprint_precheck_report,
    save_fingerprint_precheck_artifacts,
)
from fednnunet.federaser import (
    aggregate_updates as federaser_aggregate_updates,
    apply_update as federaser_apply_update,
    calibrate_update,
    elapsed_seconds,
    get_unlearning_run_dir,
    list_retained_rounds,
    load_global_checkpoint,
    load_retained_round,
    save_final_unlearned_model,
    save_unlearned_checkpoint,
    save_unlearning_report,
    start_timer,
    subtract_state_dicts as federaser_subtract_state_dicts,
)
from fednnunet.plan_diff import (
    compute_plan_diff,
    copy_generated_minus_plan,
    generate_plan_from_fingerprint,
    load_plan,
    save_plan_diff_artifacts,
)
from fednnunet.unlearning_policy import (
    DEFAULT_TAU_FP_LOW,
    DEFAULT_TAU_PLAN_HIGH,
    DEFAULT_TAU_PLAN_LOW,
    decide_unlearning_policy,
    save_unlearning_policy_artifact,
)


def state_dict_to_bytes(state_dict) -> bytes:
    bytes_io = BytesIO()
    torch.save(state_dict, bytes_io)
    return bytes_io.getvalue()


def state_dict_to_parameters(state_dict) -> Parameters:
    tensors = state_dict_to_bytes(state_dict)
    log(INFO, f"State dict to parameters: {len(tensors)} bytes")
    return Parameters(tensors=[tensors], tensor_type="whatever")


def bytes_to_state_dict(bytes_data: bytes) -> dict:
    """Converts bytes back to a PyTorch state_dict."""
    bytes_io = BytesIO(bytes_data)
    return torch.load(bytes_io)


def parameters_to_state_dict(parameters: Parameters) -> dict:
    """Converts Flower Parameters back to a PyTorch state_dict."""
    bytes_data = parameters.tensors[0]
    return bytes_to_state_dict(bytes_data)


def get_logical_client_id(
    client_proxy: fl.server.client_proxy.ClientProxy,
    fit_res: fl.common.FitRes,
) -> str:
    return str(fit_res.metrics.get("client_id", client_proxy.cid))


def get_fingerprint_history_dir() -> str:
    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed")
    if nnunet_preprocessed is None:
        nnunet_preprocessed = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fingerprint_history"
        )
        log(
            WARNING,
            "nnUNet_preprocessed is not set. Saving fingerprint history under "
            f"{nnunet_preprocessed}",
        )
    return os.path.join(nnunet_preprocessed, "fednnunet_fingerprint_history")


def get_intensity_histogram_num_bins() -> int:
    num_bins = int(os.environ.get("FEDNNUNET_INTENSITY_HISTOGRAM_NUM_BINS", "1000"))
    if num_bins <= 0:
        raise ValueError("Intensity histogram num bins must be positive")
    return num_bins

# For the first round of fingerprint extraction, we only compute basic intensity stats to keep the client computation light.
def get_federated_fingerprint_stats_config() -> Dict[str, Scalar]:
    return {
        "fingerprint_pass": "stats",
        "build_intensity_histograms": False,
    }

# For the second round of fingerprint extraction, we compute intensity histograms to get a more detailed view of the intensity distribution, which can inform better preprocessing decisions.
def get_global_intensity_histogram_config(global_fingerprint: Dict[str, Any]) -> Dict[str, Scalar]:
    num_bins = get_intensity_histogram_num_bins()
    channel_edges = {}
    for channel, stats in global_fingerprint["foreground_intensity_properties_per_channel"].items():
        minimum = float(stats["min"])
        maximum = float(stats["max"])
        if minimum == maximum:
            minimum -= 0.5
            maximum += 0.5
        channel_edges[str(channel)] = [
            minimum + (maximum - minimum) * idx / num_bins
            for idx in range(num_bins + 1)
        ]

    return {
        "fingerprint_pass": "histogram",
        "build_intensity_histograms": True,
        "intensity_histogram_bin_edges_by_channel_json": json.dumps(channel_edges),
    }


def save_client_fingerprint_history(
    server_round: int,
    client_id: str,
    fit_res: fl.common.FitRes,
) -> None:
    dataset_id = fit_res.metrics.get("dataset_id", client_id)
    dataset_name = fit_res.metrics.get("dataset_name", f"client_{client_id}")
    client_dir = os.path.join(
        get_fingerprint_history_dir(),
        f"round_{server_round:04d}",
        f"client_{client_id}",
    )
    maybe_mkdir_p(client_dir)

    fingerprint_bytes = fit_res.parameters.tensors[0]
    bytes_path = os.path.join(client_dir, "dataset_fingerprint_local.bin")
    with open(bytes_path, "wb") as f:
        f.write(fingerprint_bytes)

    fingerprint_dict = bytes_to_state_dict(fingerprint_bytes)
    dict_path = os.path.join(client_dir, "dataset_fingerprint_local.json")
    save_json(fingerprint_dict, dict_path)
    save_json(
        {
            "server_round": server_round,
            "client_id": client_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "fingerprint_bytes": bytes_path,
            "fingerprint_dict": dict_path,
        },
        os.path.join(client_dir, "metadata.json"),
    )


def save_global_fingerprint(server_round: int, parameters: Parameters) -> None:
    history_dir = get_fingerprint_history_dir()
    round_dir = os.path.join(history_dir, f"round_{server_round:04d}")
    maybe_mkdir_p(round_dir)

    global_fingerprint = parameters_to_state_dict(parameters)
    save_json(global_fingerprint, os.path.join(round_dir, "global_fingerprint_all.json"))
    save_json(global_fingerprint, os.path.join(history_dir, "global_fingerprint_all.json"))


def get_latest_fingerprint_round_dir() -> str:
    history_dir = get_fingerprint_history_dir()
    if not os.path.isdir(history_dir):
        raise RuntimeError(f"Fingerprint history directory does not exist: {history_dir}")
    round_dirs = [
        os.path.join(history_dir, entry)
        for entry in os.listdir(history_dir)
        if entry.startswith("round_") and os.path.isdir(os.path.join(history_dir, entry))
    ]
    if not round_dirs:
        raise RuntimeError(f"No fingerprint round directories found in {history_dir}")
    return sorted(round_dirs)[-1]


def load_latest_local_fingerprints() -> Tuple[str, Dict[str, Dict[str, Any]]]:
    round_dir = get_latest_fingerprint_round_dir()
    fingerprints = {}
    for entry in sorted(os.listdir(round_dir)):
        client_dir = os.path.join(round_dir, entry)
        if not entry.startswith("client_") or not os.path.isdir(client_dir):
            continue
        client_id = entry[len("client_") :]
        fingerprint_path = os.path.join(client_dir, "dataset_fingerprint_local.json")
        if os.path.isfile(fingerprint_path):
            fingerprints[client_id] = load_json(fingerprint_path)
    if not fingerprints:
        raise RuntimeError(f"No local client fingerprints found in {round_dir}")
    return round_dir, fingerprints


def load_latest_global_fingerprint() -> Dict[str, Any]:
    history_dir = get_fingerprint_history_dir()
    fingerprint_path = os.path.join(history_dir, "global_fingerprint_all.json")
    if not os.path.isfile(fingerprint_path):
        raise RuntimeError(f"Global fingerprint not found: {fingerprint_path}")
    return load_json(fingerprint_path)


def get_federaser_artifact_dir() -> str:
    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed")
    if nnunet_preprocessed is None:
        nnunet_preprocessed = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "federaser_artifacts"
        )
        log(
            WARNING,
            "nnUNet_preprocessed is not set. Saving FedEraser artifacts under "
            f"{nnunet_preprocessed}",
        )
    return os.path.join(nnunet_preprocessed, "fednnunet_federaser_artifacts")


def save_global_checkpoint(
    global_round: int,
    global_state_dict: dict,
    delta_t: int,
    total_rounds: int,
) -> None:
    checkpoint_dir = os.path.join(
        get_federaser_artifact_dir(),
        "global_checkpoints",
        f"round_{global_round:04d}",
    )
    maybe_mkdir_p(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, "global_checkpoint.pt")
    torch.save(global_state_dict, checkpoint_path)
    save_json(
        {
            "global_round": global_round,
            "delta_t": delta_t,
            "total_rounds": total_rounds,
            "checkpoint": checkpoint_path,
        },
        os.path.join(checkpoint_dir, "metadata.json"),
    )


def subtract_state_dicts(local_state_dict: dict, global_state_dict: dict) -> dict:
    update = {}
    for key, local_value in local_state_dict.items():
        global_value = global_state_dict.get(key)
        if (
            torch.is_tensor(local_value)
            and torch.is_tensor(global_value)
            and local_value.shape == global_value.shape
        ):
            update[key] = local_value.detach().cpu() - global_value.detach().cpu()
    return update


def save_client_update(
    global_round: int,
    client_id: str,
    local_state_dict: dict,
    global_state_dict: dict,
    num_examples: int,
    delta_t: int,
    total_rounds: int,
    calibration_r: float,
) -> None:
    client_dir = os.path.join(
        get_federaser_artifact_dir(),
        "client_updates",
        f"round_{global_round:04d}",
        f"client_{client_id}",
    )
    maybe_mkdir_p(client_dir)

    client_parameters_path = os.path.join(client_dir, "client_parameters.pt")
    client_update_path = os.path.join(client_dir, "client_update.pt")
    torch.save(local_state_dict, client_parameters_path)
    torch.save(subtract_state_dicts(local_state_dict, global_state_dict), client_update_path)
    save_json(
        {
            "global_round": global_round,
            "client_id": client_id,
            "num_examples": num_examples,
            "delta_t": delta_t,
            "total_rounds": total_rounds,
            "calibration_r": calibration_r,
            "client_parameters": client_parameters_path,
            "client_update": client_update_path,
        },
        os.path.join(client_dir, "metadata.json"),
    )


def save_aggregation_metadata(
    global_round: int,
    results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
) -> str:
    metadata_dir = os.path.join(
        get_federaser_artifact_dir(),
        "aggregation_metadata",
        f"round_{global_round:04d}",
    )
    maybe_mkdir_p(metadata_dir)

    metadata_path = os.path.join(metadata_dir, "metadata.json")
    aggregation_metadata_path = os.path.join(metadata_dir, "aggregation_metadata.json")
    cid_to_client_id = {
        client_proxy.cid: get_logical_client_id(client_proxy, fit_res)
        for client_proxy, fit_res in results
    }
    metadata = {
        "round": global_round,
        "client_proxy_cid_to_client_id": cid_to_client_id,
        "participating_clients": [
            cid_to_client_id[client_proxy.cid] for client_proxy, _ in results
        ],
        "num_examples": {
            cid_to_client_id[client_proxy.cid]: fit_res.num_examples
            for client_proxy, fit_res in results
        },
        "client_metadata": {
            cid_to_client_id[client_proxy.cid]: {
                "flower_cid": client_proxy.cid,
                "client_id": cid_to_client_id[client_proxy.cid],
                "dataset_id": fit_res.metrics.get("dataset_id"),
                "dataset_name": fit_res.metrics.get("dataset_name"),
            }
            for client_proxy, fit_res in results
        },
    }
    save_json(metadata, metadata_path)
    save_json(metadata, aggregation_metadata_path)
    return metadata_path


def average_dicts(dicts):
    if not dicts:
        return {}

    # Initialize a dictionary to keep track of the sum and count for each key
    totals = {}
    counts = {}

    # Iterate through each dictionary
    for d in dicts:
        for key, value in d.items():
            if key in totals:
                totals[key] += value
                counts[key] += 1
            else:
                totals[key] = value
                counts[key] = 1

    # Calculate the average for each key
    averages = {key: totals[key] / counts[key] for key in totals}

    return averages


def weighted_mean(values, weights):
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def aggregate_max(values, weights):
    return torch.tensor(values).max().item()


def aggregate_min(values, weights):
    return torch.tensor(values).min().item()


def concatenate(values, weights):
    # Concatenate the nested lists maintaining the structure
    return [item for sublist in values for item in sublist]


def pooled_std(means: List[float], variances: List[float], counts: List[int]) -> float:
    total_count = sum(counts)
    if total_count == 0:
        return float("nan")

    global_mean = weighted_mean(means, counts)
    pooled_variance = sum(
        count * (variance + (mean - global_mean) ** 2)
        for mean, variance, count in zip(means, variances, counts)
    ) / total_count
    return pooled_variance ** 0.5


def aggregate_histograms(histograms: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    if not histograms:
        return {"bin_edges": [], "counts": []}

    bin_edges = list(histograms[0]["bin_edges"])
    counts = [0.0] * len(histograms[0]["counts"])
    for histogram in histograms:
        if list(histogram["bin_edges"]) != bin_edges:
            raise ValueError("Cannot aggregate histograms with different bin edges")
        if len(histogram["counts"]) != len(counts):
            raise ValueError("Cannot aggregate histograms with different bin counts")
        counts = [
            total_count + float(client_count)
            for total_count, client_count in zip(counts, histogram["counts"])
        ]

    return {"bin_edges": bin_edges, "counts": counts}


def quantile_from_histogram(histogram: Dict[str, Any], quantile: float) -> float:
    if not 0 <= quantile <= 1:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")

    bin_edges = list(histogram["bin_edges"])
    counts = [float(i) for i in histogram["counts"]]
    total_count = sum(counts)
    if total_count == 0:
        return float("nan")

    target = quantile * total_count
    cumulative_count = 0.0
    for idx, count in enumerate(counts):
        next_cumulative_count = cumulative_count + count
        if target <= next_cumulative_count or idx == len(counts) - 1:
            if count == 0:
                return float(bin_edges[idx])
            fraction = (target - cumulative_count) / count
            left_edge = float(bin_edges[idx])
            right_edge = float(bin_edges[idx + 1])
            return left_edge + fraction * (right_edge - left_edge)
        cumulative_count = next_cumulative_count

    return float(bin_edges[-1])


def aggregate_intensity_channel_stats(
    channel_stats: List[Dict[str, Any]], weights: List[int]
) -> Dict[str, Any]:
    counts = [
        int(stats.get("count", weight))
        for stats, weight in zip(channel_stats, weights)
    ]
    result = {
        "mean": weighted_mean([stats["mean"] for stats in channel_stats], counts),
        "min": aggregate_min([stats["min"] for stats in channel_stats], counts),
        "max": aggregate_max([stats["max"] for stats in channel_stats], counts),
    }

    if all("variance" in stats for stats in channel_stats):
        variances = [stats["variance"] for stats in channel_stats]
        result["std"] = pooled_std(
            [stats["mean"] for stats in channel_stats],
            variances,
            counts,
        )
        result["variance"] = result["std"] ** 2
        result["count"] = sum(counts)
    else:
        result["std"] = weighted_mean([stats["std"] for stats in channel_stats], counts)

    if all("histogram" in stats for stats in channel_stats):
        histogram = aggregate_histograms([stats["histogram"] for stats in channel_stats])
        result["histogram"] = histogram
        result["percentile_00_5"] = quantile_from_histogram(histogram, 0.005)
        result["median"] = quantile_from_histogram(histogram, 0.5)
        result["percentile_99_5"] = quantile_from_histogram(histogram, 0.995)
    else:
        result["percentile_00_5"] = weighted_mean(
            [stats["percentile_00_5"] for stats in channel_stats], counts
        )
        result["median"] = weighted_mean(
            [stats["median"] for stats in channel_stats], counts
        )
        result["percentile_99_5"] = weighted_mean(
            [stats["percentile_99_5"] for stats in channel_stats], counts
        )

    return result


def aggregate_fingerprint_dicts(state_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not state_dicts:
        raise ValueError("Cannot aggregate an empty fingerprint list")

    num_samples = [len(sd["shapes_after_crop"]) for sd in state_dicts]
    print(f"Num samples per client: {num_samples}")

    new_state_dict = {
        "spacings": concatenate([sd["spacings"] for sd in state_dicts], num_samples),
        "shapes_after_crop": concatenate(
            [sd["shapes_after_crop"] for sd in state_dicts], num_samples
        ),
        "foreground_intensity_properties_per_channel": {},
    }

    for channel in state_dicts[0]["foreground_intensity_properties_per_channel"].keys():
        new_state_dict["foreground_intensity_properties_per_channel"][channel] = (
            aggregate_intensity_channel_stats(
                [
                    sd["foreground_intensity_properties_per_channel"][channel]
                    for sd in state_dicts
                ],
                num_samples,
            )
        )

    if all("relative_size_after_cropping_histogram" in sd for sd in state_dicts):
        histogram = aggregate_histograms(
            [sd["relative_size_after_cropping_histogram"] for sd in state_dicts]
        )
        new_state_dict["relative_size_after_cropping_histogram"] = histogram
        new_state_dict["median_relative_size_after_cropping"] = quantile_from_histogram(
            histogram, 0.5
        )
    else:
        new_state_dict["median_relative_size_after_cropping"] = weighted_mean(
            [sd["median_relative_size_after_cropping"] for sd in state_dicts],
            num_samples,
        )

    return new_state_dict


def aggregate_fingerprints(parameters: List[Parameters]) -> Parameters:
    state_dicts = [parameters_to_state_dict(p) for p in parameters]
    new_state_dict = aggregate_fingerprint_dicts(state_dicts)

    # Convert the new state_dict back to parameters
    return state_dict_to_parameters(new_state_dict)


class MyStrategy(fl.server.strategy.FedAvg):

    def __init__(
        self,
        task: str,
        target_client: Optional[int] = None,
        delta_t: int = 2,
        r: float = 0.5,
        total_rounds: int = 1000,
        planning_dataset_id: Optional[int] = None,
        plans_identifier: str = "nnUNetPlans",
        plan_diff_planner: str = "ExperimentPlanner",
        plan_diff_preprocessor_name: str = "DefaultPreprocessor",
        plan_diff_gpu_memory_target: Optional[float] = None,
        tau_fp_low: float = DEFAULT_TAU_FP_LOW,
        tau_plan_low: float = DEFAULT_TAU_PLAN_LOW,
        tau_plan_high: float = DEFAULT_TAU_PLAN_HIGH,
        calibration_epochs: Optional[int] = None,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            inplace=inplace,
        )

        self.task = task
        self.target_client = target_client
        self.delta_t = delta_t
        self.r = r
        self.total_rounds = total_rounds
        self.planning_dataset_id = planning_dataset_id
        self.plans_identifier = plans_identifier
        self.plan_diff_planner = plan_diff_planner
        self.plan_diff_preprocessor_name = plan_diff_preprocessor_name
        self.plan_diff_gpu_memory_target = plan_diff_gpu_memory_target
        self.tau_fp_low = tau_fp_low
        self.tau_plan_low = tau_plan_low
        self.tau_plan_high = tau_plan_high
        self.calibration_epochs = (
            max(1, int(calibration_epochs))
            if calibration_epochs is not None
            else max(1, int(round(self.r)))
        )
        self.saved_plan_diff_artifact = False
        self.latest_global_state_dict = None
        self.saved_initial_federaser_checkpoint = False
        self.federaser_initialized = False
        self.federaser_current_state_dict = None
        self.federaser_retained_round_ids: List[int] = []
        self.federaser_current_retained_round_id = None
        self.federaser_run_dir = None
        self.federaser_start_time = None
        self.federaser_round_reports = []

    def create_unlearning_plan_diff_artifact(self) -> None:
        if self.saved_plan_diff_artifact:
            return
        if self.task != "unlearn" or self.target_client is None:
            return

        target_client = str(self.target_client)
        round_dir, fingerprints_by_client = load_latest_local_fingerprints()
        if target_client not in fingerprints_by_client:
            raise RuntimeError(
                f"Target client {target_client} is missing from fingerprint history {round_dir}"
            )

        retained_fingerprints = [
            fingerprint
            for client_id, fingerprint in fingerprints_by_client.items()
            if client_id != target_client
        ]
        if not retained_fingerprints:
            raise RuntimeError(
                f"Cannot build minus-target fingerprint: no retained clients after excluding {target_client}"
            )

        planning_dataset_id = self.planning_dataset_id or self.target_client
        source_round_name = os.path.basename(round_dir)
        minus_fingerprint = aggregate_fingerprint_dicts(retained_fingerprints)
        artifact_dir = os.path.join(
            get_fingerprint_history_dir(),
            f"plan_diff_target_{target_client}",
            source_round_name,
        )
        maybe_mkdir_p(artifact_dir)
        save_json(
            minus_fingerprint,
            os.path.join(artifact_dir, "global_fingerprint_minus_target.json"),
            sort_keys=False,
        )
        fingerprint_precheck_dir = os.path.join(
            get_federaser_artifact_dir(),
            "plan_unlearning_precheck",
            f"target_client_{target_client}",
            source_round_name,
        )
        original_fingerprint = load_latest_global_fingerprint()
        fingerprint_precheck_report = build_fingerprint_precheck_report(
            target_client=target_client,
            source_round=source_round_name,
            original=original_fingerprint,
            excluded=minus_fingerprint,
        )
        fingerprint_precheck_paths = save_fingerprint_precheck_artifacts(
            fingerprint_precheck_dir,
            fingerprint_precheck_report,
            minus_fingerprint,
        )

        minus_plans_identifier = (
            f"{self.plans_identifier}_minus_target_{target_client}_{source_round_name}"
        )
        original_plan = load_plan(planning_dataset_id, self.plans_identifier)
        minus_plan = generate_plan_from_fingerprint(
            planning_dataset_id,
            minus_fingerprint,
            minus_plans_identifier,
            planner_class_name=self.plan_diff_planner,
            preprocessor_name=self.plan_diff_preprocessor_name,
            gpu_memory_target_in_gb=self.plan_diff_gpu_memory_target,
        )
        plan_diff = compute_plan_diff(
            original_plan,
            minus_plan,
            target_client=target_client,
            planning_dataset_id=planning_dataset_id,
            plans_identifier=self.plans_identifier,
            minus_plans_identifier=minus_plans_identifier,
        )
        unlearning_policy = decide_unlearning_policy(
            fingerprint_precheck_report,
            plan_diff,
            tau_fp_low=self.tau_fp_low,
            tau_plan_low=self.tau_plan_low,
            tau_plan_high=self.tau_plan_high,
        )
        paths = save_plan_diff_artifacts(
            artifact_dir,
            original_plan,
            minus_plan,
            plan_diff,
        )
        policy_path = save_unlearning_policy_artifact(
            artifact_dir,
            unlearning_policy,
        )
        generated_plan_snapshot = copy_generated_minus_plan(
            planning_dataset_id,
            minus_plans_identifier,
            artifact_dir,
        )
        save_json(
            {
                "target_client": target_client,
                "planning_dataset_id": planning_dataset_id,
                "source_fingerprint_round": round_dir,
                "source_fingerprint_round_name": source_round_name,
                "retained_clients": sorted(
                    client_id
                    for client_id in fingerprints_by_client
                    if client_id != target_client
                ),
                "excluded_clients": [target_client],
                "artifact_paths": {
                    **paths,
                    "unlearning_policy": policy_path,
                    "generated_minus_plan_snapshot": generated_plan_snapshot,
                    "fingerprint_precheck": fingerprint_precheck_paths,
                },
            },
            os.path.join(artifact_dir, "metadata.json"),
            sort_keys=False,
        )
        log(INFO, f"Plan diff artifact saved to {paths['plan_diff']}")
        self.saved_plan_diff_artifact = True

    def initialize_federaser_replay(self) -> None:
        if self.federaser_initialized:
            return
        if self.task != "unlearn" or self.target_client is None:
            return

        artifact_dir = get_federaser_artifact_dir()
        target_client = str(self.target_client)
        self.federaser_retained_round_ids = list_retained_rounds(artifact_dir)
        if not self.federaser_retained_round_ids:
            raise RuntimeError("FedEraser requires at least one retained client update round")
        self.federaser_current_state_dict = load_global_checkpoint(artifact_dir, 0)
        self.federaser_run_dir = get_unlearning_run_dir(artifact_dir, target_client)
        maybe_mkdir_p(self.federaser_run_dir)
        save_unlearned_checkpoint(
            self.federaser_run_dir,
            0,
            self.federaser_current_state_dict,
        )
        self.federaser_start_time = start_timer()
        self.federaser_initialized = True

    def get_federaser_round_id(self, server_round: int) -> int:
        self.initialize_federaser_replay()
        index = server_round - 1
        if index < 0 or index >= len(self.federaser_retained_round_ids):
            raise RuntimeError(
                f"FedEraser server_round {server_round} has no retained round. "
                f"Available retained rounds: {self.federaser_retained_round_ids}"
            )
        return self.federaser_retained_round_ids[index]

    def should_save_federaser_artifact(self, global_round: int) -> bool:
        return (
            global_round == 0
            or global_round % self.delta_t == 0
            or global_round == self.total_rounds
        )

    def save_initial_federaser_checkpoint(self, global_state_dict: dict) -> None:
        if self.saved_initial_federaser_checkpoint:
            return
        if not self.should_save_federaser_artifact(0):
            return
        save_global_checkpoint(
            0,
            global_state_dict,
            delta_t=self.delta_t,
            total_rounds=self.total_rounds,
        )
        self.saved_initial_federaser_checkpoint = True

    def find_common_layers(self, state_dicts):
        # Find the common keys in all state_dicts

        common_keys = set(state_dicts[0].keys())

        for sd in state_dicts[1:]:
            common_keys.intersection_update(sd.keys())
            # Print number of keys in this dictionary
            log(INFO, f"Number of keys in this dictionary: {len(sd.keys())}")

        log(INFO, f"Number of common keys: {len(common_keys)}")

        # Verify dimensions
        compatible_keys = []
        for key in common_keys:
            dimensions = [sd[key].shape for sd in state_dicts]
            if all(dim == dimensions[0] for dim in dimensions):
                compatible_keys.append(key)
        log(INFO, f"Number of compatible keys: {len(compatible_keys)}")

        return compatible_keys

    def create_compatible_state_dict(self, state_dicts, compatible_keys):
        new_state_dict = {}
        for key in compatible_keys:
            # Assuming we take the parameters from the first state_dict
            keys = [s[key] for s in state_dicts]
            new_state_dict[key] = torch.mean(torch.stack(keys, dim=0), dim=0)
        return new_state_dict

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.create_unlearning_plan_diff_artifact()
        config = {}
        if self.task == "unlearn":
            retained_round_id = self.get_federaser_round_id(server_round)
            self.federaser_current_retained_round_id = retained_round_id
            config.update(
                {
                    "federaser_mode": "calibration",
                    "retained_round_id": retained_round_id,
                    "target_client": self.target_client,
                    "delta_t": self.delta_t,
                    "r": self.r,
                    "calibration_epochs": self.calibration_epochs,
                }
            )
            fit_ins = FitIns(
                state_dict_to_parameters(self.federaser_current_state_dict),
                config,
            )
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size,
                min_num_clients=min_num_clients,
            )
            return [(client, fit_ins) for client in clients]

        if self.task == "extract_fingerprint" or self.task == "plan_and_preprocess":
            if server_round == 1:
                config.update(get_federated_fingerprint_stats_config())
            else:
                config.update(
                    get_global_intensity_histogram_config(
                        parameters_to_state_dict(parameters)
                    )
                )
        if self.task == "unlearn" and self.target_client is not None:
            config["target_client"] = self.target_client
            config["delta_t"] = self.delta_t
            config["r"] = self.r
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config.update(self.on_fit_config_fn(server_round))
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        if self.task == "extract_fingerprint" or self.task == "plan_and_preprocess":
            return [(client, fit_ins) for client in clients]

        if self.latest_global_state_dict is None and parameters.tensors:
            self.latest_global_state_dict = parameters_to_state_dict(parameters)
            self.save_initial_federaser_checkpoint(self.latest_global_state_dict)

        parms = [
            parameters_to_state_dict(
                client.get_parameters(
                    ins=fit_ins, timeout=None, group_id=None
                ).parameters
            )
            for client in clients
        ]
        for idx, sd in enumerate(parms):
            torch.save(
                sd,
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), str(idx) + ".arch"
                ),
            )

        compatible_keys = self.find_common_layers(parms)

        # here we are doing the merging already...
        new_state_dict = self.create_compatible_state_dict(parms, compatible_keys)
        if self.latest_global_state_dict is None:
            self.latest_global_state_dict = new_state_dict
            self.save_initial_federaser_checkpoint(new_state_dict)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        evaluate_instructions = super().configure_evaluate(
            server_round, parameters, client_manager
        )
        if self.task != "extract_fingerprint" and self.task != "plan_and_preprocess":
            return evaluate_instructions

        fingerprint_pass = "histogram" if server_round >= self.total_rounds else "stats"
        return [
            (
                client,
                EvaluateIns(
                    evaluate_ins.parameters,
                    {**evaluate_ins.config, "fingerprint_pass": fingerprint_pass},
                ),
            )
            for client, evaluate_ins in evaluate_instructions
        ]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[List[float], Dict[str, fl.common.Scalar]]:
        if failures:
            fl.common.logger.log(2, f"Round {rnd} had {len(failures)} failures.")

        # Filter out None results due to failures
        successful_results = [result for result in results if result is not None]

        # If there are no successful results, return None or a default value
        if not successful_results:
            fl.common.logger.log(2, f"Round {rnd} had {len(failures)} failures.")
            return None  # or some default values

        if self.task == "extract_fingerprint" or self.task == "plan_and_preprocess":
            for client_proxy, fit_res in successful_results:
                save_client_fingerprint_history(
                    rnd,
                    get_logical_client_id(client_proxy, fit_res),
                    fit_res,
                )
            aggregated_fingerprint = aggregate_fingerprints(
                [res[1].parameters for res in successful_results]
            )
            save_global_fingerprint(rnd, aggregated_fingerprint)
            return (
                aggregated_fingerprint,
                {},
            )
        if self.task == "unlearn":
            return self.aggregate_federaser_calibration_round(rnd, successful_results)

        # Perform aggregation on successful results
        save_aggregation_metadata(rnd, successful_results)
        if self.should_save_federaser_artifact(rnd) and self.latest_global_state_dict is not None:
            for client_proxy, fit_res in successful_results:
                save_client_update(
                    rnd,
                    get_logical_client_id(client_proxy, fit_res),
                    parameters_to_state_dict(fit_res.parameters),
                    self.latest_global_state_dict,
                    fit_res.num_examples,
                    delta_t=self.delta_t,
                    total_rounds=self.total_rounds,
                    calibration_r=self.r,
                )
        aggregated_weights = self.aggregate_weights(successful_results)
        self.latest_global_state_dict = parameters_to_state_dict(aggregated_weights)
        if self.should_save_federaser_artifact(rnd):
            save_global_checkpoint(
                rnd,
                self.latest_global_state_dict,
                delta_t=self.delta_t,
                total_rounds=self.total_rounds,
            )
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif rnd == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return aggregated_weights, average_dicts(
            [r[1].metrics for r in successful_results]
        )

    def aggregate_weights(self, results):

        dicts = [parameters_to_state_dict(res[1].parameters) for res in results]

        compatible_keys = self.find_common_layers(dicts)

        # here we are doing the merging already...
        new_state_dict = self.create_compatible_state_dict(dicts, compatible_keys)

        # Implement weight aggregation logic
        return state_dict_to_parameters(new_state_dict)

    def aggregate_federaser_calibration_round(
        self,
        server_round: int,
        successful_results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Parameters, Dict[str, fl.common.Scalar]]:
        self.initialize_federaser_replay()
        retained_round_id = self.get_federaser_round_id(server_round)
        retained_round = load_retained_round(get_federaser_artifact_dir(), retained_round_id)
        target_client = str(self.target_client)

        calibrated_updates = {}
        num_examples = {}
        skipped_clients = []
        used_clients = []
        for client_proxy, fit_res in successful_results:
            client_id = get_logical_client_id(client_proxy, fit_res)
            if client_id == target_client or bool(fit_res.metrics.get("is_target_client", False)):
                skipped_clients.append(client_id)
                continue
            if client_id not in retained_round.client_updates:
                skipped_clients.append(client_id)
                continue

            local_state_dict = parameters_to_state_dict(fit_res.parameters)
            current_update = federaser_subtract_state_dicts(
                local_state_dict,
                self.federaser_current_state_dict,
            )
            retained_record = retained_round.client_updates[client_id]
            calibrated_updates[client_id] = calibrate_update(
                retained_record.update,
                current_update,
            )
            num_examples[client_id] = retained_record.num_examples
            used_clients.append(client_id)

        if not calibrated_updates:
            raise RuntimeError(
                f"No retained clients produced FedEraser calibration updates for round {retained_round_id}"
            )

        aggregated_update = federaser_aggregate_updates(
            calibrated_updates,
            num_examples,
            aggregation="weighted",
        )
        self.federaser_current_state_dict = federaser_apply_update(
            self.federaser_current_state_dict,
            aggregated_update,
        )
        checkpoint_path = save_unlearned_checkpoint(
            self.federaser_run_dir,
            retained_round_id,
            self.federaser_current_state_dict,
        )
        round_report = {
            "server_round": server_round,
            "retained_round": retained_round_id,
            "used_clients": sorted(used_clients),
            "skipped_clients": sorted(set(skipped_clients)),
            "checkpoint": checkpoint_path,
            "aggregation": "weighted_by_retained_num_examples",
        }
        self.federaser_round_reports.append(round_report)

        is_last_round = server_round >= len(self.federaser_retained_round_ids)
        final_path = None
        report_path = None
        if is_last_round:
            final_path = save_final_unlearned_model(
                self.federaser_run_dir,
                self.federaser_current_state_dict,
            )
            report = {
                "method": "FedEraser",
                "target_client_id": target_client,
                "retain_interval": self.delta_t,
                "calibration_ratio": self.r,
                "calibration_epochs": self.calibration_epochs,
                "num_retained_rounds": len(self.federaser_retained_round_ids),
                "retained_rounds": self.federaser_retained_round_ids,
                "rounds": self.federaser_round_reports,
                "unlearning_time_sec": elapsed_seconds(self.federaser_start_time),
                "output_checkpoint": final_path,
                "aggregation": "weighted_by_retained_num_examples",
                "notes": [
                    "Calibrated update uses historical retained update norm and current calibration update direction.",
                    "Target client is excluded from calibration and aggregation.",
                ],
            }
            report_path = save_unlearning_report(self.federaser_run_dir, report)

        metrics = {
            "federaser_retained_round": retained_round_id,
            "federaser_used_clients": len(used_clients),
            "federaser_skipped_clients": len(skipped_clients),
            "federaser_checkpoint": checkpoint_path,
        }
        if final_path is not None:
            metrics["federaser_final_checkpoint"] = final_path
        if report_path is not None:
            metrics["federaser_report"] = report_path

        return state_dict_to_parameters(self.federaser_current_state_dict), metrics


def main() -> None:
    # Start Flower server with the custom strategy
    import argparse

    parser = argparse.ArgumentParser(description="Start Flower server")
    parser.add_argument(
        "task",
        type=str,
        choices=("extract_fingerprint", "plan_and_preprocess", "train", "unlearn"),
        help="Determines the task to be performed.",
    )
    parser.add_argument(
        "-n",
        "--num_clients",
        type=int,
        default=2,
        help="Number of clients to wait for before starting the server",
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Port number for the server to listen on"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Number of federated training rounds. Defaults to 2 for planning tasks and 1000 for training.",
    )
    parser.add_argument(
        "--target_client",
        type=int,
        default=None,
        help="Dataset id of the client to unlearn. Required for the unlearn task.",
    )
    parser.add_argument(
        "--delta_t",
        type=int,
        default=2,
        help="FedEraser unlearning interval. Used by the unlearn task. Default: 2.",
    )
    parser.add_argument(
        "--r",
        type=float,
        default=0.5,
        help="FedEraser calibration ratio. Used by the unlearn task. Default: 0.5.",
    )
    parser.add_argument(
        "--planning_dataset_id",
        type=int,
        default=None,
        help="Dataset id whose nnU-Net dataset context and P_all plan are used for unlearning plan diff.",
    )
    parser.add_argument(
        "--plans_identifier",
        type=str,
        default="nnUNetPlans",
        help="Existing P_all plans identifier used as the plan diff baseline.",
    )
    parser.add_argument(
        "--plan_diff_planner",
        type=str,
        default="ExperimentPlanner",
        help="Experiment planner class used to generate P_minus_target.",
    )
    parser.add_argument(
        "--plan_diff_preprocessor_name",
        type=str,
        default="DefaultPreprocessor",
        help="Preprocessor class name used to generate P_minus_target.",
    )
    parser.add_argument(
        "--plan_diff_gpu_memory_target",
        type=float,
        default=None,
        help="GPU memory target in GB used to generate P_minus_target.",
    )
    parser.add_argument(
        "--tau_fp_low",
        type=float,
        default=DEFAULT_TAU_FP_LOW,
        help="Fingerprint distance low threshold for Level 0 policy decisions.",
    )
    parser.add_argument(
        "--tau_plan_low",
        type=float,
        default=DEFAULT_TAU_PLAN_LOW,
        help="Plan distance low threshold for Level 0 policy decisions.",
    )
    parser.add_argument(
        "--tau_plan_high",
        type=float,
        default=DEFAULT_TAU_PLAN_HIGH,
        help="Plan distance high threshold for Level 2 policy decisions.",
    )
    parser.add_argument(
        "--calibration_epochs",
        type=int,
        default=None,
        help="Override local calibration epochs per retained FedEraser round. Defaults to max(1, round(r)).",
    )

    args = parser.parse_args()
    num_clients = args.num_clients

    if args.task == "unlearn" and args.target_client is None:
        raise ValueError("--target_client must be specified for the unlearn task")
    if args.delta_t <= 0:
        raise ValueError("--delta_t must be a positive integer")

    if args.task == "extract_fingerprint" or args.task == "plan_and_preprocess":
        num_rounds = 2
        fraction_evaluate = 1.0
    elif args.task == "unlearn":
        num_rounds = len(list_retained_rounds(get_federaser_artifact_dir()))
        if num_rounds <= 0:
            raise ValueError("FedEraser unlearn requires at least one retained update round")
        fraction_evaluate = 0.0
    else:
        # nnUNet's default training length
        num_rounds = 1000
        # Skip federated evaluation to speed up training by one less parameters transfer
        fraction_evaluate = 0.0

    if args.num_rounds is not None:
        if (
            args.task in ("extract_fingerprint", "plan_and_preprocess")
            and args.num_rounds < 2
        ):
            raise ValueError("Fingerprint extraction requires at least 2 rounds")
        num_rounds = args.num_rounds

    strategy = MyStrategy(
        args.task,
        target_client=args.target_client,
        delta_t=args.delta_t,
        r=args.r,
        total_rounds=num_rounds,
        planning_dataset_id=args.planning_dataset_id,
        plans_identifier=args.plans_identifier,
        plan_diff_planner=args.plan_diff_planner,
        plan_diff_preprocessor_name=args.plan_diff_preprocessor_name,
        plan_diff_gpu_memory_target=args.plan_diff_gpu_memory_target,
        tau_fp_low=args.tau_fp_low,
        tau_plan_low=args.tau_plan_low,
        tau_plan_high=args.tau_plan_high,
        calibration_epochs=args.calibration_epochs,
        min_available_clients=num_clients,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        fraction_evaluate=fraction_evaluate,
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        grpc_max_message_length=2147483647,  # Request a maximum message length to support sending weights from larger, more recent ResEnc architectures
    )


if __name__ == "__main__":
    main()
