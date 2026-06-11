import argparse
from datetime import datetime
import json
import os
import subprocess

# Convenience script to run federated training on a multi-gpu cluster
# Each node (data-center) is spawned on a determined GPU and communicates with server on the provided network port

VALID_TASKS = ("extract_fingerprint", "plan_and_preprocess", "train", "unlearn")


def parse_data_centers(value: str):
    return json.loads("[" + value.replace(" ", ",") + "]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "task",
        choices=VALID_TASKS,
        help="Determines the task to be performed.",
    )
    parser.add_argument(
        "data_centers",
        type=parse_data_centers,
        default="",
        help="List of dataset ids (data centers) for federated training",
    )
    parser.add_argument(
        "configuration", type=str, help="Configuration that should be trained"
    )
    parser.add_argument(
        "fold",
        type=str,
        nargs="?",
        default=None,
        help="Fold of the 5-fold cross-validation. Should be an int between 0 and 4.",
    )
    parser.add_argument(
        "--gpu_memory_target",
        type=parse_data_centers,
        default="",
        help="GPU memory target in GB for each dataset, must have the same length as data_centers",
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Port number for the server to listen on"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Number of federated training rounds to run on the server.",
    )
    parser.add_argument(
        "--clients_per_round",
        type=int,
        default=None,
        help="Number of clients to train in each federated round. Defaults to all clients.",
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
        "--calibration_epochs",
        type=int,
        default=None,
        help="Override FedEraser local calibration epochs per retained round. Defaults to max(1, round(r)).",
    )
    parser.add_argument(
        "--unlearning_level",
        choices=("auto", "0", "1", "2"),
        default="auto",
        help="Unlearning level to execute. Default: auto.",
    )
    parser.add_argument(
        "--reuse_preprocessed",
        action="store_true",
        default=False,
        help="Force reuse of existing preprocessed data for Level 1 unlearning.",
    )
    parser.add_argument(
        "--repreprocess_retained",
        action="store_true",
        default=False,
        help="Force retained-client preprocessing for Level 1 unlearning.",
    )
    parser.add_argument(
        "--correction_rounds",
        type=int,
        default=0,
        help="Correction training rounds after FedEraser replay. Default: 0.",
    )
    parser.add_argument(
        "--correction_epochs",
        type=int,
        default=1,
        help="Local epochs per correction round. Default: 1.",
    )
    parser.add_argument(
        "--level2_rounds",
        type=int,
        default=None,
        help="Retained retraining rounds for Level 2. Defaults to max(1, round(r * retained_round_count)).",
    )
    parser.add_argument(
        "--level2_epochs",
        type=int,
        default=1,
        help="Local epochs per Level 2 retained retraining round. Default: 1.",
    )
    parser.add_argument(
        "--level2_transfer_source",
        choices=("federaser", "initial", "latest", "latest_global", "original", "global"),
        default="latest_global",
        help="Checkpoint source for Level 2 compatible weight transfer. Default: latest_global.",
    )
    parser.add_argument(
        "--level2_min_transfer_ratio",
        type=float,
        default=0.0,
        help="Minimum compatible parameter ratio required from retained Level 2 clients. Default: 0.0.",
    )
    parser.add_argument(
        "--tau_fp_low",
        type=float,
        default=0.05,
        help="Fingerprint distance low threshold for planning-aware unlearning policy.",
    )
    parser.add_argument(
        "--tau_plan_low",
        type=float,
        default=0.05,
        help="Plan distance low threshold for planning-aware unlearning policy.",
    )
    parser.add_argument(
        "--tau_plan_high",
        type=float,
        default=0.25,
        help="Plan distance high threshold for planning-aware unlearning policy.",
    )

    return parser


def get_folds(task: str, fold: str):
    if task in ("extract_fingerprint", "plan_and_preprocess"):
        return [0]
    if fold == "all":
        return list(range(5))
    if fold is None:
        raise ValueError(f"Fold must be specified for the {task} task")
    return [int(fold)]


def get_option_value(args_list, flags, default=None):
    for idx, value in enumerate(args_list):
        if value in flags and idx + 1 < len(args_list):
            return args_list[idx + 1]
    return default


def get_option_values(args_list, flags, default=None):
    for idx, value in enumerate(args_list):
        if value in flags:
            values = []
            for option_value in args_list[idx + 1 :]:
                if option_value.startswith("-"):
                    break
                values.append(option_value)
            return values or default
    return default


def get_bool_option(args_list, flag):
    return flag in args_list


def get_experiment_snapshot_dir():
    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed")
    if nnunet_preprocessed is None:
        return os.path.join(os.getcwd(), "fednnunet_experiment_configs")
    return os.path.join(nnunet_preprocessed, "fednnunet_experiment_configs")


def collect_preprocessing_args(args, unknown):
    return {
        "gpu_memory_target": args.gpu_memory_target or None,
        "fingerprint_extractor": get_option_value(
            unknown, ("-fpe",), "DatasetFingerprintExtractor"
        ),
        "num_processes_fingerprint": get_option_value(unknown, ("-npfp",), 8),
        "verify_dataset_integrity": get_bool_option(
            unknown, "--verify_dataset_integrity"
        ),
        "no_pp": get_bool_option(unknown, "--no_pp"),
        "clean": get_bool_option(unknown, "--clean"),
        "planner": get_option_value(unknown, ("-pl",), "ExperimentPlanner"),
        "preprocessor_name": get_option_value(
            unknown, ("-preprocessor_name",), "DefaultPreprocessor"
        ),
        "overwrite_target_spacing": get_option_values(
            unknown, ("-overwrite_target_spacing",), None
        ),
        "overwrite_plans_name": get_option_value(
            unknown, ("-overwrite_plans_name",), None
        ),
        "configurations": get_option_values(
            unknown, ("-c",), ["2d", "3d_fullres", "3d_lowres"]
        ),
        "num_processes_preprocessing": get_option_values(unknown, ("-np",), None),
        "verbose": get_bool_option(unknown, "--verbose"),
        "raw_unknown_args": unknown,
    }


def save_experiment_config_snapshot(
    args,
    unknown,
    datasets,
    fold,
    server_command,
    client_commands,
):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = os.path.join(
        get_experiment_snapshot_dir(),
        f"{run_id}_{args.task}_{args.configuration}_fold_{fold}",
    )
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_path = os.path.join(snapshot_dir, "experiment_config_snapshot.json")

    snapshot = {
        "task": args.task,
        "dataset_ids": datasets,
        "fold": fold,
        "configuration": args.configuration,
        "trainer": get_option_value(unknown, ("-tr",), "nnUNetTrainer"),
        "plans_identifier": get_option_value(unknown, ("-p",), "nnUNetPlans"),
        "unlearning_args": {
            "clients_per_round": getattr(args, "clients_per_round", None),
            "unlearning_level": getattr(args, "unlearning_level", "auto"),
            "reuse_preprocessed": getattr(args, "reuse_preprocessed", False),
            "repreprocess_retained": getattr(args, "repreprocess_retained", False),
            "correction_rounds": getattr(args, "correction_rounds", 0),
            "correction_epochs": getattr(args, "correction_epochs", 1),
            "level2_rounds": getattr(args, "level2_rounds", None),
            "level2_epochs": getattr(args, "level2_epochs", 1),
            "level2_transfer_source": getattr(args, "level2_transfer_source", "latest_global"),
            "level2_min_transfer_ratio": getattr(args, "level2_min_transfer_ratio", 0.0),
        },
        "preprocessing_args": collect_preprocessing_args(args, unknown),
        "server_command": server_command,
        "client_commands": client_commands,
    }
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot_path


def main():
    parser = build_parser()
    args, unknown = parser.parse_known_args()

    datasets = sorted(set(args.data_centers))
    num_clients = len(datasets)
    task = args.task
    fold = args.fold

    if task == "unlearn" and args.target_client is None:
        raise ValueError("--target_client must be specified for the unlearn task")
    if args.target_client is not None and args.target_client not in datasets:
        raise ValueError("--target_client must be one of the provided data_centers")
    if args.delta_t <= 0:
        raise ValueError("--delta_t must be a positive integer")
    if getattr(args, "correction_rounds", 0) < 0:
        raise ValueError("--correction_rounds must be non-negative")
    if getattr(args, "correction_epochs", 1) <= 0:
        raise ValueError("--correction_epochs must be a positive integer")
    if getattr(args, "level2_rounds", None) is not None and args.level2_rounds <= 0:
        raise ValueError("--level2_rounds must be a positive integer")
    if getattr(args, "level2_epochs", 1) <= 0:
        raise ValueError("--level2_epochs must be a positive integer")
    if getattr(args, "level2_min_transfer_ratio", 0.0) < 0:
        raise ValueError("--level2_min_transfer_ratio must be non-negative")
    if args.clients_per_round is not None:
        if args.clients_per_round <= 0:
            raise ValueError("--clients_per_round must be a positive integer")
        if args.clients_per_round > num_clients:
            raise ValueError("--clients_per_round cannot exceed the number of data_centers")

    gpu_memory_target = None
    gpu_memory_target_mapping = {}

    if args.gpu_memory_target:
        gpu_memory_target = args.gpu_memory_target
        if len(gpu_memory_target) != num_clients:
            raise ValueError("gpu_memory_target must have the same length as data_centers")
        if task not in ("plan_and_preprocess", "unlearn"):
            print(
                f"WARNING: {task} task does not accept gpu_memory_target argument. It will be ignored."
            )
        # Create a dictionary with the dataset id as key and the gpu memory target as value
        gpu_memory_target_mapping = dict(zip(datasets, gpu_memory_target))

    folds = get_folds(task, fold)

    configuration = args.configuration
    port = args.port
    server_optional_args = ""
    if args.num_rounds is not None:
        server_optional_args += f" --num_rounds {args.num_rounds}"
    if args.clients_per_round is not None:
        server_optional_args += f" --clients_per_round {args.clients_per_round}"
    if args.target_client is not None:
        server_optional_args += f" --target_client {args.target_client}"
    if task == "unlearn":
        server_optional_args += (
            f" --delta_t {args.delta_t} --r {args.r}"
            f" --tau_fp_low {args.tau_fp_low}"
            f" --tau_plan_low {args.tau_plan_low}"
            f" --tau_plan_high {args.tau_plan_high}"
            f" --unlearning_level {args.unlearning_level}"
            f" --correction_rounds {args.correction_rounds}"
            f" --correction_epochs {args.correction_epochs}"
            f" --level2_epochs {args.level2_epochs}"
            f" --level2_transfer_source {args.level2_transfer_source}"
            f" --level2_min_transfer_ratio {args.level2_min_transfer_ratio}"
        )
        if args.level2_rounds is not None:
            server_optional_args += f" --level2_rounds {args.level2_rounds}"
        if args.reuse_preprocessed:
            server_optional_args += " --reuse_preprocessed"
        if args.repreprocess_retained:
            server_optional_args += " --repreprocess_retained"
        if args.calibration_epochs is not None:
            server_optional_args += f" --calibration_epochs {args.calibration_epochs}"
        planning_dataset_id = min(datasets)
        plan_diff_gpu_memory_target = None
        if gpu_memory_target_mapping:
            plan_diff_gpu_memory_target = gpu_memory_target_mapping.get(
                planning_dataset_id
            )
        server_optional_args += (
            f" --planning_dataset_id {planning_dataset_id}"
            f" --plans_identifier {get_option_value(unknown, ('-p',), 'nnUNetPlans')}"
            f" --plan_diff_planner {get_option_value(unknown, ('-pl',), 'ExperimentPlanner')}"
            f" --plan_diff_preprocessor_name "
            f"{get_option_value(unknown, ('-preprocessor_name',), 'DefaultPreprocessor')}"
        )
        if plan_diff_gpu_memory_target is not None:
            server_optional_args += (
                f" --plan_diff_gpu_memory_target {plan_diff_gpu_memory_target}"
            )

    multi_gpu = True

    # synthetic test datasets
    node_mapping = {301: 0, 302: 0, 303: 0}
    process_prefix = ""

    for fold in folds:
        print(f"Starting {task} for fold {fold}")
        client_processes = []
        try:
            print("Starting server")
            if multi_gpu:
                process_prefix = "CUDA_VISIBLE_DEVICES=0"
            server_command = f"{process_prefix} python fednnunet/server.py {task} -n {num_clients} --port {port}{server_optional_args}"

            client_commands = {}
            for client_dataset in datasets:
                client_id = str(client_dataset)
                if multi_gpu:
                    gpu = node_mapping[client_dataset]
                    process_prefix = f"CUDA_VISIBLE_DEVICES={gpu}"

                client_global_args = f"--port {port} --client_id {client_id}"
                optional_args = ""
                # pass the undefined arguments to the client
                if unknown:
                    optional_args += " ".join(unknown) + " "
                if gpu_memory_target:
                    optional_args += (
                        f"-gpu_memory_target {gpu_memory_target_mapping[client_dataset]} "
                    )
                if task == "unlearn" and client_dataset == args.target_client:
                    optional_args += "--is_target_client "
                if task == "unlearn":
                    optional_args += (
                        f"--delta_t {args.delta_t} --r {args.r} "
                    )
                    if args.calibration_epochs is not None:
                        optional_args += f"--calibration_epochs {args.calibration_epochs} "
                    optional_args += (
                        f"--correction_epochs {args.correction_epochs} "
                        f"--level2_epochs {args.level2_epochs} "
                    )

                if task == "plan_and_preprocess":
                    command = f"{process_prefix} python fednnunet/client.py {client_global_args} {task} -d {client_dataset} {optional_args}"
                elif task in ("train", "unlearn"):
                    command = f"{process_prefix} python fednnunet/client.py {client_global_args} {task} {client_dataset} {configuration} {fold} {optional_args}"
                else:
                    command = f"{process_prefix} python fednnunet/client.py {client_global_args} {task} -d {client_dataset} {optional_args}"
                client_commands[str(client_dataset)] = command

            snapshot_path = save_experiment_config_snapshot(
                args,
                unknown,
                datasets,
                fold,
                server_command,
                client_commands,
            )
            print(f"Experiment config snapshot saved to {snapshot_path}")

            server_process = subprocess.Popen(
                server_command,
                shell=True,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Break when "ready" is printed
            for line in server_process.stderr:
                print(line, end="")  # process line here
                if "Requesting initial parameters" in line:
                    break

            for client_dataset in datasets:
                print("Starting client " + str(client_dataset))
                if multi_gpu:
                    gpu = node_mapping[client_dataset]
                    print(
                        f"Running {task} for dataset {client_dataset} with fold {fold} on GPU {gpu}"
                    )
                command = client_commands[str(client_dataset)]
                print(command)
                client_processes.append(subprocess.Popen(command, shell=True))

            for line in server_process.stderr:
                print(line, end="")

            server_process.wait()

        except KeyboardInterrupt:
            server_process.terminate()
            server_process.wait()
            for client_process in client_processes:
                client_process.terminate()
                client_process.wait()

            print("Server and clients stopped")


if __name__ == "__main__":
    main()
