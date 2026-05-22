import argparse
import json
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

    return parser


def get_folds(task: str, fold: str):
    if task in ("extract_fingerprint", "plan_and_preprocess"):
        return [0]
    if fold == "all":
        return list(range(5))
    if fold is None:
        raise ValueError(f"Fold must be specified for the {task} task")
    return [int(fold)]


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

    gpu_memory_target = None
    gpu_memory_target_mapping = {}

    if args.gpu_memory_target:
        gpu_memory_target = args.gpu_memory_target
        if len(gpu_memory_target) != num_clients:
            raise ValueError("gpu_memory_target must have the same length as data_centers")
        if task != "plan_and_preprocess":
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
    if args.target_client is not None:
        server_optional_args += f" --target_client {args.target_client}"
    if task == "unlearn":
        server_optional_args += f" --delta_t {args.delta_t} --r {args.r}"

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
            server_process = subprocess.Popen(
                f"{process_prefix} python fednnunet/server.py {task} -n {num_clients} --port {port}{server_optional_args}",
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
                    process_prefix = f"CUDA_VISIBLE_DEVICES={gpu}"
                    print(
                        f"Running {task} for dataset {client_dataset} with fold {fold} on GPU {gpu}"
                    )

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
                    optional_args += f"--delta_t {args.delta_t} --r {args.r} "

                if task == "plan_and_preprocess":
                    command = f"{process_prefix} python fednnunet/client.py --port {port} {task} -d {client_dataset} {optional_args}"
                elif task in ("train", "unlearn"):
                    command = f"{process_prefix} python fednnunet/client.py --port {port} {task} {client_dataset} {configuration} {fold} {optional_args}"
                else:
                    command = f"{process_prefix} python fednnunet/client.py --port {port} {task} -d {client_dataset} {optional_args}"
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
