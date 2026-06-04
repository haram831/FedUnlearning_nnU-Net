import argparse
import json
import logging
import os
import sys
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import flwr as fl
import numpy as np
import torch
from flwr.common import Code, EvaluateRes, FitRes, GetParametersRes, Parameters, Status

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nnUNet"))
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprint_dataset,
    plan_experiments,
    preprocess,
)
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import (
    convert_dataset_name_to_id,
    maybe_convert_to_dataset_name,
)
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

from fednnunet.run_training import run_training

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def state_dict_to_bytes(state_dict) -> bytes:
    """Carlos: flexibility, dont want to deal with anoying converisons."""
    bytes_io = BytesIO()
    torch.save(state_dict, bytes_io)
    return bytes_io.getvalue()


def state_dict_to_parameters(state_dict) -> Parameters:
    """Carlos: flexibility, dont want to deal with anoying converisons."""
    tensors = state_dict_to_bytes(state_dict)
    return Parameters(tensors=[tensors], tensor_type="whatever")


def bytes_to_state_dict(bytes_data: bytes) -> dict:
    """Converts bytes back to a PyTorch state_dict."""
    bytes_io = BytesIO(bytes_data)
    return torch.load(bytes_io, weights_only=False)


def parameters_to_state_dict(parameters: Parameters) -> dict:
    """Converts Flower Parameters back to a PyTorch state_dict."""
    bytes_data = parameters.tensors[0]
    return bytes_to_state_dict(bytes_data)


def parse_float_list(value: Any) -> List[float]:
    if isinstance(value, str):
        return [float(i) for i in value.split(",")]
    return [float(i) for i in value]


def parse_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


class FlowerClient(fl.client.Client):

    def __init__(
        self,
        task: str = "train",
        args: argparse.Namespace = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.task = task
        self.args = args
        self.device = device
        self.dataset_name = maybe_convert_to_dataset_name(args.dataset_name_or_id)
        self.dataset_id = convert_dataset_name_to_id(self.dataset_name)
        self.client_id = getattr(args, "client_id", None) or str(self.dataset_id)
        self.num_samples = None
        self.extract_fingerprint = False
        self.plan_experiment = False
        self.preprocess_dataset = False

        self.train = False
        if self.task in ("train", "unlearn"):
            self.train = True
            self.is_target_client = getattr(args, "is_target_client", False)
            self.delta_t = getattr(args, "delta_t", None)
            self.r = getattr(args, "r", None)
            self.calibration_epochs = getattr(args, "calibration_epochs", 1)
            self.correction_epochs = getattr(args, "correction_epochs", 1)
            self.level2_epochs = getattr(args, "level2_epochs", 1)
            # this calls run_training but is not running any training, I did not change the name of the method for compatibility with regular nnUnet.
            self.trainer = run_training(
                args.dataset_name_or_id,
                args.configuration,
                args.fold,
                args.tr,
                args.p,
                args.pretrained_weights,
                args.num_gpus,
                args.use_compressed,
                args.npz,
                args.c,
                args.val,
                args.disable_checkpointing,
                args.val_best,
                device=device,
                return_trainer=True,
            )

            self.trainer.initialize()
            self.model = self.trainer.network
            self.trainer.on_train_start()
            self.current_plans_identifier = args.p

        if self.task == "plan_and_preprocess":
            self.extract_fingerprint = True
            self.plan_experiment = True
            self.preprocess_dataset = True
            self.gpu_memory_target_in_gb = args.gpu_memory_target

            if args.np is None:
                default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
                args.np = [
                    default_np[c] if c in default_np.keys() else 4 for c in args.c
                ]
            else:
                args.np = args.np
            if args.no_pp:
                self.preprocess_dataset = False

        if self.task == "extract_fingerprint" or self.extract_fingerprint:
            self.extract_fingerprint = True
            self.fingerprint = None
            self.local_fingerprint = None
            self.local_fingerprint_pass = None

        self.preprocessed_output_folder = join(nnUNet_preprocessed, self.dataset_name)

    def ensure_training_context(self, plans_identifier: str) -> None:
        if getattr(self, "current_plans_identifier", None) == plans_identifier:
            return

        self.trainer = run_training(
            self.args.dataset_name_or_id,
            self.args.configuration,
            self.args.fold,
            self.args.tr,
            plans_identifier,
            None,
            self.args.num_gpus,
            self.args.use_compressed,
            self.args.npz,
            False,
            self.args.val,
            self.args.disable_checkpointing,
            self.args.val_best,
            device=self.device,
            return_trainer=True,
        )
        self.trainer.initialize()
        self.model = self.trainer.network
        self.trainer.on_train_start()
        self.current_plans_identifier = plans_identifier

    def partial_transfer_state_dict(self, source_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        target_state_dict = self.model.state_dict()
        transferred_keys = []
        skipped_keys = []
        transferred_params = 0
        total_params = 0

        for key, target_value in target_state_dict.items():
            if torch.is_tensor(target_value):
                total_params += int(target_value.numel())
            source_value = source_state_dict.get(key)
            if (
                torch.is_tensor(target_value)
                and torch.is_tensor(source_value)
                and target_value.shape == source_value.shape
            ):
                target_state_dict[key] = source_value.detach().cpu().to(dtype=target_value.dtype)
                transferred_keys.append(key)
                transferred_params += int(target_value.numel())
            else:
                skipped_keys.append(key)

        self.model.load_state_dict(target_state_dict, strict=True)
        transfer_ratio = transferred_params / total_params if total_params > 0 else 0.0
        return {
            "transferred_keys": transferred_keys,
            "skipped_keys": skipped_keys,
            "transferred_key_count": len(transferred_keys),
            "skipped_key_count": len(skipped_keys),
            "transferred_param_count": transferred_params,
            "total_param_count": total_params,
            "transfer_ratio": transfer_ratio,
        }

    def get_overlapping_keys(self, state_dict1, state_dict2):
        """Find keys that are present in both state_dicts and have the same shape."""
        overlapping_keys = set(state_dict1.keys()).intersection(state_dict2.keys())
        compatible_keys = [
            key
            for key in overlapping_keys
            if state_dict1[key].shape == state_dict2[key].shape
        ]
        return compatible_keys

    def replace_overlapping_keys(self, target_state_dict, source_state_dict):
        """Replace keys in the target_state_dict with the values from source_state_dict for overlapping keys."""
        overlapping_keys = self.get_overlapping_keys(
            target_state_dict, source_state_dict
        )

        for key in overlapping_keys:
            target_state_dict[key] = source_state_dict[key]

        return target_state_dict

    def get_num_training_examples(self):
        dataloader = self.trainer.dataloader_train
        if hasattr(dataloader, "indices"):
            return len(dataloader.indices)

        dataset = getattr(dataloader, "_data", None)
        if dataset is None and hasattr(dataloader, "generator"):
            dataset = getattr(dataloader.generator, "_data", None)

        if hasattr(dataset, "identifiers"):
            return len(dataset.identifiers)
        if hasattr(dataset, "keys"):
            return len(dataset.keys())
        if hasattr(dataset, "__len__"):
            return len(dataset)

        return int(self.trainer.dataset_json["numTraining"])

    @staticmethod
    def get_fingerprint_extractor_kwargs(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not config:
            return {}

        extractor_kwargs = {}
        bin_edges = config.get("intensity_histogram_bin_edges")
        if bin_edges:
            extractor_kwargs["intensity_histogram_bin_edges"] = parse_float_list(bin_edges)
            return extractor_kwargs

        bin_edges_by_channel = config.get("intensity_histogram_bin_edges_by_channel_json")
        if bin_edges_by_channel:
            extractor_kwargs["intensity_histogram_bin_edges_by_channel"] = json.loads(bin_edges_by_channel)

        histogram_range = config.get("intensity_histogram_range")
        if histogram_range:
            extractor_kwargs["intensity_histogram_range"] = parse_float_list(histogram_range)

        num_bins = config.get("intensity_histogram_num_bins")
        if num_bins is not None:
            extractor_kwargs["intensity_histogram_num_bins"] = int(num_bins)

        build_histograms = config.get("build_intensity_histograms")
        if build_histograms is not None:
            extractor_kwargs["build_intensity_histograms"] = parse_bool(build_histograms)

        return extractor_kwargs

    @staticmethod
    def get_config(ins: Any) -> Dict[str, Any]:
        if isinstance(ins, dict):
            return ins
        return getattr(ins, "config", {}) or {}

    def get_fingerprint(self, config: Optional[Dict[str, Any]] = None):
        fingerprint_pass = (config or {}).get("fingerprint_pass")
        if not self.local_fingerprint or self.local_fingerprint_pass != fingerprint_pass:
            # self.local_fingerprint = extract_fingerprint_dataset(self.dataset_id, clean=True)
            fingerprint_extractor_class = recursive_find_python_class(
                join(nnunetv2.__path__[0], "experiment_planning"),
                self.args.fpe,
                current_module="nnunetv2.experiment_planning",
            )
            self.local_fingerprint = extract_fingerprint_dataset(
                self.dataset_id,
                fingerprint_extractor_class=fingerprint_extractor_class,
                num_processes=self.args.npfp,
                check_dataset_integrity=self.args.verify_dataset_integrity,
                clean=True,
                verbose=self.args.verbose,
                fingerprint_extractor_kwargs=self.get_fingerprint_extractor_kwargs(config),
            )
            self.local_fingerprint_pass = fingerprint_pass
            self.num_samples = len(self.local_fingerprint["shapes_after_crop"])
            save_json(
                self.local_fingerprint,
                join(self.preprocessed_output_folder, "dataset_fingerprint_local.json"),
            )
            logging.info(
                f"Local dataset fingerprint saved to {join(self.preprocessed_output_folder, 'dataset_fingerprint_local.json')}"
            )
            self.fingerprint = self.local_fingerprint

        return self.fingerprint

    def get_parameters(self, fi):
        if self.extract_fingerprint:
            parameters = self.get_fingerprint(self.get_config(fi))
            # print(f'Fingerprint with mean: {self.fingerprint["median_relative_size_after_cropping"]}')
        else:
            parameters = self.model.state_dict()

        parm = GetParametersRes(
            parameters=state_dict_to_parameters(parameters),
            status=Status(code=Code(0), message="caguento"),
        )
        return parm

    def set_parameters(self, parameters):

        common_state_dict = parameters_to_state_dict(parameters)

        if self.extract_fingerprint:
            self.fingerprint = common_state_dict
        else:
            # torch.save(common_state_dict,os.path.join(os.path.dirname(os.path.abspath(__file__)),'common_state_dict.arch'))
            # torch.save(self.model.state_dict(),os.path.join(os.path.dirname(os.path.abspath(__file__)),'local_state_dict.arch'))

            onset_subset_keys_dict = self.replace_overlapping_keys(
                self.model.state_dict(), common_state_dict
            )
            self.model.load_state_dict(onset_subset_keys_dict, strict=True)

    def fit(self, fi):
        config = self.get_config(fi)
        federaser_mode = config.get("federaser_mode")
        if federaser_mode == "level2_retrain":
            self.ensure_training_context(config.get("plans_identifier", self.args.p))
            self.set_parameters(fi.parameters)
        elif federaser_mode not in ("level2_preprocess", "level2_transfer"):
            self.set_parameters(fi.parameters)

        if self.extract_fingerprint:
            return FitRes(
                parameters=self.get_parameters(fi).parameters,
                status=Status(code=Code(0), message="Fingerprint extracted"),
                num_examples=0,
                metrics={
                    "client_id": self.client_id,
                    "dataset_id": self.dataset_id,
                    "dataset_name": self.dataset_name,
                },
            )
        elif config.get("federaser_mode") == "calibration":
            if self.is_target_client:
                return FitRes(
                    parameters=self.get_parameters({}).parameters,
                    status=Status(code=Code(0), message="Target client skipped FedEraser calibration"),
                    num_examples=0,
                    metrics={
                        "client_id": self.client_id,
                        "dataset_id": self.dataset_id,
                        "dataset_name": self.dataset_name,
                        "is_target_client": True,
                        "federaser_mode": "calibration",
                        "skipped": True,
                    },
                )

            calibration_epochs = max(1, int(config.get("calibration_epochs", 1)))
            try:
                for _ in range(calibration_epochs):
                    self.trainer.run_federated_train_round()
            except ValueError as e:
                logging.error(f"ValueError occurred during FedEraser calibration: {e}")
            except RuntimeError as e:
                logging.error(f"RuntimeError occurred during FedEraser calibration: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during FedEraser calibration: {e}")
                raise

            losses = self.trainer.logger.my_fantastic_logging["train_losses"]
            loss = float(np.round(losses[-1], decimals=4)) if losses else 0.0
            return FitRes(
                parameters=self.get_parameters({}).parameters,
                status=Status(code=Code(0), message="FedEraser calibration complete"),
                num_examples=self.get_num_training_examples(),
                metrics={
                    "client_id": self.client_id,
                    "dataset_id": self.dataset_id,
                    "dataset_name": self.dataset_name,
                    "loss": loss,
                    "is_target_client": False,
                    "federaser_mode": "calibration",
                    "calibration_epochs": calibration_epochs,
                    "delta_t": self.delta_t,
                    "r": self.r,
                },
            )
        elif config.get("federaser_mode") == "level2_preprocess":
            if self.is_target_client:
                return FitRes(
                    parameters=self.get_parameters({}).parameters,
                    status=Status(code=Code(0), message="Target client skipped Level 2 preprocessing"),
                    num_examples=0,
                    metrics={
                        "client_id": self.client_id,
                        "dataset_id": self.dataset_id,
                        "dataset_name": self.dataset_name,
                        "is_target_client": True,
                        "federaser_mode": "level2_preprocess",
                        "skipped": True,
                    },
                )

            plans_identifier = config.get("plans_identifier") or getattr(self.args, "p", "nnUNetPlans")
            configurations = getattr(
                self.args,
                "preprocess_configurations",
                ["2d", "3d_fullres", "3d_lowres"],
            )
            num_processes = getattr(self.args, "np", None)
            if num_processes is None:
                default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
                num_processes = [
                    default_np[c] if c in default_np else 4
                    for c in configurations
                ]
            preprocess(
                [self.dataset_id],
                plans_identifier=plans_identifier,
                configurations=configurations,
                num_processes=num_processes,
                verbose=getattr(self.args, "verbose", False),
            )
            return FitRes(
                parameters=self.get_parameters({}).parameters,
                status=Status(code=Code(0), message="Level 2 retained preprocessing complete"),
                num_examples=0,
                metrics={
                    "client_id": self.client_id,
                    "dataset_id": self.dataset_id,
                    "dataset_name": self.dataset_name,
                    "is_target_client": False,
                    "federaser_mode": "level2_preprocess",
                    "plans_identifier": plans_identifier,
                    "configurations": json.dumps(configurations),
                },
            )
        elif config.get("federaser_mode") == "level2_transfer":
            if self.is_target_client:
                return FitRes(
                    parameters=self.get_parameters({}).parameters,
                    status=Status(code=Code(0), message="Target client skipped Level 2 transfer"),
                    num_examples=0,
                    metrics={
                        "client_id": self.client_id,
                        "dataset_id": self.dataset_id,
                        "dataset_name": self.dataset_name,
                        "is_target_client": True,
                        "federaser_mode": "level2_transfer",
                        "skipped": True,
                    },
                )

            plans_identifier = config.get("plans_identifier") or getattr(self.args, "p", "nnUNetPlans")
            self.ensure_training_context(plans_identifier)
            transfer_report = self.partial_transfer_state_dict(
                parameters_to_state_dict(fi.parameters)
            )
            return FitRes(
                parameters=self.get_parameters({}).parameters,
                status=Status(code=Code(0), message="Level 2 compatible transfer complete"),
                num_examples=self.get_num_training_examples(),
                metrics={
                    "client_id": self.client_id,
                    "dataset_id": self.dataset_id,
                    "dataset_name": self.dataset_name,
                    "is_target_client": False,
                    "federaser_mode": "level2_transfer",
                    "plans_identifier": plans_identifier,
                    "transferred_key_count": transfer_report["transferred_key_count"],
                    "skipped_key_count": transfer_report["skipped_key_count"],
                    "transferred_param_count": transfer_report["transferred_param_count"],
                    "total_param_count": transfer_report["total_param_count"],
                    "transfer_ratio": transfer_report["transfer_ratio"],
                    "transfer_report_json": json.dumps(transfer_report),
                },
            )
        elif config.get("federaser_mode") == "level2_retrain":
            if self.is_target_client:
                return FitRes(
                    parameters=self.get_parameters({}).parameters,
                    status=Status(code=Code(0), message="Target client skipped Level 2 retraining"),
                    num_examples=0,
                    metrics={
                        "client_id": self.client_id,
                        "dataset_id": self.dataset_id,
                        "dataset_name": self.dataset_name,
                        "is_target_client": True,
                        "federaser_mode": "level2_retrain",
                        "skipped": True,
                    },
                )

            level2_epochs = max(1, int(config.get("level2_epochs", self.level2_epochs)))
            try:
                for _ in range(level2_epochs):
                    self.trainer.run_federated_train_round()
            except ValueError as e:
                logging.error(f"ValueError occurred during Level 2 retraining: {e}")
            except RuntimeError as e:
                logging.error(f"RuntimeError occurred during Level 2 retraining: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during Level 2 retraining: {e}")
                raise

            losses = self.trainer.logger.my_fantastic_logging["train_losses"]
            loss = float(np.round(losses[-1], decimals=4)) if losses else 0.0
            return FitRes(
                parameters=self.get_parameters({}).parameters,
                status=Status(code=Code(0), message="Level 2 retraining complete"),
                num_examples=self.get_num_training_examples(),
                metrics={
                    "client_id": self.client_id,
                    "dataset_id": self.dataset_id,
                    "dataset_name": self.dataset_name,
                    "loss": loss,
                    "is_target_client": False,
                    "federaser_mode": "level2_retrain",
                    "level2_epochs": level2_epochs,
                },
            )
        elif config.get("federaser_mode") == "preprocess_retained":
            if self.is_target_client:
                return FitRes(
                    parameters=self.get_parameters({}).parameters,
                    status=Status(code=Code(0), message="Target client skipped retained preprocessing"),
                    num_examples=0,
                    metrics={
                        "client_id": self.client_id,
                        "dataset_id": self.dataset_id,
                        "dataset_name": self.dataset_name,
                        "is_target_client": True,
                        "federaser_mode": "preprocess_retained",
                        "skipped": True,
                    },
                )

            plans_identifier = config.get("plans_identifier") or getattr(self.args, "p", "nnUNetPlans")
            configurations = getattr(
                self.args,
                "preprocess_configurations",
                ["2d", "3d_fullres", "3d_lowres"],
            )
            num_processes = getattr(self.args, "np", None)
            if num_processes is None:
                default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
                num_processes = [
                    default_np[c] if c in default_np else 4
                    for c in configurations
                ]
            preprocess(
                [self.dataset_id],
                plans_identifier=plans_identifier,
                configurations=configurations,
                num_processes=num_processes,
                verbose=getattr(self.args, "verbose", False),
            )
            return FitRes(
                parameters=self.get_parameters({}).parameters,
                status=Status(code=Code(0), message="Retained preprocessing complete"),
                num_examples=0,
                metrics={
                    "client_id": self.client_id,
                    "dataset_id": self.dataset_id,
                    "dataset_name": self.dataset_name,
                    "is_target_client": False,
                    "federaser_mode": "preprocess_retained",
                    "plans_identifier": plans_identifier,
                    "configurations": json.dumps(configurations),
                },
            )
        elif config.get("federaser_mode") == "preprocess_skip":
            return FitRes(
                parameters=self.get_parameters({}).parameters,
                status=Status(code=Code(0), message="Retained preprocessing skipped"),
                num_examples=0,
                metrics={
                    "client_id": self.client_id,
                    "dataset_id": self.dataset_id,
                    "dataset_name": self.dataset_name,
                    "is_target_client": self.is_target_client,
                    "federaser_mode": "preprocess_skip",
                    "skipped": True,
                },
            )
        elif config.get("federaser_mode") == "correction":
            if self.is_target_client:
                return FitRes(
                    parameters=self.get_parameters({}).parameters,
                    status=Status(code=Code(0), message="Target client skipped correction"),
                    num_examples=0,
                    metrics={
                        "client_id": self.client_id,
                        "dataset_id": self.dataset_id,
                        "dataset_name": self.dataset_name,
                        "is_target_client": True,
                        "federaser_mode": "correction",
                        "skipped": True,
                    },
                )

            correction_epochs = max(1, int(config.get("correction_epochs", self.correction_epochs)))
            try:
                for _ in range(correction_epochs):
                    self.trainer.run_federated_train_round()
            except ValueError as e:
                logging.error(f"ValueError occurred during correction training: {e}")
            except RuntimeError as e:
                logging.error(f"RuntimeError occurred during correction training: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during correction training: {e}")
                raise

            losses = self.trainer.logger.my_fantastic_logging["train_losses"]
            loss = float(np.round(losses[-1], decimals=4)) if losses else 0.0
            return FitRes(
                parameters=self.get_parameters({}).parameters,
                status=Status(code=Code(0), message="Correction training complete"),
                num_examples=self.get_num_training_examples(),
                metrics={
                    "client_id": self.client_id,
                    "dataset_id": self.dataset_id,
                    "dataset_name": self.dataset_name,
                    "loss": loss,
                    "is_target_client": False,
                    "federaser_mode": "correction",
                    "correction_epochs": correction_epochs,
                },
            )
        else:
            # adding try catch errors
            try:
                self.trainer.run_federated_train_round()
            except ValueError as e:
                logging.error(f"ValueError occurred: {e}")
            except RuntimeError as e:
                logging.error(f"RuntimeError occurred: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                raise

            tl = np.round(
                self.trainer.logger.my_fantastic_logging["train_losses"][-1], decimals=4
            )
            fr = FitRes(
                parameters=self.get_parameters({}).parameters,
                status=Status(code=Code(0), message=""),
                num_examples=self.get_num_training_examples(),
                metrics={
                    "client_id": self.client_id,
                    "dataset_id": self.dataset_id,
                    "dataset_name": self.dataset_name,
                    "loss": float(tl),
                    "is_target_client": self.is_target_client,
                    "delta_t": self.delta_t,
                    "r": self.r,
                },
            )
            return fr

    def evaluate(self, ei):
        # We need to update to the aggregated parameters, otherwise the model will be evaluated on local weights
        self.set_parameters(ei.parameters)
        config = self.get_config(ei)

        if self.extract_fingerprint:
            if config.get("fingerprint_pass") == "stats":
                return EvaluateRes(
                    status=Status(code=Code(0), message="Fingerprint stats pass complete"),
                    loss=0.0,
                    num_examples=1,
                    metrics={
                        "client_id": self.client_id,
                        "dataset_id": self.dataset_id,
                        "dataset_name": self.dataset_name,
                    },
                )

            save_json(
                self.fingerprint,
                join(self.preprocessed_output_folder, "dataset_fingerprint.json"),
            )
            logging.info(
                f"Federated dataset fingerprint saved to {join(self.preprocessed_output_folder, 'dataset_fingerprint.json')}"
            )
            if self.plan_experiment:
                plans_identifier = plan_experiments(
                    [self.dataset_id],
                    experiment_planner_class_name=self.args.pl,
                    gpu_memory_target_in_gb=self.args.gpu_memory_target,
                    preprocess_class_name=self.args.preprocessor_name,
                    overwrite_target_spacing=self.args.overwrite_target_spacing,
                    overwrite_plans_name=self.args.overwrite_plans_name,
                )
                logging.info(f"Experiment plan created for {self.dataset_name}")
                if self.preprocess_dataset:
                    preprocess(
                        [self.dataset_id],
                        plans_identifier=plans_identifier,
                        configurations=self.args.c,
                        num_processes=self.args.np,
                        verbose=self.args.verbose,
                    )
                    logging.info(f"Dataset {self.dataset_name} preprocessed")

            return EvaluateRes(
                status=Status(code=Code(0), message="Federated fingerprint saved"),
                loss=0.0,
                num_examples=1,
                metrics={
                    "client_id": self.client_id,
                    "dataset_id": self.dataset_id,
                    "dataset_name": self.dataset_name,
                },
            )

        vl = np.round(
            self.trainer.logger.my_fantastic_logging["val_losses"][-1], decimals=4
        )
        dc = [
            np.round(i, decimals=4)
            for i in self.trainer.logger.my_fantastic_logging[
                "dice_per_class_or_region"
            ][-1]
        ]

        er = EvaluateRes(
            status=Status(code=Code(0), message="yacasi"),
            loss=float(vl),
            num_examples=len(self.trainer.dataloader_val.generator._data),
            metrics={
                "client_id": self.client_id,
                "dataset_id": self.dataset_id,
                "dataset_name": self.dataset_name,
                "fg_dice": float(np.nanmean(dc)),
            },
        )

        return er


def run_client(args, device):

    # Initialize the client and pass all nnUNet's interface arguments
    client = FlowerClient(task=args.task, args=args, device=device)

    fl.client.start_client(
        server_address=f"0.0.0.0:{args.port}",
        client=client.to_client(),  # <-- where FlowerClient is of type flwr.client.NumPyClient object
        grpc_max_message_length=2147483647,
    )

    # Clean up after federated training and perform local validation
    if args.task in ("train", "unlearn"):
        client.trainer.on_train_end()
        client.trainer.perform_actual_validation()
