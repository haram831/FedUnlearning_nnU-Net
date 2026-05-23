import argparse
import logging
import os
import sys
from io import BytesIO
from typing import List, Optional, Tuple, Type, Union

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
    return torch.load(bytes_io)


def parameters_to_state_dict(parameters: Parameters) -> dict:
    """Converts Flower Parameters back to a PyTorch state_dict."""
    bytes_data = parameters.tensors[0]
    return bytes_to_state_dict(bytes_data)


class FlowerClient(fl.client.Client):

    def __init__(
        self,
        task: str = "train",
        args: argparse.Namespace = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.task = task
        self.args = args
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

        self.preprocessed_output_folder = join(nnUNet_preprocessed, self.dataset_name)

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

    def get_fingerprint(self):
        if not self.local_fingerprint:
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
            )
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
            parameters = self.get_fingerprint()
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
        self.set_parameters(fi.parameters)

        if self.extract_fingerprint:
            return FitRes(
                parameters=self.get_parameters({}).parameters,
                status=Status(code=Code(0), message="Fingerprint extracted"),
                num_examples=0,
                metrics={
                    "client_id": self.client_id,
                    "dataset_id": self.dataset_id,
                    "dataset_name": self.dataset_name,
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

        if self.extract_fingerprint:
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
