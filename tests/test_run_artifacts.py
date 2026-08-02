import json
import tempfile
import unittest

import torch

from fednnunet.run_artifacts import (
    commit_pending_client_checkpoint,
    get_artifact_dir,
    get_results_dir,
    get_run_dir,
    load_resume_global_checkpoint,
    resolve_client_resume_checkpoint,
    save_pending_client_checkpoint,
    save_resume_global_checkpoint,
    validate_resume_manifest,
    validate_run_id,
)


class FakeTrainer:
    def __init__(self, current_epoch):
        self.current_epoch = current_epoch

    def save_checkpoint(self, path):
        torch.save(
            {
                "current_epoch": self.current_epoch + 1,
                "network_weights": {"weight": torch.tensor([1.0])},
            },
            path,
        )


class RunArtifactsTests(unittest.TestCase):
    def test_run_layout_and_id_validation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = get_run_dir("baseline_fold0", temporary_directory)
            self.assertEqual(
                run_dir, f"{temporary_directory}/baseline_fold0"
            )
            self.assertTrue(get_artifact_dir(run_dir).endswith("baseline_fold0/artifacts"))
            self.assertTrue(get_results_dir(run_dir).endswith("baseline_fold0/nnUNet_results"))
        self.assertEqual(validate_run_id("valid.run-01"), "valid.run-01")
        with self.assertRaises(ValueError):
            validate_run_id("../escape")

    def test_resume_global_checkpoint_is_self_describing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_dir = f"{temporary_directory}/artifacts"
            state_dict = {"weight": torch.arange(3)}
            save_resume_global_checkpoint(artifact_dir, 7, 120, state_dict)

            global_round, total_rounds, restored = load_resume_global_checkpoint(
                artifact_dir
            )
            self.assertEqual(global_round, 7)
            self.assertEqual(total_rounds, 120)
            self.assertTrue(torch.equal(restored["weight"], state_dict["weight"]))
            with open(f"{artifact_dir}/run_state.json") as f:
                self.assertEqual(json.load(f)["global_round"], 7)

    def test_client_checkpoint_is_committed_only_after_server_aggregation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_dir = f"{temporary_directory}/artifacts"
            trainer = FakeTrainer(current_epoch=6)
            save_pending_client_checkpoint(trainer, artifact_dir, "301", 17)

            # The client may finish round 17 while the global model is still at 16.
            self.assertIsNone(
                resolve_client_resume_checkpoint(artifact_dir, "301", 16)
            )
            self.assertFalse(
                commit_pending_client_checkpoint(artifact_dir, "301", 16)
            )
            self.assertTrue(
                commit_pending_client_checkpoint(artifact_dir, "301", 17)
            )

            resolved = resolve_client_resume_checkpoint(artifact_dir, "301", 17)
            self.assertIsNotNone(resolved)
            checkpoint_path, metadata = resolved
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(metadata["global_round"], 17)
            self.assertEqual(metadata["current_epoch"], 6)
            self.assertEqual(checkpoint["current_epoch"], 6)
            self.assertEqual(trainer.current_epoch, 6)

    def test_resume_manifest_rejects_architecture_or_round_changes(self):
        manifest = {
            "dataset_ids": [301, 302, 303],
            "decoder": {"decoder_arch": "nnunet"},
            "total_rounds": 120,
        }
        validate_resume_manifest(manifest, dict(manifest))
        with self.assertRaisesRegex(ValueError, "decoder"):
            validate_resume_manifest(
                manifest,
                {
                    **manifest,
                    "decoder": {"decoder_arch": "effidec3d_uxnet"},
                },
            )


if __name__ == "__main__":
    unittest.main()
