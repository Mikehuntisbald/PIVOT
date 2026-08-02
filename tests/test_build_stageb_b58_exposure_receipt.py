import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import torch

from tools import build_stageb_b58_exposure_receipt as exposure


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_payload(
    *,
    step=49_539.0,
    state_count=661,
    epoch=1,
    iteration=0,
    epoch_finished=True,
    batch_size=19,
    epochs=2,
    world_size=1,
    distributed=False,
    rank=0,
    local_rank=0,
):
    return {
        "epoch": epoch,
        "iteration": iteration,
        "epoch_finished": epoch_finished,
        "args": {
            "batch_size": batch_size,
            "epochs": epochs,
            "world_size": world_size,
            "distributed": distributed,
            "rank": rank,
            "local_rank": local_rank,
        },
        "optimizer": {
            "state": {
                index: {"step": torch.tensor(step)}
                for index in range(state_count)
            }
        },
    }


class ExposureFixture:
    def __init__(self, root: Path) -> None:
        self.repo = root / "gdino"
        self.train = (
            self.repo
            / "outputs/gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch"
        )
        self.train.mkdir(parents=True)
        self.receipt = root / "pivot/outputs/paper_cvpr_v1/attestations/receipt.json"
        torch.save(
            _checkpoint_payload(),
            self.train / "checkpoint0001.pth",
        )
        torch.save(
            _checkpoint_payload(step=24_766.0, epoch=0, epochs=1),
            self.train / "checkpoint0000.pth",
        )
        torch.save(
            _checkpoint_payload(
                step=49_257.0,
                iteration=24_500,
                epoch_finished=False,
            ),
            self.train / "checkpoint_iter.pth",
        )
        (self.train / "config_args_all.json").write_text(
            json.dumps(
                {
                    "batch_size": 19,
                    "epochs": 2,
                    "world_size": 1,
                    "distributed": False,
                    "rank": 0,
                    "local_rank": 0,
                    "options": {"batch_size": 19, "epochs": 2},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (self.train / "config_cfg.py").write_text(
            "batch_size = 19\nepochs = 2\n",
            encoding="utf-8",
        )
        rows = [
            {
                "train_lr": 0.1,
                "train_loss": 2.0,
                "now_time": "t0",
                "epoch_time": "1:00",
            },
            {
                "train_lr": 0.1,
                "train_loss": 1.0,
                "now_time": "t1",
                "epoch_time": "1:00",
            },
        ]
        (self.train / "log.txt").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        (self.repo / "main.py").write_text(
            "if not args.eval:\n"
            "    batch_sampler_train = torch.utils.data.BatchSampler(\n"
            "        sampler_train, args.batch_size, drop_last=True)\n",
            encoding="utf-8",
        )
        self.refresh_hashes()

    def refresh_hashes(self) -> None:
        self.hashes = {
            "checkpoint0001": _sha256(self.train / "checkpoint0001.pth"),
            "checkpoint0000": _sha256(self.train / "checkpoint0000.pth"),
            "checkpoint_iter": _sha256(self.train / "checkpoint_iter.pth"),
            "config_args": _sha256(self.train / "config_args_all.json"),
            "config_source": _sha256(self.train / "config_cfg.py"),
            "epoch_records": _sha256(self.train / "log.txt"),
            "training_main": _sha256(self.repo / "main.py"),
        }

    def patches(self):
        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(exposure, "BASELINE_REPO_ROOT", self.repo)
        )
        stack.enter_context(
            mock.patch.object(exposure, "BASELINE_TRAIN_ROOT", self.train)
        )
        stack.enter_context(
            mock.patch.object(exposure, "CANONICAL_RECEIPT_PATH", self.receipt)
        )
        stack.enter_context(mock.patch.object(exposure, "LOCKED_SHA256", self.hashes))
        return stack


class BuildStageBB58ExposureReceiptTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = ExposureFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def _replace_checkpoint(self, name: str, payload: dict) -> None:
        torch.save(payload, self.fixture.train / f"{name}.pth")
        self.fixture.refresh_hashes()

    def test_derives_exact_validator_contract_without_writing(self):
        with self.fixture.patches():
            payload = exposure._derive_receipt_payload()
        self.assertFalse(self.fixture.receipt.exists())
        self.assertEqual(payload["schema"], exposure.SCHEMA)
        self.assertEqual(payload["status"], "verified")
        self.assertIs(payload["derived_from_hashed_optimizer_states"], True)
        self.assertEqual(
            payload["baseline"],
            {
                "checkpoint_sha256": self.fixture.hashes["checkpoint0001"],
                "batch_size": 19,
                "optimizer_updates": 49_539,
                "optimizer_state_count": 661,
                "successful_update_batch_slots": 941_241,
                "derivation": "checkpoint0001_optimizer_step49539_x_batch19",
                "drop_last": True,
                "matching_unit": "successful_optimizer_update_global_batch_slots",
            },
        )
        self.assertEqual(
            payload["candidate"],
            {
                "id": "M0",
                "architecture_objective": "S2F",
                "batch_size": 40,
                "optimizer_updates": 23_532,
                "successful_update_batch_slots": 941_280,
                "derivation": "23532_optimizer_updates_x_batch40",
            },
        )
        self.assertEqual(
            payload["candidate_minus_baseline_successful_update_batch_slots"],
            39,
        )
        self.assertEqual(
            payload["scope_limitations"],
            {
                "total_consumed_batch_slots_derived": False,
                "flops_matched": False,
                "wall_clock_compute_matched": False,
                "statement": (
                    "This receipt matches batch slots attached to successful optimizer "
                    "updates. Optimizer state does not count AMP-skipped attempted "
                    "batches, total consumed samples, FLOPs, or wall-clock compute."
                ),
            },
        )
        self.assertEqual(set(payload["evidence"]), exposure.EVIDENCE_KEYS)
        without_hash = dict(payload)
        receipt_sha256 = without_hash.pop("receipt_sha256")
        self.assertEqual(receipt_sha256, exposure.canonical_json_sha256(without_hash))

    def test_build_is_fresh_only_and_verify_replays_inputs(self):
        with self.fixture.patches():
            built = exposure.build_receipt()
            verified = exposure.verify_receipt()
            with self.assertRaisesRegex(exposure.ExposureReceiptError, "already exists"):
                exposure.build_receipt()
        self.assertEqual(built, verified)

    def test_wrong_receipt_path_is_rejected(self):
        with self.fixture.patches():
            exposure.build_receipt()
            with self.assertRaisesRegex(exposure.ExposureReceiptError, "not canonical"):
                exposure.verify_receipt(self.fixture.receipt.with_name("other.json"))

    def test_tampered_self_hash_is_rejected_before_replay(self):
        with self.fixture.patches():
            exposure.build_receipt()
            value = json.loads(self.fixture.receipt.read_text(encoding="utf-8"))
            value[
                "candidate_minus_baseline_successful_update_batch_slots"
            ] = 40
            self.fixture.receipt.write_text(json.dumps(value), encoding="utf-8")
            with mock.patch.object(
                exposure, "_derive_receipt_payload"
            ) as replay, self.assertRaisesRegex(
                exposure.ExposureReceiptError, "self-hash"
            ):
                exposure.verify_receipt()
            replay.assert_not_called()

    def test_checkpoint_hash_mismatch_prevents_pickle_load(self):
        path = self.fixture.train / "checkpoint0001.pth"
        self.fixture.hashes["checkpoint0001"] = "0" * 64
        with self.fixture.patches(), mock.patch(
            "torch.load"
        ) as torch_load, self.assertRaisesRegex(
            exposure.ExposureReceiptError, "sha256 drifted"
        ):
            exposure._inspect_checkpoint("checkpoint0001", path)
        torch_load.assert_not_called()

    def test_checkpoint_epoch_and_args_are_strict(self):
        bad = _checkpoint_payload(epoch=True)
        self._replace_checkpoint("checkpoint0001", bad)
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "epoch is bool"
        ):
            exposure._derive_receipt_payload()

        bad = _checkpoint_payload(batch_size=True)
        self._replace_checkpoint("checkpoint0001", bad)
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "batch_size is bool"
        ):
            exposure._derive_receipt_payload()

    def test_global_batch_slot_requires_single_process_checkpoint_args(self):
        for field, value in (
            ("world_size", 2),
            ("distributed", True),
            ("rank", 1),
            ("local_rank", 1),
        ):
            with self.subTest(field=field):
                bad = _checkpoint_payload(**{field: value})
                self._replace_checkpoint("checkpoint0001", bad)
                with self.fixture.patches(), self.assertRaisesRegex(
                    exposure.ExposureReceiptError, "not the locked single-process run"
                ):
                    exposure._derive_receipt_payload()

        bad = _checkpoint_payload()
        bad["args"]["gpu"] = 1
        self._replace_checkpoint("checkpoint0001", bad)
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "gpu drifted"
        ):
            exposure._derive_receipt_payload()

    def test_global_batch_slot_requires_single_process_config_args(self):
        path = self.fixture.train / "config_args_all.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["world_size"] = 2
        path.write_text(json.dumps(value), encoding="utf-8")
        self.fixture.refresh_hashes()
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "not the locked single-process run"
        ):
            exposure._derive_receipt_payload()

    def test_missing_or_mixed_optimizer_state_fails_closed(self):
        missing = _checkpoint_payload()
        missing["optimizer"]["state"][0] = {}
        self._replace_checkpoint("checkpoint0001", missing)
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "has no step"
        ):
            exposure._derive_receipt_payload()

        mixed = _checkpoint_payload()
        mixed["optimizer"]["state"][0]["step"] = torch.tensor(49_538.0)
        self._replace_checkpoint("checkpoint0001", mixed)
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "mixed steps"
        ):
            exposure._derive_receipt_payload()

    def test_bool_and_nonfinite_optimizer_steps_fail_closed(self):
        bad_bool = _checkpoint_payload(step=True)
        self._replace_checkpoint("checkpoint0001", bad_bool)
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "is bool"
        ):
            exposure._derive_receipt_payload()

        bad_nan = _checkpoint_payload(step=float("nan"))
        self._replace_checkpoint("checkpoint0001", bad_nan)
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "not finite"
        ):
            exposure._derive_receipt_payload()

    def test_all_three_checkpoint_steps_are_enforced(self):
        bad = _checkpoint_payload(step=49_256.0, iteration=24_500, epoch_finished=False)
        self._replace_checkpoint("checkpoint_iter", bad)
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "optimizer_step drifted"
        ):
            exposure._derive_receipt_payload()

    def test_checkpoint_load_uses_hashed_descriptor_and_rejects_path_replacement(self):
        path = self.fixture.train / "checkpoint0001.pth"
        backup = path.with_suffix(".original.pth")
        real_torch_load = torch.load

        def replace_path_then_load(source, *args, **kwargs):
            self.assertTrue(hasattr(source, "fileno"))
            path.rename(backup)
            torch.save(_checkpoint_payload(step=1.0), path)
            return real_torch_load(source, *args, **kwargs)

        with self.fixture.patches(), mock.patch(
            "torch.load", side_effect=replace_path_then_load
        ), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "path no longer references"
        ):
            exposure._inspect_checkpoint("checkpoint0001", path)

    def test_checkpoint_second_descriptor_hash_detects_in_place_change(self):
        path = self.fixture.train / "checkpoint0001.pth"
        real_torch_load = torch.load

        def load_then_append(source, *args, **kwargs):
            self.assertTrue(hasattr(source, "fileno"))
            value = real_torch_load(source, *args, **kwargs)
            with path.open("ab") as handle:
                handle.write(b"changed-after-load")
            return value

        with self.fixture.patches(), mock.patch(
            "torch.load", side_effect=load_then_append
        ), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "changed across trusted checkpoint load"
        ):
            exposure._inspect_checkpoint("checkpoint0001", path)

    def test_config_args_and_locked_config_source_are_enforced(self):
        config_args = self.fixture.train / "config_args_all.json"
        config_args.write_text(
            json.dumps(
                {
                    "batch_size": 18,
                    "epochs": 2,
                    "world_size": 1,
                    "distributed": False,
                    "rank": 0,
                    "local_rank": 0,
                    "options": {"batch_size": 19, "epochs": 2},
                }
            ),
            encoding="utf-8",
        )
        self.fixture.refresh_hashes()
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "contract drifted"
        ):
            exposure._derive_receipt_payload()

        config_args.write_text(
            json.dumps(
                {
                    "batch_size": 19,
                    "epochs": 2,
                    "world_size": 1,
                    "distributed": False,
                    "rank": 0,
                    "local_rank": 0,
                    "options": {"batch_size": 19, "epochs": 2},
                }
            ),
            encoding="utf-8",
        )
        self.fixture.refresh_hashes()
        self.fixture.hashes["config_source"] = "f" * 64
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "config_source sha256 drifted"
        ):
            exposure._derive_receipt_payload()

    def test_epoch_log_must_be_two_finite_jsonl_rows(self):
        log_path = self.fixture.train / "log.txt"
        log_path.write_text(
            '{"train_lr":0.1,"train_loss":NaN,"now_time":"t","epoch_time":"e"}\n'
            '{"train_lr":0.1,"train_loss":1,"now_time":"t","epoch_time":"e"}\n',
            encoding="utf-8",
        )
        self.fixture.refresh_hashes()
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "non-finite JSON constant"
        ):
            exposure._derive_receipt_payload()

    def test_ast_requires_literal_true_under_training_guard(self):
        main_path = self.fixture.repo / "main.py"
        main_path.write_text(
            "if not args.eval:\n"
            "    batch_sampler_train = torch.utils.data.BatchSampler(\n"
            "        sampler_train, args.batch_size, drop_last=1)\n",
            encoding="utf-8",
        )
        self.fixture.refresh_hashes()
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "drop_last is not exact True"
        ):
            exposure._derive_receipt_payload()

        main_path.write_text(
            "batch_sampler_train = torch.utils.data.BatchSampler(\n"
            "    sampler_train, args.batch_size, drop_last=True)\n",
            encoding="utf-8",
        )
        self.fixture.refresh_hashes()
        with self.fixture.patches(), self.assertRaisesRegex(
            exposure.ExposureReceiptError, "not guarded"
        ):
            exposure._derive_receipt_payload()


if __name__ == "__main__":
    unittest.main()
