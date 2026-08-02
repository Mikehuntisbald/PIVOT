import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from main import (
    _apply_stage_b_v15_scorer_init,
    _restore_stage_b_v15_scorer_init_audit_for_resume,
    _sha256_file,
    _validate_stage_b_v15_scorer_init_args,
    _validate_stage_b_v15_stage_a_pretrain_state,
)


class _RecordingScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_layers = 3
        self.weight = nn.Parameter(torch.zeros(()))
        self.calls = []

    def load_from_full_text_checkpoint_state(
        self,
        state_dict,
        *,
        checkpoint_label,
        source_decoder_prefix="transformer.decoder",
    ):
        self.calls.append(
            (dict(state_dict), checkpoint_label, source_decoder_prefix)
        )
        return {
            "source_decoder_num_layers": 6,
            "selected_source_layer_indices": [3, 4, 5],
            "loaded_num_layers": 3,
            "loaded_tensor_count": 9,
            "loaded_components": [
                "decoder.layers[-N:]",
                "decoder.ref_point_head",
                "decoder.norm",
            ],
        }


class _RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone_weight = nn.Parameter(torch.tensor(4.0))
        self.stage_b_fixed_text_scorer = _RecordingScorer()


class _Logger:
    def __init__(self) -> None:
        self.messages = []

    def info(self, message) -> None:
        self.messages.append(str(message))


def _args(output_dir: Path, source: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(output_dir),
        rank=0,
        resume="",
        pretrain_model_path="stage-a.pth",
        stage_b_v11_fixed_text=True,
        stage_b_v15_decoupled_confidence=True,
        stage_b_v15_scorer_init_checkpoint=source,
    )


class StageBV15ScorerWarmStartTest(unittest.TestCase):
    def test_default_disabled_is_backward_compatible(self):
        args = _args(Path("/tmp"), source="")
        args.pretrain_model_path = None
        args.stage_b_v11_fixed_text = False
        args.stage_b_v15_decoupled_confidence = False
        _validate_stage_b_v15_scorer_init_args(args)
        self.assertIsNone(
            _apply_stage_b_v15_scorer_init(_RecordingModel(), args, None)
        )

    def test_requested_init_requires_v15_and_explicit_stage_a_base(self):
        args = _args(Path("/tmp"), source="full-text.pth")
        args.stage_b_v15_decoupled_confidence = False
        with self.assertRaisesRegex(RuntimeError, "decoupled_confidence"):
            _validate_stage_b_v15_scorer_init_args(args)

        args.stage_b_v15_decoupled_confidence = True
        args.pretrain_model_path = None
        with self.assertRaisesRegex(RuntimeError, "--pretrain_model_path"):
            _validate_stage_b_v15_scorer_init_args(args)

    def test_requested_init_rejects_scorer_state_in_stage_a_pretrain(self):
        args = _args(Path("/tmp"), source="full-text.pth")
        with self.assertRaisesRegex(RuntimeError, "scorer-free Stage-A"):
            _validate_stage_b_v15_stage_a_pretrain_state(
                args,
                {
                    "backbone.weight": torch.ones(1),
                    "stage_b_fixed_text_scorer.validity_head.weight": torch.ones(1),
                },
            )
        _validate_stage_b_v15_stage_a_pretrain_state(
            args,
            {"backbone.weight": torch.ones(1)},
        )

    def test_apply_records_sha_and_resume_does_not_reopen_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            source = output_dir / "mature_full_text.pth"
            torch.save(
                {
                    "model": {
                        "module.transformer.decoder.layers.5.weight": torch.ones(1),
                        "module.backbone.weight": torch.full((1,), 99.0),
                    }
                },
                source,
            )
            expected_sha256 = _sha256_file(source)
            model = _RecordingModel()
            base_before = model.backbone_weight.detach().clone()
            args = _args(output_dir, source=str(source))
            logger = _Logger()

            audit = _apply_stage_b_v15_scorer_init(model, args, logger)

            self.assertEqual(len(model.stage_b_fixed_text_scorer.calls), 1)
            loaded_state = model.stage_b_fixed_text_scorer.calls[0][0]
            self.assertIn("transformer.decoder.layers.5.weight", loaded_state)
            self.assertIn("backbone.weight", loaded_state)
            self.assertNotIn("module.backbone.weight", loaded_state)
            self.assertTrue(torch.equal(model.backbone_weight, base_before))
            self.assertEqual(audit["source_sha256"], expected_sha256)
            self.assertEqual(audit["resolved_source_path"], str(source.resolve()))
            self.assertEqual(args.stage_b_v15_scorer_init_audit, audit)
            sidecar = output_dir / "stage_b_v15_scorer_init_audit.json"
            self.assertEqual(json.loads(sidecar.read_text()), audit)
            self.assertTrue(any("Stage-A candidate path" in item for item in logger.messages))

            resume_checkpoint = {"args": dict(vars(args))}
            source.unlink()
            resumed_model = _RecordingModel()
            # Launchers may keep the same option on resume. The missing source
            # must only be compared lexically and must never be reopened.
            resumed_args = _args(output_dir, source=str(source))
            resumed_args.resume = "checkpoint_iter.pth"
            resumed_logger = _Logger()
            restored = _restore_stage_b_v15_scorer_init_audit_for_resume(
                resumed_model,
                resumed_args,
                resume_checkpoint,
                resumed_logger,
            )
            self.assertEqual(restored, audit)
            self.assertEqual(resumed_model.stage_b_fixed_text_scorer.calls, [])
            self.assertEqual(resumed_args.stage_b_v15_scorer_init_audit, audit)
            self.assertTrue(
                any("was not reopened" in item for item in resumed_logger.messages)
            )

    def test_resume_rejects_missing_or_changed_init_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            model = _RecordingModel()
            args = _args(output_dir, source=str(output_dir / "requested.pth"))
            args.resume = "resume.pth"
            with self.assertRaisesRegex(RuntimeError, "has no scorer warm-start audit"):
                _restore_stage_b_v15_scorer_init_audit_for_resume(
                    model,
                    args,
                    {"args": {}},
                    None,
                )

            audit = {
                "schema": "stage_b_v15_scorer_init/v1",
                "status": "applied",
                "requested_source_path": "old.pth",
                "resolved_source_path": str((output_dir / "old.pth").resolve()),
                "source_sha256": "a" * 64,
                "source_size_bytes": 10,
                "source_decoder_num_layers": 6,
                "selected_source_layer_indices": [3, 4, 5],
                "loaded_num_layers": 3,
                "loaded_tensor_count": 9,
                "loaded_components": [
                    "decoder.layers[-N:]",
                    "decoder.ref_point_head",
                    "decoder.norm",
                ],
            }
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                _restore_stage_b_v15_scorer_init_audit_for_resume(
                    model,
                    args,
                    {"args": {"stage_b_v15_scorer_init_audit": audit}},
                    None,
                )


if __name__ == "__main__":
    unittest.main()
