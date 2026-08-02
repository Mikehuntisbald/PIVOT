import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from tools import eval_refcoco_stageb as ref_eval
from util.misc import NestedTensor


class _ForwardU0Model(nn.Module):
    def __init__(self, *, emit_u0: bool = True) -> None:
        super().__init__()
        self.stage_b_gdino_score_adapter = nn.Identity()
        self.stage_b_u0_patch_rank_adapter = nn.Identity()
        self.emit_u0 = bool(emit_u0)
        self.calls = []

    def forward(self, samples, **kwargs):
        self.calls.append(kwargs)
        batch_size = len(kwargs["captions"])
        outputs = {
            "stage_b_gdino_rank_score": torch.full((batch_size, 3), -10.0),
            "pred_logits_patch": torch.ones(batch_size, 3),
            "pred_boxes": torch.zeros(batch_size, 3, 4),
        }
        if self.emit_u0:
            outputs[ref_eval._U0_RANK_SCORE_KEY] = torch.arange(
                batch_size * 3, dtype=torch.float32
            ).reshape(batch_size, 3)
        return outputs


class _StrictU0Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.stage_b_u0_patch_rank_adapter = nn.Linear(2, 1)


class EvalRefCOCOStageBU0Test(unittest.TestCase):
    def test_u0_slot_score_is_authoritative_and_never_falls_back(self):
        cfg = types.SimpleNamespace(
            stage_b_u0_patch_rank=True,
            stage_b_gdino_score_adapter=True,
        )
        u0 = torch.tensor([[0.2, 0.8, 0.4]])
        outputs = {
            ref_eval._U0_RANK_SCORE_KEY: u0,
            "stage_b_gdino_rank_score": torch.tensor([[9.0, 1.0, 0.0]]),
            "pred_logits": torch.zeros(1, 3, 1),
        }
        observed = ref_eval._slot_scores(outputs, cfg, beta=999.0)
        self.assertTrue(torch.equal(observed, u0.unsqueeze(-1)))

        outputs.pop(ref_eval._U0_RANK_SCORE_KEY)
        with self.assertRaisesRegex(KeyError, "fallback.*forbidden"):
            ref_eval._slot_scores(outputs, cfg, beta=0.0)

    def test_u0_slot_score_rejects_wrong_shape_and_nonfinite_values(self):
        cfg = types.SimpleNamespace(stage_b_u0_patch_rank=True)
        with self.assertRaisesRegex(ValueError, r"must be a \(B,Q\) tensor"):
            ref_eval._slot_scores(
                {ref_eval._U0_RANK_SCORE_KEY: torch.zeros(1, 2, 1)},
                cfg,
                beta=0.0,
            )
        with self.assertRaisesRegex(ValueError, "only finite"):
            ref_eval._slot_scores(
                {ref_eval._U0_RANK_SCORE_KEY: torch.tensor([[float("nan")]])},
                cfg,
                beta=0.0,
            )

    def test_u0_forward_passes_support_and_disables_patch_only(self):
        model = _ForwardU0Model()
        samples = NestedTensor(
            torch.ones(2, 3, 8, 8),
            torch.zeros(2, 8, 8, dtype=torch.bool),
        )
        targets = [
            {
                "caption": caption,
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "patch": torch.ones(3, 4, 4) * row,
            }
            for row, caption in enumerate(("red car .", "small dog ."), 1)
        ]
        outputs, filtered_targets = ref_eval._forward(
            model,
            (samples, targets),
            torch.device("cpu"),
            amp=False,
            cfg=types.SimpleNamespace(stage_b_u0_patch_rank=True),
        )

        self.assertIn(ref_eval._U0_RANK_SCORE_KEY, outputs)
        self.assertEqual(len(filtered_targets), 2)
        self.assertEqual(len(model.calls), 1)
        call = model.calls[0]
        self.assertFalse(call["patch_only"])
        self.assertEqual(tuple(call["patches"].shape), (2, 3, 4, 4))
        self.assertIsNone(call["patch_global"])
        self.assertEqual(call["captions"], ["red car .", "small dog ."])

    def test_u0_forward_rejects_missing_support_or_rank_output(self):
        samples = NestedTensor(
            torch.ones(1, 3, 8, 8),
            torch.zeros(1, 8, 8, dtype=torch.bool),
        )
        cfg = types.SimpleNamespace(stage_b_u0_patch_rank=True)
        target = {
            "caption": "red car .",
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }
        with self.assertRaisesRegex(ValueError, "support-patch representation"):
            ref_eval._forward(
                _ForwardU0Model(),
                (samples, [target]),
                torch.device("cpu"),
                amp=False,
                cfg=cfg,
            )

        target["patch"] = torch.ones(3, 4, 4)
        with self.assertRaisesRegex(KeyError, "required rank key"):
            ref_eval._forward(
                _ForwardU0Model(emit_u0=False),
                (samples, [target]),
                torch.device("cpu"),
                amp=False,
                cfg=cfg,
            )

    def test_u0_dataset_keeps_support_while_old_adapter_does_not(self):
        old_cfg = types.SimpleNamespace(
            stage_b_gdino_score_adapter=True,
            stage_b_u0_patch_rank=False,
        )
        u0_cfg = types.SimpleNamespace(
            stage_b_gdino_score_adapter=True,
            stage_b_u0_patch_rank=True,
        )
        self.assertTrue(ref_eval._adapter_ref_eval_uses_no_support(old_cfg))
        self.assertFalse(ref_eval._adapter_ref_eval_uses_no_support(u0_cfg))

        data_root = Path("/tmp/pivot-u0-data")
        anno = Path("/tmp/pivot-u0-ref.jsonl")
        old_info = ref_eval._make_datasetinfo(
            data_root,
            "refcoco_val",
            anno,
            adapter_no_support=ref_eval._adapter_ref_eval_uses_no_support(old_cfg),
        )
        u0_info = ref_eval._make_datasetinfo(
            data_root,
            "refcoco_val",
            anno,
            adapter_no_support=ref_eval._adapter_ref_eval_uses_no_support(u0_cfg),
        )
        self.assertTrue(old_info["stage_b_gdino_adapter_no_support"])
        self.assertNotIn("support_patch_tsv", old_info)
        self.assertNotIn("stage_b_gdino_adapter_no_support", u0_info)
        self.assertIn("support_patch_tsv", u0_info)
        self.assertIn("support_patch_image_root", u0_info)
        self.assertEqual(u0_info["support_num_patches_min"], 1)
        self.assertEqual(u0_info["support_num_patches_max"], 1)

    def test_u0_loader_validates_head_and_strictly_loads_whole_model(self):
        model = _StrictU0Model()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        cfg = types.SimpleNamespace(
            modelname="u0-test",
            stage_b_u0_patch_rank=True,
            stage_b_gdino_score_adapter=False,
            stage_b_v11_fixed_text=False,
        )

        def build(_cfg):
            return _StrictU0Model(), None, None

        with mock.patch.object(
            ref_eval.MODULE_BUILD_FUNCS, "get", return_value=build
        ), mock.patch.object(
            ref_eval, "_torch_load_compat", return_value={"model": state}
        ):
            loaded = ref_eval._load_model(cfg, "/tmp/u0-valid.pth", torch.device("cpu"))
        self.assertIsInstance(loaded, _StrictU0Model)

        missing_base = dict(state)
        missing_base.pop("base.bias")
        with mock.patch.object(
            ref_eval.MODULE_BUILD_FUNCS, "get", return_value=build
        ), mock.patch.object(
            ref_eval, "_torch_load_compat", return_value={"model": missing_base}
        ):
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                ref_eval._load_model(
                    cfg, "/tmp/u0-missing-base.pth", torch.device("cpu")
                )

        missing_head = dict(state)
        missing_head.pop("stage_b_u0_patch_rank_adapter.weight")
        with mock.patch.object(
            ref_eval.MODULE_BUILD_FUNCS, "get", return_value=build
        ), mock.patch.object(
            ref_eval, "_torch_load_compat", return_value={"model": missing_head}
        ):
            with self.assertRaisesRegex(ValueError, "incompatible Stage-B U0"):
                ref_eval._load_model(
                    cfg, "/tmp/u0-missing-head.pth", torch.device("cpu")
                )


if __name__ == "__main__":
    unittest.main()
