import unittest
from types import SimpleNamespace

from PIL import Image

from datasets.coco import make_coco_transforms as make_patch_transforms
from datasets.odvg import make_coco_transforms as make_odvg_transforms
from util.slconfig import SLConfig


class CaptionSafeDataAugmentationTest(unittest.TestCase):
    def test_adapter_configs_disable_horizontal_flip(self):
        for path in (
            "config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py",
            "config/ablations/cfg_stageb_gdino_score_adapter_rank_fresh_o64_aspect_b32a2_u500.py",
            "config/ablations/cfg_stageb_gdino_score_adapter_dataft.py",
            "config/ablations/cfg_stageb_gdino_score_adapter_semantic_verified.py",
            "config/ablations/cfg_stageb_v15_decoupled_global_confidence.py",
            "config/ablations/cfg_stageb_v15_decoupled_gate_only_fpr95.py",
            "config/ablations/cfg_stageb_v16_gate_only_confidence.py",
            "config/ablations/cfg_stageb_v17_full_text_gate_only.py",
            "config/ablations/cfg_stageb_v18_strong_fpr_tail.py",
            "config/ablations/cfg_stageb_v19_full_text_base_plus_gate.py",
            "config/ablations/cfg_stageb_v20_acc50_aligned_hard_negatives.py",
        ):
            cfg = SLConfig.fromfile(path)
            self.assertEqual(cfg.data_aug_hflip_prob, 0.0, path)

    def test_patch_and_odvg_train_transforms_honor_zero_probability(self):
        args = SimpleNamespace(data_aug_hflip_prob=0.0)
        for factory in (make_patch_transforms, make_odvg_transforms):
            transform = factory("train", fix_size=True, args=args)
            self.assertEqual(transform.transforms[0].p, 0.0)

    def test_default_probability_remains_backward_compatible(self):
        for factory in (make_patch_transforms, make_odvg_transforms):
            transform = factory("train", fix_size=True, args=SimpleNamespace())
            self.assertEqual(transform.transforms[0].p, 0.5)

    def test_invalid_probability_fails_closed(self):
        for factory in (make_patch_transforms, make_odvg_transforms):
            with self.assertRaisesRegex(ValueError, "data_aug_hflip_prob"):
                factory(
                    "train",
                    fix_size=True,
                    args=SimpleNamespace(data_aug_hflip_prob=1.1),
                )

    def test_deterministic_train_resize_matches_val_aspect_contract(self):
        args = SimpleNamespace(
            data_aug_hflip_prob=0.0,
            data_aug_scales=[800],
            data_aug_max_size=1333,
            data_aug_train_deterministic_aspect_resize=True,
        )
        image = Image.new("RGB", (640, 480))
        for factory in (make_patch_transforms, make_odvg_transforms):
            train = factory("train", fix_size=False, strong_aug=False, args=args)
            val = factory("val", fix_size=False, strong_aug=False, args=args)
            self.assertEqual(train.transforms[1].sizes, [800])
            self.assertEqual(train.transforms[1].max_size, 1333)
            train_image, _ = train.transforms[1](image, None)
            val_image, _ = val.transforms[0](image, None)
            self.assertEqual(train_image.size, (1066, 800))
            self.assertEqual(train_image.size, val_image.size)

    def test_deterministic_train_resize_rejects_ambiguous_modes(self):
        args = SimpleNamespace(
            data_aug_train_deterministic_aspect_resize=True,
        )
        for factory in (make_patch_transforms, make_odvg_transforms):
            with self.assertRaisesRegex(ValueError, "requires fix_size=False"):
                factory("train", fix_size=True, strong_aug=False, args=args)
            with self.assertRaisesRegex(ValueError, "requires fix_size=False"):
                factory("train", fix_size=False, strong_aug=True, args=args)

    def test_deterministic_train_resize_flag_is_exact_boolean(self):
        args = SimpleNamespace(
            data_aug_train_deterministic_aspect_resize="true",
        )
        for factory in (make_patch_transforms, make_odvg_transforms):
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                factory("train", fix_size=False, strong_aug=False, args=args)


if __name__ == "__main__":
    unittest.main()
