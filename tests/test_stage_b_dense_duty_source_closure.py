import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from util.stage_b_dense_duty_audit import (
    build_source_closure,
    validate_formal_invocation,
    validate_source_closure,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def _minimal_repo(root: Path) -> tuple[Path, Path]:
    _write(root / "main.py", "VALUE = 'main'\n")
    _write(root / "engine.py", "VALUE = 'engine'\n")
    for directory in ("models", "datasets", "groundingdino", "util"):
        _write(root / directory / "module.py", f"VALUE = '{directory}'\n")
    common = root / "config/ablations/dense_common.py"
    _write(common, "BATCH = 16\n")
    rank = root / "config/ablations/cfg_stageb_dense_duty_rank_20260728.py"
    confidence = (
        root
        / "config/ablations/cfg_stageb_dense_duty_confidence_20260728.py"
    )
    _write(rank, "from config.ablations.dense_common import *\nPHASE = 'rank'\n")
    _write(
        confidence,
        "from config.ablations.dense_common import *\nPHASE = 'confidence'\n",
    )
    return rank, confidence


class DenseDutySourceClosureTests(unittest.TestCase):
    def test_manifest_is_deterministic_and_tracks_source_and_transitive_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rank, _confidence = _minimal_repo(root)
            first = build_source_closure(rank, repo_root=root)
            second = build_source_closure(rank, repo_root=root)
            self.assertEqual(first, second)
            self.assertEqual(validate_source_closure(first), first)
            self.assertEqual(
                [item["path"] for item in first["code"]["files"]],
                sorted(item["path"] for item in first["code"]["files"]),
            )
            self.assertEqual(
                {item["path"] for item in first["config"]["files"]},
                {
                    "config/ablations/cfg_stageb_dense_duty_rank_20260728.py",
                    "config/ablations/dense_common.py",
                },
            )

            _write(root / "models/module.py", "VALUE = 'source-drift'\n")
            source_drift = build_source_closure(rank, repo_root=root)
            self.assertNotEqual(
                first["code"]["sha256"], source_drift["code"]["sha256"]
            )
            self.assertEqual(
                first["config"]["sha256"], source_drift["config"]["sha256"]
            )

            _write(root / "config/ablations/dense_common.py", "BATCH = 8\n")
            config_drift = build_source_closure(rank, repo_root=root)
            self.assertEqual(
                source_drift["code"]["sha256"], config_drift["code"]["sha256"]
            )
            self.assertNotEqual(
                source_drift["config"]["sha256"],
                config_drift["config"]["sha256"],
            )

    def test_manifest_rejects_symlinked_or_outside_config_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            rank, _confidence = _minimal_repo(root)
            outside = Path(temporary) / "outside.py"
            _write(outside, "VALUE = 'outside'\n")
            with self.assertRaisesRegex(RuntimeError, "outside the repository"):
                build_source_closure(outside, repo_root=root)

            symlink = rank.with_name("rank_symlink.py")
            symlink.symlink_to(rank)
            with self.assertRaisesRegex(RuntimeError, "forbids symlinks"):
                build_source_closure(symlink, repo_root=root)

    def test_formal_invocation_rejects_options_and_noncanonical_phase_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rank, _confidence = _minimal_repo(root)
            valid = SimpleNamespace(
                stage_b_dense_duty_execution_scope="formal",
                stage_b_dense_duty_phase="rank",
                options=None,
                config_file=str(rank),
            )
            validate_formal_invocation(valid, repo_root=root)

            with self.assertRaisesRegex(RuntimeError, "forbids --options"):
                validate_formal_invocation(
                    SimpleNamespace(**{**vars(valid), "options": {"lr": 1e-4}}),
                    repo_root=root,
                )

            other = root / "config/ablations/other.py"
            _write(other, "PHASE = 'rank'\n")
            with self.assertRaisesRegex(RuntimeError, "exact phase config"):
                validate_formal_invocation(
                    SimpleNamespace(**{**vars(valid), "config_file": str(other)}),
                    repo_root=root,
                )

            probe = SimpleNamespace(
                **{
                    **vars(valid),
                    "stage_b_dense_duty_execution_scope": "probe",
                    "options": {"max_train_iters": 1},
                    "config_file": str(other),
                }
            )
            validate_formal_invocation(probe, repo_root=root)


if __name__ == "__main__":
    unittest.main()
