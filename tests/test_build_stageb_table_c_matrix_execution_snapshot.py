import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import build_stageb_table_c_matrix_execution_snapshot as snapshot
from tools import run_stageb_matrix_validation_queue as matrix


def _unlock_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        try:
            os.chmod(path, 0o755 if path.is_dir() else 0o644)
        except OSError:
            pass
    os.chmod(root, 0o755)


class TableCMatrixExecutionSnapshotTest(unittest.TestCase):
    def _temporary_parent(self):
        temporary = tempfile.mkdtemp()
        root = Path(temporary)

        def cleanup():
            children = list(root.iterdir()) if root.exists() else []
            for child in children:
                _unlock_tree(child)
            shutil.rmtree(root, ignore_errors=False)

        self.addCleanup(cleanup)
        return root / "snapshots", root / "execution"

    @staticmethod
    def _plan_from_manifest(manifest):
        def record(source):
            return {
                "path": source["execution_path"],
                "size_bytes": source["size_bytes"],
                "mtime_ns": source["mtime_ns"],
                "sha256": source["sha256"],
            }

        evaluation = [
            record(source)
            for source in manifest["sources"]
            if "evaluation" in source["roles"]
        ]
        controller = [
            record(source)
            for source in manifest["sources"]
            if "controller" in source["roles"]
        ]
        profile_support = [
            record(source) for source in manifest["profile_support_sources"]
        ]
        runner = next(
            item
            for item in evaluation
            if item["path"].endswith("/tools/run_stageb_paper_evaluations.py")
        )
        return {
            "provenance_scope": matrix.FORMAL_PROVENANCE_SCOPE,
            "queue_id": "relocated-snapshot-fixture",
            "repository_root": manifest["execution_root"],
            "evaluation_runner": runner,
            "evaluation_sources": evaluation,
            "controller_sources": controller,
            "profile_support_sources": profile_support,
        }

    def test_relocated_contract_is_read_only_to_the_live_controller(self):
        parent, execution_parent = self._temporary_parent()
        report = snapshot.build_snapshot(
            snapshot_parent=parent, execution_parent=execution_parent
        )
        root = Path(report["snapshot_root"])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["source_count"], 77)
        self.assertEqual(report["evaluation_source_count"], 75)
        self.assertEqual(report["controller_source_count"], 12)
        self.assertEqual(report["source_overlap_count"], 10)
        self.assertEqual(report["profile_support_source_count"], 2)
        self.assertEqual(report["snapshot_file_count"], 79)
        execution_root = Path(report["execution_root"])
        self.assertTrue((execution_root / "outputs").is_symlink())
        self.assertEqual(
            (execution_root / "outputs").resolve(),
            (snapshot.REPO_ROOT / "outputs").resolve(),
        )
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=str(execution_root),
            PYTHONDONTWRITEBYTECODE="1",
            PIVOT_ARTIFACT_REPOSITORY_ROOT="/definitely/not/a/trust/root",
        )
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json
from tools import run_stageb_paper_evaluations as evaluator
from tools import run_stageb_matrix_validation_queue as queue
from tools import stageb_screen_calibration as calibration

contract = evaluator._screen_calibration_contract(evaluator.HashCache())
print(json.dumps({
    "code_root": str(evaluator.REPO_ROOT),
    "artifact_root": str(evaluator.ARTIFACT_REPOSITORY_ROOT),
    "artifact_outputs": str(evaluator.ARTIFACT_OUTPUTS_ROOT),
    "calibration_root": str(calibration.ARTIFACT_REPOSITORY_ROOT),
    "mutation_root": str(queue._require_local_mutation_root({
        "repository_root": str(queue.REPO_ROOT),
    })),
    "source": contract["source_manifest"]["path"],
    "audit": contract["source_audit"]["path"],
}, sort_keys=True))
""",
            ],
            cwd=execution_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        roots = json.loads(probe.stdout)
        self.assertEqual(roots["code_root"], str(execution_root))
        self.assertEqual(roots["artifact_root"], str(snapshot.REPO_ROOT))
        self.assertEqual(
            roots["artifact_outputs"],
            str((snapshot.REPO_ROOT / "outputs").resolve()),
        )
        self.assertEqual(roots["calibration_root"], str(snapshot.REPO_ROOT))
        self.assertEqual(roots["mutation_root"], str(execution_root))
        self.assertTrue(roots["source"].startswith(str(snapshot.REPO_ROOT / "data")))
        self.assertTrue(roots["audit"].startswith(str(snapshot.REPO_ROOT / "data")))

        matrix_spec = parent.parent / "matrix_spec.json"
        matrix_spec.write_text(
            '{"schema":"relocated-plan-test"}\n', encoding="ascii"
        )
        dry_output = parent.parent / "dry-output"
        dry_run = subprocess.run(
            [
                sys.executable,
                str(execution_root / "tools/run_stageb_paper_evaluations.py"),
                "dry-run",
                "--training-run-root",
                str(
                    snapshot.REPO_ROOT
                    / "outputs/paper_cvpr_v1/token_ablation_frozen_v2/L0/seed17"
                ),
                "--training-queue-dir",
                str(
                    snapshot.REPO_ROOT
                    / "outputs/paper_cvpr_v1/queues/"
                    "table_c_screen_l0_l4_seed17_b40_u1000_frozen_v2"
                ),
                "--profile",
                matrix.PROFILE,
                "--python",
                sys.executable,
                "--data-root",
                str(matrix.evaluator.DEFAULT_DATA_ROOT),
                "--matrix-queue-spec",
                str(matrix_spec),
                "--output-dir",
                str(dry_output),
            ],
            cwd=execution_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertFalse(dry_output.exists())
        plan = json.loads(dry_run.stdout)
        self.assertEqual(plan["repository_root"], str(execution_root))
        self.assertEqual(
            plan["artifact_repository_root"], str(snapshot.REPO_ROOT)
        )
        self.assertEqual(
            plan["artifact_outputs_root"],
            str((snapshot.REPO_ROOT / "outputs").resolve()),
        )
        self.assertTrue(
            plan["source"]["config"].startswith(
                str(snapshot.REPO_ROOT / "config")
            )
        )
        self.assertTrue(
            plan["protocol"]["screen_calibration"]["source_manifest"][
                "path"
            ].startswith(str(snapshot.REPO_ROOT / "data"))
        )
        self.assertEqual(
            plan["commands"][0]["command"][1],
            str(execution_root / "tools/eval_text_groundingdino_refcoco_tn.py"),
        )
        input_records = plan["inputs"]["records"]
        code_roles = {
            "evaluation_code_dependency",
            "source_provenance_dependency",
        }
        code_paths = [
            record["path"]
            for record in input_records
            if code_roles.intersection(record["roles"])
        ]
        self.assertEqual(len(code_paths), 75)
        self.assertTrue(
            all(path.startswith(str(execution_root)) for path in code_paths)
        )
        execution_artifacts = [
            record["path"]
            for record in input_records
            if any(
                Path(record["path"]).is_relative_to(execution_root / relative)
                for relative in ("config", "data")
            )
        ]
        self.assertEqual(execution_artifacts, [])
        manifest = json.loads((root / "snapshot.json").read_text(encoding="ascii"))
        plan = self._plan_from_manifest(manifest)
        matrix._validate_source_contract_structure(plan)
        with self.assertRaisesRegex(
            matrix.MatrixQueueError, "mutation requires execution"
        ):
            matrix._require_local_mutation_root(plan)
        queue_dir = parent.parent / "queue"
        queue_dir.mkdir()
        plan.update(
            {
                "schema": matrix.PLAN_SCHEMA,
                "queue_dir": str(queue_dir.resolve()),
            }
        )
        queue = {
            "schema": matrix.QUEUE_SCHEMA,
            "status": "waiting_training",
            "revision": 0,
            "plan": plan,
            "plan_sha256": snapshot._canonical_sha(plan),
            "predeclared_contract_sha256": "a" * 64,
            "training_attestation": None,
            "final_verification": None,
            "failure": None,
            "items": [
                {"index": index, "status": "pending"}
                for index in range(33)
            ],
        }
        (queue_dir / "queue.json").write_text(
            json.dumps(queue, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        binding = snapshot.bind_queue(root, queue_dir)
        self.assertEqual(binding["status"], "passed")
        self.assertEqual(binding["execution_root"], str(execution_root))
        self.assertEqual(
            snapshot.verify_queue_binding(queue_dir)["binding_sha256"],
            binding["binding_sha256"],
        )
        queue["status"] = "planned"
        queue["revision"] = 1
        (queue_dir / "queue.json").write_text(
            json.dumps(queue, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        self.assertEqual(
            snapshot.verify_queue_binding(queue_dir)["binding_sha256"],
            binding["binding_sha256"],
        )

        late_queue_dir = parent.parent / "late-queue"
        late_queue_dir.mkdir()
        late_plan = copy.deepcopy(plan)
        late_plan["queue_dir"] = str(late_queue_dir.resolve())
        late_queue = copy.deepcopy(queue)
        late_queue["plan"] = late_plan
        late_queue["plan_sha256"] = snapshot._canonical_sha(late_plan)
        (late_queue_dir / "queue.json").write_text(
            json.dumps(late_queue, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(
            snapshot.ExecutionSnapshotError, "pristine revision-0"
        ):
            snapshot.bind_queue(root, late_queue_dir)
        repeated = snapshot.build_snapshot(
            snapshot_parent=parent, execution_parent=execution_parent
        )
        self.assertEqual(repeated["snapshot_root"], str(root))
        self.assertEqual(repeated["snapshot_sha256"], report["snapshot_sha256"])

    def test_snapshot_source_tamper_fails_closed(self):
        parent, execution_parent = self._temporary_parent()
        report = snapshot.build_snapshot(
            snapshot_parent=parent, execution_parent=execution_parent
        )
        root = Path(report["snapshot_root"])
        manifest = json.loads((root / "snapshot.json").read_text(encoding="ascii"))
        target = root / manifest["sources"][0]["relative_path"]
        os.chmod(target, 0o644)
        target.write_bytes(target.read_bytes() + b"\n")
        with self.assertRaisesRegex(
            snapshot.ExecutionSnapshotError, "snapshot source drifted"
        ):
            snapshot.verify_snapshot(root)

    def test_snapshot_identity_is_recomputed_from_dependency_inventory(self):
        parent, execution_parent = self._temporary_parent()
        report = snapshot.build_snapshot(
            snapshot_parent=parent, execution_parent=execution_parent
        )
        with mock.patch.object(
            snapshot, "_content_identity", return_value="f" * 64
        ), self.assertRaisesRegex(
            snapshot.ExecutionSnapshotError, "content identity differs"
        ):
            snapshot.verify_snapshot(Path(report["snapshot_root"]))

    def test_publication_rejects_live_inventory_change(self):
        parent, execution_parent = self._temporary_parent()
        inventory = snapshot._inventory_from_repository(snapshot.REPO_ROOT)
        changed = copy.deepcopy(inventory)
        changed["source_overlap_count"] += 1
        with mock.patch.object(
            snapshot,
            "_inventory_from_repository",
            side_effect=[inventory, inventory, changed],
        ):
            with self.assertRaisesRegex(
                snapshot.ExecutionSnapshotError,
                "live execution closure changed",
            ):
                snapshot.build_snapshot(
                    snapshot_parent=parent,
                    execution_parent=execution_parent,
                )
        self.assertEqual(list(parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
