import copy
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


from util import stage_b_table_b_v2_contract as contract


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tools/run_stageb_table_b_v2.py"


class TableBV2DataContractTest(unittest.TestCase):
    def _scope(self, table_b_id="D2m", seed=17):
        return contract.build_scope_binding(
            table_b_id=table_b_id,
            seed=seed,
            phase_id="joint",
            dataset_path=contract.DATASET_PATH_BY_ID[table_b_id],
            config_path=(
                REPO_ROOT
                / "config/ablations"
                / f"cfg_stageb_v24_table_b_{table_b_id.lower()}_matched.py"
            ),
            runner_path=RUNNER,
        )

    def test_exact_audit_and_both_dataset_manifests_pass(self):
        audit = contract.validate_v2_audit()
        self.assertEqual(audit["schema"], contract.AUDIT_SCHEMA)
        self.assertEqual(audit["claim_scope"], contract.CLAIM_SCOPE)
        for table_b_id in ("D2m", "D3m"):
            with self.subTest(table_b_id=table_b_id):
                dataset = contract.validate_dataset_manifest(table_b_id)
                tn = dataset["train"][-1]
                self.assertEqual(
                    tn["table_b_pair_schema"] if "table_b_pair_schema" in tn else None,
                    None,
                )
                self.assertEqual(
                    tn["paper_runtime_contract"],
                    contract.RUNTIME_DATASET_CONTRACT,
                )
                first_row = json.loads(
                    Path(tn["anno"].replace("/home/user/PIVOT", str(REPO_ROOT)))
                    .read_text(encoding="utf-8")
                    .splitlines()[0]
                )
                self.assertEqual(
                    first_row["table_b_pair_schema"], contract.TABLE_B_PAIR_SCHEMA
                )
                self.assertEqual(
                    first_row["matched_pair_schema"],
                    contract.TABLE_B_MATCHED_PAIR_SCHEMA,
                )

    def test_scope_binds_top_level_and_nested_joint_phase(self):
        scope = self._scope()
        digest = contract.canonical_sha256(scope)
        self.assertEqual(scope["phase_id"], "joint")
        self.assertEqual(scope["evidence"]["phase_id"], "joint")
        self.assertEqual(scope["claim_scope"], contract.CLAIM_SCOPE)
        self.assertEqual(
            contract.validate_scope_binding(scope, expected_sha256=digest), scope
        )

    def test_returned_audit_copy_cannot_mutate_cached_authority(self):
        first = contract.validate_v2_audit()
        first["claim_scope"]["generalization_to_unmatched_d3_parent_rows_supported"] = True
        second = contract.validate_v2_audit()
        self.assertEqual(second["claim_scope"], contract.CLAIM_SCOPE)

    def test_scope_tampering_fails_closed(self):
        scope = self._scope()
        digest = contract.canonical_sha256(scope)
        cases = []
        phase = copy.deepcopy(scope)
        phase["evidence"]["phase_id"] = "final"
        cases.append(phase)
        claim = copy.deepcopy(scope)
        claim["claim_scope"]["generalization_to_unmatched_d3_parent_rows_supported"] = True
        cases.append(claim)
        source = copy.deepcopy(scope)
        source["training_source"]["sha256"] = "0" * 64
        cases.append(source)
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(contract.TableBContractError):
                    contract.validate_scope_binding(value, expected_sha256=digest)

    def test_process_guard_rejects_training_imports_before_scope(self):
        script = r'''
import sys, types
from pathlib import Path
from util import stage_b_table_b_v2_contract as c
root = Path.cwd()
b = c.build_scope_binding(
    table_b_id="D2m", seed=17, phase_id="joint",
    dataset_path=c.DATASET_PATH_BY_ID["D2m"],
    config_path=root / "config/ablations/cfg_stageb_v24_table_b_d2m_matched.py",
    runner_path=root / "tools/run_stageb_table_b_v2.py",
)
sys.modules["engine"] = types.ModuleType("engine")
try:
    c.establish_process_scope(b, c.canonical_sha256(b))
except c.TableBContractError as error:
    assert "after training imports" in str(error), error
else:
    raise AssertionError("late scope establishment unexpectedly passed")
'''
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_scoped_alias_validates_runtime_args_and_dataset_binding(self):
        script = r'''
import runpy, sys
from pathlib import Path
from types import SimpleNamespace
from util import stage_b_table_b_v2_contract as c
root = Path.cwd()
b = c.build_scope_binding(
    table_b_id="D3m", seed=42, phase_id="joint",
    dataset_path=c.DATASET_PATH_BY_ID["D3m"],
    config_path=root / "config/ablations/cfg_stageb_v24_table_b_d3m_matched.py",
    runner_path=root / "tools/run_stageb_table_b_v2.py",
)
sha = c.canonical_sha256(b)
c.establish_process_scope(b, sha)
c.install_as_training_contract()
assert sys.modules["util.stage_b_table_b_contract"] is c
cfg = runpy.run_path(str(root / "config/ablations/cfg_stageb_v24_table_b_d3m_matched.py"))
cfg.update(
    stage_b_v19_table_b_audit=str(c.AUDIT_PATH.relative_to(root)),
    stage_b_v19_table_b_audit_sha256=c.AUDIT_SHA256,
    stage_b_v2_scope_contract_sha256=sha,
    stage_b_v2_phase_id="joint",
    stage_b_v2_profile=c.NONFORMAL_PROFILE,
    stage_b_v2_training_queue_id="none",
    stage_b_v2_training_queue_plan_sha256="none",
    stage_b_v2_formal_source_plan_sha256="none",
)
args = SimpleNamespace(**{k: v for k, v in cfg.items() if not k.startswith("__")})
dataset = c.validate_dataset_manifest("D3m")
for source in dataset["train"][:-1]:
    assert c.validate_table_b_dataset_binding(args, source) is None
bound = c.validate_table_b_dataset_binding(args, dataset["train"][-1])
assert bound.table_b_id == "D3m"
assert bound.audit_sha256 == c.AUDIT_SHA256
'''
        environment = dict(os.environ)
        environment.pop(contract.SCOPE_SHA_ENV, None)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
