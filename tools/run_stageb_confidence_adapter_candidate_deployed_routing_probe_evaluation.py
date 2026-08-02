"""Fail-closed v43 promotion binding; populated only after U400 evaluation."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / (
    "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_deployed_routing_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)


def verify_admission_report(path: Path | None = None):
    del path
    raise RuntimeError(
        "v43 formal promotion is unavailable until the immutable U400 strict1607 "
        "evaluation passes"
    )
