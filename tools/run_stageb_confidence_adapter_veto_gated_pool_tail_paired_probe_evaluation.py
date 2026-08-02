"""Fail closed until the v18 U300 promotion surface is sealed."""

from pathlib import Path


REPORT = (
    Path(__file__).resolve().parents[1]
    / "outputs/paper_cvpr_v1/dense_duty_adapter_veto_gated_pool_tail_paired_highmem_20260731/probe_evaluation/u000300_strict1607_report.json"
)


def verify_admission_report(*_args, **_kwargs):
    raise RuntimeError("v18 U300 promotion surface is not sealed")
