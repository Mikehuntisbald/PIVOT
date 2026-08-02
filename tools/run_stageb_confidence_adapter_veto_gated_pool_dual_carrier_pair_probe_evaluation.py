"""Fail-closed placeholder until the v10 U300 promotion surface is sealed."""


def verify_admission_report(*_args, **_kwargs):
    raise RuntimeError("v10 U300 promotion surface is not sealed")
