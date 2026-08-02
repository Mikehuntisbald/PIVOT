from pathlib import Path

from tools import (
    run_stageb_confidence_adapter_fulltext_global_absolute_highmem_formal as formal,
)
from tools import (
    run_stageb_confidence_adapter_fulltext_global_absolute_probe_evaluation as promotion,
)


def test_v53_formal_controller_uses_exact_promoted_config_and_output():
    assert formal.CONFIG.resolve() == promotion.FORMAL_CONFIG.resolve()
    assert formal.UPDATES == 4412
    assert formal.CHECKPOINT == formal.OUTPUT / "checkpoint_iter.pth"
    assert formal.OUTPUT == Path(
        "outputs/paper_cvpr_v1/"
        "dense_duty_adapter_fulltext_global_absolute_highmem_20260802/"
        "formal/confidence"
    ).resolve()


def test_v53_formal_controller_requires_the_probe_admission_verifier(monkeypatch):
    sentinel = {"formal_training_admitted": True}
    monkeypatch.setattr(promotion, "verify_admission_report", lambda: sentinel)

    assert formal.verify_probe_admission() is sentinel
