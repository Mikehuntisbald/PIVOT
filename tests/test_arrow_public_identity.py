from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_readme_is_arrow_u2_landing_page() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_screen = "\n".join(readme.splitlines()[:55])

    assert readme.startswith("# ARROW\n")
    assert "Responsibility-Isolated Admission, Ranking, and Abstention" in first_screen
    assert "**Paper model:** `ARROW-U2`" in first_screen
    assert "**Current paper implementation:** `U2-v5`" in first_screen
    assert "frozen B58 candidate generator" in first_screen
    assert "isolated D3 Abstention" in first_screen


def test_root_readme_does_not_embed_historical_manual() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert len(readme.splitlines()) < 250
    assert "# Patch Episode Training" not in readme
    assert "## Stage A" not in readme
    assert "## Stage B" not in readme
    assert "Historical Grounding DINO / Stage-A/B implementation notes" not in readme
    assert "docs/historical/README.md" in readme


def test_public_model_card_separates_name_from_legacy_abi() -> None:
    model_card = (ROOT / "docs/arrow_u2_model_card.md").read_text(encoding="utf-8")
    historical = (ROOT / "docs/historical/README.md").read_text(encoding="utf-8")
    legacy_manual = (
        ROOT / "docs/historical/legacy_stage_ab_readme.md"
    ).read_text(encoding="utf-8")

    assert "**ARROW-U2** is the complete sealed model" in model_card
    assert "| Complete paper model | ARROW-U2 | `U2-v5 A / A5+C3` |" in model_card
    assert "not a second model name" in model_card
    assert "do **not** define the current paper model" in historical
    assert "ARROW-U2" in historical
    assert "# Patch Episode Training" in legacy_manual
    assert "## Stage A" in legacy_manual
    assert "## Stage B" in legacy_manual
    assert len(legacy_manual.splitlines()) > 900
