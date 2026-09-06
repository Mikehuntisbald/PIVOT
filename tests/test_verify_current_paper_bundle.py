import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"paper/scripts"))
import verify_current_paper_bundle as bundle


def fixture(tmp_path):
    generator=tmp_path/"generator.py";generator.write_text("# frozen\n")
    output=tmp_path/"table.tex";output.write_text("frozen table\n")
    sources={"analysis":{"path":"/new/checkout/result.json","sha256":"a"*64}}
    receipt={"generator":{"path":"/old/host/generator.py","sha256":bundle.v8.old.sha(generator)},
        "sources":{"analysis":{"path":"/old/host/result.json","sha256":"a"*64}},
        "outputs":{"table.tex":bundle.v8.old.sha(output)}}
    (tmp_path/"receipt.json").write_text(json.dumps(receipt))
    return generator,sources,receipt


def test_rebase_keeps_old_receipt_bytes(tmp_path):
    generator,sources,_=fixture(tmp_path);before=(tmp_path/"receipt.json").read_bytes()
    assert bundle.verify_assets(tmp_path,generator,sources)==1
    assert before==(tmp_path/"receipt.json").read_bytes()


@pytest.mark.parametrize("change",["source","generator","output","key","traversal"])
def test_drift_still_fails_closed(tmp_path,change):
    generator,sources,receipt=fixture(tmp_path)
    if change=="source":sources["analysis"]["sha256"]="b"*64
    elif change=="generator":generator.write_text("changed")
    elif change=="output":(tmp_path/"table.tex").write_text("changed")
    elif change=="key":sources["different"]=sources.pop("analysis")
    else:
        receipt["outputs"]={"../escape.tex":"a"*64}
        (tmp_path/"receipt.json").write_text(json.dumps(receipt))
    with pytest.raises((ValueError,FileNotFoundError)):bundle.verify_assets(tmp_path,generator,sources)
