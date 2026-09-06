#!/usr/bin/env python3
"""Seal MDETR assets/runtime/non-heldout parity before any formal cache forward."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.extract_mdetr_readout_cache import load_json, record, verify_record, write_json_new
from tools.mdetr_frozen_runtime import CHECKPOINT_MD5, PINNED_COMMIT, file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--environment", type=Path, required=True)
    p.add_argument("--positive-smoke", type=Path, required=True)
    p.add_argument("--negative-smoke", type=Path, required=True)
    p.add_argument("--train-source", type=Path, required=True)
    p.add_argument("--val-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if file_digest(args.protocol) != args.protocol_sha256:
        raise ValueError("protocol hash drift")
    if load_json(args.protocol).get("schema") != "arrow.confidence_readout.study_protocol/v1":
        raise ValueError("wrong study protocol")
    assets = load_json(args.assets / "download_receipt.json")
    checkpoint = assets["checkpoint"]
    verify_record(checkpoint)
    if checkpoint["md5"] != CHECKPOINT_MD5 or assets["upstream_commit"] != PINNED_COMMIT:
        raise ValueError("official checkpoint identity drift")
    environment = load_json(args.environment / "runtime_receipt.json")
    smokes = [load_json(path) for path in (args.positive_smoke, args.negative_smoke)]
    fixtures = []
    for index, smoke in enumerate(smokes):
        if smoke.get("status") != "passed" or smoke["runtime"]["checkpoint"]["sha256"] != checkpoint["sha256"]:
            raise ValueError("failed or wrong-checkpoint smoke")
        for flag in ("raw_postprocess_bitwise_parity", "native_tuple_selection_parity", "preprocess_bitwise_parity", "repeat_bitwise_parity"):
            if smoke.get(flag) is not True:
                raise ValueError("incomplete official parity probe")
        if smoke["runtime"]["state_key"] != "model_ema" or not smoke["runtime"]["strict_state_load"]:
            raise ValueError("EMA strict load not verified")
        for key in ("runtime_code", "extractor_code", "fixture"):
            verify_record(smoke[key])
        fixture = load_json(smoke["fixture"]["path"])
        if fixture["scope"] != "finecops_train_nonheldout" or fixture["kind"] != ("positive" if index == 0 else "negative_text"):
            raise ValueError("wrong non-heldout fixture")
        verify_record(fixture["image"])
        fixtures.append(fixture)
    if smokes[0]["runtime"] != smokes[1]["runtime"] or fixtures[0]["parent_positive_id"] != fixtures[1]["parent_positive_id"] or fixtures[0]["image"] != fixtures[1]["image"]:
        raise ValueError("paired smoke must share runtime and image")
    for name in ("torch", "torchvision", "transformers"):
        if smokes[0]["runtime"][name] != environment["versions"][name]:
            raise ValueError("smoke environment drift")
    sources = {}
    for split, path, count in (("train", args.train_source, 163792), ("val", args.val_source, 18455)):
        source = load_json(path)
        if source.get("split") != split or source.get("records") != count or source.get("status") != "complete":
            raise ValueError("source count/split drift")
        for key in ("annotation", "index", "extractor"):
            verify_record(source[key])
        sources[split] = {"manifest": record(path), "index": source["index"], "records": count}
    code = {name: record(ROOT / "tools" / name) for name in ("mdetr_frozen_runtime.py", "extract_mdetr_readout_cache.py", "setup_mdetr_readout_preparation.py")}
    receipt = {"schema": "arrow.confidence_readout.mdetr_asset_preflight/v1", "status": "ready_for_formal_cache",
               "study_protocol": record(args.protocol), "assets": record(args.assets / "download_receipt.json"),
               "environment": record(args.environment / "runtime_receipt.json"), "checkpoint": checkpoint,
               "sources": sources, "code": code,
               "parity": {"positive": record(args.positive_smoke), "paired_negative": record(args.negative_smoke)},
               "formal_cache_forward_performed": False, "FineCops_Test_opened": False, "gRef_forward_performed": False}
    write_json_new(args.output, receipt)
    print(record(args.output), flush=True)


if __name__ == "__main__":
    main()
