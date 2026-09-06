"""Cache evaluator tests only; no benchmark records or detector are accessed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("remote frozen PyTorch environment required")

from tools.confidence_readout import GLOBAL, MDETR, MMGDINO, SELECTED, make_readout_heads
from tools.finecops_fixed_rank_targets import make_heads
from tools.confidence_readout_metrics import CELLS, _validate
from tools.evaluate_confidence_readout_cache import (
    SEEDS, check_legacy_parity, evaluation_groups, load_panel, row_geometry, score_groups,
    source_metadata, tensor_sha, verify_all_heads_sealed, write_jsonl,
)
from tools.train_confidence_readout_heads import bind
from tools.train_finecops_bce_l2_heads import _tensor_state_sha256


class CacheEvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def row(self, sid, annotation, kind="positive", parent=None, image=1, q=5):
        boxes=torch.tensor([[.2,.2,.1,.1],[.8,.8,.1,.1]]+[[.4,.4,.1,.1]]*(q-2))
        row={"sample_id":sid,"annotation_id":annotation,"parent_positive_id":annotation if parent is None else parent,
             "kind":kind,"cluster_image_id":image,"level":2,"query_features":torch.randn(q,256,dtype=torch.float16),
             "native_score":torch.tensor([.8,.7]+[.1]*(q-2)),"candidate_mask":torch.ones(q,dtype=torch.bool),
             "boxes":boxes,"gt_boxes":torch.tensor([[.2,.2,.1,.1]])}
        if kind != "positive":
            row.pop("gt_boxes")
        return row

    def models(self):
        out={}
        for seed in SEEDS:
            heads=make_readout_heads(int(seed),"cpu",MMGDINO)
            old=make_heads(int(seed),"cpu")
            for target,model in old.items():heads[f"{GLOBAL}__{target}"]=model
            for arm,model in heads.items():
                with torch.no_grad():
                    model.factorized.confidence_head.weight.fill_(.01 if arm.endswith("exists") else -.01)
                model.eval().requires_grad_(False)
            out[seed]=heads
        return out

    def test_val_batches_preserve_order_and_tail(self):
        rows=[{"sample_id":f"v{i}"} for i in range(18455)]
        groups=evaluation_groups(rows,{"split":"val"},"evaluation")
        self.assertEqual(len(groups),577)
        self.assertEqual(len(groups[-1]),23)
        self.assertEqual([r["sample_id"] for g in groups for r in g],[r["sample_id"] for r in rows])

    def test_gref_batches_modulo_four_and_nineteen_tail(self):
        rows=[{"sample_id":f"g{i:05d}"} for i in range(20684)]
        ids=[r["sample_id"] for r in rows]
        declared=[ids[worker::4][start:start+32] for worker in range(4)
                  for start in range(0,len(ids[worker::4]),32)]
        manifest={"split":"gref_testab","evaluation_groups":declared}
        groups=evaluation_groups(rows,manifest,"evaluation")
        self.assertEqual(sum(len(g)==19 for g in groups),4)
        self.assertEqual(len(groups),648)
        self.assertEqual([r["sample_id"] for r in groups[0]],ids[:128:4])
        manifest["evaluation_groups"][0][0],manifest["evaluation_groups"][0][1]=declared[0][1],declared[0][0]
        with self.assertRaises(ValueError):evaluation_groups(rows,manifest,"evaluation")

    def test_train_positive_only_no_pair_expansion(self):
        rows=[self.row("p1",1),self.row("n1",11,"text",1),self.row("p2",2)]
        groups=evaluation_groups(rows,{"split":"train"},"train_statistics")
        self.assertEqual([r["sample_id"] for r in groups[0]],["p1","p2"])
        with self.assertRaises(ValueError):evaluation_groups(rows,{"split":"val"},"train_statistics")
        with self.assertRaises(ValueError):evaluation_groups(rows,{"split":"train"},"evaluation")

    def test_original_parent_identity_and_difficulty(self):
        rows=[self.row("p1",1),self.row("n1",11,"text",1)]
        rows[1]["level"]=9
        metadata=source_metadata(rows,"val")
        self.assertIsNone(metadata["p1"]["parent_positive_id"])
        self.assertEqual(metadata["n1"]["parent_positive_id"],"p1")
        self.assertEqual(metadata["n1"]["parent_positive_level"],2)
        self.assertEqual(metadata["n1"]["negative_edit_level"],9)
        self.assertIsNone(metadata["n1"]["level"])
        rows[1]["cluster_image_id"]=2
        with self.assertRaises(ValueError):source_metadata(rows,"val")

    def test_no_target_does_not_require_or_invent_gt(self):
        row=self.row("testA:1:1",1,"no_target")
        row.update(split="gref_testab",stratum="testA",source_split="testA",finecops_train_val_source_disjoint=True)
        metadata=source_metadata([row],"gref_testab")
        self.assertIsNone(metadata[row["sample_id"]]["parent_positive_id"])
        geometry,ious=row_geometry(row,0)
        self.assertIsNone(ious)
        self.assertIsNone(geometry["correct"])
        self.assertIsNone(geometry["native_gt_iou"])
        row["gt_boxes"]=torch.ones(1,4)
        with self.assertRaises(ValueError):row_geometry(row,0)

    def test_all_readouts_geometry_and_analyzer_wire_shape(self):
        rows=[self.row("p1",1),self.row("p2",2),self.row("n1",11,"text",1)]
        rows[1]["gt_boxes"]=torch.tensor([[.8,.8,.1,.1]])
        metadata=source_metadata(rows,"val")
        models=self.models()
        before={seed:{arm:{k:v.clone() for k,v in model.state_dict().items()} for arm,model in heads.items()}
                for seed,heads in models.items()}
        result=score_groups(models,[rows],metadata,localizer=MMGDINO,mode="evaluation",device=torch.device("cpu"))
        _validate({MMGDINO:result},SEEDS)
        for seed in SEEDS:
            self.assertEqual([r["correct"] for r in result[seed]],[True,False,None])
            for record in result[seed]:
                self.assertEqual(set(record["scores"]),set(CELLS))
                self.assertEqual(set(record["readout_diagnostics"]),set(CELLS))
                for arm,d in record["readout_diagnostics"].items():
                    self.assertGreaterEqual(d["max_logit"],d["selected_logit"])
                    read="max_logit" if arm.startswith(GLOBAL) else "selected_logit"
                    self.assertEqual(record["scores"][arm],d[read])
                    self.assertEqual(d["native_selected_index"],record["native_selected_index"])
                    if record["kind"] != "positive":self.assertIsNone(d["winner_gt_iou"])
            for arm,model in models[seed].items():
                self.assertTrue(all(torch.equal(value,before[seed][arm][key]) for key,value in model.state_dict().items()))

    def test_train_statistics_scores_only_global_exists(self):
        rows=[self.row("p1",1),self.row("p2",2)]
        result=score_groups(self.models(),[rows],source_metadata(rows,"train"),localizer=MMGDINO,
                            mode="train_statistics",device=torch.device("cpu"))
        for records in result.values():
            for record in records:
                self.assertEqual(set(record["scores"]),{"global_max__exists"})
                self.assertNotIn("correct",record)
                self.assertNotIn("readout_diagnostics",record)

    def test_legacy_global_parity_rejects_single_float_drift(self):
        rows=[self.row("p1",1)]
        record=score_groups(self.models(),[rows],source_metadata(rows,"val"),localizer=MMGDINO,
                            mode="evaluation",device=torch.device("cpu"))["17"][0]
        reference={"sample_id":"p1","correct":True,"baseline_score":record["scores"]["global_max__exists"],
                   "candidate_score":record["scores"]["global_max__emit"]}
        check_legacy_parity(record,reference,"17")
        reference["baseline_score"]+=1e-15
        with self.assertRaises(ValueError):check_legacy_parity(record,reference,"17")

    def test_legacy_gref_native_geometry_hash_is_required(self):
        row=self.row("p1",1)
        record=score_groups(self.models(),[[row]],source_metadata([row],"val"),localizer=MMGDINO,
                            mode="evaluation",device=torch.device("cpu"))["17"][0]
        old={"sample_id":"p1","correct":True,"scores":{"17":{"exists":record["scores"]["global_max__exists"],
              "emit":record["scores"]["global_max__emit"]}},"native_score":record["native_score"],
              "native_top1_query":record["native_selected_index"],"native_box":record["native_box"],
              "native_iou":record["native_gt_iou"],"boxes_sha256":record["boxes_sha256"],
              "candidate_mask_sha256":record["candidate_mask_sha256"]}
        check_legacy_parity(record,old,"17")
        old["boxes_sha256"]="0"*64
        with self.assertRaises(ValueError):check_legacy_parity(record,old,"17")

    def test_append_only_jsonl_and_finite_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"rows.jsonl"
            binding=write_jsonl(path,[{"sample_id":"x","score":.1}])
            self.assertEqual(binding["path"],str(path))
            self.assertEqual(json.loads(path.read_text())["score"],.1)
            with self.assertRaises(ValueError):write_jsonl(path,[])
            with self.assertRaises(ValueError):write_jsonl(Path(tmp)/"bad.jsonl",[{"score":float("nan")}])

    def test_complete_checkpoint_panel_and_locked_legacy_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def put(name,value):
                path=root/name
                path.write_text(json.dumps(value))
                return bind(path)
            old_bindings={}
            for seed in SEEDS:
                old=make_heads(int(seed),"cpu")
                path=root/f"legacy{seed}.pt"
                torch.save({"schema":"arrow.fixed_rank_targets.checkpoint/v1","epoch":5,
                            "seed":int(seed),"updates":12575,
                            "models":{target:model.state_dict() for target,model in old.items()}},path)
                old_bindings[seed]=bind(path)
            protocol=put("protocol.json",{"schema":"arrow.confidence_readout.study_protocol/v1",
                "localizers":{MMGDINO:{"reused_global_heads":{
                    seed:{"checkpoint":old_bindings[seed]} for seed in SEEDS}}}})
            panel={"schema":"arrow.confidence_readout.checkpoint_panel/v1","localizer":MMGDINO,
                   "study_protocol":protocol,"seeds":{}}
            for seed in SEEDS:
                heads=make_readout_heads(int(seed),"cpu",MMGDINO)
                design=put(f"design{seed}.json",{"study_protocol":protocol,"checkpoint":{"sha256":"parent"},
                                               "caches":{"train_cache":{},"val_cache":{}}})
                path=root/f"new{seed}.pt"
                torch.save({"schema":"arrow.confidence_readout.head_checkpoint/v1","epoch":5,"updates":12575,
                    "seed":int(seed),"localizer":MMGDINO,"design":design,
                    "models":{arm:model.state_dict() for arm,model in heads.items()}},path)
                ownership={arm:{"frozen_rank_sha256":_tensor_state_sha256(dict(model.named_task_parameters("rank"))),
                                "confidence_sha256":_tensor_state_sha256(dict(model.named_task_parameters("confidence")))}
                           for arm,model in heads.items()}
                post=put(f"post{seed}.json",{"schema":"arrow.confidence_readout.training_postflight/v1",
                    "status":"complete","seed":int(seed),"localizer":MMGDINO,"updates_per_head":12575,
                    "checkpoint":bind(path),"design":design,"ownership":ownership})
                panel["seeds"][seed]={"postflight":post,"readout":bind(path),"legacy_global":old_bindings[seed]}
            path=root/"panel.json";path.write_text(json.dumps(panel))
            _,models,legacy,_=load_panel(path,root/"protocol.json",MMGDINO,torch.device("cpu"))
            self.assertEqual(set(models),set(SEEDS))
            self.assertFalse(legacy)
            self.assertTrue(all(set(heads)==set(CELLS) for heads in models.values()))
            self.assertTrue(all(not p.requires_grad for heads in models.values() for model in heads.values() for p in model.parameters()))
            panel["seeds"]["17"]["legacy_global"]=old_bindings["42"]
            path.write_text(json.dumps(panel))
            with self.assertRaises(ValueError):load_panel(path,root/"protocol.json",MMGDINO,torch.device("cpu"))


if __name__=="__main__":
    unittest.main()
