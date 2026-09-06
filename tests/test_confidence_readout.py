"""Synthetic readout/ownership tests, with no detector or benchmark forward."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("PyTorch is available in the frozen remote head environment")

from tools.confidence_readout import (
    CACHE_SCHEMA, GLOBAL, MDETR, MMGDINO, ROW_SCHEMA, SELECTED, SHARD_SCHEMA,
    load_readout_cache, make_readout_heads, native_labels, native_selected_index,
    parse_arm, readout_scores, reduce_dense_logits, training_arms,
)
from tools.finecops_fixed_rank_targets import make_heads, maxima_and_parity, target_loss
from tools.train_confidence_readout_heads import (
    parameter_audit, restore_rng, rng_state, train_step,
)
from tools.train_finecops_bce_l2_heads import _epoch_schedule


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, q=5):
        return (torch.randn(4, q, 256, requires_grad=True),
                torch.rand(4, q, requires_grad=True),
                torch.ones(4, q, dtype=torch.bool), torch.tensor([q-1, 0, 1, 2]))

    def test_factory_exact_legacy_rng_and_state(self):
        legacy = make_heads(17, "cpu")
        rng = torch.get_rng_state().clone()
        for localizer in (MMGDINO, MDETR):
            heads = make_readout_heads(17, "cpu", localizer)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(len(heads), 2 if localizer == MMGDINO else 4)
            for arm, model in heads.items():
                _, target = parse_arm(arm)
                self.assertTrue(all(torch.equal(v, legacy[target].state_dict()[k])
                                    for k, v in model.state_dict().items()))
                self.assertEqual(len(model.task_parameters("confidence")), 8)
                self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 50179)
            tensors = [{id(p) for p in model.parameters()} for model in heads.values()]
            self.assertTrue(all(not a.intersection(b) for i,a in enumerate(tensors) for b in tensors[i+1:]))

    def test_dense_readout_and_native_rank_parity_dynamic_q(self):
        for q in (100, 900):
            model = make_heads(42, "cpu")["emit"]
            f,n,m,j = self.fixture(q)
            output = readout_scores(model, f,n,m,j)
            self.assertTrue(torch.equal(output[GLOBAL], maxima_and_parity(model,f,n,m)))
            self.assertTrue(torch.equal(output[SELECTED], torch.zeros(4)))
            self.assertTrue(torch.equal(output["native_selected"], output[SELECTED]))
            self.assertEqual(output["confidence_logits"].shape, (4,q))
            with torch.no_grad():
                model.factorized.confidence_head.weight.fill_(.01)
            output = readout_scores(model,f,n,m,j)
            self.assertTrue(torch.equal(output[GLOBAL], output["confidence_logits"].max(1).values))
            self.assertTrue(torch.equal(output[SELECTED], output["confidence_logits"].gather(1,j[:,None]).squeeze(1)))

    def test_selected_loss_only_selected_dense_logits(self):
        logits = torch.randn(4,5,requires_grad=True)
        mask = torch.ones_like(logits,dtype=torch.bool)
        selected = torch.tensor([4,3,2,1])
        for target in ("exists", "emit"):
            scores = reduce_dense_logits(logits,mask,selected)[SELECTED]
            loss = target_loss(scores[:2],scores[2:],torch.tensor([True,False]),target)
            grad, = torch.autograd.grad(loss,logits)
            allowed = torch.zeros_like(mask).scatter(1,selected[:,None],True)
            self.assertEqual(int(torch.count_nonzero(grad[~allowed])),0)
            self.assertTrue((grad[allowed] != 0).all())

    def test_global_tie_gradient_uses_first_valid_query(self):
        logits=torch.zeros(2,4,requires_grad=True)
        mask=torch.tensor([[False,True,True,True],[True,True,True,False]])
        out=reduce_dense_logits(logits,mask,torch.tensor([2,2]))
        self.assertEqual(out["confidence_winner_index"].tolist(),[1,0])
        out[GLOBAL].sum().backward()
        self.assertEqual(torch.where(logits.grad)[1].tolist(),[1,0])

    def test_bad_mask_index_and_logits_fail(self):
        logits=torch.zeros(2,3); mask=torch.ones(2,3,dtype=torch.bool); j=torch.tensor([0,1])
        for bad in (torch.zeros_like(mask), mask.float()):
            with self.assertRaises(ValueError): reduce_dense_logits(logits,bad,j)
        for bad in (torch.tensor([-1,0]),torch.tensor([0,3]),j.float()):
            with self.assertRaises(ValueError): reduce_dense_logits(logits,mask,bad)
        mask[1,1]=False
        with self.assertRaises(ValueError): reduce_dense_logits(logits,mask,j)
        with self.assertRaises(ValueError): reduce_dense_logits(logits+float("nan"),torch.ones_like(mask),j)

    def test_mmgdino_native_tie_and_mask(self):
        row={"native_score":torch.tensor([.8,.8,.9]),"candidate_mask":torch.tensor([True,True,False])}
        self.assertEqual(native_selected_index(row,MMGDINO),0)
        row["native_selected_index"]=1
        with self.assertRaises(ValueError): native_selected_index(row,MMGDINO)
        row["native_selected_index"]=.9
        with self.assertRaises(ValueError): native_selected_index(row,MMGDINO)

    def test_mdetr_official_box_secondary_and_duplicate_tie(self):
        row={"native_score":torch.tensor([.8,.8,.8]),"candidate_mask":torch.ones(3,dtype=torch.bool),
             "boxes":torch.tensor([[.2,.2,.1,.1],[.8,.2,.1,.1],[.8,.2,.1,.1]]),"image_size":[80,100]}
        self.assertEqual(native_selected_index(row,MDETR),1)
        row["native_selected_index"]=1
        self.assertEqual(native_selected_index(row,MDETR),1)
        row["native_selected_index"]=0
        with self.assertRaises(ValueError): native_selected_index(row,MDETR)

    def test_native_labels_positive_wrong_and_no_target(self):
        base={"boxes":torch.tensor([[.2,.2,.1,.1],[.8,.8,.1,.1]]),
              "gt_boxes":torch.tensor([[.8,.8,.1,.1]]),"native_score":torch.tensor([.8,.2]),
              "candidate_mask":torch.ones(2,dtype=torch.bool)}
        wrong=dict(base,sample_id="p1",kind="positive")
        correct=dict(base,sample_id="p2",kind="positive",native_score=torch.tensor([.2,.8]))
        negative=dict(base,sample_id="n1",kind="text")
        negative.pop("gt_boxes")
        labels,indices=native_labels([wrong,correct,negative],MMGDINO)
        self.assertEqual(labels,{"p1":False,"p2":True,"n1":None})
        self.assertEqual(indices,{"p1":0,"p2":1,"n1":0})
        with self.assertRaises(ValueError): native_labels([wrong,wrong],MMGDINO)
        with self.assertRaises(ValueError): native_labels([dict(wrong,kind="image")],MMGDINO)

    def test_training_step_resume_and_frozen_inputs(self):
        f,n,m,j=self.fixture()
        y=torch.tensor([True,False])
        for readout in (GLOBAL,SELECTED):
            model=make_heads(73,"cpu")["emit"]
            rank={k:p.detach().clone() for k,p in model.named_task_parameters("rank")}
            opt=torch.optim.AdamW(model.task_parameters("confidence"),lr=1e-4,weight_decay=0,foreach=False)
            def step(model,opt):
                return train_step(model,opt,f,n,m,j,y,target="emit",readout=readout,positive_count=2)
            step(model,opt)
            payload=io.BytesIO()
            torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),
                        "rng":rng_state(torch.device("cpu"))},payload)
            payload.seek(0)
            loaded=torch.load(payload,map_location="cpu",weights_only=False)
            clone=make_heads(73,"cpu")["emit"]
            clone.load_state_dict(loaded["model"])
            opt2=torch.optim.AdamW(clone.task_parameters("confidence"),lr=1e-4,weight_decay=0,foreach=False)
            opt2.load_state_dict(loaded["optimizer"])
            before=rng_state(torch.device("cpu"))
            a=step(model,opt)
            restore_rng(before,torch.device("cpu"))
            b=step(clone,opt2)
            self.assertEqual(a,b)
            self.assertTrue(all(torch.equal(v,clone.state_dict()[k]) for k,v in model.state_dict().items()))
            parameter_audit(model,opt,rank)
            self.assertIsNone(f.grad);self.assertIsNone(n.grad)
            self.assertEqual({int(s["step"]) for s in opt.state.values()},{2})
            opt.param_groups[0]["weight_decay"]=1e-4
            with self.assertRaises(ValueError): parameter_audit(model,opt,rank)

    def test_global_step_exact_legacy_trajectory(self):
        f,n,m,j=self.fixture()
        y=torch.tensor([True,False])
        legacy=make_heads(17,"cpu")["emit"]
        candidate=make_readout_heads(17,"cpu",MDETR)["global_max__emit"]
        options=dict(lr=1e-4,weight_decay=0,foreach=False)
        oldopt=torch.optim.AdamW(legacy.task_parameters("confidence"),**options)
        newopt=torch.optim.AdamW(candidate.task_parameters("confidence"),**options)
        for _ in range(3):
            oldopt.zero_grad(set_to_none=True)
            scores=maxima_and_parity(legacy,f,n,m)
            loss=target_loss(scores[:2],scores[2:],y,"emit")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(legacy.task_parameters("confidence"),.1)
            oldopt.step()
            measured,_=train_step(candidate,newopt,f,n,m,j,y,target="emit",readout=GLOBAL,positive_count=2)
            self.assertEqual(float(loss.detach()),measured)
            self.assertTrue(all(torch.equal(value,candidate.state_dict()[key])
                                for key,value in legacy.state_dict().items()))

    def test_schedule_exact_rank_rng_and_last_batch(self):
        import numpy as np
        events,report=_epoch_schedule(seed=17,epoch=1,rank_count=83341,confidence_count=80451)
        confidence=[indices for task,indices in events if task=="confidence"]
        self.assertEqual(len(confidence),2515)
        self.assertEqual(len(confidence[-1]),3)
        rng=np.random.Generator(np.random.PCG64(170001))
        rng.permutation(83341)
        expected=rng.permutation(80451)
        self.assertTrue(np.array_equal(np.concatenate(confidence),expected))
        self.assertNotEqual(report["confidence_permutation_sha256"],
                            hashlib.sha256(np.random.default_rng(170001).permutation(80451).tobytes()).hexdigest())

    def test_cache_new_abi_hash_and_split_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            row={"schema":ROW_SCHEMA,"sample_id":"train:p:1","kind":"positive",
                 "native_selected_index":0,"query_features":torch.zeros(100,256,dtype=torch.float16),
                 "native_score":torch.zeros(100),"candidate_mask":torch.ones(100,dtype=torch.bool),
                 "boxes":torch.zeros(100,4)}
            shard=root/"shard.pt"
            torch.save({"schema":SHARD_SCHEMA,"split":"train","start":0,"rows":[row]},shard)
            manifest={"schema":CACHE_SCHEMA,"status":"complete","formal":True,"split":"train",
                      "localizer":MDETR,"records":1,"shards":[{"path":"shard.pt","rows":1,
                          "sha256":hashlib.sha256(shard.read_bytes()).hexdigest()}]}
            path=root/"manifest.json";path.write_text(json.dumps(manifest))
            rows,_=load_readout_cache(path,split="train",localizer=MDETR)
            self.assertEqual(len(rows),1)
            with self.assertRaises(ValueError): load_readout_cache(path,split="test",localizer=MDETR)
            with self.assertRaises(ValueError): load_readout_cache(path,split="train",localizer=MMGDINO)
            manifest["shards"][0]["sha256"]="0"*64;path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): load_readout_cache(path,split="train",localizer=MDETR)

    def test_mm_only_new_selected_heads(self):
        self.assertEqual(training_arms(MMGDINO),("native_selected__exists","native_selected__emit"))
        self.assertEqual(len(training_arms(MDETR)),4)
        with self.assertRaises(ValueError): training_arms("other")
        with self.assertRaises(ValueError): parse_arm("exists")


if __name__ == "__main__":
    unittest.main()
