"""Pure-Python tests for the append-only diagnostic metadata erratum."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.reconcile_confidence_readout_metadata import (
    ALLOWED, REVISION, ROW_SCHEMA, canonical, correct_record, official_lookup,
    parse_json, protected_projection, reconcile_rows, write_new_records,
)


class MetadataErratumTest(unittest.TestCase):
    def fixtures(self):
        annotation={"annotations":[{"id":1,"level":3},
            {"id":2,"level":1,"positive_id":1,"negative_type":"replace"},
            {"id":3,"level":1,"positive_id":1,"negative_type":"order","negative_level":2},
            {"id":4,"level":1,"positive_id":1,"negative_type":"order","negative_level":1}]}
        index=[{"sample_id":f"finecops-val:{aid}","annotation_id":aid,"kind":"positive" if aid==1 else "text",
                "level":3 if aid==1 else 1,"cluster_image_id":"99","parent_positive_id":1} for aid in range(1,5)]
        rows=[{"sample_id":r["sample_id"],"cluster_id":"99","kind":r["kind"],"level":3 if r["kind"]=="positive" else None,
               "parent_positive_id":None if r["kind"]=="positive" else "finecops-val:1",
               "parent_positive_level":None if r["kind"]=="positive" else 3,
               "negative_edit_level":None if r["kind"]=="positive" else 1,"split":"val","stratum":"validation",
               "scores":{"global_max__exists":-0.0,"native_selected__emit":.12345678901234566},
               "readout_diagnostics":{"global_max__exists":{"native_selected_index":2,"winner_gt_iou":None}},
               "native_score":.9,"correct":True if r["kind"]=="positive" else None} for r in index]
        return annotation,index,rows

    def test_real_semantics_distinguish_missing_one_two_and_raw_level(self):
        annotation,index,rows=self.fixtures()
        fixed,audit=reconcile_rows(rows,official_lookup(annotation,index))
        self.assertEqual([r["negative_edit_level"] for r in fixed],[None,None,2,1])
        self.assertEqual([r["raw_annotation_level"] for r in fixed],[3,1,1,1])
        self.assertEqual(audit["changed_edit_level_records"],2)
        self.assertTrue(all(r["schema"]==ROW_SCHEMA and r["metadata_revision"]==REVISION for r in fixed))

    def test_every_other_field_and_float_bits_identical(self):
        annotation,index,rows=self.fixtures()
        before=copy.deepcopy(rows)
        fixed,audit=reconcile_rows(rows,official_lookup(annotation,index))
        self.assertEqual(rows,before)
        for old,new in zip(rows,fixed):
            self.assertEqual(canonical(protected_projection(old)),canonical(protected_projection(new)))
            self.assertEqual(old["level"],new["level"])
            self.assertEqual(old["parent_positive_level"],new["parent_positive_level"])
            for name in old["scores"]:
                self.assertEqual(struct.pack("!d",old["scores"][name]),struct.pack("!d",new["scores"][name]))
            changed={key for key in set(old)|set(new) if key not in old or key not in new or old[key]!=new[key]}
            self.assertTrue(changed.issubset(ALLOWED))
        self.assertTrue(audit["all_other_fields_canonical_bitwise_identical"])

    def test_no_inference_of_missing_edit_level(self):
        annotation,index,rows=self.fixtures()
        source=official_lookup(annotation,index)
        self.assertIsNone(correct_record(rows[1],source)["negative_edit_level"])
        self.assertEqual(correct_record(rows[1],source)["parent_positive_level"],3)

    def test_already_corrected_and_unknown_pattern_rejected(self):
        annotation,index,rows=self.fixtures();source=official_lookup(annotation,index)
        fixed=correct_record(rows[2],source)
        with self.assertRaises(ValueError):correct_record(fixed,source)
        bad=copy.deepcopy(rows[2]);bad["negative_edit_level"]=2
        with self.assertRaises(ValueError):correct_record(bad,source)
        bad=copy.deepcopy(rows[2]);bad["parent_positive_level"]=1
        with self.assertRaises(ValueError):correct_record(bad,source)

    def test_duplicate_missing_scope_and_parent_rejected(self):
        annotation,index,rows=self.fixtures();source=official_lookup(annotation,index)
        with self.assertRaises(ValueError):reconcile_rows(rows[:-1],source)
        with self.assertRaises(ValueError):reconcile_rows(rows+[rows[0]],source)
        bad=copy.deepcopy(index);bad[1]["cluster_image_id"]="100"
        with self.assertRaises(ValueError):official_lookup(annotation,bad)
        bad=copy.deepcopy(index);bad[0]["sample_id"]="finecops-test:1"
        with self.assertRaises(ValueError):official_lookup(annotation,bad)
        bad=copy.deepcopy(annotation);bad["annotations"][2]["negative_level"]=True
        with self.assertRaises(ValueError):official_lookup(bad,index)

    def test_strict_json_and_append_only_export(self):
        with self.assertRaises(ValueError):parse_json('{"x":1,"x":2}')
        with self.assertRaises(ValueError):parse_json('{"x":NaN}')
        annotation,index,rows=self.fixtures();fixed,_=reconcile_rows(rows,official_lookup(annotation,index))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"new.jsonl";write_new_records(path,fixed)
            loaded=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(loaded,fixed)
            with self.assertRaises(FileExistsError):write_new_records(path,fixed)


if __name__ == "__main__":unittest.main()
