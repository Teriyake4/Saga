import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[2]/"tools"/"m0_quality.py"
spec=importlib.util.spec_from_file_location("m0_quality",SCRIPT)
assert spec is not None and spec.loader is not None
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
q=module

def prereg():
    return {"freeze_record":{"review":"frozen"},"baseline":{"checkpoint_revision":"rev1","files":[{"path":"weights.bin","sha256":"a"*64}],"precision":"BF16","harness_revision":"h1","environment":{"host":"declared"}},"dataset":{"revision":"ds1","sha256":"b"*64},"calibration_ids":["cal"],"calibration_content_sha256":["c"*64],"heldout_ids":["x","y"],"metrics":{"primary":"token_weighted_mean_nll","direction":"lower"},"confidence":{"method":"paired_percentile_bootstrap","level":0.95,"replicates":100},"seeds":[4],"sample_count":2,"stopping_rule":"fixed-full-heldout", "budget_approval":{"approved":True,"reviewer":"reviewer","date":"2026-01-01","baseline_evidence":"approved baseline evidence"},"regression_budget":{"max_upper_nll_regression":1.0}}

def records():
    base=[]; var=[]
    for ident,content,toks,nll,delta in [("x","held out x",2,2.,1.),("y","held out y",1,3.,0.)]:
        common={"id":ident,"content":content,"context_tokens":6,"token_count":toks,"seed":4}
        base.append({**common,"nll_sum":nll}); var.append({**common,"nll_sum":nll+delta})
    return base,var

class QualityTests(unittest.TestCase):
    def test_known_answer_and_deterministic_ci(self):
        p=prereg(); a,b=records(); result=q._stats(*q._validate_records(a,b,p),p)
        self.assertAlmostEqual(result["baseline_mean_nll"],5/3)
        self.assertAlmostEqual(result["variant_mean_nll"],2.)
        self.assertAlmostEqual(result["baseline_perplexity"],math.exp(5/3))
        self.assertEqual(result["paired_bootstrap_ci"],q._stats(*q._validate_records(a,b,p),p)["paired_bootstrap_ci"])
    def test_split_overlap_and_exact_content_leakage(self):
        p=prereg(); p["calibration_ids"].append("x")
        with self.assertRaises(q.QualityError): q.validate_prereg(p)
        p=prereg(); p["calibration_content_sha256"]=[hashlib.sha256(b"held out x").hexdigest()]
        a,b=records()
        with self.assertRaisesRegex(q.QualityError,"leaks"): q._validate_records(a,b,p)
    def test_pair_mismatch_and_missing_row_fail(self):
        p=prereg(); a,b=records(); b[0]["token_count"]=3
        with self.assertRaisesRegex(q.QualityError,"token_count mismatch"): q._validate_records(a,b,p)
        a,b=records(); b.pop()
        with self.assertRaises(q.QualityError): q._validate_records(a,b,p)
    def test_sample_count_must_match_full_heldout_set(self):
        p=prereg(); p["sample_count"]=999
        with self.assertRaisesRegex(q.QualityError,"sample_count"):
            q.validate_prereg(p)
    def test_nonfinite_aggregate_and_perplexity_fail_closed(self):
        p=prereg(); a,b=records()
        a[0]["nll_sum"]=1e308; a[1]["nll_sum"]=1e308
        with self.assertRaisesRegex(q.QualityError,"baseline NLL sum"):
            q._stats(*q._validate_records(a,b,p),p)
        a,b=records()
        for row in a+b: row["nll_sum"]=3000.0
        with self.assertRaisesRegex(q.QualityError,"perplexity overflow"):
            q._stats(*q._validate_records(a,b,p),p)
    def test_missing_later_required_fields_return_not_evaluated_without_records(self):
        p=prereg(); del p["regression_budget"]; del p["calibration_content_sha256"]
        with tempfile.TemporaryDirectory() as d:
            pp=Path(d)/"p"; pp.write_text(json.dumps(p)); digest=hashlib.sha256(pp.read_bytes()).hexdigest()
            result=q.evaluate(pp,digest,"must-not-open","must-not-open")
        self.assertEqual(set(result["missing_preregistration_fields"]),{"regression_budget","calibration_content_sha256"})
        self.assertEqual(result["comparison"],"not_evaluated")
    def test_unapproved_budget_not_comparable(self):
        p=prereg(); p["budget_approval"]["approved"]=False
        with self.assertRaises(q.QualityError): q.validate_prereg(p)
    def test_strict_json_rejects_duplicate_nonfinite_and_bools(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"unused_extra":1e999}'):
            with self.assertRaises(q.QualityError): q.strict_json(raw)
        p=prereg(); p["confidence"]["level"]=True
        with self.assertRaises(q.QualityError): q.validate_prereg(p)
    def test_hash_checked_before_records_opened(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"p.json"; path.write_text(json.dumps(prereg()))
            with self.assertRaisesRegex(q.QualityError,"hash mismatch"):
                q.evaluate(path,"0"*64,"does-not-exist","does-not-exist")
    def test_missing_prereg_retains_not_evaluated(self):
        p=prereg(); del p["budget_approval"]
        with tempfile.TemporaryDirectory() as d:
            pp=Path(d)/"p"; pp.write_text(json.dumps(p)); digest=hashlib.sha256(pp.read_bytes()).hexdigest()
            result=q.evaluate(pp,digest,"unused","unused")
        self.assertEqual(result["comparison"],"not_evaluated")
        self.assertEqual(result["qualification"],"unqualified")

if __name__=="__main__": unittest.main()
