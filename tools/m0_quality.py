#!/usr/bin/env python3
"""Bounded paired token-NLL analysis; not a model runner or qualification tool."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

MAX_PREREG_BYTES = 1_000_000
MAX_RECORD_BYTES = 1_000_000
MAX_TOTAL_BYTES = 256_000_000
MAX_ROWS = 100_000
MAX_BOOTSTRAP = 100_000
REQUIRED = ("freeze_record", "baseline", "dataset", "calibration_ids", "calibration_content_sha256", "heldout_ids", "metrics", "confidence", "seeds", "sample_count", "stopping_rule", "budget_approval", "regression_budget")

class QualityError(ValueError): pass

def _pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out: raise QualityError(f"duplicate JSON key: {k}")
        out[k] = v
    return out

def _constant(value): raise QualityError(f"non-finite JSON number: {value}")

def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed): raise QualityError(f"non-finite JSON number: {value}")
    return parsed

def strict_json(raw: bytes) -> Any:
    try: return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_finite_float)
    except (UnicodeDecodeError, json.JSONDecodeError) as e: raise QualityError(f"invalid JSON: {e}") from e

def _positive_int(v, label):
    if type(v) is not int or v <= 0: raise QualityError(f"{label} must be a positive integer")
    return v

def _mapping(v, label):
    if not isinstance(v, dict): raise QualityError(f"{label} must be an object")
    return v

def _sha(v, label):
    if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v): raise QualityError(f"{label} must be a lowercase SHA-256")

def validate_prereg(p):
    _mapping(p, "preregistration")
    missing = [k for k in REQUIRED if k not in p or p[k] is None]
    if missing: return missing
    if p["freeze_record"] == {}: raise QualityError("freeze_record must be substantive")
    base = _mapping(p["baseline"], "baseline")
    for k in ("checkpoint_revision", "files", "precision", "harness_revision", "environment"):
        if not base.get(k): raise QualityError(f"baseline.{k} is required")
    if not isinstance(base["files"], list) or not base["files"]: raise QualityError("baseline.files must list checkpoint file hashes")
    for f in base["files"]:
        _mapping(f, "baseline file")
        if not f.get("path"): raise QualityError("baseline file path required")
        _sha(f.get("sha256"), "baseline file sha256")
    if str(base["precision"]).lower() not in ("bf16", "bfloat16", "original bf16"): raise QualityError("baseline precision must explicitly identify original BF16")
    d = _mapping(p["dataset"], "dataset")
    if not d.get("revision"): raise QualityError("dataset revision required")
    _sha(d.get("sha256"), "dataset sha256")
    for split in ("calibration_ids", "heldout_ids"):
        if not isinstance(p[split], list) or not p[split] or any(not isinstance(x,str) or not x for x in p[split]) or len(set(p[split])) != len(p[split]): raise QualityError(f"{split} must contain unique nonempty string IDs")
    if set(p["calibration_ids"]) & set(p["heldout_ids"]): raise QualityError("calibration and heldout IDs overlap")
    calibration_hashes=p.get("calibration_content_sha256")
    if not isinstance(calibration_hashes,list) or not calibration_hashes: raise QualityError("calibration_content_sha256 must list exact normalized calibration-content hashes")
    for h in calibration_hashes: _sha(h,"calibration content hash")
    metrics = _mapping(p["metrics"], "metrics")
    if metrics.get("primary") != "token_weighted_mean_nll" or metrics.get("direction") != "lower": raise QualityError("metric must preregister lower-is-better token_weighted_mean_nll")
    c = _mapping(p["confidence"], "confidence")
    if c.get("method") != "paired_percentile_bootstrap": raise QualityError("confidence method must be paired_percentile_bootstrap")
    level = c.get("level"); reps = c.get("replicates")
    if type(level) not in (float,int) or isinstance(level,bool): raise QualityError("confidence.level must be between 0 and 1")
    level_number = cast(float, level)
    if not 0 < level_number < 1: raise QualityError("confidence.level must be between 0 and 1")
    reps = _positive_int(reps,"confidence.replicates")
    if reps > MAX_BOOTSTRAP: raise QualityError("bootstrap replicate cap exceeded")
    if not isinstance(p["seeds"],list) or not p["seeds"] or any(type(s) is not int or s < 0 for s in p["seeds"]): raise QualityError("seeds must be nonnegative integers")
    if len(set(p["seeds"])) != len(p["seeds"]): raise QualityError("seeds must be unique")
    _positive_int(p["sample_count"],"sample_count")
    if p["sample_count"] != len(p["heldout_ids"]): raise QualityError("sample_count must exactly equal the heldout ID count")
    if p["stopping_rule"] != "fixed-full-heldout": raise QualityError("only stopping_rule fixed-full-heldout is supported")
    a = _mapping(p["budget_approval"],"budget_approval")
    for k in ("approved", "reviewer", "date", "baseline_evidence"):
        if k not in a or a[k] in (None, "", False): raise QualityError(f"budget approval missing {k}")
    if a["approved"] is not True: raise QualityError("budget approval must be explicit true")
    b = _mapping(p.get("regression_budget"),"regression_budget")
    v = b.get("max_upper_nll_regression")
    if type(v) not in (float,int) or isinstance(v,bool): raise QualityError("approved regression_budget.max_upper_nll_regression must be finite nonnegative number")
    budget = cast(float, v)
    if not math.isfinite(budget) or budget < 0: raise QualityError("approved regression_budget.max_upper_nll_regression must be finite nonnegative number")
    return []

def _rows(path):
    total=0; rows=[]
    with open(path,"rb") as f:
        for line_no,line in enumerate(f,1):
            total += len(line)
            if total > MAX_TOTAL_BYTES: raise QualityError("JSONL byte cap exceeded")
            if len(line)>MAX_RECORD_BYTES: raise QualityError(f"row {line_no} exceeds byte cap")
            if not line.strip(): continue
            r=_mapping(strict_json(line),f"row {line_no}")
            if len(rows)>=MAX_ROWS: raise QualityError("row cap exceeded")
            rows.append(r)
    return rows

def _normalize_content(v):
    if not isinstance(v,str): raise QualityError("content must be a string")
    return " ".join(v.split())

def _validate_records(base_rows,var_rows,p):
    if not base_rows or len(base_rows)!=len(var_rows): raise QualityError("paired records must be nonempty and equal length")
    expected=set(p["heldout_ids"])
    def index(rows,label):
        out={}; contents=set()
        for r in rows:
            if set(r)!={"id","content","context_tokens","token_count","nll_sum","seed"}: raise QualityError(f"{label} row fields must be exactly id, content, context_tokens, token_count, nll_sum, seed")
            i=r["id"]
            if not isinstance(i,str) or not i or i in out: raise QualityError(f"{label} IDs must be unique nonempty strings")
            h=hashlib.sha256(_normalize_content(r["content"]).encode()).hexdigest()
            if h in contents: raise QualityError(f"duplicate normalized content in {label}")
            contents.add(h)
            for k in ("context_tokens","token_count","seed"):
                _positive_int(r[k],f"{label}.{k}")
            n=r["nll_sum"]
            if type(n) not in (int,float) or isinstance(n,bool) or not math.isfinite(n) or n<0: raise QualityError(f"{label}.nll_sum must be finite nonnegative number")
            out[i]=r
        if set(out)!=expected: raise QualityError(f"{label} IDs must exactly match heldout IDs; missing rows are not passes")
        return out,contents
    a,ac=index(base_rows,"baseline"); b,bc=index(var_rows,"variant")
    if ac!=bc: raise QualityError("baseline/variant normalized content mismatch")
    calibration_hashes=set(p["calibration_content_sha256"])
    if ac & calibration_hashes: raise QualityError("exact normalized content leaks between calibration and heldout")
    for i in expected:
        for k in ("context_tokens","token_count","seed"):
            if a[i][k]!=b[i][k]: raise QualityError(f"paired {k} mismatch for {i}")
        if a[i]["seed"] not in p["seeds"]: raise QualityError(f"undeclared seed for {i}")
    return a,b

def _quantile(xs,q):
    xs=sorted(xs); pos=(len(xs)-1)*q; lo=int(pos); hi=min(lo+1,len(xs)-1)
    return xs[lo]+(xs[hi]-xs[lo])*(pos-lo)

def _finite(value, label):
    if not math.isfinite(value): raise QualityError(f"non-finite {label}; refusing evaluation")
    return value

def _sum_finite(values, label):
    try: value=math.fsum(values)
    except (OverflowError, ValueError) as e: raise QualityError(f"non-finite {label}; refusing evaluation") from e
    return _finite(value,label)

def _stats(a,b,p):
    ids=sorted(a); tokens=sum(a[i]["token_count"] for i in ids)
    baseline_total=_sum_finite((a[i]["nll_sum"] for i in ids),"baseline NLL sum")
    variant_total=_sum_finite((b[i]["nll_sum"] for i in ids),"variant NLL sum")
    bn=_finite(baseline_total/tokens,"baseline mean NLL"); vn=_finite(variant_total/tokens,"variant mean NLL")
    per={i:_finite((b[i]["nll_sum"]-a[i]["nll_sum"])/a[i]["token_count"],"paired NLL difference") for i in ids}
    rng=random.Random(sum(p["seeds"]))
    reps=[]
    for _ in range(p["confidence"]["replicates"]):
        sample=[ids[rng.randrange(len(ids))] for _ in ids]
        den=sum(a[i]["token_count"] for i in sample)
        boot_total=_sum_finite((per[i]*a[i]["token_count"] for i in sample),"bootstrap NLL sum")
        reps.append(_finite(boot_total/den,"bootstrap mean NLL difference"))
    alpha=(1-p["confidence"]["level"])/2
    ci=[_finite(_quantile(reps,alpha),"lower confidence bound"),_finite(_quantile(reps,1-alpha),"upper confidence bound")]
    try: bp=math.exp(bn); vp=math.exp(vn)
    except OverflowError as e: raise QualityError("perplexity overflow; refusing evaluation") from e
    regression=_finite(vn-bn,"mean NLL regression")
    return {"heldout_rows":len(ids),"tokens":tokens,"baseline_mean_nll":bn,"variant_mean_nll":vn,"baseline_perplexity":_finite(bp,"baseline perplexity"),"variant_perplexity":_finite(vp,"variant perplexity"),"mean_nll_regression":regression,"paired_bootstrap_ci":ci,"confidence_level":p["confidence"]["level"]}

def evaluate(prereg_path, prereg_hash, baseline_path, variant_path):
    raw=Path(prereg_path).read_bytes()
    if len(raw)>MAX_PREREG_BYTES: raise QualityError("preregistration byte cap exceeded")
    actual=hashlib.sha256(raw).hexdigest()
    if actual!=prereg_hash: raise QualityError("preregistration hash mismatch; evaluation records were not read")
    p=strict_json(raw); missing=validate_prereg(p)
    if missing:
        return {"inventory_validation":"valid_records_required","comparison":"not_evaluated","qualification":"unqualified","reason":"incomplete preregistration","missing_preregistration_fields":missing,"baseline_provenance":"supplied assertion; not physically verified","full_model_quality":"not evaluated"}
    # Only after the pinned, complete preregistration has been validated do we open records.
    a,b=_validate_records(_rows(baseline_path),_rows(variant_path),p)
    stats=_stats(a,b,p)
    budget=p["regression_budget"]["max_upper_nll_regression"]
    approved=validate_prereg(p)==[]
    result="not_evaluated"
    if approved: result="within_preregistered_budget" if stats["paired_bootstrap_ci"][1] <= budget else "exceeds_preregistered_budget"
    return {"inventory_validation":"passed","comparison":result,"qualification":"unqualified","budget_approved":approved,"approved_upper_nll_regression_budget":budget if approved else None,"statistics":stats,"baseline_identity":p["baseline"],"baseline_provenance":"supplied assertion; not physically verified","full_model_quality":"not evaluated","original_bf16_execution_claimed":False}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prereg",required=True); ap.add_argument("--prereg-sha256",required=True)
    ap.add_argument("--baseline-jsonl",required=True); ap.add_argument("--variant-jsonl",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args(argv)
    out=Path(args.output)
    if out.exists(): ap.error("output already exists; refusing to overwrite")
    temp_path=None
    try:
        result=evaluate(args.prereg,args.prereg_sha256,args.baseline_jsonl,args.variant_jsonl)
        serialized=json.dumps(result,indent=2,allow_nan=False)+"\n"
        fd,temp_name=tempfile.mkstemp(prefix=f".{out.name}.",suffix=".tmp",dir=out.parent)
        temp_path=Path(temp_name)
        with os.fdopen(fd,"w",encoding="utf-8") as f: f.write(serialized); f.flush(); os.fsync(f.fileno())
        os.link(temp_path,out)
        temp_path.unlink(); temp_path=None
    except (OSError,QualityError,ValueError,OverflowError) as e:
        if temp_path is not None:
            try: temp_path.unlink()
            except OSError: pass
        print(f"m0_quality: {e}",file=sys.stderr); return 2
    return 0
if __name__=="__main__": raise SystemExit(main())
