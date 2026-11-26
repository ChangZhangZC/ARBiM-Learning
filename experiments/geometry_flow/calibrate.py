"""Separate per-arm thresholds from successful rollout episode maxima."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .common import write_json

def calibrate(root, quantile):
    root=Path(root)
    if not 0<quantile<1:
        raise ValueError("Quantile must be between 0 and 1")
    with (root/"episodes.jsonl").open() as f:
        episodes=[json.loads(line) for line in f]
    maxima={s:[] for s in ("left","right")}
    skipped=[]
    for ep in episodes:
        if not ep["success"]:
            continue
        with (root/f"episode_{ep['episode_index']:06d}.csv").open() as f:
            rows=[r for r in csv.DictReader(f) if r["ready"]=="True"]
        if not rows:
            skipped.append(ep["episode_index"])
            continue
        for side in maxima:
            values=np.array([float(r[f"{side}_nll"]) for r in rows])
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite calibration score")
            maxima[side].append(float(values.max()))
    if not maxima["left"]:
        raise ValueError("No successful episodes with complete history")
    # Empirical episode-max quantile; not a finite-sample conformal guarantee.
    return {"method":"empirical_success_episode_max_quantile","quantile":quantile,
            "n_episodes":len(maxima["left"]),"skipped_warmup":skipped,
            "thresholds":{s:float(np.quantile(v,quantile,method="higher")) for s,v in maxima.items()},
            "warning":"Use independent test rollouts; few calibration episodes give unreliable tail estimates."}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout-dir",required=True)
    p.add_argument("--quantile",type=float,default=.95)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    result=calibrate(a.rollout_dir,a.quantile)
    write_json(a.output,result)
    print(result)
