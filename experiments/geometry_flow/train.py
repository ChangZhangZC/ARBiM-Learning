"""Fit independent left/right flows on successful expert windows."""
import argparse
from pathlib import Path
import shutil
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import load_config,write_json
from .dataset import GeometryDataset
from .conditional_flow import BimanualFlow

def run_epoch(model,loader,device,optimizers=None):
    training=optimizers is not None
    model.train(training)
    total={s:0.0 for s in ("left","right")}
    count=0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            scores=model.scores(**batch)
            for side,score in scores.items():
                loss=score.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite {side} NLL")
                if training:
                    optimizers[side].zero_grad(set_to_none=True)
                    loss.backward()
                    optimizers[side].step()
                total[side]+=float(score.detach().sum())
            count+=len(batch["joint"])
    return {s:v/count for s,v in total.items()}

def train(cfg):
    d,tc,tr=cfg["data"],cfg["temporal"],cfg["training"]
    torch.manual_seed(d["seed"])
    np.random.seed(d["seed"])
    trainset=GeometryDataset(d["feature_root"],tc,"train")
    valset=GeometryDataset(d["feature_root"],tc,"val")
    meta=trainset.meta
    if meta["roi"] != cfg["encoder"]["roi"] or meta["feature_dim"] != cfg["encoder"]["feature_dim"]:
        raise ValueError("Feature config mismatch; regenerate feature cache")
    device=torch.device(tr["device"])
    model=BimanualFlow(tc["window_size"],meta["feature_dim"],meta["state_dim"],cfg["flow"]).to(device)
    opts={s:torch.optim.Adam(f.parameters(),lr=tr["learning_rate"]) for s,f in model.flows.items()}
    loaders=[DataLoader(ds,batch_size=tr["batch_size"],shuffle=(i==0))
             for i,ds in enumerate((trainset,valset))]
    out=Path(tr["output"]).expanduser()
    if tr["epochs"] < 1:
        raise ValueError("epochs must be positive")
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(d["feature_root"])/"preprocess.npz",out/"preprocess.npz")
    contract={"feature_meta":meta,"temporal":tc,"flow":cfg["flow"],"expert_stride":trainset.stride}
    write_json(out/"contract.json",contract)
    best={s:float("inf") for s in opts}
    with (out/"metrics.jsonl").open("w") as log:
        for epoch in range(1,tr["epochs"]+1):
            train_loss=run_epoch(model,loaders[0],device,opts)
            val_loss=run_epoch(model,loaders[1],device)
            for s in opts:
                if val_loss[s] < best[s]:
                    best[s]=val_loss[s]
                    torch.save(model.flows[s].state_dict(),out/f"best_{s}.pt")
            row={"epoch":epoch,"train_nll":train_loss,"val_nll":val_loss}
            log.write(json.dumps(row,allow_nan=False)+"\n")
            log.flush()
            print(row,flush=True)
    # Expert validation is diagnostic, not a calibrated rollout failure threshold.
    write_json(out/"best_val_nll.json",best)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="experiments/geometry_flow/config.yaml")
    train(load_config(p.parse_args().config))
