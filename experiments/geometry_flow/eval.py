"""Hinge-Chest shadow evaluation using the existing DA3 chunk hook."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from .common import load_config, write_json
from .detector import GeometryMonitor
from .viewer import live_overlay

def monitor_observation(raw, spec, camera_map):
    """Match raw expert state; never change the policy's gripper convention."""
    from arbim.envs.duobench import to_dataset_observation
    return to_dataset_observation(raw, spec, camera_map, binarize_state_gripper=False)

def main():
    from arbim.envs.duobench import DuoBenchAdapter,create_duobench_env
    from arbim.policy.inference_rl100_dp2d import load_inference_policy,seed_inference
    from experiments.da3.rollout_da3 import run_single_episode_da3
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout-config",default="experiments/da3/hinge_chest_da3.yaml")
    p.add_argument("--model-dir",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--episodes",type=int)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--live-view",action="store_true",help="Overlay chunk-aligned NLL and DA3 depth in native MuJoCo GUI")
    args=p.parse_args()
    cfg=load_config(args.rollout_config)
    ec,pc,rc=(cfg[k] for k in ("env","policy","rollout"))
    monitor=GeometryMonitor(args.model_dir,args.device,cfg["da3"]["device"],capture=args.live_view)
    tc=monitor.contract["temporal"]
    if rc["execute_steps"]!=tc["sampling_stride"] or ec["frequency"]!=tc["rollout_frequency"]:
        raise ValueError("Rollout timing differs from trained Flow contract")
    policy,spec,_=load_inference_policy(args.checkpoint or pc["checkpoint_path"],
                                        device=args.device,weights=pc.get("weights","raw"),
                                        n_action_steps=pc.get("n_action_steps"),
                                        num_inference_steps=pc.get("num_inference_steps"))
    expert=monitor.expert
    if expert["state_key"]!=spec["state_key"] or monitor.meta["state_dim"]!=spec["state_dim"]:
        raise ValueError("Expert/checkpoint state mismatch")
    names=expert["state_names"]
    if isinstance(names,dict):
        names=names.get("state_names")
    if names and tuple(names)!=tuple(spec.get("state_names",())):
        raise ValueError("Expert/checkpoint joint order mismatch")
    for side in ("left","right"):
        key=expert["camera_keys"][side]
        if ec["camera_map"].get(key)!=f"{side}_wrist":
            raise ValueError("Expert/rollout wrist mapping mismatch")
        c,h,w=spec["camera_shapes"][key]
        if [h,w,c]!=expert["rgb_shapes"][side]:
            raise ValueError("Expert/checkpoint RGB resolution mismatch")
    n=args.episodes if args.episodes is not None else rc["num_episodes"]
    if n<1:
        raise ValueError("Positive episode count required")
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"run_config.json",{"rollout":cfg,"model_contract":monitor.contract,
                                      "model_dir":str(Path(args.model_dir).resolve()),
                                      "checkpoint":args.checkpoint or pc["checkpoint_path"],
                                      "monitor_binarize_state_gripper":False,"live_view":args.live_view})
    seed=rc.get("seed")
    seed_inference(seed)
    class StageAdapter(DuoBenchAdapter):
        def reset(self, **kwargs):
            observation,info=super().reset(**kwargs)
            self.monitor_stage=info.get("stage")
            return observation,info
        def step(self, action):
            result=super().step(action)
            self.monitor_stage=result[-1].get("stage")
            return result
    with live_overlay(args.live_view,ec) as overlay:
        env=create_duobench_env(ec)
        try:
            adapter=(StageAdapter if overlay else DuoBenchAdapter)(env,spec,ec["camera_map"],
                                   binarize_state_gripper=ec.get("binarize_state_gripper",True),
                                   gripper_threshold=ec.get("gripper_threshold",.9))
            with (out/"episodes.jsonl").open("w") as log:
                for ep in range(n):
                    monitor.reset()
                    if overlay:
                        overlay.reset()
                    path=out/f"episode_{ep:06d}.csv"
                    with path.open("w",newline="") as f:
                        writer=csv.DictWriter(f,fieldnames=["chunk_index","env_step","ready","left_nll","right_nll","monitor_ms"])
                        writer.writeheader()
                        def on_chunk(raw,step,chunk):
                            start=time.perf_counter()
                            # Dataset-aligned observation, not native unresized RGB.
                            score=monitor.observe(monitor_observation(raw,spec,ec["camera_map"]),step)
                            monitor_ms=(time.perf_counter()-start)*1000
                            writer.writerow({"chunk_index":chunk,"env_step":step,**score,
                                             "monitor_ms":monitor_ms})
                            f.flush()
                            if overlay:
                                overlay.publish(monitor.last_sample,monitor.meta["roi"],
                                                ep,chunk,adapter.monitor_stage,monitor_ms)
                        result=run_single_episode_da3(adapter,policy,spec,
                             execute_steps=rc["execute_steps"],max_env_steps=rc["max_env_steps"],
                             post_success_steps=rc.get("post_success_steps",0),
                             seed=None if seed is None else seed+ep,
                             deterministic=pc.get("deterministic",True),on_chunk=on_chunk)
                    row={"episode_index":ep,**result}
                    log.write(json.dumps(row,allow_nan=False)+"\n")
                    log.flush()
                    print(row,flush=True)
        finally:
            env.close()

if __name__=="__main__":
    main()
