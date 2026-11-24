"""Pure GA benchmark operating directly on 900-point physical field waveforms.

Three methods have the same starting population and exactly the same measured
physical evaluation budget. No DKL surrogate and no waveform repair/clipping.
A source/config-fingerprinted directory and atomic checkpoint allow restart.
"""
import argparse
import json
import os
import pickle
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from representation import initial_design,fields,complexity,proposed_child
from oracle import evaluate_fields

METHODS=["Hard-Crossover-GA","Smooth-Crossover-GA","Mutation-Only-GA"]

def _atomic_pickle(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix="checkpoint_")
    try:
        with os.fdopen(fd,"wb") as h:
            pickle.dump(state,h,pickle.HIGHEST_PROTOCOL)
            h.flush();os.fsync(h.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def _atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix="arrays_")
    try:
        with os.fdopen(fd,"wb") as h:
            np.savez_compressed(h,**arrays)
            h.flush();os.fsync(h.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def measure(waves,defects):
    x=np.atleast_2d(np.asarray(waves,float))
    if x.ndim!=2 or x.shape[1]!=900 or not np.isfinite(x).all():
        raise ValueError("Physical evaluation requires finite 900-sample waveforms")
    result=evaluate_fields(fields(x,900),method="rk4",fidelity=1,
                           defects_ec=defects,max_substeps=100000)
    valid=np.asarray(result["valid"],bool)&~np.asarray(result["native_guard_exceeded"],bool)
    return dict(reward=np.where(valid,result["reward"],np.nan),valid=valid,
        interior=np.asarray(result["interior"],float),
        status=np.asarray(result["status"],int),
        rhs_calls=np.asarray(result["rhs_calls"],int),
        substeps=np.asarray(result["substeps"],int))

def initial_cache(out,mapname,seed,defects,count,run_id):
    path=Path(out)/mapname/f"shared_initial_seed_{seed}.npz"
    if path.is_file():
        with np.load(path,allow_pickle=False) as data:
            if data["run_id"].item()!=run_id:raise ValueError("Incompatible initial cache")
            wave=data["waveforms"].copy()
            res={key:data[key].copy() for key in ("reward","valid","interior","status","rhs_calls","substeps")}
        if len(wave)!=count:raise ValueError("Initial-cache size mismatch")
        return wave,res
    waves=initial_design(seed,count)
    result=measure(waves,defects)
    _atomic_npz(path,run_id=np.asarray(run_id),waveforms=waves,**result)
    return waves,result

def records(waves,result,mapname,method,seed,generation,offset,ops,selection,parents):
    descriptors=complexity(waves)
    out=[]
    for i in range(len(waves)):
        pa,pb=parents[i]
        out.append({
            "map":str(mapname),"method":str(method),"seed":int(seed),
            "evaluation":offset+i+1,"generation":int(generation),
            "reward":float(result["reward"][i]),"valid":bool(result["valid"][i]),
            "interior":float(result["interior"][i]),"status":int(result["status"][i]),
            "rhs_calls":int(result["rhs_calls"][i]),"substeps":int(result["substeps"][i]),
            "turns":int(descriptors["turns"][i]),"complexity":int(descriptors["turns"][i]),
            "raw_turns":int(descriptors["raw_turns"][i]),
            "peak_ec":float(descriptors["peak_ec"][i]),
            "rms_ec":float(descriptors["rms_ec"][i]),
            "operation":str(ops[i]),"proposal_source":str(ops[i]),
            "selection_source":str(selection[i]),
            "parent_evaluation_a":int(pa),"parent_evaluation_b":int(pb),
            "predicted_mean":np.nan,"predicted_std":np.nan})
    return out

def tournament(waves,rewards,rng):
    eligible=np.flatnonzero(np.isfinite(rewards))
    if not len(eligible):raise RuntimeError("No physically valid parents available for GA")
    chosen=rng.choice(eligible,size=3,replace=len(eligible)<3)
    index=int(chosen[np.argmax(rewards[chosen])])
    return waves[index],index+1

def _export(folder,rows,waves):
    if len(rows)!=len(waves):raise RuntimeError("Mismatch of waveforms and physical observations")
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(folder/"history.csv",index=False)
    _atomic_npz(folder/"waveforms.npz",waveforms=np.asarray(waves,float))
    return df

def run_one(out,mapname,defects,method,seed,initial_count,generations,batch,run_id):
    folder=Path(out)/mapname/method/f"seed_{seed}"
    folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/"checkpoint.pkl"
    rng_seed=(int(seed)*1000003+1009*sum(map(ord,method))
              +97*sum(map(ord,mapname)))%(2**63-1)
    rng=np.random.default_rng(rng_seed)
    if checkpoint.is_file():
        # Trusted LOCAL pickle only; never accept an untrusted external checkpoint.
        with checkpoint.open("rb") as h:state=pickle.load(h)
        if state["run_id"]!=run_id or state["seed"]!=seed or state["method"]!=method:
            raise ValueError("Checkpoint source/configuration mismatch")
        rows=state["rows"];waves=state["waves"]
        done=int(state["completed_generation"])
        rng.bit_generator.state=state["rng_state"]
        if len(rows)!=initial_count+done*batch:raise RuntimeError("Corrupt checkpoint history")
    else:
        waves,initial=initial_cache(out,mapname,seed,defects,initial_count,run_id)
        rows=records(waves,initial,mapname,method,seed,0,0,
                     ["initial_design"]*len(waves),["shared_initial"]*len(waves),
                     [(0,0)]*len(waves))
        done=0
        _atomic_pickle(checkpoint,dict(run_id=run_id,seed=seed,method=method,
            completed_generation=done,rows=rows,waves=waves,
            rng_state=rng.bit_generator.state))
    if done>generations:raise ValueError("Checkpoint extends beyond requested run")
    for generation in range(done+1,generations+1):
        reward=np.asarray([r["reward"] for r in rows],float)
        new=[];ops=[];parent_ids=[]
        for _ in range(batch):
            a,aid=tournament(waves,reward,rng)
            b,bid=tournament(waves,reward,rng)
            child,op=proposed_child(method,a,b,rng)
            new.append(child);ops.append(op)
            parent_ids.append((aid,bid if "crossover" in op else 0))
        new=np.asarray(new,float)
        measured=measure(new,defects)
        rows.extend(records(new,measured,mapname,method,seed,generation,len(rows),
                            ops,["genetic_search"]*batch,parent_ids))
        waves=np.vstack([waves,new])
        _atomic_pickle(checkpoint,dict(run_id=run_id,seed=seed,method=method,
            completed_generation=generation,rows=rows,waves=waves,
            rng_state=rng.bit_generator.state))
        print(f"{mapname}/{method}/seed{seed}: generation={generation},"
              f" valid={int(measured['valid'].sum())}/{batch}",flush=True)
    return _export(folder,rows,waves)

def run(out,map_dir,mapnames,seeds,initial_count,generations,batch,run_id):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results=[]
    for mapname in mapnames:
        defects=np.load(Path(map_dir)/(mapname+".npy"),allow_pickle=False)
        for seed in seeds:
            for method in METHODS:
                print("Direct waveform GA:",mapname,seed,method,flush=True)
                results.append(run_one(out,mapname,defects,method,int(seed),
                    initial_count,generations,batch,run_id))
    df=pd.concat(results,ignore_index=True)
    planned=len(mapnames)*len(seeds)*len(METHODS)*(initial_count+generations*batch)
    if len(df)!=planned:raise RuntimeError(f"Incomplete physical history {len(df)} != {planned}")
    df.to_csv(out/"all_evaluations.csv",index=False)
    print("Recorded method-history observations:",len(df),
          "valid:",int(df["valid"].sum()),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True)
    p.add_argument("--map-dir",required=True)
    p.add_argument("--map-names",required=True)
    p.add_argument("--seeds",required=True)
    p.add_argument("--initial",type=int,default=64)
    p.add_argument("--generations",type=int,default=32)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--run-id",required=True)
    a=p.parse_args()
    if a.initial<2 or a.generations<0 or a.batch<1:p.error("Need initial>=2, generations>=0, batch>=1")
    seeds=[int(s) for s in a.seeds.split(",")]
    if not seeds or min(seeds)<0 or len(seeds)!=len(set(seeds)):
        p.error("Seeds must be unique nonnegative integers")
    maps=a.map_names.split(",")
    run(a.out,a.map_dir,maps,seeds,a.initial,a.generations,a.batch,a.run_id)
