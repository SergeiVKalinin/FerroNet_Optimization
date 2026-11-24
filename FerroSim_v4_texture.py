"""FerroSim v4 texture evolution from optimizer incumbents.
For every disorder x method x seed, save the terminal Px, Py and curl whenever
the best-so-far *measured* reward improves. There is no change to the reward.
Identical module is embedded in all five standalone notebooks.
"""
from __future__ import annotations
from pathlib import Path
import math
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from oracle import evaluate_fields,curl_field
from representation import fields

def safe_slug(s):
    return re.sub(r"[^0-9a-zA-Z_-]+","_",str(s)).strip("_")

def radial_correlation_length(P):
    """Mean-subtracted, overlap-corrected spatial polarization correlation.
    Reports first 1/e crossing in lattice-pixel units; NaN if unresolved.
    No periodic wraparound is introduced.
    """
    x=np.asarray(P,dtype=float)
    q=x-x.mean(axis=(1,2),keepdims=True)
    variance=float(np.mean(np.sum(q*q,axis=0)))
    if variance<1e-12:return np.nan
    corr=np.zeros((2*x.shape[1]-1,2*x.shape[2]-1),float)
    for component in q:
        corr+=fftconvolve(component,component[::-1,::-1],mode="full")
    ones=np.ones(x.shape[1:])
    overlap=fftconvolve(ones,ones[::-1,::-1],mode="full")
    c=(corr/np.maximum(overlap,1.))/variance
    ny,nx=x.shape[1:]
    iy,ix=np.indices(c.shape)
    rr=np.hypot(iy-(ny-1),ix-(nx-1))
    prev=1.
    for r in range(1,max(2,min(nx,ny)//2+1)):
        ring=(rr>=r-.5)&(rr<r+.5)
        value=float(np.nanmean(c[ring]))
        if not np.isfinite(value):continue
        if value<=math.exp(-1):
            if prev<=math.exp(-1):return float(r)
            return float((r-1)+(prev-math.exp(-1))/max(prev-value,1e-12))
        prev=value
    return np.nan

def state_metrics(P, C, threshold):
    P=np.asarray(P,float);absC=np.abs(np.asarray(C,float))
    total=float(absC.sum())
    squares=float((absC*absC).sum())
    n=absC.size
    localization=float(total*total/(n*squares)) if squares>0 else np.nan
    high=float(np.mean(absC>threshold)) if np.isfinite(threshold) else np.nan
    return {"curl_total":total,"curl_localization":localization,
            "high_curl_area":high,"high_curl_threshold":float(threshold),
            "polarization_rms":float(np.sqrt(np.mean(np.sum(P*P,axis=0)))),
            "polarization_spatial_variance":float(np.mean(np.sum((P-P.mean(axis=(1,2),keepdims=True))**2,axis=0))),
            "correlation_length_px":radial_correlation_length(P)}

def _physical_state(row,defects,max_substeps=100000):
    w=np.asarray(fields(np.asarray(row.coefficients,float),n=900),float).reshape(-1)
    result=evaluate_fields(w,method="rk4",defects_ec=defects,max_substeps=max_substeps)
    good=bool(result["valid"][0]) and not bool(result["native_guard_exceeded"][0])
    if not good:
        raise RuntimeError(f"Previously valid winner failed independent physics reevaluation "
                           f"map={row['map']} method={row.method} seed={row.seed} eval={row.evaluation}")
    P=np.asarray(result["final_P"][0],float)
    C=np.asarray(curl_field(P),float)
    measured=float(result["reward"][0])
    return P,C,w,measured,float(result["interior"][0])

def record_texture_histories(df,map_ids,map_dir,out,*,max_substeps=100000):
    """Build index of every generation and save each distinct incumbent once.
    Returns (generation-level rows, distinct saved states) as DataFrames.
    """
    out=Path(out);archive=out/"states";archive.mkdir(parents=True,exist_ok=True)
    maps=set(map_ids);map_dir=Path(map_dir)
    history=df[df["map"].isin(maps)].copy()
    if history.empty:raise RuntimeError("No optimizer histories available for texture analysis")
    thresholds={}
    baseline_rows=history[(history.generation==0)&history.valid&np.isfinite(history.reward)]
    for (mid,seed),group in baseline_rows.groupby(["map","seed"]):
        best=group.loc[group.reward.idxmax()]
        P,C,w,reward,interior=_physical_state(best,np.load(map_dir/(str(mid)+".npy")),max_substeps)
        threshold=float(np.percentile(np.abs(C),90.))
        thresholds[(str(mid),int(seed))]=threshold
    saved=[];timeline=[];cache={}
    groups=history.groupby(["map","method","seed"],sort=False)
    for (mid,method,seed),group in groups:
        mid=str(mid);method=str(method);seed=int(seed)
        group=group.sort_values("evaluation")
        defects=np.load(map_dir/(mid+".npy"),allow_pickle=False)
        threshold=thresholds.get((mid,seed),np.nan)
        previous=None;previous_c=None;best_reward=-np.inf;winner=None
        for generation,block in group.groupby("generation",sort=True):
            generation=int(generation)
            candidates=block[block.valid & np.isfinite(block.reward)]
            if len(candidates):
                candidate=candidates.loc[candidates.reward.idxmax()]
                if float(candidate.reward)>best_reward:
                    best_reward=float(candidate.reward);winner=candidate
            if winner is None:
                timeline.append({"map":mid,"method":method,"seed":seed,"generation":generation,
                                 "source_evaluation":np.nan,"reward":np.nan,"state_path":None,
                                 "new_record":False})
                continue
            evaluation=int(winner.evaluation)
            key=(mid,method,seed,evaluation)
            is_new=key not in cache
            if is_new:
                P,C,w,physical_reward,interior=_physical_state(winner,defects,max_substeps)
                if not np.isclose(physical_reward,float(winner.reward),rtol=1e-7,atol=1e-8):
                    raise RuntimeError(f"Replay reward disagrees with archived reward for {key}: "
                                       f"{physical_reward} vs {winner.reward}; check oracle and checkpoints")
                state_path=Path("states")/safe_slug(mid)/safe_slug(method)/f"seed_{seed}_eval_{evaluation:06d}.npz"
                target=out/state_path;target.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target,final_P=P,Px=P[0],Py=P[1],curl=C,abs_curl=np.abs(C),
                                    waveform_Ec=w,reward=np.float64(physical_reward),
                                    interior_curl=np.float64(interior),
                                    generation=np.int64(generation),
                                    evaluation=np.int64(evaluation))
                metrics=state_metrics(P,C,threshold)
                dist=float(np.sqrt(np.mean((P-previous)**2))/
                           max(float(np.sqrt(np.mean(previous**2))),1e-12)) if previous is not None else np.nan
                curl_change=float(np.mean(np.abs(np.abs(C)-np.abs(previous_c)))) if previous_c is not None else np.nan
                saved.append({"map":mid,"method":method,"seed":seed,"generation_first":generation,
                     "evaluation":evaluation,"reward":float(physical_reward),
                     "interior_curl":float(interior),"boundary_curl":float(physical_reward-interior),
                     "state_path":str(state_path),"state_change_from_previous":dist,
                     "curl_change_from_previous":curl_change,**metrics})
                cache[key]={"path":str(state_path),"metrics":metrics,
                            "distance":dist,"curl_change":curl_change}
                previous=P;previous_c=C
            info=cache[key]
            timeline.append({"map":mid,"method":method,"seed":seed,"generation":generation,
                 "source_evaluation":evaluation,"reward":float(best_reward),
                 "state_path":info["path"],"new_record":is_new,
                 "state_change_from_previous":info["distance"] if is_new else 0.,
                 "curl_change_from_previous":info["curl_change"] if is_new else 0.,
                 **info["metrics"]})
    T=pd.DataFrame(timeline);S=pd.DataFrame(saved)
    T.to_csv(out/"texture_generation_index.csv",index=False)
    S.to_csv(out/"texture_distinct_states.csv",index=False)
    (out/"texture_protocol.json").write_text(json.dumps({
       "reward":"terminal sum(abs(periodic centered curl))",
       "what_is_saved":"terminal polarization and curl per unique best-so-far incumbent; unchanged incumbents reference the previous NPZ",
       "timeline":"optimization generation, not within-waveform physical time",
       "curl_localization":"(sum(abs(C))**2)/(pixels*sum(abs(C)**2))",
       "high_curl_threshold":"90th percentile of abs curl of best VALID initial control, across all methods for the same map and seed",
       "correlation_length":"overlap-normalized mean-subtracted vector autocorrelation 1/e crossing; open spatial shifts",
       "p_change":"RMS(Pnew-Pold)/RMS(Pold) for distinct improved records",
       "physical_validation":"independent RK4 re-evaluation with reward agreement check",
       "fields_stored":["Px","Py","curl","abs_curl","waveform_Ec","reward","interior_curl"]
       },indent=2))
    return T,S

def _read(out,record):
    with np.load(Path(out)/str(record.state_path),allow_pickle=False) as z:
        return {"P":z["final_P"].copy(),"C":z["curl"].copy(),"W":z["waveform_Ec"].copy()}

def figure_T1(out,S,*,max_snapshots=6,all_disorders=True):
    """For each (map,method,seed): snapshots of E, Px, Py, |curl|, delta |curl|."""
    out=Path(out);S=S.sort_values(["map","method","seed","generation_first"])
    first=str(S["map"].iloc[0]) if len(S) else "reference"
    count=0
    for (mid,method,seed),g in S.groupby(["map","method","seed"],sort=False):
        if not all_disorders and str(mid)!=first:continue
        inds=np.unique(np.linspace(0,len(g)-1,min(max_snapshots,len(g)),dtype=int))
        chosen=g.iloc[inds]
        stages=[_read(out,row) for _,row in chosen.iterrows()]
        amp=max(max(np.max(np.abs(z["P"])) for z in stages),1e-8)
        curlmax=max(max(np.max(np.abs(z["C"])) for z in stages),1e-8)
        rows=len(stages)
        fig,axs=plt.subplots(rows,5,figsize=(16,2.75*rows),squeeze=False,layout="constrained")
        last=None
        for j,((_,row),state) in enumerate(zip(chosen.iterrows(),stages)):
            P=state["P"];absC=np.abs(state["C"]);w=state["W"]
            axs[j,0].plot(np.linspace(0,1,len(w)),w,color="#007F83",lw=1.2)
            axs[j,0].set(title=f"Gen {int(row.generation_first)}, R={row.reward:.3g}" if j==0 else
                               f"Gen {int(row.generation_first)}, R={row.reward:.3g}",
                         xlabel="t/T",ylabel="E/Ec")
            axs[j,1].imshow(P[0],vmin=-amp,vmax=amp,cmap="coolwarm",origin="lower")
            axs[j,2].imshow(P[1],vmin=-amp,vmax=amp,cmap="coolwarm",origin="lower")
            axs[j,3].imshow(absC,vmin=0,vmax=curlmax,cmap="magma",origin="lower")
            delta=absC-last if last is not None else np.zeros_like(absC)
            dmax=max(curlmax*.5,float(np.max(np.abs(delta))),1e-8)
            axs[j,4].imshow(delta,vmin=-dmax,vmax=dmax,cmap="coolwarm",origin="lower")
            last=absC
        for ax,title in zip(axs[0],["Winning field","P_x","P_y","|curl P|","Change in |curl|"]):
            ax.set_title(title)
        fig.suptitle(f"T1 — Measured physical states as reward improves | {mid} | {method} | seed {seed}")
        p=out/"figures"/"T1"
        p.mkdir(parents=True,exist_ok=True)
        fig.savefig(p/f"T1_{safe_slug(mid)}_{safe_slug(method)}_seed{seed}.png",dpi=135,bbox_inches="tight")
        plt.show();plt.close(fig);count+=1
    print("T1 figures:",count)

def figure_T2(out,T,*,all_disorders=True):
    out=Path(out);figdir=out/"figures"/"T2";figdir.mkdir(parents=True,exist_ok=True)
    for (mid,seed),g in T.groupby(["map","seed"],sort=False):
        if not all_disorders and mid!="reference":continue
        fig,axs=plt.subplots(2,3,figsize=(14,8),layout="constrained")
        fields=[
            ("reward","Best measured curl"),("curl_localization","Curl localization (participation fraction)"),
            ("high_curl_area","Area above fixed initial high-curl threshold"),
            ("state_change_from_previous","Relative P change at new improvements"),
            ("correlation_length_px","P correlation length / pixels"),
            ("curl_change_from_previous","Mean change in |curl| at improvements")]
        for method,h in g.groupby("method"):
            h=h.sort_values("generation")
            for ax,(column,title) in zip(axs.flat,fields):
                ax.plot(h.generation,h[column],marker=".",lw=1.2,label=method)
        for ax,(column,title) in zip(axs.flat,fields):
            ax.set(xlabel="Optimization generation",ylabel=title,title=title)
            ax.grid(alpha=.2)
        axs[0,0].legend(fontsize=7)
        fig.suptitle(f"T2 — Texture descriptors through optimization | {mid} | seed {seed}")
        fig.savefig(figdir/f"T2_{safe_slug(mid)}_seed{seed}.png",dpi=140,bbox_inches="tight")
        plt.show();plt.close(fig)

def figure_T3(out,S):
    out=Path(out)
    props=[("curl_localization","Curl localization"),("high_curl_area","High-curl area"),
           ("state_change_from_previous","Relative change in P"),("correlation_length_px","Correlation length")]
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout="constrained")
    for method,g in S.groupby("method"):
        for ax,(key,label) in zip(axs.flat,props):
            ax.scatter(g[key],g.reward,s=15,alpha=.45,label=method)
    for ax,(key,label) in zip(axs.flat,props):
        ax.set(xlabel=label,ylabel="Best-measured reward",
               title="Measured reward vs "+label)
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7)
    fig.suptitle("T3 — Texture descriptors and reward (repeated incumbents excluded)")
    fig.savefig(out/"figures"/"T3_reward_texture_descriptors.png",dpi=150,bbox_inches="tight")
    plt.show();plt.close(fig)

def figure_T4(out,S,*,reference_only=True):
    out=Path(out)
    maps=["reference"] if reference_only else sorted(S["map"].unique())
    from matplotlib.markers import MarkerStyle
    markers=["o","s","^","D","v","P","X","*"]
    records=[]
    for mid in maps:
        subset=S[S["map"]==mid].copy()
        if len(subset)<2:continue
        vectors=[]
        for _,row in subset.iterrows():
            q=_read(out,row)
            vectors.append(q["P"].reshape(-1))
        X=np.asarray(vectors,float)
        X=X-X.mean(axis=0,keepdims=True)
        if np.linalg.norm(X)<1e-12:
            print("T4 PCA skipped: identical states on",mid);continue
        U,s,Vt=np.linalg.svd(X,full_matrices=False)
        coords=U[:,:2]*s[:2]
        if coords.shape[1]==1:coords=np.c_[coords,np.zeros(len(coords))]
        frac=(s[:2]**2)/(s**2).sum()
        methods=list(dict.fromkeys(subset.method))
        fig,ax=plt.subplots(figsize=(9,6),layout="constrained")
        for j,m in enumerate(methods):
            rows=np.flatnonzero((subset.method==m).to_numpy())
            sc=ax.scatter(coords[rows,0],coords[rows,1],
                c=subset.iloc[rows].reward.to_numpy(float),cmap="viridis",
                marker=markers[j%len(markers)],s=32,alpha=.85,label=m)
            for seed,g in subset[subset.method==m].groupby("seed"):
                ids=[subset.index.get_loc(v) for v in g.sort_values("generation_first").index]
                if len(ids)>1:ax.plot(coords[ids,0],coords[ids,1],alpha=.3,lw=1,color="gray")
        fig.colorbar(sc,ax=ax,label="Measured terminal curl")
        ax.set(xlabel=f"PC1 ({100*frac[0]:.1f}% variance)",
               ylabel=f"PC2 ({100*frac[1]:.1f}% variance)" if len(frac)>1 else "PC2",
               title=f"T4 — Polarization-state PCA | disorder {mid}")
        ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.savefig(out/"figures"/f"T4_texture_state_PCA_{safe_slug(mid)}.png",dpi=150,bbox_inches="tight")
        plt.show();plt.close(fig)
        for (_,row),(x,y) in zip(subset.iterrows(),coords):
            records.append({"map":mid,"method":row.method,"seed":row.seed,"generation":row.generation_first,
                "evaluation":row.evaluation,"reward":row.reward,"PC1":x,"PC2":y,
                "variance_fraction_PC1":frac[0],"variance_fraction_PC2":frac[1] if len(frac)>1 else 0.})
    pd.DataFrame(records).to_csv(out/"texture_PCA_coordinates.csv",index=False)

def analyze_textures(df,maps,map_dir,root,*,max_substeps=100000,
                     max_snapshots=6,plot_each_disorder=True,plot_pca_reference_only=True):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    timeline,saved=record_texture_histories(df,maps,map_dir,root,max_substeps=max_substeps)
    if saved.empty:
        print("No valid incumbent physical states to plot.")
        return timeline,saved
    (root/"figures").mkdir(parents=True,exist_ok=True)
    figure_T1(root,saved,max_snapshots=max_snapshots,all_disorders=plot_each_disorder)
    figure_T2(root,timeline,all_disorders=plot_each_disorder)
    figure_T3(root,saved)
    figure_T4(root,saved,reference_only=plot_pca_reference_only)
    return timeline,saved
