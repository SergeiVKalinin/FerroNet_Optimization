"""
FerroSim v3: shared analysis & figure protocol.
Embedded verbatim in the v3 notebooks for standalone Anaconda/Jupyter execution.
Methods differ among representations, but every figure has the same scientific meaning.
No synthetic rewards; every optimization curve uses saved oracle evaluations.
"""
from __future__ import annotations
from functools import lru_cache
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _valid(df):
    return df[df.valid & np.isfinite(df.reward)].copy()

def _front(df):
    z=_valid(df).sort_values(["complexity","reward"],ascending=[True,False])
    if z.empty:return z
    keep=[]; incumbent=-np.inf
    for complexity,g in z.groupby("complexity",sort=True):
        v=g.iloc[0]
        if float(v.reward)>incumbent+1e-12:
            keep.append(v);incumbent=float(v.reward)
    return pd.DataFrame(keep) if keep else z.iloc[:0]

def _incumbent(g):
    g=g.sort_values("evaluation")
    vals=np.where(g.valid & np.isfinite(g.reward),g.reward,-np.inf)
    return g, np.maximum.accumulate(vals)

def _save(fig,out,name):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/name,dpi=150,bbox_inches="tight")
    plt.show();plt.close(fig)
    return out/name

def _champion(g):
    z=_valid(g)
    return None if z.empty else z.loc[z.reward.idxmax()]

def _scalar(row,key,default=np.nan):
    try:return float(row[key])
    except (TypeError,ValueError,KeyError):return default

def _zero_reward(ctx,map_id):
    cache=ctx.setdefault("zero_cache",{})
    if map_id not in cache:
        from oracle import evaluate_fields
        result=evaluate_fields(np.zeros(900),defects_ec=ctx["defect"](map_id),
                               method="rk4",max_substeps=ctx["max_substeps"])
        cache[map_id]=float(result["reward"][0]) if bool(result["valid"][0]) and not bool(result["native_guard_exceeded"][0]) else np.nan
    return cache[map_id]

def _selected(ctx,map_id):
    d=_valid(ctx["data"])
    z=d[d["map"]==map_id]
    if z.empty:return {}
    ans={"best":z.loc[z.reward.idxmax()]}
    simple=z[z.complexity<=1]
    if len(simple):ans["simple"]=simple.loc[simple.reward.idxmax()]
    return ans

def _grid(count,maxcols=5):
    cols=max(1,min(maxcols,int(count)))
    return int(math.ceil(count/cols)),cols

def figure01_disorders(ctx):
    maps=ctx["maps"];r,c=_grid(len(maps))
    fig,axs=plt.subplots(r,c,figsize=(3.1*c,2.9*r),squeeze=False,layout="constrained")
    for ax in axs.flat:ax.axis("off")
    for ax,mid in zip(axs.flat,maps):
        z=ctx["defect"](mid);ax.imshow(np.linalg.norm(z,axis=2),origin="lower")
        ax.set_title(mid);ax.axis("on")
    _save(fig,ctx["figdir"],"F01_disorder_maps.png")

def figure02_convergence(ctx):
    df=ctx["data"];maps=ctx["maps"];methods=ctx["methods"]
    fig,axs=plt.subplots(1,2 if len(maps)>1 else 1,figsize=(12 if len(maps)>1 else 7,4.4),squeeze=False)
    ax=axs[0,0]
    for meth in methods:
        g=df[(df["map"]==maps[0])&(df.method_key==meth)]
        for seed,s in g.groupby("seed"):
            s,b=_incumbent(s); ax.step(s.evaluation,np.where(np.isfinite(b),b,np.nan),
                                    where="post",alpha=.45,label=meth if seed==g.seed.min() else None)
    ax.set(xlabel="Attempted oracle evaluations / run",ylabel="Best measured reward",title="Reference-map convergence")
    ax.grid(alpha=.2);ax.legend(fontsize=7)
    if len(maps)>1:
        ax=axs[0,1]
        for meth in methods:
            traces=[]
            for (mid,seed),g in df[df.method_key==meth].groupby(["map","seed"]):
                g,b=_incumbent(g);traces.append(b)
            if not traces:continue
            minlen=min(map(len,traces));a=np.array([t[:minlen] for t in traces]);a[~np.isfinite(a)]=np.nan
            mu=np.nanmean(a,axis=0);n=np.sum(np.isfinite(a),axis=0)
            se=np.nanstd(a,axis=0,ddof=1)/np.sqrt(np.maximum(n,1)) if len(a)>1 else np.zeros(minlen)
            ax.plot(np.arange(1,minlen+1),mu,label=meth)
            if len(a)>1:ax.fill_between(np.arange(1,minlen+1),mu-se,mu+se,alpha=.12)
        ax.set(xlabel="Attempted oracle evaluations / run",ylabel="Mean best reward",title="Disorder + seed ensemble (descriptive)")
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    _save(fig,ctx["figdir"],"F02_reward_vs_evaluations.png")

def figure03_reference_pareto(ctx):
    df=ctx["data"];g=df[df["map"]==ctx["maps"][0]]
    fig,ax=plt.subplots(figsize=(8,5))
    for meth,s in g.groupby("method_key"):
        v=_valid(s);ax.scatter(v.complexity,v.reward,s=8,alpha=.15)
        f=_front(s)
        if len(f):ax.plot(f.complexity,f.reward,"o-",lw=1.5,ms=3,label=meth)
    ax.set(xlabel=ctx["complexity_label"],ylabel="Measured terminal curl",
           title="F03 — Reference-disorder reward–complexity Pareto fronts")
    ax.grid(alpha=.2);ax.legend(fontsize=7)
    _save(fig,ctx["figdir"],"F03_reference_pareto.png")

def figure04_waveform_histories(ctx,history_steps=40):
    """One waveform-history figure per disorder, with one panel per algorithm."""
    df=ctx["data"]
    for mid in ctx["maps"]:
        methods=ctx["methods"];rows,cols=_grid(len(methods),maxcols=2)
        fig,axs=plt.subplots(rows,cols,figsize=(6*cols,3.4*rows),squeeze=False,layout="constrained")
        for ax in axs.flat:ax.axis("off")
        for ax,meth in zip(axs.flat,methods):
            g=_valid(df[(df["map"]==mid)&(df.method_key==meth)&(df.seed==df.seed.min())])
            if g.empty:continue
            g=g.sort_values("evaluation")
            checkpoints=np.unique(np.linspace(0,len(g)-1,min(history_steps,len(g)),dtype=int))
            best=g.reward.to_numpy()
            idx=[]; incumbent=-np.inf;winner=0
            for i,y in enumerate(best):
                if y>incumbent:incumbent=y;winner=i
                idx.append(winner)
            rows_to_plot=[g.iloc[idx[i]] for i in checkpoints]
            waves=np.stack([ctx["field"](row) for row in rows_to_plot])
            amp=max(float(np.max(np.abs(waves))),1e-8)
            ax.imshow(waves,aspect="auto",origin="lower",cmap="coolwarm",vmin=-amp,vmax=amp,
                      extent=[0,1,float(g.evaluation.iloc[checkpoints[0]]),float(g.evaluation.iloc[checkpoints[-1]])])
            ax.axis("on");ax.set(title=meth,xlabel="Normalized control time",ylabel="Evaluation of incumbent")
        fig.suptitle("F04 — Best-so-far waveform history: "+str(mid))
        _save(fig,ctx["figdir"],"F04_waveform_history_"+str(mid)+".png")

def figure05_winners(ctx):
    picks=_selected(ctx,ctx["maps"][0]);fig,axs=plt.subplots(1,2,figsize=(12,4.3))
    for label,row in picks.items():
        w=ctx["field"](row)
        axs[0].plot(np.linspace(0,1,len(w)),w,label=f"{label}, R={row.reward:.4g}, C={int(row.complexity)}")
        axs[1].plot(np.linspace(0,1,len(w)),w/max(np.max(np.abs(w)),1e-12),label=label)
    axs[0].set(xlabel="Normalized control time",ylabel="E/Ec",title="Physical amplitude")
    axs[1].set(xlabel="Normalized control time",ylabel="Normalized E/Ec",title="Shape only (not reward comparable)")
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    _save(fig,ctx["figdir"],"F05_best_and_simple_waveforms.png")

def figure06_incumbent_diagnostics(ctx):
    df=ctx["data"]; fig,axs=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    labels=[("reward","Best reward"),("complexity",ctx["complexity_label"]),
            ("peak_ec","Peak |E| / Ec"),("rms_ec","RMS E / Ec")]
    for meth,g in df[(df["map"]==ctx["maps"][0])&(df.seed==df.seed.min())].groupby("method_key"):
        g=g.sort_values("evaluation");inc=-np.inf;winner=None;traces={k:[] for k,_ in labels}
        for _,row in g.iterrows():
            if bool(row.valid) and np.isfinite(row.reward) and row.reward>inc:
                inc=float(row.reward);winner=row
            for key,_ in labels:traces[key].append(_scalar(winner,key) if winner is not None else np.nan)
        for ax,(key,_) in zip(axs.flat,labels):
            ax.plot(g.evaluation,traces[key],label=meth)
    for ax,(key,title) in zip(axs.flat,labels):
        ax.set(xlabel="Attempted evaluations / run",ylabel=title,title=title);ax.grid(alpha=.2);ax.legend(fontsize=6)
    _save(fig,ctx["figdir"],"F06_incumbent_field_complexity_reward.png")

def figure07_surrogate_calibration(ctx):
    d=_valid(ctx["data"]);d=d[(d.generation>0)&np.isfinite(d.predicted_mean)&np.isfinite(d.predicted_std)]
    if d.empty:
        print("F07 skipped: no finite held-out proposal predictions.");return
    fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    for meth,g in d.groupby("method_key"):
        sample=g.iloc[::max(1,len(g)//250)]
        axs[0].scatter(sample.predicted_mean,sample.reward,s=8,alpha=.3,label=meth)
    lo=min(float(d.predicted_mean.min()),float(d.reward.min()));hi=max(float(d.predicted_mean.max()),float(d.reward.max()))
    axs[0].plot([lo,hi],[lo,hi],"k--",lw=1)
    axs[0].set(xlabel="DKL predicted reward before measurement",ylabel="Actual oracle reward",title="Proposal calibration (selection-biased)")
    for meth,g in d.groupby("method_key"):
        e=np.abs(g.reward-g.predicted_mean).to_numpy()
        std=np.maximum(g.predicted_std.to_numpy(),1e-12)
        axs[1].plot([1,2,3],[np.mean(e<=k*std) for k in (1,2,3)],"o-",label=meth)
    from math import erf,sqrt
    axs[1].plot([1,2,3],[erf(k/sqrt(2)) for k in (1,2,3)],"k--",label="Gaussian nominal")
    axs[1].set(xlabel="Credible interval ± k × std",ylabel="Empirical fraction covered",ylim=(0,1.03),
               title="On-policy selected candidates; not IID holdout")
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
    _save(fig,ctx["figdir"],"F07_DKL_prediction_calibration.png")

def figure08_09_10_physical(ctx):
    if not ctx.get("run_physical",True):
        print("F08–F10 physical revalidation disabled by configuration.");return
    from oracle import evaluate_fields,curl_field
    picks=_selected(ctx,ctx["maps"][0])
    if not picks:return
    row=picks["best"];wave=ctx["field"](row);defects=ctx["defect"](ctx["maps"][0])
    v=evaluate_fields(wave,defects_ec=defects,method="rk4",
                      return_history=True,max_substeps=ctx["max_substeps"])
    if not bool(v["valid"][0]) or bool(v["native_guard_exceeded"][0]):
        print("F08–F10 skipped: champion invalid under independent reevaluation.");return
    P=v["final_P"][0];curl=np.abs(curl_field(P))
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout="constrained")
    for ax,arr,title in zip(axs,[P[0],P[1],curl],["P_x","P_y","|curl P|"]):
        im=ax.imshow(arr,origin="lower");ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.75)
    _save(fig,ctx["figdir"],"F08_terminal_polarization_fields.png")
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].plot(v["time"],v["curl_history"][0],label="whole grid")
    axs[0].plot(v["time"],v["interior_history"][0],label="interior only")
    axs[0].set(xlabel="Physical time",ylabel="Total |curl|",title="Reward formation")
    axs[1].plot(v["time"],v["mean_P"][0,0],label="Mean P_x")
    axs[1].plot(v["time"],v["mean_P"][0,1],label="Mean P_y")
    axs[1].set(xlabel="Physical time",ylabel="Mean polarization",title="Polarization dynamics")
    for ax in axs:ax.grid(alpha=.2);ax.legend()
    _save(fig,ctx["figdir"],"F09_curl_and_polarization_trajectories.png")
    checks=[("fidelity 1",float(v["reward"][0]),float(v["interior"][0]),True)]
    if ctx.get("run_refinement",True):
        z=evaluate_fields(wave,defects_ec=defects,method="rk4",fidelity=2,
                          max_substeps=max(10000,2*ctx["max_substeps"]))
        good=bool(z["valid"][0]) and not bool(z["native_guard_exceeded"][0])
        checks.append(("fidelity 2",float(z["reward"][0]) if good else np.nan,
                      float(z["interior"][0]) if good else np.nan,good))
    pd.DataFrame([dict(fidelity=f,whole=r,interior=i,valid=valid,boundary=r-i)
                  for f,r,i,valid in checks]).to_csv(Path(ctx["figdir"])/"F10_numerical_validation.csv",index=False)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    pos=np.arange(len(checks)); whole=[z[1] for z in checks];inside=[z[2] for z in checks]
    axs[0].bar(pos,inside,label="Interior")
    axs[0].bar(pos,np.array(whole)-inside,bottom=inside,label="Boundary")
    axs[0].set(xticks=pos,xticklabels=[z[0] for z in checks],ylabel="Total absolute curl",
               title="Integration fidelity / boundary decomposition");axs[0].legend()
    fine=np.linspace(0,1,1801);coarse=np.linspace(0,1,len(wave))
    reference=ctx["field"](row,1801);interp=np.interp(fine,coarse,wave)
    axs[1].plot(fine,reference-interp,lw=1);axs[1].set(xlabel="Normalized control time",
                     ylabel="Fine rendering − 900-sample interpolation",
                     title="Waveform resolution (diagnostic only)")
    for ax in axs:ax.grid(alpha=.2)
    _save(fig,ctx["figdir"],"F10_numerical_and_sampling_diagnostics.png")

def figure11_18_ensemble(ctx):
    maps=ctx["maps"]
    if len(maps)<=1:
        print("F11–F18 skipped: N=1 cannot estimate disorder variability or cross-disorder generalization.")
        return
    df=ctx["data"];valid=_valid(df);methods=ctx["methods"];out=ctx["figdir"]
    runbest=valid.sort_values("reward").groupby(["map","method_key","seed"],sort=False).tail(1).copy()
    # F11: one observation per disorder per method, averaging optimizer seeds.
    means=runbest.groupby(["map","method_key"],as_index=False).reward.mean()
    fig,ax=plt.subplots(figsize=(10,4.5))
    vals=[means[means.method_key==m].reward.to_numpy() for m in methods]
    ax.boxplot(vals,showmeans=True);ax.set_xticks(np.arange(1,len(methods)+1),methods,rotation=30)
    ax.set(ylabel="Best measured reward",title="F11 — Per-disorder best reward (averaged over seeds)")
    ax.grid(alpha=.2);_save(fig,out,"F11_reward_distributions.png")
    # F12 paired difference on the SAME disorder maps, against GA where present.
    baseline=next((m for m in methods if "GA" in m.upper() and "QD" not in m.upper()),methods[0])
    pivot=means.pivot(index="map",columns="method_key",values="reward").reindex(index=maps)
    fig,ax=plt.subplots(figsize=(9,4.7))
    for m in methods:
        if m==baseline:continue
        d=pivot[m]-pivot[baseline]
        ax.plot(range(len(maps)),d,"o-",label=m)
    ax.axhline(0,color="black",lw=1);ax.set(xticks=range(len(maps)),xticklabels=maps,
        ylabel="Paired reward difference",title="F12 — Method minus "+baseline)
    ax.tick_params(axis="x",rotation=70);ax.grid(alpha=.2);ax.legend(fontsize=7)
    _save(fig,out,"F12_paired_reward_differences.png")
    # F13 winning complexity across disorders.
    fig,ax=plt.subplots(figsize=(10,4.5))
    for m in methods:
        sub=runbest[runbest.method_key==m]
        x=[maps.index(v) for v in sub["map"]];ax.scatter(x,sub.complexity,s=22,alpha=.5,label=m)
    ax.set(xticks=range(len(maps)),xticklabels=maps,ylabel=ctx["complexity_label"],
           title="F13 — Winner complexity versus disorder")
    ax.tick_params(axis="x",rotation=70);ax.grid(alpha=.2);ax.legend(fontsize=7)
    _save(fig,out,"F13_winner_complexity.png")
    # F14 fraction of observed gain retained with <=1 complexity unit.
    rows=[]
    for mid in maps:
        d=valid[valid["map"]==mid]
        if d.empty:continue
        best=float(d.reward.max());small=d[d.complexity<=1]
        zero=_zero_reward(ctx,mid)
        rows.append((mid,(float(small.reward.max())-zero)/(best-zero) if len(small) and np.isfinite(zero) and best>zero else np.nan))
    fig,ax=plt.subplots(figsize=(9,4.3))
    ax.bar([p[0] for p in rows],[p[1] for p in rows])
    ax.set(ylabel="Fraction of discovered improvement retained",
           title="F14 — Parsimony (complexity ≤1) at same disorder")
    ax.tick_params(axis="x",rotation=70);ax.grid(alpha=.2)
    _save(fig,out,"F14_simple_gain_retention.png")
    # F15: optional measured champion cross-disorder transfer, never surrogate predictions.
    if ctx.get("run_transfer",False):
        from oracle import evaluate_fields
        transfers=[]
        for source in maps:
            chosen=_selected(ctx,source)
            if not chosen:continue
            for kind,row in chosen.items():
                field=ctx["field"](row)
                for target in maps:
                    physical=evaluate_fields(field,defects_ec=ctx["defect"](target),
                                             max_substeps=ctx["max_substeps"])
                    good=bool(physical["valid"][0]) and not bool(physical["native_guard_exceeded"][0])
                    transfers.append(dict(source=source,target=target,kind=kind,
                        reward=float(physical["reward"][0]) if good else np.nan,valid=good))
        tr=pd.DataFrame(transfers);tr.to_csv(Path(out)/"F15_transfer_records.csv",index=False)
        fig,axs=plt.subplots(1,len(tr.kind.unique()),figsize=(5.5*len(tr.kind.unique()),5),squeeze=False)
        for ax,kind in zip(axs.flat,tr.kind.unique()):
            pivot=tr[(tr.kind==kind)&tr.valid].pivot(index="source",columns="target",values="reward").reindex(index=maps,columns=maps)
            im=ax.imshow(pivot,aspect="auto")
            ax.set(title=kind,xticks=range(len(maps)),xticklabels=maps,yticks=range(len(maps)),yticklabels=maps)
            ax.tick_params(axis="x",rotation=90);fig.colorbar(im,ax=ax)
        fig.suptitle("F15 — Validated measured cross-disorder transfer")
        _save(fig,out,"F15_cross_disorder_transfer.png")
    else:print("F15 transfer skipped; enable RUN_TRANSFER to perform expensive extra oracle evaluations.")
    # F16 random-effects-inspired corrected between-disorder variance.
    stats=[]
    for meth,g in runbest.groupby("method_key"):
        means_by_map=g.groupby("map").reward.mean()
        within=g.groupby("map").reward.var(ddof=1)
        nseeds=g.groupby("map").reward.size()
        variance_mean=float(means_by_map.var(ddof=1))
        sampling=float((within/nseeds).mean()) if within.notna().any() else np.nan
        stats.append(dict(method=meth,observed_between_map_mean_variance=variance_mean,
                     mean_within_seed_variance=float(within.mean()) if within.notna().any() else np.nan,
                     corrected_between_estimate=variance_mean-sampling if np.isfinite(sampling) else np.nan))
    stats=pd.DataFrame(stats);stats.to_csv(Path(out)/"F16_variance_decomposition.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,4.5))
    xx=np.arange(len(stats));ax.bar(xx-.18,stats.observed_between_map_mean_variance,width=.36,label="Between map means")
    ax.bar(xx+.18,stats.mean_within_seed_variance,width=.36,label="Within-map optimizer seeds")
    ax.set(xticks=xx,xticklabels=stats.method,ylabel="Variance of reward",title="F16 — Descriptive variance components")
    ax.tick_params(axis="x",rotation=30);ax.legend(fontsize=8);ax.grid(alpha=.2)
    _save(fig,out,"F16_variance_components.png")
    # F17 fixed pooled target per map; unsuccessful runs treated as censored.
    records=[]
    for mid in maps:
        g=valid[valid["map"]==mid]
        if g.empty:continue
        zero=_zero_reward(ctx,mid);top=float(g.reward.max())
        threshold=zero+.95*(top-zero)
        for (meth,seed),full in df[df["map"]==mid].groupby(["method_key","seed"]):
            hit=full[full.valid & (full.reward>=threshold)]
            records.append(dict(map=mid,method=meth,seed=seed,reached=not hit.empty,
                                first_evaluation=int(hit.evaluation.min()) if not hit.empty else np.nan))
    events=pd.DataFrame(records);events.to_csv(Path(out)/"F17_target_attainment.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,4.5))
    horizon=int(df.evaluation.max())
    for meth,g in events.groupby("method"):
        xs=np.arange(1,horizon+1)
        times=g.first_evaluation.dropna().to_numpy()
        yy=np.array([(times<=i).sum()/len(g) for i in xs])
        ax.step(xs,yy,where="post",label=meth)
    ax.set(xlabel="Attempted oracle evaluations",ylabel="Fraction of runs reaching target",
           ylim=(0,1.05),title="F17 — 95% pooled target attainment (unreached = censored)")
    ax.grid(alpha=.2);ax.legend(fontsize=8);_save(fig,out,"F17_time_to_target.png")
    # F18: common physical disorder descriptor and achieved reward.
    records=[]
    for mid in maps:
        a=ctx["defect"](mid);pos=np.argwhere(np.linalg.norm(a,axis=2)>0)
        if len(pos)<2:continue
        dist=np.sqrt(np.sum((pos[:,None,:]-pos[None,:,:])**2,axis=2))
        np.fill_diagonal(dist,np.inf)
        records.append(dict(map=mid,mean_neighbor_distance=float(np.mean(dist.min(axis=1))),
                            max_reward=float(valid[valid["map"]==mid].reward.max())))
    desc=pd.DataFrame(records);desc.to_csv(Path(out)/"F18_disorder_descriptors.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for _,r in desc.iterrows():
        ax.scatter(r.mean_neighbor_distance,r.max_reward)
        ax.annotate(r["map"],(r.mean_neighbor_distance,r.max_reward),fontsize=7)
    ax.set(xlabel="Mean nearest defect-site distance (pixels)",ylabel="Maximum observed curl",
           title="F18 — Disorder geometry and accessible reward")
    ax.grid(alpha=.2);_save(fig,out,"F18_disorder_structure_vs_reward.png")

def run_common(ctx):
    d=ctx["data"]
    if len(d)==0:raise RuntimeError("No measured oracle histories; execute the experiment cell first.")
    for k in ("map","method_key","seed","evaluation","generation","reward","valid","complexity","peak_ec","rms_ec","predicted_mean","predicted_std"):
        if k not in d.columns:raise KeyError("Missing common data column "+k)
    figure01_disorders(ctx)
    figure02_convergence(ctx)
    figure03_reference_pareto(ctx)
    figure04_waveform_histories(ctx)
    figure05_winners(ctx)
    figure06_incumbent_diagnostics(ctx)
    figure07_surrogate_calibration(ctx)
    figure08_09_10_physical(ctx)
    figure11_18_ensemble(ctx)
    print("Standard v3 figures saved:",Path(ctx["figdir"]).resolve())

def appendable_per_disorder_figures(ctx,only_disorders=None,history_steps=30):
    """
    F19: per-disorder best-so-far waveform history for each algorithm.
    F20: per-disorder reward–complexity Pareto fronts with method and pooled fronts.
    Can be called as the last code cell in any FerroSim v3 notebook.
    """
    df=ctx["data"];out=Path(ctx["figdir"]);maps=ctx["maps"] if only_disorders is None else list(only_disorders)
    for mid in maps:
        if mid not in ctx["maps"]:raise ValueError(f"Unknown disorder {mid}")
        current=df[df["map"]==mid]
        if current.empty:continue
        methods=ctx["methods"];fig,axs=plt.subplots(len(methods),2,figsize=(12,3.1*len(methods)),squeeze=False,layout="constrained")
        for i,meth in enumerate(methods):
            g=current[(current.method_key==meth)&(current.seed==current.seed.min())].sort_values("evaluation")
            if g.empty:continue
            valid=_valid(g)
            if valid.empty:continue
            checkpoint=np.unique(np.linspace(0,len(g)-1,min(history_steps,len(g)),dtype=int))
            bestrow=None;best=-np.inf;wave=[]
            checkpoints=set(checkpoint.tolist())
            for j,(_,r) in enumerate(g.iterrows()):
                if bool(r.valid) and np.isfinite(r.reward) and float(r.reward)>best:
                    best=float(r.reward);bestrow=r
                if j in checkpoints:wave.append(ctx["field"](bestrow) if bestrow is not None else np.zeros(900))
            waves=np.asarray(wave);extent=[0,1,int(g.evaluation.iloc[checkpoint[0]]),int(g.evaluation.iloc[checkpoint[-1]])]
            axs[i,0].imshow(waves,origin="lower",aspect="auto",extent=extent,cmap="coolwarm")
            axs[i,0].set(title=meth+" — incumbent waveform",xlabel="Control time/T",ylabel="Evaluation")
            p=_front(g);axs[i,1].scatter(valid.complexity,valid.reward,s=7,alpha=.18)
            if len(p):axs[i,1].plot(p.complexity,p.reward,"o-",ms=3)
            axs[i,1].set(title=meth+" — measured Pareto front",xlabel=ctx["complexity_label"],ylabel="Curl reward")
            axs[i,1].grid(alpha=.2)
        fig.suptitle("F19 — Per-disorder waveform and method Pareto history: "+str(mid))
        _save(fig,out,"F19_history_and_method_pareto_"+str(mid)+".png")
    rows,cols=_grid(len(maps))
    fig,axs=plt.subplots(rows,cols,figsize=(4.3*cols,3.4*rows),squeeze=False,layout="constrained")
    for ax in axs.flat:ax.axis("off")
    for ax,mid in zip(axs.flat,maps):
        part=df[df["map"]==mid]
        for meth,g in part.groupby("method_key"):
            v=_valid(g);ax.scatter(v.complexity,v.reward,s=4,alpha=.08)
            f=_front(g)
            if len(f):ax.plot(f.complexity,f.reward,lw=1,marker=".",label=meth)
        globalfront=_front(part)
        if len(globalfront):ax.plot(globalfront.complexity,globalfront.reward,"k--",lw=2,label="pooled")
        ax.axis("on");ax.set(title=mid,xlabel=ctx["complexity_label"],ylabel="Curl reward");ax.grid(alpha=.2)
    if len(maps)<=4:axs.flat[0].legend(fontsize=6)
    fig.suptitle("F20 — Pareto fronts across disorder realizations (within-representation)")
    _save(fig,out,"F20_all_disorder_pareto_fronts.png")
