"""FerroSim v4 common analysis and representation adapters.
Embedded in each standalone notebook. The Legendre Try statistics remain the baseline.
No surrogate predictions are represented as measured physics.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from representation import fields, complexity

METHODS=[]
NAMES={}

def configure_methods(representation):
    global METHODS,NAMES
    if representation in ("Fourier","Legendre","Spline"):
        METHODS[:]=["ga_dkl","gflownet_dkl","diffusion_dkl","qd_dkl"]
        NAMES.clear()
        NAMES.update({"ga_dkl":"GA–DKL","gflownet_dkl":"GFlowNet–DKL",
                      "diffusion_dkl":"Diffusion–DKL","qd_dkl":"QD–DKL"})
    elif representation=="Segments":
        METHODS[:]=["GA-DKL","QD-DKL","Sequential-GFlowNet-inspired-DKL",
                   "Denoising-diffusion-inspired-DKL"]
        NAMES.clear();NAMES.update({m:m for m in METHODS})
    elif representation=="Grammar":
        METHODS[:]=["simple/GA-DKL","simple/QD-DKL","full/GA-DKL","full/QD-DKL"]
        NAMES.clear();NAMES.update({m:m.replace("/"," / ") for m in METHODS})
    else:raise ValueError("Unknown representation "+str(representation))
    return METHODS,NAMES

def load_histories(root,representation):
    root=Path(root)
    configure_methods(representation)
    if representation in ("Fourier","Legendre","Spline"):
        from analysis import load
        df=load(root)
        if df.empty:raise RuntimeError("No physical histories in "+str(root))
    else:
        path=root/"all_evaluations.csv"
        if not path.is_file():raise FileNotFoundError(path)
        df=pd.read_csv(path)
        if df.empty:raise RuntimeError("No recorded evaluations")
        df["native_method"]=df["method"].astype(str)
        if representation=="Grammar":
            df["method"]=df["grammar"].astype(str)+"/"+df["native_method"]
        df["complexity"]=df["turns"].astype(int)
        if "raw_turns" not in df:df["raw_turns"]=np.nan
        if "selection_source" not in df:df["selection_source"]="dkl_acquisition"
        if "proposal_source" not in df:
            src=("operator" if "operator" in df else "operation")
            df["proposal_source"]=df[src].fillna("unknown").astype(str)
        if "substeps" not in df:df["substeps"]=0
        if "rhs_calls" not in df:df["rhs_calls"]=0
        vectors=[None]*len(df)
        cols=["map","seed","method"] + (["grammar"] if representation=="Grammar" else [])
        for key,g in df.groupby(cols,sort=False):
            row=g.iloc[0]
            mid=str(row["map"]);seed=int(row["seed"]);native=str(row["native_method"])
            if representation=="Segments":
                file=root/mid/native/("seed_"+str(seed))/"coordinates.npz"
                key_name="x"
            else:
                file=root/mid/(str(row["grammar"])+"__"+native)/("seed_"+str(seed))/"programs.npz"
                key_name="programs"
            with np.load(file,allow_pickle=False) as archive:arr=archive[key_name].copy()
            order=g.sort_values("evaluation")
            if len(arr)!=len(order):
                raise RuntimeError("Coordinate archive length disagrees with measured history: "+str(file))
            for idx,vector in zip(order.index,arr):vectors[int(idx)]=vector
        if any(v is None for v in vectors):
            raise RuntimeError("Missing serialized controls in historical evaluation rows")
        df["coefficients"]=vectors
    for key,default in (("predicted_mean",np.nan),("predicted_std",np.nan),
                        ("interior",np.nan),("peak_ec",np.nan),("rms_ec",np.nan),
                        ("substeps",0),("rhs_calls",0),("selection_source","not_recorded"),
                        ("proposal_source","not_recorded")):
        if key not in df:df[key]=default
    if df["valid"].dtype==object:
        df["valid"]=df["valid"].astype(str).str.strip().str.lower().eq("true")
    else:df["valid"]=df["valid"].astype(bool)
    df["method"]=df["method"].astype(str)
    if not set(df.method).issubset(set(METHODS)):
        raise ValueError("Unexpected method labels "+str(set(df.method)-set(METHODS)))
    df["evaluation"]=df["evaluation"].astype(int)
    df["generation"]=df["generation"].astype(int)
    df["seed"]=df["seed"].astype(int)
    return df

def valid(df):
    return df[df.valid & np.isfinite(df.reward)].copy()

def pareto(df):
    q=valid(df)
    if q.empty:return q
    rows=[];best=-np.inf
    for comp in sorted(q.complexity.unique()):
        g=q[q.complexity==comp];r=g.loc[g.reward.idxmax()]
        if float(r.reward)>best+1e-12:
            rows.append(r);best=float(r.reward)
    return pd.DataFrame(rows) if rows else q.iloc[:0]

def incumbent_history(df):
    z=df.sort_values("evaluation");best=-np.inf;idx=None;out=[]
    for _,row in z.iterrows():
        if bool(row.valid) and np.isfinite(row.reward) and float(row.reward)>best:
            best=float(row.reward);idx=row
        out.append(idx)
    return z,out

def _save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.show();plt.close(fig)

def plot_convergence(df,figures,baseline_fn):
    fig,axs=plt.subplots(1,2,figsize=(13,4.7),layout="constrained")
    for method in METHODS:
        g=df[(df.method==method)&(df["map"]=="reference")]
        for j,(seed,sub) in enumerate(g.groupby("seed")):
            sub,history=incumbent_history(sub)
            vals=[float(r.reward) if r is not None else np.nan for r in history]
            axs[0].step(sub.evaluation,vals,where="post",alpha=.6,label=NAMES[method] if j==0 else None)
        traces=[]
        for (mid,seed),sub in df[df.method==method].groupby(["map","seed"]):
            context=df[(df["map"]==mid)&(df.seed==seed)]
            baseline=float(baseline_fn[(mid,seed)])
            observed=context.loc[context.valid & np.isfinite(context.reward),"reward"]
            target=float(observed.max()) if len(observed) else np.nan
            sub,h=incumbent_history(sub)
            reward=np.array([float(r.reward) if r is not None else np.nan for r in h])
            denom=target-baseline
            normalized=(reward-baseline)/denom if denom>1e-12 else np.zeros(len(reward))
            traces.append(normalized)
        if traces:
            n=min(map(len,traces));a=np.asarray([z[:n] for z in traces])
            axs[1].plot(np.arange(1,n+1),np.nanmean(a,axis=0),label=NAMES[method])
            if len(a)>1:
                se=np.nanstd(a,axis=0,ddof=1)/np.sqrt(len(a))
                axs[1].fill_between(np.arange(1,n+1),np.nanmean(a,axis=0)-se,
                     np.nanmean(a,axis=0)+se,alpha=.1)
    axs[0].set(title="Reference disorder: best measured curl",xlabel="Physical evaluations / method",
               ylabel="Best terminal curl")
    axs[1].set(title="Fraction of pooled gain above best starting control",
               xlabel="Physical evaluations / method",ylabel="Normalized observed gain")
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
    _save(fig,Path(figures)/"02_convergence.png")

def plot_reference_pareto(df,figures):
    ref=valid(df[df["map"]=="reference"])
    fig,axs=plt.subplots(2,2,figsize=(12,9),squeeze=False,layout="constrained")
    for ax,method in zip(axs.flat,METHODS):
        sub=ref[ref.method==method]
        for generation,g in sub.groupby("generation"):
            ax.scatter(g.complexity,g.reward,s=9,alpha=.3)
        front=pareto(sub)
        if len(front):ax.plot(front.complexity,front.reward,"ko-",lw=1.2,ms=3)
        ax.set(title=NAMES[method],xlabel="Resolved turns (2% prominence)",ylabel="Curl reward")
        ax.grid(alpha=.2)
    _save(fig,Path(figures)/"03_reference_pareto.png")

def plot_control_evolution(df,figures,representation,seed=None):
    """Reference-disorder, single-seed (not pooled) coefficient+waveform evolution."""
    ref=df[df["map"]=="reference"]
    seed=int(ref.seed.min()) if seed is None else int(seed)
    fig,axs=plt.subplots(4,2,figsize=(13,12),layout="constrained")
    for i,m in enumerate(METHODS):
        g=ref[(ref.method==m)&(ref.seed==seed)]
        if g.empty:continue
        waves=[];coordinates=[]
        for gen in sorted(g.generation.unique()):
            upto=valid(g[g.generation<=gen])
            if upto.empty:continue
            w=upto.loc[upto.reward.idxmax()]
            wave=np.asarray(fields(w.coefficients,n=900),float).reshape(-1)
            waves.append(wave)
            if representation in ("Fourier","Legendre","Spline"):
                from representation import full_coefficients
                coordinates.append(np.asarray(full_coefficients(w.coefficients),float).ravel())
            else:coordinates.append(np.asarray(w.coefficients,float).ravel())
        if not waves:continue
        coords=np.stack(coordinates);wavs=np.stack(waves)
        for ax,arr,label in [(axs[i,0],coords,"Physical basis coefficients" if representation in ("Fourier","Legendre","Spline") else "Encoded program/segments"),
                             (axs[i,1],wavs,"Applied waveform E/Ec")]:
            vmax=max(float(np.nanmax(np.abs(arr))),1e-10)
            ax.imshow(arr,aspect="auto",origin="lower",cmap="coolwarm",vmin=-vmax,vmax=vmax)
            ax.set(title=NAMES[m]+" | "+label,xlabel="Index / control time",ylabel="Generation")
    fig.suptitle("Best-so-far evolution on reference map, optimizer seed "+str(seed))
    _save(fig,Path(figures)/"04_control_evolution.png")

def plot_champion_and_descriptor(df,figures,representation):
    ref=valid(df[df["map"]=="reference"])
    if ref.empty:return
    champion=ref.loc[ref.reward.idxmax()]
    simplest=ref[ref.complexity<=1]
    choose=[("Best",champion)]
    if len(simplest):choose.append(("Best <=1 turn",simplest.loc[simplest.reward.idxmax()]))
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for label,row in choose:
        waveform=np.asarray(fields(row.coefficients,900),float).reshape(-1)
        axs[0].plot(np.linspace(0,1,len(waveform)),waveform,label=label+" R="+format(row.reward,".4g"))
        if representation in ("Fourier","Legendre","Spline"):
            from representation import full_coefficients
            z=np.asarray(full_coefficients(row.coefficients)).ravel()
        else:z=np.asarray(row.coefficients).ravel()
        axs[1].plot(np.arange(len(z)),z,label=label)
    axs[0].set(title="Champion and simple physical controls",xlabel="Normalized time",ylabel="E/Ec")
    axs[1].set(title="Descriptor/encoded parameters",xlabel="Descriptor index",ylabel="Parameter value")
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    _save(fig,Path(figures)/"05_best_waveform_and_descriptor.png")

def plot_modal_energy(df,figures,representation):
    if representation not in ("Fourier","Legendre","Spline"):
        print("Modal energy plot not applicable to variable-topology representation.")
        return
    from representation import full_coefficients
    ref=valid(df[df["map"]=="reference"])
    champion=ref.loc[ref.reward.idxmax()]
    simpler=ref[ref.complexity<=1]
    comparisons=[("Best",champion)]
    if len(simpler):comparisons.append(("Simple",simpler.loc[simpler.reward.idxmax()]))
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
    for label,row in comparisons:
        a=np.asarray(full_coefficients(row.coefficients)).ravel()
        axs[0].plot(np.arange(len(a)),a,label=label)
        if representation=="Legendre":
            energy=a*a/(2*np.arange(len(a))+1)
        elif representation=="Fourier":
            energy=a*a/2
        else:
            # Spline basis is not orthogonal; show physically rendered RMS and label.
            w=np.asarray(fields(row.coefficients,900)).ravel()
            energy=np.full(len(a),np.nan)
        if representation!="Spline":axs[1].plot(np.arange(len(energy)),energy,label=label)
    axs[0].set(title="Physical expansion coefficients",xlabel="Mode / basis index",ylabel="Amplitude")
    axs[1].set(title="Orthogonal modal energy" if representation!="Spline" else "Spline modal energy not additive",
               xlabel="Mode index",ylabel="Contribution to mean squared field")
    if representation=="Spline":
        axs[1].text(.5,.5,"Spline basis is nonorthogonal.\nCompare rendered RMS E/Ec.",ha="center",va="center",transform=axs[1].transAxes)
    for ax in axs:ax.grid(alpha=.2)
    axs[0].legend()
    _save(fig,Path(figures)/"05_modal_descriptors.png")

def plot_disorder_summary(df,figures,tables,maps):
    v=valid(df);rows=[]
    for mid in maps:
        g=v[v["map"]==mid]
        if g.empty:continue
        best=g.loc[g.reward.idxmax()]
        simple=g[g.complexity<=1]
        rows.append({"map":mid,"best_reward":best.reward,"best_turns":best.complexity,
         "best_simple_reward":float(simple.reward.max()) if len(simple) else np.nan})
    t=pd.DataFrame(rows);t.to_csv(Path(tables)/"disorder_summary.csv",index=False)
    if len(maps)>1:
        fig,axs=plt.subplots(1,2,figsize=(12,4),layout="constrained")
        axs[0].plot(t["map"],t.best_reward,"o-",label="Best")
        axs[0].plot(t["map"],t.best_simple_reward,"s--",label="Simple")
        axs[1].bar(t["map"],t.best_turns)
        axs[0].set(title="Disorder-dependent reward",ylabel="Curl")
        axs[1].set(title="Winner complexity",ylabel="Resolved turns")
        for ax in axs:ax.tick_params(axis="x",rotation=65);ax.grid(alpha=.2)
        axs[0].legend()
        _save(fig,Path(figures)/"06_disorder_summary.png")
    return t

def plot_all_disorder_pareto(df,maps,figures):
    import math
    columns=min(4,len(maps));rows=int(math.ceil(len(maps)/columns))
    fig,axs=plt.subplots(rows,columns,figsize=(4.3*columns,3.6*rows),squeeze=False,layout="constrained")
    for ax in axs.flat:ax.axis("off")
    for ax,mid in zip(axs.flat,maps):
        g=valid(df[df["map"]==mid])
        for method,z in g.groupby("method"):
            p=pareto(z)
            if len(p):ax.plot(p.complexity,p.reward,marker=".",lw=1.1,label=NAMES.get(method,method))
        pooled=pareto(g)
        if len(pooled):ax.plot(pooled.complexity,pooled.reward,"k--",lw=1.7,label="Pooled")
        ax.axis("on");ax.set(title=mid,xlabel="Resolved turns",ylabel="Curl");ax.grid(alpha=.2)
    if len(maps)<=4:axs.flat[0].legend(fontsize=6)
    _save(fig,Path(figures)/"13_all_disorder_pareto_montage.png")


def write_selected(df,tables):
    """Save physically evaluated winners and low-turn alternatives, for every disorder."""
    tables=Path(tables)
    selected=tables/"selected_controls";selected.mkdir(parents=True,exist_ok=True)
    rows=[]
    for mid,g in valid(df).groupby("map"):
        champion=g.loc[g.reward.idxmax()]
        simple=g[g.complexity<=1]
        picks=[("best",champion)]
        if len(simple):picks.append(("simple",simple.loc[simple.reward.idxmax()]))
        for label,row in picks:
            wave=np.asarray(fields(row.coefficients,n=900),float).reshape(-1)
            fname=("".join(q if q.isalnum() or q in "-_" else "_" for q in str(mid))+"__"+label+".npz")
            np.savez_compressed(selected/fname,
                coordinates=np.asarray(row.coefficients,float),waveform_Ec=wave,
                measured_reward=float(row.reward),generation=int(row.generation),
                evaluation=int(row.evaluation),seed=int(row.seed))
            rows.append(dict(map=mid,kind=label,method=row.method,seed=int(row.seed),
                             evaluation=int(row.evaluation),generation=int(row.generation),
                             reward=float(row.reward),complexity=int(row.complexity),
                             waveform_file=str(Path("selected_controls")/fname)))
    result=pd.DataFrame(rows)
    result.to_csv(tables/"selected_controls.csv",index=False)
    return result
