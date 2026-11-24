"""Transparent, offline analyses of observed FerroSim histories.

Everything in this module consumes measured physical outcomes. Figures handle a
single map, a single seed, and zero optimization generations. No synthetic
rewards, extrapolation or fabricated uncertainty intervals are generated.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from representation import fields, complexity
from common_analysis import METHODS, NAMES, pareto


def _saved(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def _valid(df):
    return df.loc[df.valid & np.isfinite(df.reward)].copy()


def _best(g):
    valid = _valid(g)
    return None if valid.empty else valid.loc[valid.reward.idxmax()]


def _safe_name(x):
    return ''.join(c if (c.isalnum() or c in ('-','_')) else '_' for c in str(x))


def audit_budget(df, maps, seeds, initial_count, generations, tables, figures, shared_initial_cache=True):
    tables,figures = Path(tables),Path(figures)
    plan = len(maps)*len(seeds)*len(METHODS)*(initial_count+32*generations)
    distinct = (len(maps)*len(seeds)*(initial_count+len(METHODS)*32*generations)
                if shared_initial_cache else plan)
    rows=[]
    for (mp, seed, method),g in df.groupby(['map','seed','method'],sort=False):
        good = _valid(g)
        rows.append({'map':mp,'seed':seed,'method':method,'attempts':len(g),
            'valid':len(good),'invalid':len(g)-len(good),
            'initialization':int((g.generation==0).sum()),
            'new_queries':int((g.generation>0).sum()),
            'best_curl':float(good.reward.max()) if len(good) else np.nan,
            'best_interior_curl':float(good.interior.max()) if len(good) else np.nan,
            'best_turns':int(good.loc[good.reward.idxmax()].complexity) if len(good) else np.nan,
            'peak_ec_best':float(good.loc[good.reward.idxmax()].peak_ec) if len(good) else np.nan,
            'rhs_calls':int(g.rhs_calls.sum()) if 'rhs_calls' in g else 0,
            'substeps':int(g.substeps.sum()) if 'substeps' in g else 0})
    summary = pd.DataFrame(rows)
    summary.to_csv(tables/'run_audit.csv',index=False)
    status = {'recorded_method_observations':int(len(df)),
        'expected_method_observations':int(plan),
        'expected_distinct_physical_calls':int(distinct),
        'valid_recorded_observations':int(df.valid.sum()),
        'invalid_recorded_observations':int((~df.valid).sum()),
        'shared_initial_cache':bool(shared_initial_cache),
        'n_maps':int(len(maps)),'n_seeds':int(len(seeds)),
        'initial_controls_per_map_seed':int(initial_count),
        'new_evaluations_per_method_generation':32,
        'generations':int(generations)}
    (tables/'budget.json').write_text(json.dumps(status,indent=2))
    if len(df)!=plan: raise AssertionError(f'Incomplete physical histories: {len(df)} / {plan}')
    if len(summary)!=len(maps)*len(seeds)*len(METHODS): raise AssertionError('Missing method/seed/map history')
    if (summary.initialization!=initial_count).any(): raise AssertionError('Missing initial observations')
    if (summary.new_queries!=32*generations).any(): raise AssertionError('Missing new observations')
    print('Evaluation accounting:', status)
    return summary


def shared_initial_baselines(df):
    """Per-map, per-seed best VALID initial control pooled across methods.
    Modal runs share the initial observations; Grammar may not.
    """
    df = _valid(df)
    init=df[df.generation==0]
    # Common baseline across methods; initial control populations may differ for Grammar.
    return init.groupby(['map','seed']).reward.max().to_dict()


def exact_target_attainment(df,tables):
    b=shared_initial_baselines(df)
    records=[]
    for mp,g in df.groupby('map'):
        gv=_valid(g)
        target=float(gv.reward.max()) if len(gv) else np.nan
        for (method,seed),r in g.groupby(['method','seed']):
            baseline=float(b[(mp,seed)])
            for fraction in (.95,.99):
                threshold=baseline+fraction*(target-baseline)
                hit=r[r.valid & (r.reward>=threshold)]
                records.append({'map':mp,'method':method,'seed':int(seed),
                    'baseline_best_initial':baseline,'pooled_observed_target':target,
                    'fraction':fraction,'threshold':threshold,
                    'reached':not hit.empty,
                    'first_evaluation':int(hit.evaluation.min()) if len(hit) else np.nan,
                    'initially_attained':bool(len(hit) and int(hit.evaluation.min())<=int((r.generation==0).sum()))})
    t=pd.DataFrame(records)
    t.to_csv(Path(tables)/'exact_target_attainment.csv',index=False)
    return t


def plot_pareto_by_disorder(df, map_ids, figures, tables, all_maps=True):
    root=Path(figures)/'pareto_by_disorder';root.mkdir(parents=True,exist_ok=True)
    p_rows=[]
    selected=map_ids if all_maps else map_ids[:1]
    for mid in selected:
        g=_valid(df[df['map']==mid]); fig,axs=plt.subplots(2,2,figsize=(11,8),squeeze=False,layout='constrained')
        for ax,method in zip(axs.flat,METHODS):
            h=g[g.method==method]
            if not h.empty:
                ax.scatter(h.complexity,h.reward,s=11,alpha=.20)
                p=pareto(h)
                if len(p): ax.plot(p.complexity,p.reward,'o-',ms=3.5,lw=1.4)
                for _,x in p.iterrows():
                    p_rows.append(dict(map=mid,method=method,complexity=int(x.complexity),
                        reward=float(x.reward),seed=int(x.seed),evaluation=int(x.evaluation)))
            ax.set_title(NAMES[method]);ax.set(xlabel='Prominent turns (2%)',ylabel='Measured curl');ax.grid(alpha=.2)
        fig.suptitle('Observed reward–complexity Pareto fronts | '+str(mid))
        _saved(fig,root/(_safe_name(mid)+'.png'))
    pd.DataFrame(p_rows,columns=['map','method','complexity','reward','seed','evaluation']).to_csv(Path(tables)/'pareto_by_disorder.csv',index=False)


def plot_best_waveforms_by_disorder(df, map_ids, figures, tables, all_maps=True):
    root=Path(figures)/'champions_by_disorder';root.mkdir(parents=True,exist_ok=True)
    result=[];selected=map_ids if all_maps else map_ids[:1]
    for mid in selected:
        g=_valid(df[df['map']==mid]);fig,axs=plt.subplots(2,2,figsize=(11,7.2),squeeze=False,layout='constrained')
        for ax,method in zip(axs.flat,METHODS):
            h=g[g.method==method];w=_best(h)
            if w is not None:
                wave=np.asarray(fields(w.coefficients,n=900),float).reshape(-1)
                ax.plot(np.linspace(0,1,len(wave)),wave,lw=1.5,label='Best')
                simple=h[h.complexity<=1];s=_best(simple)
                if s is not None:
                    sw=np.asarray(fields(s.coefficients,n=900),float).reshape(-1)
                    ax.plot(np.linspace(0,1,len(sw)),sw,lw=1.0,ls='--',label='Best ≤1 turn')
                ax.legend(fontsize=8)
                result.append({'map':mid,'method':method,'seed':int(w.seed),
                    'evaluation':int(w.evaluation),'best_curl':float(w.reward),
                    'turns':int(w.complexity),'simple_curl':float(s.reward) if s is not None else np.nan})
            ax.set_title(NAMES[method]);ax.set(xlabel='Time / T',ylabel='E / Ec');ax.grid(alpha=.2)
        fig.suptitle('Per-method champion waveforms | '+str(mid))
        _saved(fig,root/(_safe_name(mid)+'.png'))
    pd.DataFrame(result,columns=['map','method','seed','evaluation','best_curl','turns','simple_curl']).to_csv(
        Path(tables)/'champions_by_disorder.csv',index=False)


def complexity_retention(df, figures, tables):
    """Find minimum complexity retaining 95/99% of gain above initialization."""
    base=shared_initial_baselines(df);rows=[]
    for (mid,seed,method),g in df.groupby(['map','seed','method']):
        v=_valid(g)
        if v.empty:continue
        best=v.loc[v.reward.idxmax()]
        baseline=base[(mid,seed)]
        for turns in (0,1,3):
            q=v[v.complexity<=turns]
            selected=_best(q)
            rows.append({'map':mid,'seed':seed,'method':method,'criterion':f'≤{turns} turns',
               'best_reward':float(selected.reward) if selected is not None else np.nan,
               'unrestricted_reward':float(best.reward),'baseline':float(baseline),
               'resolved_turns':int(selected.complexity) if selected is not None else np.nan})
        for fraction in (.95,.99):
            threshold=baseline+fraction*(float(best.reward)-baseline)
            q=v[v.reward>=threshold]
            selected=q.loc[q.complexity.idxmin()] if len(q) else None
            rows.append({'map':mid,'seed':seed,'method':method,
               'criterion':f'{int(100*fraction)}% gain retained',
               'best_reward':float(selected.reward) if selected is not None else np.nan,
               'unrestricted_reward':float(best.reward),'baseline':float(baseline),
               'resolved_turns':int(selected.complexity) if selected is not None else np.nan})
    t=pd.DataFrame(rows);t.to_csv(Path(tables)/'complexity_retention.csv',index=False)
    subset=t[t.criterion.isin(['95% gain retained','99% gain retained'])]
    fig,axs=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for ax,(crit,h) in zip(axs,subset.groupby('criterion',sort=True)):
        a=[h[h.method==m].resolved_turns.dropna().to_numpy() for m in METHODS]
        a=[arr if len(arr) else np.array([np.nan]) for arr in a]
        ax.boxplot(a,positions=np.arange(1,5),showfliers=False)
        ax.set_xticks(range(1,5),[NAMES[m] for m in METHODS],rotation=25,ha='right')
        ax.set(title=crit,ylabel='Minimum observed prominent turns');ax.grid(axis='y',alpha=.2)
    _saved(fig,Path(figures)/'complexity_retention.png')
    return t


def seed_disorder_variance(df, figures, tables):
    """Separates variation of best observed per-run rewards over seed and disorder."""
    winners=[]
    for (mid,method,seed),g in df.groupby(['map','method','seed']):
        w=_best(g)
        if w is not None: winners.append({'map':mid,'method':method,'seed':int(seed),
            'best_reward':float(w.reward),'best_turns':int(w.complexity)})
    winners=pd.DataFrame(winners);winners.to_csv(Path(tables)/'run_champions.csv',index=False)
    summary=[]
    for method,g in winners.groupby('method'):
        per_map=g.groupby('map').best_reward
        means=per_map.mean()
        within=per_map.agg(lambda x: float(x.var(ddof=1)) if len(x)>1 else np.nan)
        summary.append({'method':method,'maps':len(means),'seeds_per_map':int(g.seed.nunique()),
            'mean_best_curl':float(g.best_reward.mean()),
            'between_map_variance_of_seed_means':float(means.var(ddof=1)) if len(means)>1 else np.nan,
            'mean_within_map_seed_variance':float(within.mean()) if within.notna().any() else np.nan,
            'single_map_or_seed_warning':bool(len(means)<2 or g.seed.nunique()<2)})
    parts=pd.DataFrame(summary);parts.to_csv(Path(tables)/'seed_disorder_variance.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    mids=list(dict.fromkeys(winners['map']))
    for method in METHODS:
        h=winners[winners.method==method]
        means=h.groupby('map').best_reward.mean().reindex(mids)
        axs[0].plot(range(len(mids)),means,'o-',ms=3,label=NAMES[method])
    axs[0].set(xticks=range(len(mids)),xticklabels=mids,title='Best curl per map (averaged over seeds)',ylabel='Curl')
    axs[0].tick_params(axis='x',rotation=60);axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
    arrays=[winners[winners.method==method].best_reward.to_numpy() for method in METHODS]
    axs[1].boxplot(arrays,showfliers=False)
    axs[1].set(xticks=range(1,len(METHODS)+1),xticklabels=[NAMES[m] for m in METHODS],
        title='Best reward across observed map–seed runs',ylabel='Curl')
    axs[1].tick_params(axis='x',rotation=25);axs[1].grid(alpha=.2)
    _saved(fig,Path(figures)/'seed_disorder_variance.png')
    return winners,parts


def proposal_attribution(df, figures, tables):
    # Archive uses two complementary notions: candidate origin and selection channel.
    h=df.copy()
    if 'selection_source' not in h: h['selection_source']='not_recorded'
    if 'proposal_source' not in h: h['proposal_source']='not_recorded'
    rows=[]
    for (m,stage,source),g in h.groupby(['method','selection_source','proposal_source']):
        v=_valid(g)
        rows.append({'method':m,'selection_source':stage,'proposal_source':source,
            'evaluations':len(g),'valid':len(v),
            'best_reward':float(v.reward.max()) if len(v) else np.nan,
            'mean_reward':float(v.reward.mean()) if len(v) else np.nan})
    t=pd.DataFrame(rows);t.to_csv(Path(tables)/'proposal_attribution.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    new=h[h.generation>0]
    if len(new):
        x=new.pivot_table(index='method',columns='proposal_source',values='reward',aggfunc='size',fill_value=0).reindex(METHODS)
        x.plot.bar(stacked=True,ax=axs[0],legend=True)
        axs[0].legend(fontsize=7)
        y=new.pivot_table(index='method',columns='selection_source',values='reward',aggfunc='size',fill_value=0).reindex(METHODS)
        y.plot.bar(stacked=True,ax=axs[1],legend=True)
        axs[1].legend(fontsize=7)
        for ax in axs: ax.tick_params(axis='x',rotation=20)
    else:
        for ax in axs: ax.text(.5,.5,'No optimization generations',ha='center',va='center',transform=ax.transAxes)
    axs[0].set_title('Candidates evaluated by origin');axs[1].set_title('Candidates evaluated by selection channel')
    for ax in axs:ax.set(xlabel='Optimizer',ylabel='Physical evaluations');ax.grid(axis='y',alpha=.2)
    _saved(fig,Path(figures)/'proposal_attribution.png')
    return t


def surrogate_calibration(df,figures,tables):
    h=df[(df.generation>0)&df.valid].copy()
    if not {'predicted_mean','predicted_std'}.issubset(h.columns):
        print('Prediction logging absent: skipping calibration.')
        return pd.DataFrame()
    h=h[np.isfinite(h.predicted_mean)&np.isfinite(h.predicted_std)&(h.predicted_std>=0)]
    rows=[];fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for method,g in h.groupby('method'):
        delta=g.reward-g.predicted_mean
        rows.append({'method':method,'n':len(g),'mean_error':float(delta.mean()),
            'rmse':float(np.sqrt(np.mean(delta**2))),'mae':float(np.mean(np.abs(delta))),
            'inside_1_sigma':float(np.mean(np.abs(delta)<=g.predicted_std)),
            'inside_2_sigma':float(np.mean(np.abs(delta)<=2*g.predicted_std))})
        axs[0].scatter(g.predicted_mean,g.reward,s=10,alpha=.25,label=NAMES.get(method,method))
        axs[1].scatter(g.predicted_std,abs(delta),s=10,alpha=.25,label=NAMES.get(method,method))
    if len(h):
        bounds=[float(min(h.predicted_mean.min(),h.reward.min())),float(max(h.predicted_mean.max(),h.reward.max()))]
        axs[0].plot(bounds,bounds,'k--',lw=1)
        axs[0].legend(fontsize=7)
    else:
        for ax in axs:ax.text(.5,.5,'No predicted/test observations',ha='center',va='center',transform=ax.transAxes)
    axs[0].set(title='DKL selected-candidate prediction vs reality',xlabel='Predicted curl',ylabel='Measured curl')
    axs[1].set(title='Absolute prediction error vs predicted uncertainty',xlabel='Predicted σ',ylabel='Absolute error')
    for ax in axs:ax.grid(alpha=.2)
    _saved(fig,Path(figures)/'surrogate_calibration.png')
    t=pd.DataFrame(rows,columns=['method','n','mean_error','rmse','mae','inside_1_sigma','inside_2_sigma'])
    t.to_csv(Path(tables)/'surrogate_calibration.csv',index=False)
    return t


def plot_additional_complexity_metrics(df,figures,tables):
    """Raw reversals, amplitude and interior curl guard against metric gaming."""
    records=[]
    for (mp,method,seed),g in df.groupby(['map','method','seed']):
        w=_best(g)
        if w is not None:
            wave=np.asarray(fields(w.coefficients,n=900),dtype=float).reshape(-1)
            d=np.diff(wave)
            records.append({'map':mp,'method':method,'seed':seed,'reward':float(w.reward),
                'interior_reward':float(w.interior),'boundary_contribution':float(w.reward-w.interior),
                'boundary_fraction':float((w.reward-w.interior)/w.reward) if w.reward!=0 else np.nan,
                'peak_ec':float(np.max(np.abs(wave))),'total_variation_ec':float(np.sum(np.abs(d))),
                'max_step_ec':float(np.max(np.abs(d))),
                'turns_2pct':int(w.complexity),
                'raw_turns':int(complexity(w.coefficients)['raw_turns'][0])})
    t=pd.DataFrame(records);t.to_csv(Path(tables)/'physical_complexity_metrics.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(15,4.0),layout='constrained')
    for method,g in t.groupby('method'):
        name=NAMES.get(method,method)
        axs[0].scatter(g.turns_2pct,g.reward,label=name,s=18,alpha=.65)
        axs[1].scatter(g.total_variation_ec,g.reward,label=name,s=18,alpha=.65)
        axs[2].scatter(g.boundary_fraction,g.reward,label=name,s=18,alpha=.65)
    for ax,xlabel in zip(axs,['Prominent turns','Total variation (E / Ec)','Boundary fraction of curl']):
        ax.set(xlabel=xlabel,ylabel='Best reward per run');ax.grid(alpha=.2)
    axs[0].legend(fontsize=7)
    _saved(fig,Path(figures)/'physical_complexity_metrics.png')
    return t


def export_index(figures,tables,root):
    figs=sorted(str(p.relative_to(root)) for p in Path(figures).rglob('*.png'))
    tabs=sorted(str(p.relative_to(root)) for p in Path(tables).rglob('*') if p.is_file())
    (Path(root)/'output_manifest.json').write_text(json.dumps({'figures':figs,'tables':tabs},indent=2))
    print('Saved',len(figs),'figures and',len(tabs),'tables. Output index:',Path(root)/'output_manifest.json')


def paired_method_comparison(winners, figures, tables, bootstrap_draws=1000):
    """Six paired contrasts with disorder-block bootstrap CIs.
    Seeds are repeated runs on the same disorder, *not* independent disorder
    replicates. Report a confidence interval only for >=2 distinct disorder maps.
    """
    from itertools import combinations
    t=winners.pivot_table(index=['map','seed'],columns='method',values='best_reward',aggfunc='first')
    rows=[];rng=np.random.default_rng(20261124)
    for a,b in combinations(METHODS,2):
        if a not in t or b not in t:continue
        pair=t[[a,b]].dropna()
        if pair.empty:continue
        diff=(pair[a]-pair[b]).astype(float)
        # Within each disorder, average across the available optimizer seeds.
        by_map=diff.groupby(level="map").mean()
        independent_maps=len(by_map)
        if independent_maps>1:
            map_means=by_map.to_numpy()
            ids=rng.integers(independent_maps,size=(bootstrap_draws,independent_maps))
            simulated=map_means[ids].mean(axis=1)
            low,high=np.quantile(simulated,[.025,.975])
        else:low=high=np.nan
        rows.append({'method_a':a,'method_b':b,'paired_map_seed_runs':len(diff),
            'independent_disorder_maps':independent_maps,
            'mean_difference_a_minus_b':float(by_map.mean()),
            'median_difference':float(diff.median()),
            'a_wins':int((diff>0).sum()),'ties':int((diff==0).sum()),
            'b_wins':int((diff<0).sum()),'bootstrap_CI95_low':float(low),
            'bootstrap_CI95_high':float(high),
            'CI_valid':bool(independent_maps>1),
            'interpretation':'Descriptive paired differences, block-bootstrap across disorder maps; no global optimum claim'})
    tab=pd.DataFrame(rows)
    tab.to_csv(Path(tables)/'paired_algorithm_differences.csv',index=False)
    M=np.zeros((len(METHODS),len(METHODS)),dtype=float)
    for _,z in tab.iterrows():
        i=METHODS.index(z.method_a);j=METHODS.index(z.method_b)
        M[i,j]=z.mean_difference_a_minus_b;M[j,i]=-M[i,j]
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    scale=max(float(np.max(abs(M))),1e-8)
    im=ax.imshow(M,cmap='coolwarm',vmin=-scale,vmax=scale)
    ax.set_xticks(range(len(METHODS)),[NAMES[m] for m in METHODS],rotation=35,ha='right')
    ax.set_yticks(range(len(METHODS)),[NAMES[m] for m in METHODS])
    ax.set_title('Paired map-level mean differences: row method minus column method')
    for i in range(len(METHODS)):
        for j in range(len(METHODS)):
            ax.text(j,i,f'{M[i,j]:+.4f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,shrink=.75,label='Difference in terminal curl')
    _saved(fig,Path(figures)/'paired_algorithm_differences.png')
    return tab
