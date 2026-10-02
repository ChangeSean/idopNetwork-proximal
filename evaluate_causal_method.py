"""Comparison of idop representation, proxy construction and information rank.

python evaluate_causal_method.py --outcome linear --reps 200 --boot 100
python evaluate_causal_method.py --outcome rmst --reps 200 --boot 100

Every replicate is saved. Reporting, proxy-role validity and accuracy use separate
denominators. The five fixed systems vary sample size, rank, topology and loading
scale; comparisons within a system use the same patients and bootstrap draws.
"""
import argparse
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from dgp import make_system
from idop_core import data_transformation, data_quasi_dynamic
from proximal import latent_state, proxy_roles, screen, proxy_strength, bridge_information_rank, proximal_bridge, ols_effect
from run_application import analyse_cohort
from run_causal_application import configuration, configuration_grid
from causal_survival import survival_pseudo_outcomes

ROOT=Path(__file__).resolve().parent
SCENARIOS=[
    dict(name='rank1_n400_branch',r=1,n=400,topology='branch',latent=.25,seed=11),
    dict(name='rank1_n1000_chain',r=1,n=1000,topology='chain',latent=.25,seed=23),
    dict(name='rank1_n1000_modules',r=1,n=1000,topology='modules',latent=.50,seed=37),
    dict(name='rank2_n1000_branch',r=2,n=1000,topology='branch',latent=.25,seed=41),
    dict(name='rank2_n1000_branch_strong',r=2,n=1000,topology='branch',latent=.35,seed=7),
]
ALPHAS=(.15,.20,.25,.30)


def system_for(setting):
    system=make_system(r=setting['r'],seed=setting['seed'])
    rng=np.random.default_rng(setting['seed']+500)
    O=system['iO']; theta=system['Theta']; theta[np.ix_(O,O)]=0
    for j in range(1,len(O)):
        if setting['topology']=='branch': parent=(j-1)//2
        elif setting['topology']=='chain': parent=j-1
        else:
            if j%6==0: continue
            parent=(j//6)*6
        theta[O[parent],O[j]]=rng.choice([-1,1])*rng.uniform(.3,.7)
    system['B']=np.linalg.inv(np.eye(system['p'])-theta)
    return system


def sample_system(system,setting,rep,outcome):
    rng=np.random.default_rng(710000+10000*setting['seed']+rep)
    n,r,p=setting['n'],system['r'],system['p']; a=system['a0']; scale=setting['latent']
    F=rng.uniform(-np.sqrt(3),np.sqrt(3),(n,r))
    noise=np.ones(p); noise[np.r_[system['iW'],system['iZ']]]=.5
    eps=rng.uniform(-np.sqrt(3),np.sqrt(3),(n,p))*noise
    X=(scale*F@system['Lam'].T+eps)@system['B']
    if outcome=='linear':
        Y=system['gamma']*X[:,a]+X@system['pi']+F@system['lam_Y']+rng.normal(size=n)
        return dict(X=X,F=F,Y=Y,target=system['gamma'],T=None,D=None)
    # This bounded mixture gives an exact linear restricted-mean bridge.
    bound=(scale*np.sqrt(3)*np.abs(system['Lam']).sum(1)+np.sqrt(3)*noise)@np.abs(system['B'])
    pi=.10*system['pi']/np.dot(np.abs(system['pi']),bound)
    bf=np.full(r,.12/(np.sqrt(3)*r)); coefficient=1.5/24
    probability=.5-coefficient*X[:,a]-F@bf-X@pi
    assert coefficient*bound[a]+np.sqrt(3)*np.abs(bf).sum()+np.dot(np.abs(pi),bound)<.5
    assert np.all((probability>0)&(probability<1))
    early=rng.uniform(size=n)<probability
    true_time=np.where(early,12+rng.uniform(-2,2,n),60+rng.uniform(-5,5,n))
    censor_time=rng.exponential(100,n)
    observed=np.minimum(true_time,censor_time); events=(true_time<=censor_time).astype(int)
    return dict(X=X,F=F,Y=events,target=1.5,T=observed,D=events)


def truth_graph(system):
    graph=nx.DiGraph(); p,r=system['p'],system['r']; y=p+r
    graph.add_nodes_from(range(y+1))
    graph.add_edges_from((int(i),int(j)) for i,j in np.argwhere(system['Theta']!=0))
    graph.add_edges_from((p+k,j) for j,k in np.argwhere(system['Lam']!=0))
    graph.add_edges_from((int(j),y) for j in np.flatnonzero(system['pi']))
    graph.add_edge(system['a0'],y); graph.add_edges_from((p+k,y) for k in range(r))
    return graph,set(range(p,p+r)),y


def role_validity(graph,factors,y,a,Z,W):
    if not Z or not W:return dict(z_valid=np.nan,w_valid=np.nan,set_valid=np.nan)
    valid_z=[nx.is_d_separator(graph,{v},{y},factors|{a}) for v in Z]
    valid_w=[nx.is_d_separator(graph,{v},{a}|set(Z),factors) for v in W]
    return dict(z_valid=float(np.mean(valid_z)),w_valid=float(np.mean(valid_w)),
                set_valid=float(all(valid_z) and all(valid_w)))


def proxy_spec(X,a,Z,W,rank=None,legacy=False,overcomplete=False):
    if not screen(Z,W,rank if legacy else 1,overcomplete=overcomplete):return None
    rank=bridge_information_rank(X[:,W],X[:,Z],X[:,a])['r'] if rank is None else rank
    if not rank:return None
    nu=proxy_strength(X[:,W],X[:,Z],X[:,a],rank)
    if nu<.05:return None
    return dict(Z=Z,W=W,rank=rank,nu=nu,legacy=legacy)


def configurations(cohort,a,system):
    new=None; old=None; unanchored=None; first=None; cached=[]
    for alpha in ALPHAS:
        res=analyse_cohort(cohort,alpha=alpha,rank_mode='bridge',estimate_legacy=False); cached.append(res)
        row,Z,W=configuration(res,a)
        if first is None and row['status']=='reported':first=dict(Z=Z,W=W,rank=row['r_bridge'],nu=row['nu'],legacy=False)
        if unanchored is None:unanchored=proxy_spec(res['X'],a,Z,W,overcomplete=True)
        if old is None:
            old_Z,old_W=proxy_roles(res['Und'],a,np.ones(system['p'],bool),np.linalg.norm(res['latent']['Lam'],axis=1),res['latent']['r'])
            old=proxy_spec(res['X'],a,old_Z,old_W,res['latent']['r'],legacy=True)
    row,Z,W,res=configuration_grid(cached,a)
    if row['status']=='reported':new=dict(Z=Z,W=W,rank=row['r_bridge'],nu=row['nu'],legacy=False,alpha=row['alpha'],rank_grid=row['r_grid'])
    X=res['X'];global_window=None;global_grid=[]
    for alpha in ALPHAS:
        global_grid.append(analyse_cohort(cohort,alpha=alpha,k=1,rank_mode='bridge',estimate_legacy=False))
    row,Z,W,global_res=configuration_grid(global_grid,a)
    if row['status']=='reported':global_window=dict(Z=Z,W=W,rank=row['r_bridge'],nu=row['nu'],legacy=False,alpha=row['alpha'],rank_grid=row['r_grid'])
    correlation=np.abs(np.corrcoef(X,rowvar=False)[a]); candidates=np.argsort(correlation)
    Z=[int(v) for v in candidates[::-1] if v!=a][:6]
    W=[int(v) for v in candidates if v!=a and v not in Z][:6]
    simple=proxy_spec(X,a,Z,W)
    designated=proxy_spec(X,a,list(system['iZ']),list(system['iW']),rank=system['r'])
    return dict(idop_proximal=new,first_passing_proximal=first,unanchored_rank_proximal=unanchored,pca_rank_proximal=old,global_window_proximal=global_window,
                correlation_proximal=simple,designated_proximal=designated),res


def effect_for(method,spec,y,X,A,F):
    if method=='unadjusted':return ols_effect(y,A)[0]
    if method=='factor_adjustment':return ols_effect(y,A,latent_state(X)['F'])[0]
    if method=='oracle_factor':return ols_effect(y,A,F)[0]
    return proximal_bridge(y,A,X[:,spec['W']],X[:,spec['Z']],spec['rank']+(1 if spec['legacy'] else 0),
                            min_sv_ratio=.1 if spec['legacy'] else 1e-8)[0]


def evaluate(outcome='linear',reps=200,boots=100,scenario=None,reuse_baselines=None):
    warnings.filterwarnings('ignore'); started=time.time(); records=[]
    settings=SCENARIOS if scenario is None else [SCENARIOS[scenario]]
    result_dir=ROOT/'results';result_dir.mkdir(exist_ok=True)
    suffix='' if scenario is None else f'_part{scenario}'
    out_path=result_dir/f'sim_method_{outcome}_replicates{suffix}.csv'
    reuse={}
    reusable={'first_passing_proximal','pca_rank_proximal','correlation_proximal','designated_proximal','unadjusted','factor_adjustment','oracle_factor'}
    if reuse_baselines:
        for row in pd.read_csv(reuse_baselines).to_dict('records'):
            if row['method']=='idop_proximal':row['method']='first_passing_proximal'
            if row['method'] in reusable:reuse[(row['scenario'],row['replicate'],row['method'])]=row
    for setting in settings:
        system=system_for(setting);a=system['a0'];graph,factors,y_node=truth_graph(system)
        for rep in range(reps):
            data=sample_system(system,setting,rep,outcome);raw=data['X']
            scaled=data_transformation(pd.DataFrame(raw,columns=[f'v{j}' for j in range(system['p'])]),'Zscore_shift')
            order=np.argsort(scaled.sum(1).to_numpy(),kind='stable')
            cohort=dict(qd=data_quasi_dynamic(scaled),names=list(scaled.columns),Y=data['Y'][order],
                        Cov=np.zeros((setting['n'],0)),T=data['T'][order] if outcome=='rmst' else None)
            specifications,res=configurations(cohort,a,system)
            X=res['X'];A=raw[order,a];F=data['F'][order]
            if outcome=='rmst':
                T,D=data['T'][order],data['D'][order]
                response=survival_pseudo_outcomes(T,D,36,censor='km')['rmst']
            else:response=data['Y'][order]
            methods={**specifications,'unadjusted':{},'factor_adjustment':{},'oracle_factor':{}}
            copied={m:reuse[(setting['name'],rep,m)] for m in methods if (setting['name'],rep,m) in reuse}
            methods={m:s for m,s in methods.items() if m not in copied}
            estimates={m:effect_for(m,s,response,X,A,F) for m,s in methods.items() if s is not None}
            draws={m:[] for m in estimates};rng=np.random.default_rng(910000+rep+10000*setting['seed'])
            for _ in range(boots):
                ix=rng.integers(0,setting['n'],setting['n'])
                try:
                    boot_y=survival_pseudo_outcomes(T[ix],D[ix],36,censor='km')['rmst'] if outcome=='rmst' else response[ix]
                    values={m:effect_for(m,methods[m],boot_y,X[ix],A[ix],F[ix]) for m in estimates}
                    if not np.isfinite(list(values.values())).all():continue
                    for m,v in values.items():draws[m].append(v)
                except (ValueError,np.linalg.LinAlgError):continue
            for method,spec in methods.items():
                row=dict(scenario=setting['name'],replicate=rep,outcome=outcome,method=method,
                         n=setting['n'],r=setting['r'],topology=setting['topology'],latent_scale=setting['latent'],
                         target=data['target'],reported=spec is not None,boot_requested=boots)
                if spec is not None:
                    e=estimates[method];v=draws[method]
                    se=np.std(v,ddof=1) if len(v)>=max(20,.9*boots) else np.nan
                    row.update(estimate=e,se_boot=se,error=e-data['target'],
                               covered=float(abs(e-data['target'])<=1.96*se) if np.isfinite(se) else np.nan,
                               width=3.92*se,boot_success=len(v))
                    if method not in ('unadjusted','factor_adjustment','oracle_factor'):
                        row.update(rank=spec['rank'],nu=spec['nu'],nZ=len(spec['Z']),nW=len(spec['W']),
                                   alpha=spec.get('alpha',np.nan),rank_grid=spec.get('rank_grid',np.nan),
                                   Z=';'.join(str(v) for v in spec['Z']),W=';'.join(str(v) for v in spec['W']),
                                   **role_validity(graph,factors,y_node,a,spec['Z'],spec['W']))
                records.append(row)
            records.extend(copied.values())
            if (rep+1)%25==0:
                pd.DataFrame(records).to_csv(out_path,index=False)
                print(outcome,setting['name'],rep+1,'/',reps,round(time.time()-started,1),'seconds',flush=True)
        pd.DataFrame(records).to_csv(out_path,index=False)
    summary=summarize(pd.DataFrame(records))
    summary.to_csv(result_dir/f'sim_method_{outcome}_summary{suffix}.csv',index=False)
    print(summary[['scenario','method','report_rate','bias','rmse','coverage']].to_string(index=False),flush=True)
    return summary


def summarize(records):
    def wilson(successes,n):
        if not n:return np.nan,np.nan
        p=successes/n;z=stats.norm.ppf(.975);den=1+z*z/n
        center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return max(0.,center-half),min(1.,center+half)
    rows=[]
    for (scenario,method),group in records.groupby(['scenario','method'],sort=False):
        n=len(group); reported=group[group.reported]; p=len(reported)/n
        row=dict(scenario=scenario,method=method,replicates=n,n_reported=len(reported),report_rate=p,
                 report_mcse=np.sqrt(p*(1-p)/n),target=group.target.iloc[0],n=group.n.iloc[0],r=group.r.iloc[0])
        row['report_ci_low'],row['report_ci_high']=wilson(len(reported),n)
        if len(reported):
            errors=reported.error.to_numpy();m=len(errors);squared=errors**2;rmse=np.sqrt(squared.mean())
            covered=reported.covered.dropna();cp=covered.mean()
            row['coverage_mc_ci_low'],row['coverage_mc_ci_high']=wilson(covered.sum(),len(covered))
            row.update(bias=errors.mean(),bias_mcse=errors.std(ddof=1)/np.sqrt(m) if m>1 else np.nan,
                       rmse=rmse,rmse_mcse=squared.std(ddof=1)/(2*rmse*np.sqrt(m)) if m>1 and rmse else np.nan,
                       n_intervals=len(covered),coverage=cp,coverage_mcse=np.sqrt(cp*(1-cp)/len(covered)) if len(covered) else np.nan,
                       mean_width=reported.width.mean(),rank_mean=reported['rank'].mean() if 'rank' in reported else np.nan)
            for key in ('z_valid','w_valid','set_valid'):
                if key in reported:
                    values=reported[key].dropna()
                    row[key]=values.mean()
                    row[key+'_mcse']=values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else np.nan
            if 'rank' in reported:
                selected=reported[reported['rank'].notna()]
                if len(selected):
                    correct=(selected['rank']==selected.r).astype(float)
                    row['rank_correct_rate']=correct.mean()
                    row['rank_correct_mcse']=np.sqrt(correct.mean()*(1-correct.mean())/len(correct))
                    row['rank_correct_ci_low'],row['rank_correct_ci_high']=wilson(correct.sum(),len(correct))
        rows.append(row)
    return pd.DataFrame(rows)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--outcome',choices=['linear','rmst'],default='linear')
    ap.add_argument('--reps',type=int,default=200);ap.add_argument('--boot',type=int,default=100)
    ap.add_argument('--scenario',type=int,choices=range(len(SCENARIOS)))
    ap.add_argument('--reuse-baselines',type=Path)
    args=ap.parse_args();evaluate(args.outcome,args.reps,args.boot,args.scenario,args.reuse_baselines)
