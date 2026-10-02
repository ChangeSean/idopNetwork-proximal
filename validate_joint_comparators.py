"""Paired proximal and simple-screen comparators on frozen validation seeds."""
import argparse
import json
import hashlib
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from validate_joint_readout_bridge import ROOT,OUT,SETTINGS,summaries
from evaluate_window_prototype import extended_system
from evaluate_causal_method import sample_system,truth_graph,role_validity
from run_application import analyse_cohort
from joint_readout_bridge import select_joint_space,estimate
from proximal import bridge_information_rank,proximal_bridge,proxy_strength
from causal_survival import survival_pseudo_outcomes
from idop_core import data_transformation,data_quasi_dynamic


def run(parts,reps=200,boots=100):
    OUT.mkdir(exist_ok=True,parents=True);start=time.time()
    manifest=dict(parts=parts,reps=reps,boots=boots,offset=30000,
                  paired_with='validate_joint_readout_bridge.py; identical patients and bootstrap indices',
                  methods=['global_joint_readout','correlation_joint_readout','designated_joint_readout','designated_proximal'],
                  source_sha256=hashlib.sha256(__file__.encode()).hexdigest())
    # Hash the source bytes, never the pathname.
    manifest['source_sha256']=hashlib.sha256((ROOT/'validate_joint_comparators.py').read_bytes()).hexdigest()
    (OUT/f'comparators_manifest_{parts[0]}.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for part in parts:
        setting=SETTINGS[part];system=extended_system(setting);a=system['a0'];graph,Fnodes,y_node=truth_graph(system)
        records=[]
        for rep in range(reps):
            lin=sample_system(system,setting,30000+rep,'linear');surv=sample_system(system,setting,30000+rep,'rmst')
            names=[f'v{j}' for j in range(system['p'])]
            scaled=data_transformation(pd.DataFrame(lin['X'],columns=names),'Zscore_shift')
            order=np.argsort(scaled.sum(1).to_numpy(),kind='stable')
            coh=dict(qd=data_quasi_dynamic(scaled),names=names,Y=lin['Y'][order],Cov=np.zeros((setting['n'],0)),T=None)
            grid=[analyse_cohort(coh,alpha=alpha,k=1,rank_mode='bridge',estimate_legacy=False) for alpha in (.15,.20,.25,.30)]
            X=grid[0]['X'];A=lin['X'][order,a]
            row,Z,W,_=select_joint_space(grid,a)
            specs={'global_joint_readout':(row,Z,W)}
            correlations=abs(np.corrcoef(X,rowvar=False)[a]);idx=np.argsort(correlations)
            Zc=[int(j) for j in idx[::-1] if j!=a][:6];Wc=[int(j) for j in idx if j!=a and j not in Zc][:6]
            r=bridge_information_rank(X[:,Wc],X[:,[a]+Zc],None,alpha=min(.05,len(X)**(-.5)))['r']
            rc=dict(status='reported' if r and r<=len(Zc) else 'no_shared_direction',r_bridge=r,
                    nu=proxy_strength(X[:,Wc],X[:,Zc],A,max(r,1)))
            specs['correlation_joint_readout']=(rc,Zc,Wc)
            Zt=list(system['iZ']);Wt=list(system['iW'])
            rt=dict(status='reported',r_bridge=system['r'],nu=proxy_strength(X[:,Wt],X[:,Zt],A,system['r']))
            specs['designated_joint_readout']=(rt,Zt,Wt);specs['designated_proximal']=(rt,Zt,Wt)
            rng=np.random.default_rng(10200000+10000*setting['seed']+rep+30000)
            draws=[rng.integers(0,len(X),len(X)) for _ in range(boots)]
            T,D=surv['T'][order],surv['D'][order]
            responses={'linear':lin['Y'][order],'rmst':survival_pseudo_outcomes(T,D,36,censor='km')['rmst']}
            boot_rmst=[survival_pseudo_outcomes(T[ix],D[ix],36,censor='km')['rmst'] for ix in draws]
            for outcome,y in responses.items():
                target=lin['target'] if outcome=='linear' else surv['target']
                for method,(rr,zz,ww) in specs.items():
                    entry={**rr,'scenario':setting['name'],'replicate':rep,'outcome':outcome,'method':method,
                           'target':target,'true_r':system['r'],'n':setting['n'],'p':system['p'],
                           'boot_requested':boots,'Z':';'.join(names[j] for j in zz),'W':';'.join(names[j] for j in ww),
                           **role_validity(graph,Fnodes,y_node,a,zz,ww)}
                    if rr['status']=='reported':
                        def calculate(yy,aa,xx):
                            if method=='designated_proximal':return proximal_bridge(yy,aa,xx[:,ww],xx[:,zz],rr['r_bridge'],min_sv_ratio=1e-8)[0]
                            return estimate(yy,aa,xx[:,ww],xx[:,zz],rr['r_bridge'])
                        try:
                            e=calculate(y,A,X);values=[]
                            for b,ix in enumerate(draws):
                                try:
                                    v=calculate(y[ix] if outcome=='linear' else boot_rmst[b],A[ix],X[ix])
                                    if np.isfinite(v):values.append(v)
                                except (ValueError,np.linalg.LinAlgError):pass
                            se=float(np.std(values,ddof=1)) if len(values)>1 else np.nan
                            entry.update(estimate=e,error=e-target,se=se,boot_success=len(values),
                                         covered=float(abs(e-target)<=norm.ppf(.975)*se),width=2*norm.ppf(.975)*se,
                                         rank_correct=float(rr['r_bridge']==system['r']))
                            if len(values)<max(20,.9*boots):entry['status']='bootstrap_incomplete'
                        except (ValueError,np.linalg.LinAlgError) as error:entry.update(status='numerical_support',reason=str(error))
                    records.append(entry)
            if (rep+1)%50==0:print('comparators',part,rep+1,'elapsed',round(time.time()-start,1),flush=True)
        table=pd.DataFrame(records);table.to_csv(OUT/f'comparators_part{part}.csv',index=False)
        summaries(table).to_csv(OUT/f'comparators_summary_part{part}.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parts',nargs='+',type=int,required=True)
    args=p.parse_args();run(args.parts)
