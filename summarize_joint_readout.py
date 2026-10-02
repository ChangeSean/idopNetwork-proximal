"""Combine independent validation; preserve all denominators and MC uncertainty."""
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from validate_joint_readout_bridge import ROOT,OUT,SETTINGS,summaries


def wilson(k,n):
    if not n:return np.nan,np.nan
    z=norm.ppf(.975);p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half


def summarize(data):
    result=summaries(data)
    for i,row in result.iterrows():
        attempted=data[(data.scenario==row.scenario)&(data.outcome==row.outcome)&(data.method==row.method)]
        v=attempted[attempted.status.eq('reported')];err=v.error.to_numpy();n=len(v)
        result.loc[i,'n_intervals']=n
        result.loc[i,'report_mcse']=np.sqrt(row.report_rate*(1-row.report_rate)/row.attempts)
        result.loc[i,'rmse_mcse']=(err**2).std(ddof=1)/(2*row.rmse*np.sqrt(n)) if n>1 and row.rmse else np.nan
        for key,k in [('report',n),('coverage',float(v.covered.sum()))]:
            lo,hi=wilson(k,row.attempts if key=='report' else n)
            result.loc[i,key+'_wilson_low']=lo;result.loc[i,key+'_wilson_high']=hi
        result.loc[i,'width_mcse']=v.width.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
    return result


def main():
    frames=[];checks=0
    for p in range(7):
        manifest=json.loads((OUT/f'manifest_part{p}.json').read_text())
        for name,sha in manifest['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha;checks+=1
        data=pd.read_csv(OUT/f'replicates_part{p}.csv');comp=pd.read_csv(OUT/f'comparators_part{p}.csv')
        assert len(data)==200*2*6 and len(comp)==200*2*4;checks+=2
        frames.extend([data,comp])
    data=pd.concat(frames,ignore_index=True)
    assert not data.duplicated(['scenario','outcome','method','replicate']).any();checks+=1
    for (_,_,_),v in data.groupby(['scenario','outcome','method']):
        assert len(v)==200 and set(v.replicate)==set(range(200));checks+=1
    successful=data[data.status.eq('reported')]
    assert np.allclose(successful.error,successful.estimate-successful.target);checks+=1
    assert successful.boot_success.between(90,100).all();checks+=1
    assert np.array_equal(successful.covered.to_numpy(),(abs(successful.error)<=norm.ppf(.975)*successful.se).astype(float));checks+=1
    assert np.allclose(successful.width,2*norm.ppf(.975)*successful.se);checks+=1
    data.to_csv(OUT/'all_replicates.csv',index=False);summary=summarize(data)
    summary.to_csv(OUT/'all_summary.csv',index=False)
    strata=[]
    for (s,o,r),v in data[data.method.eq('joint_readout')&data.status.eq('reported')].groupby(['scenario','outcome','r_bridge']):
        copy=v.copy();copy['method']=f'rank{int(r)}'
        row=summarize(copy).iloc[0].to_dict();row['r_bridge']=r;strata.append(row)
    pd.DataFrame(strata).to_csv(OUT/'rank_strata.csv',index=False)
    paired=[]
    for (s,o),v in data.groupby(['scenario','outcome']):
        j=v[v.method.eq('joint_readout')&v.status.eq('reported')].set_index('replicate')
        for method in v.method.unique():
            if method=='joint_readout':continue
            other=v[v.method.eq(method)&v.status.eq('reported')].set_index('replicate')
            common=j.index.intersection(other.index);delta=j.loc[common,'error']**2-other.loc[common,'error']**2
            paired.append(dict(scenario=s,outcome=o,comparator=method,paired=len(common),
                               mse_difference=delta.mean(),mse_difference_mcse=delta.std(ddof=1)/np.sqrt(len(common)) if len(common)>1 else np.nan))
    pd.DataFrame(paired).to_csv(OUT/'paired_differences.csv',index=False)
    audit=dict(passed=True,checks=checks,attempts=len(data),settings=7,methods=10,reps=200,bootstrap=100,
               valid_joint_roles=bool(successful[successful.method.eq('joint_readout')].set_valid.eq(1).all()),
               source_sha256=manifest['source_sha256'])
    (OUT/'validation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(summary[summary.method.eq('joint_readout')].to_string(index=False));print(audit)


if __name__=='__main__':main()
