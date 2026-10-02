"""Apply the same mechanistic boundaries inside the final discovery split."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_discovery_estimation_application import prepare
from run_structured_clinical_cases import apply_boundary
from multiscale_bridge import design_grid
from causal_survival import survival_pseudo_outcomes
import independent_design_bridge as ID

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/discovery_estimation_application_20261001'


def main():
    cases=json.loads((ROOT/'clinical_case_boundaries_20261001.json').read_text())['cases']
    rows=[]
    for case in cases:
        cohort,X,C,T,D,meta=prepare(case['study'])
        name=case['exposure']
        row=dict(study=case['study'],exposure=name,boundary=case['boundary'],
                 n_discovery=meta['n_discovery'],n_estimation=len(X))
        if name not in cohort['names']:
            row['status']='exposure_outside_discovery_panel'
            design=None
        else:
            a=cohort['names'].index(name)
            grid=design_grid(cohort)
            original=ID.freeze(grid,a)
            masked,eligible=apply_boundary(grid,cohort['names'],case)
            design=ID.freeze(masked,a,eligible)
            row.update(design['row'],reference_dimension=original['row']['r_grid'])
            if design['row']['r_grid']<original['row']['r_grid']:
                design['ready']=False
                row['status']='boundary_readout_dimension'
        results={target:ID.unavailable(row['status']) for target in ('rmst','survival')}
        if design and design['ready']:
            rng=np.random.default_rng(20261301)
            draws=[rng.integers(0,len(X),len(X)) for _ in range(300)]
            try:
                py=survival_pseudo_outcomes(T,D,36,np.column_stack([X[:,a],C]))
                responses=[]
                for ix in draws:
                    try:
                        responses.append(survival_pseudo_outcomes(T[ix],D[ix],36,np.column_stack([X[ix,a],C[ix]])))
                    except (ValueError,np.linalg.LinAlgError):
                        responses.append({t:np.full(len(ix),np.nan) for t in results})
                for target in results:
                    results[target]=ID.fit(py[target],X[:,a],X[:,design['W']],X[:,design['Z']],C,
                                           design,draws,[p[target] for p in responses])
                row['status']=results['rmst']['status']
            except (ValueError,np.linalg.LinAlgError):
                row['status']='censoring_support'
        for target,result in results.items():
            row.update({target+'_'+k:result[k] for k in ('kind','estimate','status','boot_success')})
            row[target+'_intervals']=json.dumps(result['intervals'])
        rows.append(row)
        print(case['study'],name,row['status'],flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/'structured_cases.csv',index=False)
    files=('run_independent_structured_cases.py','clinical_case_boundaries_20261001.json',
           'run_discovery_estimation_application.py','independent_design_bridge.py')
    (OUT/'structured_manifest.json').write_text(json.dumps(dict(boots=300,split='75:25',
         source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}),indent=2))


if __name__=='__main__':main()
