"""Clinical record checks and direct isolation of discovery from held-out data."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from run_discovery_estimation_application import prepare

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/discovery_estimation_application_20261001'


def main():
    checks=0
    for study in ('ov','luad'):
        meta=json.loads((OUT/f'{study}_manifest.json').read_text())
        data=pd.read_csv(OUT/f'{study}_all_exposures.csv')
        assert len(data)==60 and not data.exposure.duplicated().any()
        assert meta['n_discovery']==3*meta['n']//4
        checks+=2
        for f,sha in meta['source_sha256'].items():
            assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==sha,f
            checks+=1
        for row in data.to_dict('records'):
            for target in ('rmst','survival'):
                status=row[target+'_status']
                intervals=json.loads(row[target+'_intervals'])
                if status=='estimated':
                    assert row['design_ready'] and row[target+'_boot_success']>=293
                    assert np.isfinite(row[target+'_estimate'])
                else:
                    assert intervals==[[-np.inf,np.inf]] and np.isnan(row[target+'_estimate'])
                for lo,hi in intervals:assert lo<=hi
                checks+=2
        original=prepare(study)
        real_read=pd.read_csv
        rp=real_read(ROOT/f'data/tcga_{study}_rppa_long.csv')
        cl=real_read(ROOT/f'data/tcga_{study}_clinical.csv',index_col=0)
        wide=rp.pivot_table(index='patientId',columns='gene',values='value')
        included=cl.loc[sorted(set(cl.index)&set(wide.index))]
        included=included[included.OS_STATUS.notna() & pd.to_numeric(included.OS_MONTHS,errors='coerce').notna()]
        allocation=np.random.default_rng(20261201).permutation(len(included))
        held_out=included.index[allocation[3*len(included)//4:]]
        altered=rp.copy()
        mask=altered.patientId.isin(held_out)
        altered.loc[mask,'value']=1000+altered.loc[mask,'value']*9
        changed_cl=cl.copy()
        changed_cl.loc[held_out,'OS_MONTHS']=1.0
        changed_cl.loc[held_out,'OS_STATUS']='0:LIVING'
        def fake_read(path,*args,**kwargs):
            if Path(path).name==f'tcga_{study}_rppa_long.csv':return altered.copy()
            if Path(path).name==f'tcga_{study}_clinical.csv':return changed_cl.copy()
            return real_read(path,*args,**kwargs)
        with patch('pandas.read_csv',side_effect=fake_read):
            changed=prepare(study)
        assert original[0]['qd'].equals(changed[0]['qd'])
        assert np.array_equal(original[0]['Cov'],changed[0]['Cov'])
        assert original[0]['names']==changed[0]['names']
        for field in ('discovery_mean','discovery_sd','discovery_id_sha256','estimation_id_sha256'):
            assert original[-1][field]==changed[-1][field]
        assert not np.array_equal(original[1],changed[1])
        assert not np.array_equal(original[3],changed[3])
        checks+=9
    meta=json.loads((OUT/'structured_manifest.json').read_text())
    for f,sha in meta['source_sha256'].items():
        assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==sha,f
        checks+=1
    result=dict(passed=True,checks=checks,cohorts=2,exposures=120,endpoint_sets=240,
                discovery_isolation='Held-out molecular values and survival labels perturbed; discovery panel, scaling, imputation and quasi-dynamic data exactly unchanged',
                patient_identifiers='Not included in audit outputs')
    (OUT/'application_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
