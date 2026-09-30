"""Fetch RPPA protein levels and clinical outcomes for TCGA pan-can atlas studies
from the cBioPortal REST API (public, no auth).

Usage:  python fetch_tcga.py ov lgg luad ...      (study prefixes)
        python fetch_tcga.py --all                (every study with RPPA)

Writes data/tcga_<study>_rppa_long.csv and data/tcga_<study>_clinical.csv per
study, and data/tcga_pooled_rppa_wide.csv + data/tcga_pooled_clinical.csv for
everything fetched in this run (cancer type kept as a column).
"""
import sys, os, json, time, urllib.request
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
BASE = "https://www.cbioportal.org/api"
CLIN = ['DFS_STATUS', 'DFS_MONTHS', 'PFS_STATUS', 'PFS_MONTHS', 'OS_STATUS', 'OS_MONTHS',
        'DSS_STATUS', 'AGE', 'SEX', 'SUBTYPE', 'AJCC_PATHOLOGIC_TUMOR_STAGE', 'CANCER_TYPE_ACRONYM']


def get(url, tries=3):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'Accept': 'application/json'}), timeout=180) as r:
                return json.load(r)
        except Exception as e:
            if t == tries - 1: raise
            time.sleep(3)


def post(url, body, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except Exception as e:
            if t == tries - 1: raise
            time.sleep(3)


def fetch_study(prefix, gene_ids, sym):
    sid = f"{prefix}_tcga_pan_can_atlas_2018"
    prof = f"{sid}_rppa"
    sls = get(f"{BASE}/studies/{sid}/sample-lists")
    rl = [s['sampleListId'] for s in sls if 'rppa' in s['sampleListId'].lower()]
    if not rl:
        print(f"  {prefix}: no RPPA sample list"); return None, None
    records = []
    for i in range(0, len(gene_ids), 3000):
        d = post(f"{BASE}/molecular-profiles/{prof}/molecular-data/fetch?projection=SUMMARY",
                 {"sampleListId": rl[0], "entrezGeneIds": gene_ids[i:i + 3000]})
        records.extend(d); time.sleep(0.2)
    if not records:
        print(f"  {prefix}: no RPPA records"); return None, None
    rp = pd.DataFrame(records)[['patientId', 'sampleId', 'entrezGeneId', 'value']]
    rp['gene'] = rp.entrezGeneId.map(sym)
    cl = pd.DataFrame(get(f"{BASE}/studies/{sid}/clinical-data?clinicalDataType=PATIENT&pageSize=100000&projection=SUMMARY"))
    cl = cl[['patientId', 'clinicalAttributeId', 'value']].pivot(index='patientId', columns='clinicalAttributeId', values='value')
    cl = cl[[c for c in CLIN if c in cl.columns]]
    cl['study'] = prefix
    rp.to_csv(os.path.join(DATA, f'tcga_{prefix}_rppa_long.csv'), index=False)
    cl.to_csv(os.path.join(DATA, f'tcga_{prefix}_clinical.csv'))
    return rp, cl


if __name__ == '__main__':
    args = sys.argv[1:]
    genes = get(f"{BASE}/genes?pageSize=100000&projection=SUMMARY")
    gene_ids = [g['entrezGeneId'] for g in genes if g.get('type') == 'protein-coding' and g['entrezGeneId'] > 0]
    sym = {g['entrezGeneId']: g['hugoGeneSymbol'] for g in genes}
    if args == ['--all']:
        studies = [s['studyId'].replace('_tcga_pan_can_atlas_2018', '') for s in get(f"{BASE}/studies?pageSize=1000")
                   if s['studyId'].endswith('_tcga_pan_can_atlas_2018')]
    else:
        studies = args
    wides, clins = [], []
    for prefix in studies:
        t0 = time.time()
        rp, cl = fetch_study(prefix, gene_ids, sym)
        if rp is None: continue
        w = rp.pivot_table(index='patientId', columns='gene', values='value')
        w['study'] = prefix
        wides.append(w); clins.append(cl)
        print(f"  {prefix}: {w.shape[0]} patients x {w.shape[1]-1} proteins, {time.time()-t0:.0f}s", flush=True)
    if wides:
        W = pd.concat(wides); C = pd.concat(clins)
        W.to_csv(os.path.join(DATA, 'tcga_pooled_rppa_wide.csv'))
        C.to_csv(os.path.join(DATA, 'tcga_pooled_clinical.csv'))
        print(f"\npooled: {W.shape[0]} patients, {W.shape[1]-1} protein columns (union), {len(studies)} studies")
