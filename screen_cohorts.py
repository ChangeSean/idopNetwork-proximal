"""Diagnostics for every downloaded TCGA cohort: run the pipeline on overall survival
(Cox second stage, no bootstrap) and report the quantities that decide whether the
method is operable — before looking at any estimate.

    python screen_cohorts.py                # every data/tcga_<study>_rppa_long.csv
    python screen_cohorts.py ov skcm lgg    # a subset

Writes results/cohort_screen.csv.
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
warnings.filterwarnings('ignore')
from run_application import load_cohort, analyse_cohort
from proximal import components

studies = sys.argv[1:] or sorted(os.path.basename(f)[5:-14] for f in glob.glob(os.path.join(HERE, 'data', 'tcga_*_rppa_long.csv')))
rows = []
for st in studies:
    try:
        coh = load_cohort(st, 'OS', survival=True)
    except Exception as e:
        print(f'{st}: skipped ({e})'); continue
    if len(coh['Y']) < 100 or coh['Y'].sum() < 40:
        print(f"{st}: n={len(coh['Y'])} events={int(coh['Y'].sum())} -> too few"); continue
    res = analyse_cohort(coh, survival=True); d = res['estimates']; lat = res['latent']
    comp = components(res['Und']); sizes = np.bincount(comp)
    rows.append(dict(study=st, n=len(coh['Y']), events=int(coh['Y'].sum()), p=len(coh['names']), r_hat=lat['r'],
                     rho_bar=lat['rho'].mean(), edges=int(res['A'].sum()), components=len(sizes), main_component=int(sizes.max()),
                     n_pass=len(d), n_weak=res['n_weak'], nu_median=d.nu.median() if len(d) else np.nan,
                     n_plugin_q10=int((d.q_prox < 0.1).sum()) if len(d) else 0, min_plugin_q=d.q_prox.min() if len(d) else np.nan,
                     top=('; '.join(f'{r.exposure}({r.proximal:+.2f})' for r in d.reindex(d.t_prox.abs().sort_values(ascending=False).index).head(3).itertuples()) if len(d) else '')))
    print(f"{st}: n={rows[-1]['n']} ev={rows[-1]['events']} r={rows[-1]['r_hat']} rho={rows[-1]['rho_bar']:.2f} comps={rows[-1]['components']} "
          f"pass={rows[-1]['n_pass']} weak={rows[-1]['n_weak']} nu_med={rows[-1]['nu_median']:.3f} q10={rows[-1]['n_plugin_q10']} minq={rows[-1]['min_plugin_q']:.3f}", flush=True)
out = pd.DataFrame(rows); out.to_csv(os.path.join(HERE, 'results', 'cohort_screen.csv'), index=False)
print(out.drop(columns='top').round(3).to_string(index=False))
