"""Generate every figure for the manuscript into figures/.

    python make_figures.py             # ovarian cancer, OS (Cox second stage, bootstrap se), all figures
    python make_figures.py ov PFS
"""
import os, sys, warnings
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
warnings.filterwarnings('ignore')
from run_application import load_cohort, analyse_cohort, bootstrap_se
import figures as F

STUDY = sys.argv[1] if len(sys.argv) > 1 else 'ov'
OUTCOME = sys.argv[2] if len(sys.argv) > 2 else 'OS'
BOOT = 300
RES = os.path.join(HERE, 'results')

print('Figure 1: schematic'); F.save(F.fig_schematic(), 'fig1_schematic')

coh = load_cohort(STUDY, OUTCOME, survival=True)
res = analyse_cohort(coh, solve_ode=True, survival=True)
res['qd'] = coh['qd']
est = bootstrap_se(res, coh, BOOT)
est.to_csv(os.path.join(RES, f'{STUDY}_{OUTCOME}_cox_estimates.csv'), index=False)
print(f"{STUDY}/{OUTCOME}: n={len(coh['Y'])}, events={int(coh['Y'].sum())}, r={res['latent']['r']}, "
      f"{len(est)} exposures pass")

print('Figures 2 and 3: power-law fits and effect decompositions')
sup = res['supports']; sup['n'] = sup.source.str.strip('{}').apply(lambda s: 0 if not s else len(s.split(',')))
if (STUDY, OUTCOME) == ('ov', 'OS'):
    # proteins tied to the reported results: the two OV hits and nominated Z/W candidates
    fits = ['LCK', 'CDH2', 'CTNNB1', 'TP53BP1', 'CDH1', 'STAT5A', 'GAPDH', 'VHL']
    decs = ['LCK', 'CTNNB1', 'TP53BP1', 'MSH6', 'CDH2', 'CDH1', 'STAT5A', 'CLDN7']
    tags = {'LCK': 'exposure', 'CDH2': 'exposure', 'CTNNB1': 'Z for LCK', 'TP53BP1': 'Z for LCK', 'MSH6': 'child of CTNNB1',
            'CDH1': 'Z for CDH2', 'STAT5A': 'Z for CDH2', 'CLDN7': 'child of CDH1', 'GAPDH': 'W', 'VHL': 'W'}
else:
    b = res['params']['b'].reindex(res['names']).sort_values(); fits = list(b.index[:4]) + list(b.index[-4:])
    decs = list(sup.sort_values('n', ascending=False).target.head(8)); tags = {}
F.save(F.fig_fits(res, fits, tags), 'fig2_fits')

# exposure for the proxy-candidate views: LCK in OV, else the top |t|
top = est.reindex(est.t_boot.abs().sort_values(ascending=False).index)
lead = 'LCK' if (STUDY, OUTCOME) == ('ov', 'OS') and 'LCK' in set(est.exposure) else top.exposure.iloc[0]
exposures = ['LCK', 'CDH2'] if (STUDY, OUTCOME) == ('ov', 'OS') else list(top.exposure.head(2))
print(f'Figure 3: stacked planes for {exposures} + decompositions'); F.save(F.fig_causal(res, exposures, decs, tags), 'fig3_causal')
print('Figure S2: network by component'); F.save(F.fig_network(res, lead), 'figS2_network')

print('Figure 4: simulation')
F.save(F.fig_simulation(os.path.join(RES, 'sim_coverage_vs_delta.csv'), os.path.join(RES, 'sim_tradeoff.csv'), os.path.join(RES, 'sim_p1_vs_p2.csv')), 'fig4_simulation')
print('Figure 5: cohort screen'); F.save(F.fig_cohorts(os.path.join(RES, 'cohort_screen.csv')), 'fig5_cohorts')

# stability of the leading exposures across the support window
print('Figure 6: application (stability scan)')
settings = [(0.10, 5, 0.6), (0.15, 5, 0.6), (0.20, 5, 0.6), (0.25, 5, 0.6), (0.15, 10, 0.6)]
watch = list(top.exposure.head(3))
rows = []
for alpha, k, thr in settings:
    r_ = analyse_cohort(coh, alpha, k, thr, survival=True)['estimates']
    for e in watch:
        m = r_[r_.exposure == e]
        rows.append(dict(exposure=e, setting=f'{alpha:.2f}, {k}', proximal=m.proximal.iloc[0] if len(m) else np.nan,
                         se=m.se.iloc[0] if len(m) else np.nan))
stab = pd.DataFrame(rows)
stab.to_csv(os.path.join(RES, f'{STUDY}_{OUTCOME}_stability.csv'), index=False)
F.save(F.fig_application(est, top=14, stability=stab, survival=True), 'fig6_application')
# second cohort: LUAD overall survival (Figure 7)
print('Figure 7: second cohort (luad OS)')
coh2 = load_cohort('luad', 'OS', survival=True); res2 = analyse_cohort(coh2, survival=True)
est2 = bootstrap_se(res2, coh2, BOOT); est2.to_csv(os.path.join(RES, 'luad_OS_cox_estimates.csv'), index=False)
top2 = est2.reindex(est2.t_boot.abs().sort_values(ascending=False).index); rows = []
for alpha, k, thr in settings:
    r_ = analyse_cohort(coh2, alpha, k, thr, survival=True)['estimates']
    for e in list(top2.exposure.head(3)):
        m = r_[r_.exposure == e] if len(r_) else r_
        rows.append(dict(exposure=e, setting=f'{alpha:.2f}, {k}', proximal=m.proximal.iloc[0] if len(m) else np.nan, se=m.se.iloc[0] if len(m) else np.nan))
stab2 = pd.DataFrame(rows); stab2.to_csv(os.path.join(RES, 'luad_OS_stability.csv'), index=False)
F.save(F.fig_application(est2, top=14, stability=stab2, survival=True), 'fig7_luad')

# supplementary: the idopNetwork on every protein that survives the filters
print('Figure S1: full-protein network')
coh_full = load_cohort(STUDY, OUTCOME, p_keep=10000, survival=True)
res_full = analyse_cohort(coh_full, solve_ode=True, survival=True)
res_full['edgelist'].to_csv(os.path.join(RES, f'{STUDY}_{OUTCOME}_edgelist_allproteins.csv'), index=False)
F.save(F.fig_full_network(res_full), 'figS1_full_network')
print('done ->', F.FIG)
