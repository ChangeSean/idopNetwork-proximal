"""Application: idopNetwork + proximal inference on a TCGA RPPA cohort.

    python run_application.py ov PFS            # primary cohort
    python run_application.py brca DFS
    python run_application.py ov PFS --alpha 0.15 --k 5 --thr 0.6 --p 60

Writes results/<study>_<outcome>_estimates.csv and results/<study>_<outcome>_edgelist.csv
and returns everything the figures need via analyse_cohort().
"""
import os, sys, argparse, warnings
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings('ignore')

from idop_core import (data_transformation, data_quasi_dynamic, get_power_function_params,
                       get_power_function_samples, power_law, edge_select, ode_solver)
from proximal import (latent_state, adjacency_from_supports, components, proxy_roles, screen,
                      outcome_equation, ols_effect, proximal_bridge, bh)
from survival import cox_effect, proximal_cox


def load_cohort(study, outcome, p_keep=60, data_dir=os.path.join(HERE, 'data'), survival=False):
    """survival=True also returns the follow-up time T (months) and drops patients without it."""
    rp = pd.read_csv(os.path.join(data_dir, f'tcga_{study}_rppa_long.csv'))
    cl = pd.read_csv(os.path.join(data_dir, f'tcga_{study}_clinical.csv'), index_col=0)
    wide = rp.pivot_table(index='patientId', columns='gene', values='value')
    cl = cl.loc[cl.index.intersection(wide.index)]
    cl = cl[cl[f'{outcome}_STATUS'].notna()]
    if survival: cl = cl[pd.to_numeric(cl[f'{outcome}_MONTHS'], errors='coerce').notna()]
    wide = wide.loc[cl.index]
    wide = wide.loc[:, wide.notna().mean() >= 0.9].fillna(wide.median())
    # collapse duplicate antibody columns (one antibody mapped to several genes)
    c = wide.corr().values.copy(); np.fill_diagonal(c, 0); drop = set()
    for i in range(c.shape[0]):
        for j in range(i + 1, c.shape[0]):
            if c[i, j] > 0.999 and wide.columns[i] not in drop: drop.add(wide.columns[j])
    wide = wide.drop(columns=sorted(drop))
    keep = wide.var().sort_values(ascending=False).index[:p_keep]
    Xt = data_transformation(wide[keep], 'Zscore_shift')
    # niche ordering permutes rows: outcome and covariates must follow
    order = np.argsort(Xt.sum(axis=1).values, kind='stable'); pid = wide.index[order]
    qd = data_quasi_dynamic(Xt)
    Y = cl.loc[pid][f'{outcome}_STATUS'].str.startswith('1').astype(float).values
    parts = []
    age = pd.to_numeric(cl.loc[pid].get('AGE'), errors='coerce')
    if age is not None and age.notna().mean() > 0.5: parts.append(age.fillna(age.median()).values)
    for cat in ('SUBTYPE', 'SEX'):
        if cat in cl.columns and cl.loc[pid][cat].notna().mean() > 0.5 and cl.loc[pid][cat].nunique() > 1:
            parts.append(pd.get_dummies(cl.loc[pid][cat].fillna('NA'), drop_first=True).astype(float).values)
    Cov = np.column_stack(parts) if parts else np.zeros((len(Y), 0))
    T = pd.to_numeric(cl.loc[pid][f'{outcome}_MONTHS'], errors='coerce').values if survival else None
    return dict(qd=qd, Y=Y, Cov=Cov, names=list(qd.columns), pid=pid, study=study, outcome=outcome, T=T)


BASIS_ORDER, ODE_RIDGE = 0, 1e-6   # weak-form ODE basis order and ridge, as in the original idop main() (basis_order=0, default ridge)


def analyse_cohort(coh, alpha=0.15, k=5, threshold=0.6, amplification=0.8, solve_ode=False, support='levels',
                   nu_min=0.05, extra_rank=1, survival=False, equivalence_margin=None,
                   latent_source='deviations'):
    """support: 'levels' learns the support graph from the transformed levels; 'xi' from the
    deviations with the latent state partialled out.
    nu_min: proxy-strength floor; exposures below it abstain (weak proxies give unusable estimates).
    extra_rank: What is truncated at r + extra_rank; one spare direction guards against the
    spectral-gap rule under-estimating r (under-truncation leaves a confounder direction,
    over-truncation only costs precision).
    survival: second stage is a Cox partial likelihood in (A, What, C) on (T, Y=event); estimates
    are working-model log hazard ratios per unit. The role screen still uses the event indicator.
    equivalence_margin: optional standardized TOST margin for an outcome-based Z sensitivity
    screen.  None keeps the primary proxy nomination outcome-blind and treats validity as an
    identifying assumption rather than something established by a non-significance test.
    latent_source: 'deviations' (default) or 'levels', provided for the idop ablation study."""
    qd, Y, Cov, names = coh['qd'], coh['Y'], coh['Cov'], coh['names']
    X = qd.values; s = qd.index.to_numpy(float); p = len(names)
    zc = stats.norm.ppf(0.975)

    # curves and deviations
    params = get_power_function_params(qd)
    C = np.column_stack([power_law(s, params.at[c, 'a'], params.at[c, 'b']) for c in names])
    U = X - C
    if latent_source not in ('deviations', 'levels'):
        raise ValueError("latent_source must be 'deviations' or 'levels'")
    lat = latent_state(U if latent_source == 'deviations' else X)
    r, F, Lam = lat['r'], lat['F'], lat['Lam']
    lam_norm = np.linalg.norm(Lam, axis=1)
    if equivalence_margin is None:
        outcome_screen = None
        pi_absent = np.ones(p, dtype=bool)
    else:
        outcome_screen = outcome_equation(Y, U, F, Cov, equivalence_margin=equivalence_margin,
                                          return_diagnostics=True)
        pi_absent = outcome_screen['absent']

    # support graph from patient-level edge selection
    G = X if support == 'levels' else lat['Xi']
    es = edge_select(pd.DataFrame(G, columns=names), alpha=alpha, k=k, threshold=threshold)
    A = adjacency_from_supports(es, names); Und = A | A.T
    ncomp = len(set(components(Und)))

    rows = []; n_weak = 0
    for a0 in range(p):
        # Nodewise regressions estimate adjacency, not causal direction.  The
        # graph therefore nominates candidates; it does not justify descendant
        # claims from asymmetric regression coefficients.
        Z, W = proxy_roles(Und, a0, pi_absent, lam_norm, r)
        if not screen(Z, W, r): continue
        if survival:
            e, se, nu = proximal_cox(coh['T'], Y, X[:, a0], X[:, W], X[:, Z], r=r + extra_rank, C=Cov, r_nu=r)
        else:
            e, se, nu = proximal_bridge(Y, X[:, a0], X[:, W], X[:, Z], r=r + extra_rank, C=Cov, r_nu=r)
        if nu < nu_min: n_weak += 1; continue
        nb, nse = cox_effect(coh['T'], Y, X[:, a0], C=Cov) if survival else ols_effect(Y, X[:, a0], C=Cov)
        rows.append(dict(exposure=names[a0], nZ=len(Z), nW=len(W), naive=nb, naive_se=nse,
                         proximal=e, se=se, nu=nu, Z=';'.join(names[z] for z in Z), W=';'.join(names[w] for w in W)))
    d = pd.DataFrame(rows)
    if len(d):
        d['t_prox'] = d.proximal / d.se; d['t_naive'] = d.naive / d.naive_se
        d['q_prox'] = bh(2 * stats.norm.sf(np.abs(d.t_prox))); d['q_naive'] = bh(2 * stats.norm.sf(np.abs(d.t_naive)))
        d['delta_star'] = d.proximal.abs() / amplification

    out = dict(estimates=d, params=params, curves=C, U=U, latent=lat, supports=es, A=A, Und=Und,
               ncomp=ncomp, pi_absent=pi_absent, names=names, X=X, s=s, Y=Y, Cov=Cov,
               alpha=alpha, k=k, threshold=threshold, support=support, n_weak=n_weak, nu_min=nu_min,
               outcome_screen=outcome_screen, equivalence_margin=equivalence_margin,
               latent_source=latent_source)
    if solve_ode:
        samples = get_power_function_samples(params, qd.index, n_samples=100)
        out['samples'] = samples
        out['edgelist'] = ode_solver(qd, samples, es, basis_order=BASIS_ORDER, ridge=ODE_RIDGE); out['basis_order'] = BASIS_ORDER; out['ode_ridge'] = ODE_RIDGE
    return out


def bootstrap_se(res, coh, n_boot=300, survival=True, extra_rank=1):
    """Patient bootstrap of the proximal estimate with the first stage inside the loop
    (roles fixed). Adds se_boot, t_boot, q_boot to res['estimates'] in place."""
    d = res['estimates']; X, Y, Cov, names = res['X'], res['Y'], res['Cov'], res['names']; r = res['latent']['r']
    for i in d.index:
        row = d.loc[i]; a0 = names.index(row.exposure)
        Z = [names.index(z) for z in row.Z.split(';')]; W = [names.index(w) for w in row.W.split(';')]
        if survival:
            _, se, _ = proximal_cox(coh['T'], Y, X[:, a0], X[:, W], X[:, Z], r=r + extra_rank, C=Cov, r_nu=r, n_boot=n_boot)
        else:
            se = row.se
        d.loc[i, 'se_boot'] = se
    d['t_boot'] = d.proximal / d.se_boot; d['q_boot'] = bh(2 * stats.norm.sf(np.abs(d.t_boot)))
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('study'); ap.add_argument('outcome')
    ap.add_argument('--p', type=int, default=60); ap.add_argument('--alpha', type=float, default=0.15)
    ap.add_argument('--k', type=int, default=5); ap.add_argument('--thr', type=float, default=0.6)
    ap.add_argument('--ode', action='store_true', help='also solve the weak-form ODE for signed edge weights')
    ap.add_argument('--survival', action='store_true', help='Cox second stage on follow-up time (log hazard ratios)')
    ap.add_argument('--boot', type=int, default=0, help='bootstrap se (first stage included) for every exposure')
    ap.add_argument('--equivalence-margin', type=float, default=None,
                    help='optional TOST margin (outcome SD per candidate SD) for a strict Z sensitivity screen')
    a = ap.parse_args()
    coh = load_cohort(a.study, a.outcome, a.p, survival=a.survival)
    res = analyse_cohort(coh, a.alpha, a.k, a.thr, solve_ode=a.ode, survival=a.survival,
                         equivalence_margin=a.equivalence_margin)
    tag = f'{a.study}_{a.outcome}' + ('_cox' if a.survival else '')
    d = res['estimates']; lat = res['latent']
    print(f"{a.study}/{a.outcome}: n={len(coh['Y'])} events={int(coh['Y'].sum())} p={len(coh['names'])} "
          f"| r={lat['r']} rho_bar={lat['rho'].mean():.3f} | support {int(res['A'].sum())} edges, {res['ncomp']} components "
          f"| {res['n_weak']} exposures abstain on nu < {res['nu_min']}")
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    if a.boot and len(d): bootstrap_se(res, coh, a.boot, survival=a.survival)
    d.to_csv(os.path.join(HERE, 'results', f'{tag}_estimates.csv'), index=False)
    if len(d):
        print(f"{len(d)} exposures pass | nu median {d.nu.median():.3f} | q<0.10: {int((d.q_prox<0.1).sum())} | min q {d.q_prox.min():.3f}")
        top = d.reindex(d[('t_boot' if a.boot else 't_prox')].abs().sort_values(ascending=False).index).head(8)
        cols = ['exposure', 'nZ', 'nW', 'naive', 'proximal', 'se', 't_prox', 'q_prox', 'nu', 'delta_star'] + (['se_boot', 't_boot', 'q_boot'] if a.boot else [])
        print(('log hazard ratio per unit' if a.survival else 'risk difference per unit') + ':')
        print(top[cols].round(3).to_string(index=False))
    if a.ode:
        res['edgelist'].to_csv(os.path.join(HERE, 'results', f'{tag}_edgelist.csv'), index=False)
        print(f"signed edgelist: {len(res['edgelist'])} edges")
