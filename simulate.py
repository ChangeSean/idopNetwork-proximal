"""Simulation study, run through the same pipeline as the application.

    python simulate.py coverage [n] [reps]     # coverage against the violation magnitude delta
    python simulate.py tradeoff [n] [reps]     # identifying share vs proxy strength (Proposition 2)
    python simulate.py breadth [reps]          # n x r grid at delta = 0 and 0.1
    python simulate.py survival [n] [reps]     # Cox second stage: plug-in vs bootstrap coverage
    python simulate.py ablation [n] [reps]     # contribution of ordering, curves and windowing
    python simulate.py safeguards [n] [reps]   # nu floor x truncation rule
    python simulate.py violations [n] [reps]   # P1 and P2 violations separately
    python simulate.py all

Every replicate goes through analyse_cohort(): Zscore_shift -> niche ordering ->
power curves -> deviations -> latent state -> edge_select -> proxy roles -> bridge.
The transformed exposure is A/sd(A) up to a shift, so estimates and standard
errors are divided by sd(A) to return to the units of the data-generating model.

Writes results/sim_coverage_vs_delta.csv and results/sim_tradeoff.csv.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
warnings.filterwarnings('ignore')
from dgp import make_system, draw
from idop_core import data_transformation, data_quasi_dynamic
from run_application import analyse_cohort
from proximal import ols_effect, proximal_bridge
from survival import cox_effect, proximal_cox

RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
ALPHAS, K, THR = (0.15, 0.20, 0.25, 0.30), 5, 0.6   # alpha escalates until the screen passes (outcome-blind)
LAT_X = 0.25                          # latent share of the levels in the coverage study
Z95 = stats.norm.ppf(0.975)


def run_pipeline(X, Y, alpha=ALPHAS[0], k=K, threshold=THR, support='levels', T=None,
                 latent_source='deviations', ordering='niche', rng=0,
                 nu_min=0.05, extra_rank=1):
    names = [f'v{j}' for j in range(X.shape[1])]
    Xt = data_transformation(pd.DataFrame(X, columns=names), 'Zscore_shift')
    if ordering == 'niche':
        order = np.argsort(Xt.sum(axis=1).values, kind='stable')
        qd = data_quasi_dynamic(Xt)
    elif ordering == 'random':
        order = np.random.default_rng(rng).permutation(len(Xt))
        qd = Xt.iloc[order].copy()
        # Keep a positive increasing index so the same power-law fitter runs;
        # only the patient-to-index assignment is broken.
        qd.index = np.sort(Xt.sum(axis=1).to_numpy())
    else:
        raise ValueError("ordering must be 'niche' or 'random'")
    coh = dict(qd=qd, Y=Y[order], Cov=np.zeros((len(Y), 0)), names=names,
               T=None if T is None else T[order])
    return analyse_cohort(coh, alpha, k, threshold, support=support, survival=T is not None,
                          latent_source=latent_source, nu_min=nu_min, extra_rank=extra_rank)


def proposed(res, X, a0):
    """Estimate, se, nu and the selected sets for exposure a0, in raw units; None if screened out."""
    est = res['estimates']
    if not len(est): return None
    row = est[est.exposure == f'v{a0}']
    if not len(row): return None
    row = row.iloc[0]; sd = X[:, a0].std()
    W = [int(w[1:]) for w in row.W.split(';')]; Z = [int(z[1:]) for z in row.Z.split(';')]
    return dict(est=row.proximal / sd, se=row.se / sd, nu=row.nu, W=W, Z=Z)


def run_escalating(X, Y, a0, alphas=ALPHAS, support='levels', T=None,
                   latent_source='deviations', ordering='niche', rng=0, k=K,
                   nu_min=0.05, extra_rank=1):
    """Run the pipeline at increasing alpha until exposure a0 passes the proxy screen.
    Returns (res, proposed-or-None, alpha_used); res is at the last alpha tried."""
    for alpha in alphas:
        res = run_pipeline(X, Y, alpha=alpha, support=support, T=T, k=k,
                           latent_source=latent_source, ordering=ordering, rng=rng,
                           nu_min=nu_min, extra_rank=extra_rank)
        p_ = proposed(res, X, a0)
        if p_ is not None: return res, p_, alpha
    return res, None, np.nan


# ------------------------------------------------------------------ coverage against delta
def coverage(n=1000, reps=100, deltas=(0.0, 0.05, 0.10, 0.15, 0.20)):
    rows = []; t0 = time.time()
    for delta in deltas:
        S = make_system(delta=delta, seed=0)
        a0, iW, iZ, r = S['a0'], S['iW'], S['iZ'], S['r']
        # the (P1) violation makes Z_0 a mediator, so the target is the total effect
        target = S['gamma'] + S['pi'][iZ[0]] * S['Theta'][a0, iZ[0]]
        viol = set(iW)                                          # the W-block carries the (P2) violation
        acc = {m: [] for m in ('proposed', 'designated', 'unadjusted')}
        extra = dict(nu=[], nZ=[], nW=[], W_invalid=[], r_hat=[], alpha=[])
        for rep in range(reps):
            d = draw(S, n, rng=90000 + rep, lat_x=LAT_X); X, Y, A = d['X'], d['Y'], d['X'][:, a0]
            e, se, _ = proximal_bridge(Y, A, X[:, iW], X[:, iZ], r=r); acc['designated'].append((e, se))
            acc['unadjusted'].append(ols_effect(Y, A))
            res, p_, alpha_used = run_escalating(X, Y, a0)
            extra['r_hat'].append(res['latent']['r'])
            if p_ is not None:
                acc['proposed'].append((p_['est'], p_['se'])); extra['nu'].append(p_['nu']); extra['alpha'].append(alpha_used)
                extra['nZ'].append(len(p_['Z'])); extra['nW'].append(len(p_['W']))
                extra['W_invalid'].append(np.mean([w in viol for w in p_['W']]))
        for m, v in acc.items():
            v = np.array(v)
            row = dict(delta=delta, method=m, n_ok=len(v), abstain=1 - len(v) / reps)
            if len(v):
                lo, hi = v[:, 0] - Z95 * v[:, 1], v[:, 0] + Z95 * v[:, 1]
                row.update(bias=v[:, 0].mean() - target, coverage=np.mean((lo <= target) & (target <= hi)),
                           width=np.mean(hi - lo))
            if m == 'proposed':
                row.update(nu=np.mean(extra['nu']) if extra['nu'] else np.nan, nZ=np.mean(extra['nZ']) if extra['nZ'] else np.nan,
                           nW=np.mean(extra['nW']) if extra['nW'] else np.nan,
                           W_invalid_frac=np.mean(extra['W_invalid']) if extra['W_invalid'] else np.nan,
                           r_hat=np.mean(extra['r_hat']), alpha_used=np.mean(extra['alpha']) if extra['alpha'] else np.nan)
            rows.append(row)
        print(f'delta={delta:.2f} done  {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, 'sim_coverage_vs_delta.csv'), index=False)
    print(df.round(3).to_string(index=False)); return df


# ------------------------------------------------------------------ identifying share vs proxy strength
def tradeoff(n=1000, reps=60, lat=(1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10)):
    S = make_system(delta=0.0, seed=0); a0, iW, iZ, r = S['a0'], S['iW'], S['iZ'], S['r']
    truth = (np.abs(S['Theta']) > 1e-9) | (np.abs(S['Theta']).T > 1e-9); np.fill_diagonal(truth, False)
    rows = []; t0 = time.time()
    for lat_x in lat:
        acc = dict(share=[], f1=[], wfp=[], r_hat=[], nu_des=[], nu_prop=[], ols=[], des=[], prop=[], alpha=[])
        for rep in range(reps):
            d = draw(S, n, rng=40000 + rep, lat_x=lat_x); X, Y, F, A = d['X'], d['Y'], d['F'], d['X'][:, a0]
            acc['share'].append((lat_x * (F @ S['Lam'].T)).var(0).sum() / X.var(0).sum())
            res, p_, alpha_used = run_escalating(X, Y, a0); adj = res['Und'].copy(); np.fill_diagonal(adj, False)
            tp = (truth & adj).sum() / 2; fp = ((~truth) & adj).sum() / 2; fn = (truth & ~adj).sum() / 2
            acc['f1'].append(2 * tp / max(2 * tp + fp + fn, 1e-9)); acc['wfp'].append(int(adj[iW].sum()))
            acc['r_hat'].append(res['latent']['r'])
            e, _, nu = proximal_bridge(Y, A, X[:, iW], X[:, iZ], r=r); acc['des'].append(e); acc['nu_des'].append(nu)
            acc['ols'].append(ols_effect(Y, A)[0])
            if p_ is not None: acc['prop'].append(p_['est']); acc['nu_prop'].append(p_['nu']); acc['alpha'].append(alpha_used)
        rows.append(dict(lat_x=lat_x, x_latent_share=np.mean(acc['share']), edge_F1=np.mean(acc['f1']),
                         W_false_edges=np.mean(acc['wfp']), rank_hat=np.mean(acc['r_hat']),
                         nu=np.mean(acc['nu_des']), nu_proposed=np.mean(acc['nu_prop']) if acc['nu_prop'] else np.nan,
                         proposed_abstain=1 - len(acc['prop']) / reps, alpha_used=np.mean(acc['alpha']) if acc['alpha'] else np.nan,
                         ols_bias=np.mean(acc['ols']) - S['gamma'], prox_bias=np.mean(acc['des']) - S['gamma'],
                         proposed_bias=np.mean(acc['prop']) - S['gamma'] if acc['prop'] else np.nan))
        print(f'lat_x={lat_x} done  {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, 'sim_tradeoff.csv'), index=False)
    print(df.round(3).to_string(index=False)); return df


# ------------------------------------------------------------------ breadth: n x r grid
def breadth(reps=60, ns=(400, 1000, 2000), rs=(1, 2, 3), deltas=(0.0, 0.1)):
    """Coverage of the proposed and designated estimators over sample size and latent rank."""
    rows = []; t0 = time.time()
    for r in rs:
        for n in ns:
            for delta in deltas:
                S = make_system(delta=delta, seed=0, r=r); a0, iW, iZ = S['a0'], S['iW'], S['iZ']
                target = S['gamma'] + S['pi'][iZ[0]] * S['Theta'][a0, iZ[0]]
                acc = {'proposed': [], 'designated': []}; nus = []; rhat = []
                for rep in range(reps):
                    d = draw(S, n, rng=70000 + rep, lat_x=LAT_X); X, Y, A = d['X'], d['Y'], d['X'][:, a0]
                    e, se, _ = proximal_bridge(Y, A, X[:, iW], X[:, iZ], r=r); acc['designated'].append((e, se))
                    res, p_, au = run_escalating(X, Y, a0); rhat.append(res['latent']['r'])
                    if p_ is not None: acc['proposed'].append((p_['est'], p_['se'])); nus.append(p_['nu'])
                for m, v in acc.items():
                    v = np.array(v); row = dict(r=r, n=n, delta=delta, method=m, n_ok=len(v), abstain=1 - len(v) / reps, r_hat=np.mean(rhat))
                    if len(v):
                        lo, hi = v[:, 0] - Z95 * v[:, 1], v[:, 0] + Z95 * v[:, 1]
                        row.update(bias=v[:, 0].mean() - target, coverage=np.mean((lo <= target) & (target <= hi)), width=np.mean(hi - lo),
                                   nu=np.mean(nus) if m == 'proposed' and nus else np.nan)
                    rows.append(row)
                print(f'r={r} n={n} delta={delta} done  {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, 'sim_breadth.csv'), index=False)
    print(df.round(3).to_string(index=False)); return df


# ------------------------------------------------------------------ idop component ablation
def ablation(n=1000, reps=60):
    """Ablate the three idop-specific inputs to proxy nomination.

    ``levels latent`` replaces curve deviations by raw levels for PCA;
    ``random ordering`` destroys the niche order while keeping the algorithm;
    ``one window`` removes stability across local niche windows.  The target
    exposure and data-generating system are identical across arms and repeats.
    """
    S = make_system(delta=0.0, seed=0)
    a0 = S['a0']; target = S['gamma']
    arms = (
        ('full', dict()),
        ('levels latent', dict(latent_source='levels')),
        ('random ordering', dict(ordering='random')),
        ('one window', dict(k=1)),
    )
    acc = {name: [] for name, _ in arms}; diag = {name: [] for name, _ in arms}
    t0 = time.time()
    for rep in range(reps):
        d = draw(S, n, rng=80000 + rep, lat_x=LAT_X); X, Y = d['X'], d['Y']
        for name, kw in arms:
            res, p_, au = run_escalating(X, Y, a0, rng=81000 + rep, **kw)
            diag[name].append((res['latent']['r'], res['ncomp']))
            if p_ is not None: acc[name].append((p_['est'], p_['se'], p_['nu']))
        if rep % 10 == 9:
            print(f'ablation rep {rep + 1}/{reps}  {time.time() - t0:.0f}s', flush=True)
    rows = []
    for name, _ in arms:
        v = np.asarray(acc[name]); dg = np.asarray(diag[name])
        row = dict(arm=name, n_ok=len(v), abstain=1 - len(v) / reps,
                   rank_hat=dg[:, 0].mean(), components=dg[:, 1].mean())
        if len(v):
            lo, hi = v[:, 0] - Z95 * v[:, 1], v[:, 0] + Z95 * v[:, 1]
            row.update(bias=v[:, 0].mean() - target,
                       coverage=np.mean((lo <= target) & (target <= hi)),
                       width=np.mean(hi - lo), nu=v[:, 2].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'sim_ablation.csv'), index=False)
    print(df.round(3).to_string(index=False)); return df


# ------------------------------------------------------------------ safeguard calibration
def safeguards(n=1000, reps=100, deltas=(0.0, 0.2), floors=(0.0, 0.03, 0.05, 0.08)):
    """Cross the proxy-strength floor with truncation at rhat or rhat+1."""
    raw = []; t0 = time.time()
    for delta in deltas:
        S = make_system(delta=delta, seed=0); a0, iZ = S['a0'], S['iZ']
        target = S['gamma'] + S['pi'][iZ[0]] * S['Theta'][a0, iZ[0]]
        for rep in range(reps):
            d = draw(S, n, rng=50000 + rep, lat_x=LAT_X)
            for extra_rank, truncation in ((0, 'rhat'), (1, 'rp1')):
                _, p_, _ = run_escalating(d['X'], d['Y'], a0, nu_min=0.0,
                                           extra_rank=extra_rank)
                if p_ is not None:
                    raw.append(dict(delta=delta, rep=rep, truncation=truncation,
                                    est=p_['est'], se=p_['se'], nu=p_['nu'], target=target))
        print(f'safeguards delta={delta:.2f} done  {time.time() - t0:.0f}s', flush=True)
    d = pd.DataFrame(raw); rows = []
    for delta in deltas:
        for floor in floors:
            for truncation in ('rhat', 'rp1'):
                g = d[(d.delta == delta) & (d.truncation == truncation) & (d.nu >= floor)]
                row = dict(delta=delta, nu_min=floor, truncation=truncation, keep=len(g))
                if len(g):
                    lo, hi = g.est - Z95 * g.se, g.est + Z95 * g.se
                    row.update(bias=(g.est - g.target).mean(),
                               coverage=np.mean((lo <= g.target) & (g.target <= hi)),
                               width=np.mean(hi - lo))
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'sim_safeguard_sensitivity.csv'), index=False)
    print(out.round(3).to_string(index=False)); return out


# ------------------------------------------------------------------ violations separately
def violations(n=1000, reps=100, deltas=(0.1, 0.2)):
    """Separate treatment-proxy (P1) and outcome-proxy (P2) violations."""
    rows = []; t0 = time.time()
    for label, which in (('P1 only', 'p1'), ('P2 only', 'p2')):
        for delta in deltas:
            S = make_system(seed=0, delta_p1=delta if which == 'p1' else 0.0,
                            delta_p2=delta if which == 'p2' else 0.0)
            a0, iW, iZ, r = S['a0'], S['iW'], S['iZ'], S['r']
            target = S['gamma'] + S['pi'][iZ[0]] * S['Theta'][a0, iZ[0]]
            acc = {'proposed': [], 'designated': []}
            for rep in range(reps):
                d = draw(S, n, rng=30000 + rep, lat_x=LAT_X); X, Y, A = d['X'], d['Y'], d['X'][:, a0]
                e, se, _ = proximal_bridge(Y, A, X[:, iW], X[:, iZ], r=r)
                acc['designated'].append((e, se))
                _, p_, _ = run_escalating(X, Y, a0)
                if p_ is not None: acc['proposed'].append((p_['est'], p_['se']))
            for method, vals in acc.items():
                v = np.asarray(vals); row = dict(violation=label, delta=delta, method=method,
                                                  n_ok=len(v), abstain=1 - len(v) / reps)
                if len(v):
                    lo, hi = v[:, 0] - Z95 * v[:, 1], v[:, 0] + Z95 * v[:, 1]
                    row.update(bias=v[:, 0].mean() - target,
                               coverage=np.mean((lo <= target) & (target <= hi)),
                               width=np.mean(hi - lo))
                rows.append(row)
            print(f'{label} delta={delta:.2f} done  {time.time() - t0:.0f}s', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'sim_p1_vs_p2.csv'), index=False)
    print(out.round(3).to_string(index=False)); return out


# ------------------------------------------------------------------ survival outcome
def survival_sim(n=1000, reps=60, n_boot=100, delta=0.0):
    """Time-to-event outcome with hazard exp(gamma A + pi'X + lam_Y'F); independent censoring.
    Coverage of the true log hazard ratio gamma with plug-in and bootstrap Cox standard errors."""
    S = make_system(delta=delta, seed=0); a0, iW, iZ, r = S['a0'], S['iW'], S['iZ'], S['r']
    target = S['gamma'] + S['pi'][iZ[0]] * S['Theta'][a0, iZ[0]]
    rows_rep = []; t0 = time.time()
    for rep in range(reps):
        d = draw(S, n, rng=60000 + rep, lat_x=LAT_X); X, F, A = d['X'], d['F'], d['X'][:, a0]
        rng = np.random.default_rng(60000 + rep)
        lin = S['gamma'] * A + X @ S['pi'] + F @ S['lam_Y']; lin = lin - lin.mean()
        T0 = rng.exponential(np.exp(-lin)); Cc = rng.exponential(np.quantile(T0, 0.7) * 2.0)
        E = (T0 <= Cc).astype(float); T = np.minimum(T0, Cc)
        # designated proxies, Cox second stage
        e_d, se_d, _ = proximal_cox(T, E, A, X[:, iW], X[:, iZ], r=r)
        _, seb_d, _ = proximal_cox(T, E, A, X[:, iW], X[:, iZ], r=r, n_boot=n_boot, rng=rep)
        e_u, se_u = cox_effect(T, E, A)
        # working-model estimand of Proposition 4: Cox on (A, E[F | A, Z*]) with the true F projected
        D0 = np.column_stack([np.ones(n), A, X[:, iZ]]); Fproj = D0 @ np.linalg.lstsq(D0, F, rcond=None)[0]
        beta0 = cox_effect(T, E, A, C=Fproj)[0]
        e_o, se_o = cox_effect(T, E, A, C=np.column_stack([F, X[:, S['pi'] != 0]]))   # oracle: every term of the hazard
        row = dict(rep=rep, events=E.mean(), beta0=beta0, des_est=e_d, des_se=se_d, des_seb=seb_d, unadj_est=e_u, unadj_se=se_u, orc_est=e_o, orc_se=se_o)
        res, p_, au = run_escalating(X, E, a0, T=T)
        if p_ is not None:
            W, Z = p_['W'], p_['Z']; sd = A.std(); rr = res['latent']['r']
            e_p, se_p, nu = proximal_cox(T, E, A, X[:, W], X[:, Z], r=rr + 1, r_nu=rr)
            _, seb_p, _ = proximal_cox(T, E, A, X[:, W], X[:, Z], r=rr + 1, r_nu=rr, n_boot=n_boot, rng=rep)
            row.update(prop_est=e_p, prop_se=se_p, prop_seb=seb_p, nu=nu)
        rows_rep.append(row)
        if rep % 10 == 9: print(f'survival rep {rep + 1}/{reps}  {time.time() - t0:.0f}s', flush=True)
    d = pd.DataFrame(rows_rep); out = []
    def cov(est, se, tgt): return np.mean(np.abs(est - tgt) <= Z95 * se)
    for m, e, s1, s2 in (('proposed', 'prop_est', 'prop_se', 'prop_seb'), ('designated', 'des_est', 'des_se', 'des_seb')):
        g = d.dropna(subset=[e])
        out.append(dict(method=m, n_ok=len(g), abstain=1 - len(g) / reps, bias_vs_gamma=g[e].mean() - target, bias_vs_beta0=(g[e] - g.beta0).mean(),
                        coverage_beta0_plugin=cov(g[e], g[s1], g.beta0), coverage_beta0_bootstrap=cov(g[e], g[s2], g.beta0),
                        se_ratio=np.median(g[s2] / g[s1]), width_bootstrap=np.mean(2 * Z95 * g[s2])))
    for m, e, s1 in (('oracle (A, F, X_pi)', 'orc_est', 'orc_se'), ('unadjusted', 'unadj_est', 'unadj_se')):
        out.append(dict(method=m, n_ok=reps, abstain=0.0, bias_vs_gamma=d[e].mean() - target, bias_vs_beta0=(d[e] - d.beta0).mean(),
                        coverage_beta0_plugin=cov(d[e], d[s1], target if m.startswith('oracle') else d.beta0), coverage_beta0_bootstrap=np.nan, se_ratio=np.nan, width_bootstrap=np.nan))
    df = pd.DataFrame(out); df['events'] = d.events.mean(); df['gamma'] = target; df['beta0_mean'] = d.beta0.mean()
    df.to_csv(os.path.join(RES, 'sim_survival.csv'), index=False); print(df.round(3).to_string(index=False)); return df


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else None
    if what in ('coverage', 'all'): coverage(n, reps or 100)
    if what in ('tradeoff', 'all'): tradeoff(n, reps or 60)
    if what == 'breadth': breadth(reps or 60)
    if what == 'survival': survival_sim(n, reps or 60)
    if what == 'ablation': ablation(n, reps or 60)
    if what == 'safeguards': safeguards(n, reps or 100)
    if what == 'violations': violations(n, reps or 100)
