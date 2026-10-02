"""Conditionally complete designs with joint readout coordinates.

The joint grid dimension is retained. A design must supply that dimension
conditionally, and its smallest conditional canonical root determines quality.
No clinical outcome enters this selection. Rank tests estimate information,
not biological proxy exclusions.
"""
import numpy as np
from proximal import bridge_information_rank, proxy_strength
from multiscale_bridge import candidate
from joint_readout_bridge import joint_projection, estimate


def select_conditional_design(grid, a, treatment_eligible=None):
    if not grid:
        raise ValueError('Empty molecular design grid')
    marginal = [(*candidate(res, a, treatment_eligible), res) for res in grid]
    r = max(row['r_joint'] for row, _, _, _ in marginal)
    candidates = []
    for row, Z, W, res in marginal:
        row = dict(row, r_grid=r, conditional_root=np.nan)
        if r == 0:
            row['status'] = 'no_shared_direction'
        elif row['r_joint'] != r or min(len(Z), len(W)) < r:
            row['status'] = 'proxy_dimensions'
        else:
            try:
                X, C = res['X'], res['Cov']
                P = joint_projection(X, a, Z, W, r, C)
                info = bridge_information_rank(X[:, W] @ P, X[:, Z], X[:, a], C,
                                             alpha=min(.05, len(X)**(-.5)))
                row.update(r_bridge=r, r_signal=info['r'],
                           conditional_roots=';'.join(map(str, info['roots'])),
                           conditional_pvalues=';'.join(map(str, info['pvalues'])),
                           nu_raw=proxy_strength(X[:, W], X[:, Z], X[:, a], r, C),
                           nu=proxy_strength(X[:, W] @ P, X[:, Z], X[:, a], r, C),
                           conditional_rank_agreement=info['r'] == r,
                           conditional_root=float(info['roots'][r-1]) if len(info['roots']) >= r else 0.)
                row['status'] = 'reported' if info['r'] == r else 'conditional_information_dimension'
            except (ValueError, np.linalg.LinAlgError) as error:
                row.update(status='numerical_support', reason=str(error))
        candidates.append((row, Z, W, res))
    eligible = [v for v in candidates if v[0]['status'] == 'reported']
    # Python max retains the first item for exact ties, matching grid order.
    selected = max(eligible, key=lambda v: v[0]['conditional_root']) if eligible else max(
        candidates, key=lambda v: (v[0]['r_joint'], np.nan_to_num(v[0]['conditional_root'], nan=-1.)))
    selected[0]['eligible_designs'] = len(eligible)
    return selected


def bootstrap_interval(point, values, attempts, level=.95):
    """Normal interval for the complete pipeline, with explicit availability.

    Finite draws cannot estimate the tails of a non-estimable estimator.
    A finite interval therefore requires >=97.5% successful complete fits and
    at least 20 values. Incomplete results retain all attempts and point fits;
    no CI or p value is inferred from a conditional subset of draws. This is
    an operational availability rule, not a weak-identification guarantee.
    """
    from scipy.stats import norm
    finite = np.asarray(values, float)
    if not np.isfinite(finite).all():
        raise ValueError('Only recorded finite successful draws are accepted')
    rate = len(finite)/attempts if attempts else 0.
    enough = len(finite) >= 20 and rate >= .975
    se = float(np.std(finite, ddof=1)) if enough else np.nan
    radius = norm.ppf((1+level)/2)*se
    return dict(estimate=float(point), se_boot=se, ci_low=point-radius,
                ci_high=point+radius, boot_success=len(finite),
                boot_attempts=attempts, availability=rate,
                interval_status='complete_pipeline_interval' if enough else 'construction_incomplete')
