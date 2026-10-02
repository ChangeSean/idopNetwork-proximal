"""Figure 3 from the frozen ovarian discovery sample and selected designs."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figures as F
import independent_design_bridge as ID
from niche_ode import DEFAULT_DEGREE, DEFAULT_RIDGE, solve_niche_ode, decomposition_edges
from multiscale_bridge import design_grid
from proximal import adjacency_from_supports
from run_discovery_estimation_application import prepare

ROOT = Path(__file__).resolve().parent
APP = ROOT / 'results/discovery_estimation_application_20261001'
EXPOSURES = ('PTEN', 'SERPINE1', 'CCNE1')


def result_hashes():
    """Clinical and simulation coefficients must remain unchanged by plotting."""
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / 'results').rglob('*.csv')}


def network_edges(selected, decomposition):
    """Display the frozen regression supports independently of curve weights."""
    source, target = np.nonzero(selected['A'])
    names = selected['names']
    return pd.DataFrame(dict(source=[names[i] for i in source],
                             target=[names[j] for j in target], weight=np.ones(len(source))))


def neutral_support_arrows(figure):
    for ax in figure.axes:
        for annotation in ax.texts:
            arrow = getattr(annotation, 'arrow_patch', None)
            if arrow is not None:
                arrow.set_color(F.MUTED)
                arrow.set_linewidth(.6)
        for location in ('left', 'center', 'right'):
            title = ax.get_title(loc=location)
            if 'signed edges' in title:
                ax.set_title(title.replace('signed edges', 'support edges'), loc=location, fontsize=8)


def discovery_networks(degree=DEFAULT_DEGREE, ridge=DEFAULT_RIDGE):
    cohort, _, _, _, _, info = prepare('ov')
    manifest = json.loads((APP / 'ov_manifest.json').read_text(encoding='utf-8'))
    frozen = json.loads((APP / 'ov_discovery_designs.json').read_text(encoding='utf-8'))
    for key in ('names', 'n_discovery', 'n_estimation', 'split_seed',
                'discovery_id_sha256', 'estimation_id_sha256'):
        assert info[key] == manifest[key], 'Discovery replay: ' + key
    for key in ('discovery_mean', 'discovery_sd'):
        assert np.allclose([info[key][n] for n in info['names']],
                           [manifest[key][n] for n in info['names']], rtol=1e-12, atol=1e-12)
    grid = design_grid(cohort)
    mapped, provenance = {}, []
    cache = {}
    for exposure in EXPOSURES:
        saved = frozen[exposure]
        replay = ID.freeze(grid, info['names'].index(exposure))
        assert saved['ready'] and replay['ready'], exposure
        for key in ('a', 'Z', 'W', 'pivots'):
            assert replay[key] == saved[key], f'{exposure}: {key}'
        assert np.allclose(replay['P'], saved['P'], rtol=1e-8, atol=1e-8), exposure
        row = saved['row']
        for key in ('window_fraction', 'alpha', 'r_grid', 'r_signal', 'Z', 'W'):
            assert replay['row'][key] == row[key], f'{exposure}: {key}'
        selected = next(r for r in grid if r['alpha'] == row['alpha'] and
                        r.get('window_fraction', .2) == row['window_fraction'])
        design_key = (row['window_fraction'], row['alpha'])
        if design_key not in cache:
            names = selected['names']
            supports = pd.DataFrame([
                dict(target=names[j], source='{' + ','.join(names[i] for i in
                     np.flatnonzero(selected['A'][:, j])) + '}')
                for j in range(len(names))])
            assert np.array_equal(adjacency_from_supports(supports, names), selected['A'])
            dec = solve_niche_ode(cohort['qd'], supports, degree=degree, ridge=ridge, n_grid=50)
            edges = network_edges(selected, dec)
            cache[design_key] = dict(supports=supports, decomposition=dec, edgelist=edges)
        res = dict(selected, qd=cohort['qd'], estimates=pd.DataFrame([row]),
                   ode_degree=degree, ode_ridge=ridge, **cache[design_key])
        mapped[exposure] = res
        provenance.append(dict(exposure=exposure, n_discovery=info['n_discovery'],
                               n_proteins=len(info['names']), Z=row['Z'], W=row['W'],
                               r_grid=row['r_grid'], r_signal=row['r_signal'],
                               conditional_root=row['conditional_root'],
                               window_fraction=row['window_fraction'], alpha=row['alpha'],
                               support_edges=int(np.triu(selected['Und'], 1).sum()),
                               displayed_support_edges=len(res['edgelist']),
                               signed_ode_edges=len(decomposition_edges(res['decomposition'])),
                               decomposition_diagnostics=res['decomposition']['diagnostics'][exposure]))
    return mapped, provenance, info


def draw_decomposition(ax, res, exposure):
    dec = res['decomposition']
    s = np.asarray(dec['sample_tau'])
    j = list(dec['features']).index(exposure)
    predicted = np.asarray(dec['predicted_states'])[:, j]
    ax.scatter(res['qd'].index, res['qd'][exposure], s=5, color=F.MUTED,
               alpha=.45, linewidths=0)
    ax.plot(s, predicted, color=F.INK, lw=1.35)
    self_part = dec['intercepts'][j] + dec['interaction_functions'].get(
        (exposure, exposure), np.zeros_like(s))
    if dec['settings']['degree']:
        ax.plot(s, self_part, color=F.INK2, lw=1, ls='--')
    palette = (F.BLUE, F.AQUA, F.YELLOW, '#9b4567', '#7859a8')
    endpoints = []
    for i, source in enumerate(dec['support_sets'].get(exposure, [])):
        if dec['settings']['degree'] == 0:
            continue
        if source == exposure:
            continue
        contribution = np.asarray(dec['interaction_functions'][(exposure, source)])
        color = palette[i % len(palette)]
        ax.plot(s, contribution, color=color, lw=1)
        endpoints.append((float(contribution[-1]), source, color))
    endpoints.sort()
    span = max(np.ptp(ax.get_ylim()), 1.)
    labels = [v[0] for v in endpoints]
    for i in range(1, len(labels)):
        labels[i] = max(labels[i], labels[i-1] + .10 * span)
    dx = s[-1] - s[0]
    for (value, source, color), position in zip(endpoints, labels):
        ax.text(s[-1] + .035 * dx, position, source, color=color, fontsize=8,
                ha='left', va='center')
        if abs(value-position) > 1e-10:
            ax.plot([s[-1], s[-1] + .028 * dx], [value, position], lw=.45,
                    color=F.MUTED, ls=':')
    if labels:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, min(labels)-.05*span), max(hi, max(labels)+.05*span))
    ax.set_xlim(s[0], s[-1] + .34 * dx)
    ax.axhline(0, color=F.GRID, lw=.7)
    if dec['settings']['degree'] == 0:
        ax.text(.98, .06, 'Source terms merged into baseline', transform=ax.transAxes,
                ha='right', fontsize=7, color=F.INK2)
    ax.grid(axis='y', color=F.GRID, lw=.4)
    ax.set_xlabel('Niche index $s$', fontsize=8)
    ax.set_ylabel('Protein level' if dec['settings']['degree'] == 0 else 'Cumulative niche contribution', fontsize=8)
    ax.set_title(exposure + (' zero-degree niche trend' if dec['settings']['degree'] == 0 else ' curve decomposition'),
                 fontsize=9, loc='left', pad=9)
    ax.tick_params(labelsize=8)


def reference_network_figures():
    """Use the same ODE refit for the full-cohort network reference panels."""
    from run_application import load_cohort, analyse_cohort
    from proximal import components, proxy_roles
    reference = ROOT / 'results/joint_readout_application_20261001'
    cohort = load_cohort('ov', 'OS', p_keep=140, survival=True)
    res = analyse_cohort(cohort, alpha=.15, k=5, estimate_legacy=False, solve_ode=False)
    dec = solve_niche_ode(cohort['qd'], res['supports'], degree=DEFAULT_DEGREE, ridge=DEFAULT_RIDGE)
    res['edgelist'] = network_edges(res, dec)
    figure = F.fig_full_network(res)
    neutral_support_arrows(figure)
    for text in figure.texts:
        if 'signed edges' in text.get_text():
            text.set_text(f'Full ovarian reference: {len(res["names"])} proteins; '
                          f'{len(res["edgelist"])} directed support edges; {res["ncomp"]} components\n'
                          'Node size and hubs: total support degree')
            text.set_position((.5, .99)); text.set_ha('center'); text.set_fontsize(9)
    for legend in list(figure.legends):
        legend.remove()
    figure.legend(handles=[
        Line2D([], [], marker='o', ls='', color=F.YELLOW, mec=F.INK, ms=6, label='Top 10 hubs by degree'),
        Line2D([], [], marker='o', ls='', color='#e3d9b8', ms=4, label='Protein (size ~ degree)'),
        Line2D([], [], color=F.MUTED, lw=.6, label='Directed LASSO support')],
        loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(.5, -.01))
    F.save(figure, 'figS1_full_network')
    provenance = [dict(figure='figS1_full_network', n=len(cohort['qd']),
                       n_proteins=len(res['names']), support_edges=int(res['A'].sum()),
                       ode_edges=len(decomposition_edges(dec)), components=res['ncomp'])]
    cohort = load_cohort('ov', 'OS', survival=True)
    grid = design_grid(cohort)
    saved = pd.read_csv(reference/'figure_network_provenance.csv').set_index('exposure').loc['LCK']
    row = dict(saved, exposure='LCK')
    res = dict(next(r for r in grid if r['alpha'] == saved.alpha and
                    r.get('window_fraction', .2) == saved.window_fraction))
    names = res['names']
    z, w = proxy_roles(res['Und'], names.index('LCK'), np.ones(len(names), bool),
                       np.linalg.norm(res['latent']['Lam'], axis=1), 1, limit_to_treatment=False)
    assert ';'.join(names[i] for i in z) == saved.Z
    assert ';'.join(names[i] for i in w) == saved.W
    supports = pd.DataFrame([dict(target=names[j], source='{' + ','.join(
        names[i] for i in np.flatnonzero(res['A'][:, j])) + '}') for j in range(len(names))])
    dec = solve_niche_ode(cohort['qd'], supports, degree=DEFAULT_DEGREE, ridge=DEFAULT_RIDGE)
    res.update(estimates=pd.DataFrame([row]), edgelist=network_edges(res, dec))
    figure = F.fig_network(res, 'LCK')
    neutral_support_arrows(figure)
    for text in figure.texts:
        if 'signed edges' in text.get_text():
            text.set_text(f'{len(res["edgelist"])} directed LASSO support edges; {res["ncomp"]} components in total')
    for label in figure.axes[0].texts:
        if label.get_text() == 'IGFBP2':
            label.set_position((-10, -12))
            label.set_ha('right')
            label.set_va('top')
    for legend in list(figure.legends):
        legend.remove()
    figure.legend(handles=[
        Line2D([], [], marker='o', ls='', color=F.ORANGE, mec=F.INK, ms=6, label='Exposure $A$ = LCK'),
        Line2D([], [], marker='o', ls='', color=F.BLUE, mec=F.INK, ms=6, label='Treatment proxy $Z$'),
        Line2D([], [], marker='o', ls='', color=F.AQUA, mec=F.INK, ms=6, label='Outcome proxy $W$'),
        Line2D([], [], color=F.MUTED, lw=.6, label='Directed LASSO support')],
        loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(.5, -.02))
    F.save(figure, 'figS2_network')
    provenance.append(dict(figure='figS2_network', n=len(cohort['qd']), n_proteins=len(names),
                           Z=saved.Z, W=saved.W, alpha=saved.alpha,
                           window_fraction=saved.window_fraction, support_edges=len(res['edgelist']),
                           ode_edges=len(decomposition_edges(dec)),
                           components=res['ncomp'], selection='saved full-cohort reference roles'))
    return provenance


def trial(degree, ridge=DEFAULT_RIDGE):
    """Compare an alternative degree or ridge without replacing paper figures."""
    canonical = [ROOT/'manuscript.md', ROOT/'manuscript_SiM.docx']
    canonical += [p for p in (ROOT/'figures').iterdir() if p.is_file()]
    before = result_hashes()
    artifacts = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in canonical}
    mapped, _, _ = discovery_networks()
    F.set_style()
    fig, axes = plt.subplots(3, 2, figsize=(F.DOUBLE, 8.9))
    fig.subplots_adjust(left=.10, right=.98, bottom=.12, top=.90, hspace=.52, wspace=.40)
    comparisons = []
    cache = {}
    for i, exposure in enumerate(EXPOSURES):
        res = mapped[exposure]
        design = (res.get('window_fraction', .2), res['alpha'])
        if design not in cache:
            cache[design] = solve_niche_ode(res['qd'], res['supports'], degree=degree, ridge=ridge)
        alternative = cache[design]
        for j, item in enumerate((res, dict(res, decomposition=alternative))):
            draw_decomposition(axes[i, j], item, exposure)
            axes[i, j].set_title(exposure + f'   degree {DEFAULT_DEGREE if j == 0 else degree}, ridge {DEFAULT_RIDGE if j == 0 else ridge:g}',
                                 fontsize=9, loc='left', pad=9)
        comparisons.append(dict(exposure=exposure, baseline=res['decomposition']['diagnostics'][exposure],
                                trial=alternative['diagnostics'][exposure]))
    comparison = f'Legendre degree {DEFAULT_DEGREE} versus {degree}' if degree != DEFAULT_DEGREE else f'Ridge {DEFAULT_RIDGE:g} versus {ridge:g}'
    fig.suptitle(f'{comparison}: ovarian discovery curves', fontsize=10, y=.98)
    subtitle = 'Degree 0: constant basis merged into the linear niche baseline' if 0 in (degree, DEFAULT_DEGREE) else 'Same discovery support'
    fig.text(.5, .944, subtitle, ha='center', fontsize=8, color=F.INK2)
    handles = [Line2D([], [], color=F.INK, lw=1.3, label='Reconstructed curve'),
               Line2D([], [], color=F.INK2, lw=1, ls='--', label='Intrinsic contribution with baseline'),
               Line2D([], [], color=F.MUTED, marker='o', ls='none', ms=3, label='Observed protein'),
               Line2D([], [], color=F.BLUE, lw=1, label='Source contributions labelled by protein')]
    if degree == DEFAULT_DEGREE == 0:
        handles = [handles[0], handles[2]]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=7.5,
               bbox_to_anchor=(.5, .025), columnspacing=1.2)
    output = ROOT/('results/niche_ode_degree_trial' if degree != DEFAULT_DEGREE else 'results/niche_ode_ridge_trial')
    output.mkdir(parents=True, exist_ok=True)
    filename = f'degree_{DEFAULT_DEGREE}_vs_{degree}' if degree != DEFAULT_DEGREE else f'ridge_{DEFAULT_RIDGE:g}_vs_{ridge:g}'
    for suffix in ('pdf', 'png'):
        fig.savefig(output/f'{filename}.{suffix}', bbox_inches='tight', pad_inches=.02)
    plt.close(fig)
    assert before == result_hashes(), 'Trial changed an analysis CSV'
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in artifacts.items())
    document = dict(baseline_settings=mapped[EXPOSURES[0]]['decomposition']['settings'],
                    settings=next(iter(cache.values()))['settings'], panels=comparisons,
                    analysis_csv_files_unchanged=len(before), publication_artifacts_unchanged=True,
                    zero_degree_interpretation='Constant basis shared by all sources, represented once as baseline; zero source contributions are a parameterisation convention' if 0 in (degree, DEFAULT_DEGREE) else None)
    (output/f'{filename}_diagnostics.json').write_text(json.dumps(document, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), **document), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-degree', type=int, help='Compare a nonnegative degree with the published degree 1')
    parser.add_argument('--trial-ridge', type=float, help='Compare a ridge penalty with the published value')
    args = parser.parse_args()
    if args.trial_degree is not None or args.trial_ridge is not None:
        degree = DEFAULT_DEGREE if args.trial_degree is None else args.trial_degree
        ridge = DEFAULT_RIDGE if args.trial_ridge is None else args.trial_ridge
        if degree < 0:
            parser.error('Trial degree must be nonnegative')
        if not np.isfinite(ridge) or ridge < 0:
            parser.error('Trial ridge must be finite and nonnegative')
        trial(degree, ridge)
        return
    before = result_hashes()
    mapped, provenance, info = discovery_networks()
    F.set_style()
    fig = plt.figure(figsize=(F.DOUBLE, 9.3))
    grid = fig.add_gridspec(3, 2, left=.045, right=.985, top=.92, bottom=.12,
                           width_ratios=(1.06, 1), wspace=.27, hspace=.57)
    for i, exposure in enumerate(EXPOSURES):
        res = mapped[exposure]
        ax = fig.add_subplot(grid[i, 0])
        F._draw_planes(ax, res, exposure, label_others=False,
                       w_in=F.DOUBLE/2, h_in=2.4, compact=True)
        # Spread the eight W labels over the gap between the outer planes.
        row = res['estimates'].iloc[0]
        names_w = row.W.split(';')
        labels_w = sorted([t for t in ax.texts if t.get_text() in names_w],
                          key=lambda t: -t.get_position()[1])
        leaders_w = ax.lines[-len(names_w):]
        for label, leader, y in zip(labels_w, leaders_w, np.linspace(2.48, .82, len(names_w))):
            label.set_y(y)
            xy = leader.get_ydata().copy()
            xy[-1] = y
            leader.set_ydata(xy)
        # Replace the old full-cohort annotation with frozen discovery information.
        for text in ax.texts:
            if 'signed edges' in text.get_text():
                text.remove()
        ax.set_title(f'{exposure}   joint / conditional rank {int(row.r_grid)} / {int(row.r_signal)}',
                     fontsize=9, loc='left', x=.025, pad=7)
        F._panel_label(ax, 'ace'[i], x=-.065, y=1.04)
        curve = fig.add_subplot(grid[i, 1])
        draw_decomposition(curve, res, exposure)
        F._panel_label(curve, 'bdf'[i], x=-.17, y=1.04)
    fig.suptitle('Ovarian discovery supports and monotone curve contributions', fontsize=10, y=.982)
    neutral_support_arrows(fig)
    fig.text(.5, .954, f'{info["n_discovery"]} discovery patients; graph width 0.4, LASSO 0.20; '
             f'ODE degree {DEFAULT_DEGREE}, ridge {DEFAULT_RIDGE:g}', ha='center', fontsize=8, color=F.INK2)
    handles = [
        Line2D([], [], ls='none', marker='o', color=F.ORANGE, mec=F.INK, ms=5, label='Exposure $A$'),
        Line2D([], [], ls='none', marker='o', color=F.BLUE, mec=F.INK, ms=5, label='Treatment proxy $Z$'),
        Line2D([], [], ls='none', marker='o', color=F.AQUA, mec=F.INK, ms=5, label='Outcome proxy $W$'),
        Line2D([], [], color=F.MUTED, lw=.6, label='Directed LASSO support'),
        Line2D([], [], color=F.INK, lw=1.3, label='Reconstructed curve'),
        Line2D([], [], color=F.INK2, lw=1, ls='--', label='Intrinsic contribution'),
        Line2D([], [], color=F.BLUE, lw=1, label='Source contributions (labelled)'),
        Line2D([], [], color=F.MUTED, marker='o', ls='none', ms=3, label='Observed discovery protein')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=7.5,
               bbox_to_anchor=(.52, .019), columnspacing=1.1, handlelength=1.7)
    F.save(fig, 'fig3_causal')
    references = reference_network_figures()
    assert before == result_hashes(), 'Plotting changed an analysis CSV'
    document = dict(figure='fig3_causal', cohort='ov', source='frozen discovery sample',
                    split_seed=info['split_seed'], discovery_id_sha256=info['discovery_id_sha256'],
                    design_file_sha256=hashlib.sha256((APP/'ov_discovery_designs.json').read_bytes()).hexdigest(),
                    clinical_records_sha256=hashlib.sha256((APP/'ov_all_exposures.csv').read_bytes()).hexdigest(),
                    analysis_csv_files_unchanged=len(before), all_frozen_designs_replayed=True,
                    panels=provenance, reference_panels=references,
                    ode_settings=mapped[EXPOSURES[0]]['decomposition']['settings'],
                    edge_interpretation='Directed LASSO regression supports; monotone state-integrated contributions fitted separately',
                    figures={suffix: hashlib.sha256((ROOT/f'figures/fig3_causal.{suffix}').read_bytes()).hexdigest()
                             for suffix in ('pdf', 'png')},
                    reference_figures={f'{name}.{suffix}': hashlib.sha256(
                        (ROOT/f'figures/{name}.{suffix}').read_bytes()).hexdigest()
                        for name in ('figS1_full_network', 'figS2_network') for suffix in ('pdf', 'png')})
    (APP/'figure3_discovery_provenance.json').write_text(json.dumps(document, indent=2), encoding='utf-8')
    print(json.dumps(document, indent=2))


if __name__ == '__main__':
    main()
