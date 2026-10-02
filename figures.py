"""Publication figures for the idopNetwork + proximal inference paper.

Every function takes the output of run_application.analyse_cohort (or a results
DataFrame) and returns a matplotlib Figure. save(fig, name) writes PDF and PNG.

Style: 8 pt sans, single column 3.3 in, double column 6.9 in, colour-blind-safe
palette validated for adjacent-pair separation; role colours are fixed across
figures (exposure orange, treatment proxy Z blue, outcome proxy W
aqua) so a reader can carry them from the schematic to the network to the forest
plot without a second legend.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')

# ---------------------------------------------------------------- palette and style
INK, INK2, MUTED, GRID, PAPER = '#0b0b0b', '#52514e', '#8a8985', '#e5e4e0', '#ffffff'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
ROLE = dict(exposure=ORANGE, Z=BLUE, W=AQUA, other='#c9c8c3')
SINGLE, DOUBLE = 3.3, 6.9


def set_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2,
        'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
        'figure.facecolor': PAPER, 'axes.facecolor': PAPER,
    })


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return os.path.join(FIG, f'{name}.pdf')


def _panel_label(ax, s, x=-0.18, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left')


# ================================================================ Figure 1: schematic
def fig_schematic():
    """Data representation, network construction and exposure-specific estimation.

    ODE fitting uses both power curves and selected supports. Complete-data and
    inverse-censoring-weighted outcomes enter the same reduced mean bridge.
    All lettering is at least 8 pt at the 6.9 in publication width.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(DOUBLE, 7.0))
    ax.set_xlim(0, 104); ax.set_ylim(-10, 104); ax.axis('off')

    def frame(x, y, w, h, fc=PAPER, ec=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle='round,pad=0.3,rounding_size=0.8',
                     fc=fc, ec=ec, lw=0.7))

    def label(x, y, value, color=INK2, bold=False, ha='center'):
        return ax.text(x, y, value, ha=ha, va='center', fontsize=8,
                       color=color, fontweight='bold' if bold else 'normal')

    def arrow(start, end, color=INK2):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                     mutation_scale=9, color=color, lw=0.8,
                     shrinkA=2, shrinkB=2))

    def routed(points, color=INK2):
        ax.plot(*zip(*points[:-1]), color=color, lw=0.8,
                solid_joinstyle='round', solid_capstyle='round')
        arrow(points[-2], points[-1], color)

    def step(x, title, lines):
        frame(x, 81, 18, 11)
        label(x + 9, 89.5, title, color=INK, bold=True)
        for y, line in zip((86, 83), lines):
            label(x + 9, y, line)

    # A: transformed patient levels, curves and estimated latent representation.
    label(1, 96, 'A  Data representation', color=INK, bold=True, ha='left')
    for x0, lab, color in ((1, 'exposure $A$', ORANGE),
                           (27, 'treatment proxy $Z$', BLUE),
                           (63, 'outcome proxy $W$', AQUA)):
        ax.add_patch(plt.Rectangle((x0, 101), 1.4, 1.4, fc=color, ec='none'))
        label(x0 + 2.3, 101.7, lab, ha='left')
    step(1, 'Protein panel', ['$X$: standardised', 'and shifted levels'])
    step(21, 'Niche ordering', ['$s_k=\\sum_j X_{kj}$', 'sort patient rows'])
    step(41, 'Power curves', ['$c_j(s)=a_j s^{b_j}$'])
    step(61, 'Deviations', ['$U=X-C(s)$', 'centre columns: $U_c$'])
    step(81, 'Latent space', ['$U_c\\approx\\widehat F\\widehat\\Lambda^\\top$', '$\\widehat r_{\\mathrm{PCA}}$; residual $\\rho_i$'])
    for x in (19, 39, 59, 79):
        arrow((x, 86.5), (x + 2, 86.5))

    # B: niche-ordered levels determine supports; supports and curves feed ODE.
    label(1, 76, 'B  Networks', color=INK, bold=True, ha='left')
    frame(21, 58, 20, 13)
    label(31, 68, 'Selected support', color=INK, bold=True)
    label(31, 64, 'nodewise LASSO')
    label(31, 60.5, '$k$ niche windows')
    arrow((30, 81), (30, 71))
    frame(46, 58, 29, 13, fc='#f3f3f1')
    label(60.5, 68, 'Weak-form ODE', color=INK, bold=True)
    label(60.5, 64.2, '$\\dot x_j=Q_{jj}(x_j)$')
    label(60.5, 61, '$+\\sum_{k\\in\\mathrm{pa}(j)} Q_{jk}(x_k)$')
    arrow((50, 81), (60.5, 71))
    arrow((41, 64.5), (46, 64.5))
    frame(81, 58, 18, 13, fc='#f3f3f1')
    label(90, 68, 'ODE network', color=INK, bold=True)
    label(90, 64, 'signed, weighted,')
    label(90, 60.5, 'directed context')
    arrow((75, 64.5), (81, 64.5))

    # C: exposure, symmetrised support and latent loading/rank nominate proxies.
    label(1, 53, 'C  Estimation', color=INK, bold=True, ha='left')
    frame(1, 29, 15, 14, fc='#fdeee8', ec=ORANGE)
    label(8.5, 39, 'Select exposure', color=INK, bold=True)
    label(8.5, 33.5, '$A=X_a$', color=ORANGE)
    frame(20, 29, 29, 14)
    label(34.5, 40, 'Exposure-specific proxies', color=INK, bold=True)
    label(34.5, 35.8, '$Z_A$: eligible neighbours of $A$', color=BLUE)
    label(34.5, 31.5, '$W_A$: separate components', color=AQUA)
    arrow((16, 36), (20, 36), ORANGE)
    arrow((31, 58), (31, 43))
    label(29.5, 49, r'$G_\alpha$: support grid', ha='right')
    routed([(90, 81), (90, 77.5), (101.5, 77.5),
            (101.5, 48), (42, 48), (42, 43)], MUTED)
    label(70, 49.5, '$\\widehat\\Lambda$: loading eligibility', color=MUTED)

    frame(55, 29, 44, 14)
    label(77, 40, 'Grid information and first stage', color=INK, bold=True)
    label(77, 35.8, '$\\widehat r_* = \\max_\\alpha \\widehat r_{\\mathrm{joint},\\alpha}$; select full coverage')
    label(77, 31.5, 'LS first stage $\\widehat W$; $R=\\widehat W P$ has $\\widehat r_*$ directions')
    arrow((49, 36), (55, 36))
    label(77, 45.5, 'Measured covariates $C$')
    arrow((77, 44.2), (77, 43))

    # First-stage output branches to distinct linear and survival outcome fits.
    routed([(77,29),(77,25.5),(43,25.5),(43,22)])
    frame(25, 1, 36, 21)
    label(43, 19, 'Linear outcome bridge', color=INK, bold=True)
    label(43, 14.8, '$E[Y\\mid A,Z_A,C]$')
    label(43, 11, '$=E[h(W_A,A,C)\\mid A,Z_A,C]$')
    label(43, 6.8, '$Y$ or $V_L$ on $(A,R,C)$')
    label(43, 3, 'Total intervention contrast $\\widehat\\tau_{A,L}$', color=ORANGE, bold=True)
    frame(66, 1, 34, 21)
    label(83, 19, 'Censored clinical outcomes', color=INK, bold=True)
    label(83, 14.8, 'follow-up $O$, event indicator $D$')
    label(83, 10.6, 'fit $\\widehat G(t\\mid A,C)$; IPCW $V_L$')
    label(83, 5.3, 'RMST and survival probability', color=ORANGE, bold=True)
    arrow((66,11),(61,11),AQUA)
    frame(1, 5, 19, 16)
    label(10.5, 18, 'Clinical outcomes', color=INK, bold=True)
    label(10.5, 13.5, '$Y$ (linear)')
    label(10.5, 9, '$(O,D)$ (survival)')
    arrow((20, 14), (25, 14))
    routed([(10.5, 5), (10.5, -3), (83, -3), (83, 1)])
    label(46.5, -1.2, 'survival outcome $(O,D)$')
    label(50.5, -8, 'Report: matching proxy ranks, $\\nu\\geq0.05$, censoring positivity; patient bootstrap.')
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    return fig


# ================================================================ Figure 2: power-law fits
def fig_powerfit(res, features=None, ncol=3, nrow=2):
    """Patient values against the niche index with the fitted power curve."""
    set_style()
    X, s, names, C = res['X'], res['s'], res['names'], res['curves']
    if features is None:
        # the most and least steeply changing, plus the middle, for a representative spread
        b = res['params']['b'].reindex(names).values
        order = np.argsort(b)
        pick = np.r_[order[:2], order[len(order) // 2 - 1: len(order) // 2 + 1], order[-2:]]
        features = [names[i] for i in pick]
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, 1.75 * nrow + 0.45), sharex=True)
    for ax, f in zip(axes.ravel(), features):
        j = names.index(f)
        ax.scatter(s, X[:, j], s=4, color=MUTED, alpha=0.45, lw=0)
        ax.plot(s, C[:, j], color=ORANGE, lw=1.4)
        a, b = res['params'].loc[f, ['a', 'b']]
        ax.text(0.03, 0.95, f, transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='top')
        ax.text(0.03, 0.79, f'$b$ = {b:.2f}', transform=ax.transAxes, fontsize=8, va='top', color=INK2)
        ax.grid(axis='y', color=GRID, lw=0.4)
        ax.tick_params(length=2)
    for ax in axes[-1]: ax.set_xlabel('niche index $s$')
    for ax in axes[:, 0]: ax.set_ylabel('level (shifted z-score)')
    fig.subplots_adjust(hspace=0.25, wspace=0.28)
    return fig


# ================================================================ Figure 3: network with roles
def _edge_colour(res, source, target, weight):
    sign = res.get('edge_signs', {}).get((source, target), np.sign(weight))
    return BLUE if sign > 0 else ORANGE if sign < 0 else MUTED


def fig_network(res, exposure, seed=7):
    """Signed weighted dynamic network with the proxy roles for one exposure.

    Left: the connected component of the support graph that contains the
    exposure, with the exposure pinned at the centre. Right: every other
    component, stacked. The outcome-inducing proxies must lie on the right,
    which is (P2) made visible."""
    from proximal import components as _components
    from matplotlib.gridspec import GridSpec
    set_style()
    el = res['edgelist']; names = res['names']; est = res['estimates']; Und = res['Und']
    row = est[est.exposure == exposure]
    Z = row.Z.iloc[0].split(';') if len(row) and isinstance(row.Z.iloc[0], str) and row.Z.iloc[0] else []
    W = row.W.iloc[0].split(';') if len(row) and isinstance(row.W.iloc[0], str) and row.W.iloc[0] else []
    role = {n: 'other' for n in names}
    role.update({z: 'Z' for z in Z}); role.update({w: 'W' for w in W}); role[exposure] = 'exposure'

    comp = _components(Und)
    groups = {}
    for n, c in zip(names, comp): groups.setdefault(c, []).append(n)
    ca = comp[names.index(exposure)]
    others = [g for c, g in groups.items() if c != ca and (len(g) > 1 or any(role[n] != 'other' for n in g))]
    others.sort(key=lambda g: (-sum(role[n] != 'other' for n in g), -len(g)))

    G = nx.DiGraph(); G.add_nodes_from(names)
    for r in el.itertuples(): G.add_edge(r.source, r.target, weight=float(r.weight))
    wmax = max(abs(el.weight).max(), 1e-9)

    def draw(ax, nodes, pos, label_roles=True, node_scale=1.0):
        for u, v, d in G.subgraph(nodes).edges(data=True):
            w = d['weight']; col = _edge_colour(res, u, v, w)
            ax.annotate('', xy=pos[v], xytext=pos[u], zorder=1,
                        arrowprops=dict(arrowstyle='-|>', color=col, alpha=0.55, lw=0.35 + 1.4 * abs(w) / wmax,
                                        shrinkA=4, shrinkB=4, mutation_scale=6, connectionstyle='arc3,rad=0.12'))
        # labels for role nodes, offsets alternating to keep neighbours apart
        rn = sorted([n for n in nodes if role[n] != 'other'], key=lambda n: pos[n][0])
        for n in nodes:
            r_ = role[n]; big = r_ != 'other'
            ax.scatter(*pos[n], s=(100 if big else 26) * node_scale, color=ROLE[r_], ec=INK if big else PAPER,
                       lw=0.6 if big else 0.4, zorder=3)
        for n in (rn if label_roles else []):
            # label on the side facing away from the nearest other role node
            others_ = [m for m in rn if m != n]
            if others_:
                nearest = min(others_, key=lambda m: np.linalg.norm(np.asarray(pos[m]) - np.asarray(pos[n])))
                v = np.asarray(pos[n]) - np.asarray(pos[nearest]); v = v / max(np.linalg.norm(v), 1e-9)
            else:
                v = np.array([0.0, 1.0])
            dx, dy = 9 * v[0], 9 * v[1]
            ax.annotate(n, pos[n], xytext=(dx, dy), textcoords='offset points',
                        ha='left' if dx > 2 else 'right' if dx < -2 else 'center',
                        va='bottom' if dy > 2 else 'top' if dy < -2 else 'center',
                        fontsize=8, fontweight='bold', color=INK, zorder=4,
                        bbox=dict(boxstyle='round,pad=0.12', fc=PAPER, ec='none', alpha=0.75))

    fig = plt.figure(figsize=(DOUBLE, max(4.4,.4*len(others)+1)))
    gs = GridSpec(max(len(others), 1), 2, figure=fig, width_ratios=[2.6, 1], wspace=0.04, hspace=0.12)
    # ---- exposure component, exposure pinned at the centre
    axL = fig.add_subplot(gs[:, 0]); axL.axis('off')
    main = groups[ca]; sub = G.subgraph(main)
    pos = nx.spring_layout(sub, seed=seed, k=1.1 / np.sqrt(len(main)), iterations=500)
    # push role nodes apart when the layout stacks them (tightly connected
    # triads land on one point), then fit the component to the box
    rn = [n for n in main if role[n] != 'other']
    for _ in range(50):
        moved = False
        for a_ in rn:
            for b_ in rn:
                if a_ >= b_: continue
                d = pos[b_] - pos[a_]; dist = np.linalg.norm(d)
                if dist < 0.22:
                    step = (d / max(dist, 1e-9) if dist > 1e-9 else np.array([1.0, 0.3])) * (0.22 - dist) / 2
                    pos[a_] = pos[a_] - step; pos[b_] = pos[b_] + step; moved = True
        if not moved: break
    arr = np.array(list(pos.values())); lo, hi = arr.min(0), arr.max(0)
    pos = {n: (xy - (lo + hi) / 2) / np.maximum(hi - lo, 1e-9) * 1.85 for n, xy in pos.items()}
    draw(axL, main, pos)
    axL.set_title(f'component containing the exposure ({len(main)} nodes)', fontsize=8, loc='left', color=INK2)
    axL.add_patch(plt.Rectangle((-1.08, -1.08), 2.16, 2.16, fc='#fdf3ee', ec=ORANGE, lw=0.5, zorder=0,
                                transform=axL.transData))
    axL.set_xlim(-1.1, 1.1); axL.set_ylim(-1.1, 1.1)
    # ---- the other components, stacked on the right
    for i_, g in enumerate(others):
        ax = fig.add_subplot(gs[i_, 1]); ax.axis('off')
        sg = G.subgraph(g)
        p = nx.spring_layout(sg, seed=seed, k=0.9, iterations=200) if len(g) > 1 else {g[0]: np.zeros(2)}
        arr = np.array(list(p.values())); span = max((arr.max(0) - arr.min(0)).max(), 1e-9)
        p = {n: (xy - arr.mean(0)) / span * 1.5 for n, xy in p.items()}
        if len(g)==1:
            p={g[0]:np.array([-.6,0.])}
        elif len(g)<=4:
            p={node:np.array([x,.3]) for node,x in zip(g,np.linspace(-.72,.72,len(g)))}
        draw(ax, g, p, label_roles=False, node_scale=0.85)
        for node in g:
            ax.annotate(node,p[node],xytext=(12,0) if len(g)==1 else (0,-5),textcoords='offset points',
                        ha='left' if len(g)==1 else 'center',va='center' if len(g)==1 else 'top',fontsize=8,fontweight='bold')
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.0, 1.0)
        ax.add_patch(plt.Rectangle((-1.08, -0.98), 2.16, 1.96, fc='#faf9f6', ec='#ebe9e3', lw=0.5, zorder=0))
        if i_ == 0: ax.set_title('other components', fontsize=8, loc='left', color=INK2)
    h = [plt.Line2D([], [], marker='o', ls='', color=ROLE['exposure'], mec=INK, mew=0.6, ms=6.5, label=f'exposure $A$ = {exposure}'),
         plt.Line2D([], [], marker='o', ls='', color=ROLE['Z'], mec=INK, mew=0.6, ms=6.5, label='treatment proxy $Z$'),
         plt.Line2D([], [], marker='o', ls='', color=ROLE['W'], mec=INK, mew=0.6, ms=6.5, label='outcome proxy $W$'),
         plt.Line2D([], [], color=BLUE, lw=1.3, label='positive effect'),
         plt.Line2D([], [], color=ORANGE, lw=1.3, label='negative effect')]
    fig.legend(handles=h, loc='lower center', ncol=5, fontsize=8, handlelength=1.5, columnspacing=1.0,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.text(0.01, 0.995, f'{len(el)} signed edges from the weak-form ODE on the support graph; '
             f'{len(set(comp))} components in total', ha='left', va='top', fontsize=8, color=MUTED)
    return fig


def _spread(pos, min_dist, box, iters=300):
    """Fit a layout to [-box, box] and push nodes apart until no pair is closer than min_dist."""
    names = list(pos); P = np.array([pos[n] for n in names], float)
    def fit(P):
        lo, hi = P.min(0), P.max(0); return (P - (lo + hi) / 2) / np.maximum(hi - lo, 1e-9) * box
    P = fit(P)
    for _ in range(iters):
        d = P[:, None, :] - P[None, :, :]; dist = np.linalg.norm(d, axis=-1); np.fill_diagonal(dist, 1e9)
        close = dist < min_dist
        if not close.any(): break
        push = np.where(close[..., None], d / np.maximum(dist, 1e-9)[..., None] * (min_dist - dist)[..., None] / 2, 0).sum(1)
        P = np.clip(P + push, -box, box)
    return dict(zip(names, fit(P)))


# ================================================================ Figure 3b: the whole network
def fig_full_network(res, seed=7, top_hubs=10):
    """Every protein, every signed ODE edge, no proxy roles: the idopNetwork itself.

    Main component laid out by a spring embedding; small components in a
    strip below it; proteins with no support edge in a row at the bottom.
    Node size ~ total strength (sum of |weights| in and out); the top hubs
    are labelled in bold."""
    from proximal import components as _components
    set_style()
    el = res['edgelist']; names = res['names']; Und = res['Und']; p = len(names)
    G = nx.DiGraph(); G.add_nodes_from(names)
    for r in el.itertuples(): G.add_edge(r.source, r.target, weight=float(r.weight))
    wmax = max(abs(el.weight).max(), 1e-9)
    strength = {n: sum(abs(d['weight']) for _, _, d in G.in_edges(n, data=True)) +
                   sum(abs(d['weight']) for _, _, d in G.out_edges(n, data=True)) for n in names}
    smax = max(max(strength.values()), 1e-9)
    hubs = set(sorted(names, key=lambda n: -strength[n])[:top_hubs])

    comp = _components(Und); groups = {}
    for n, c in zip(names, comp): groups.setdefault(c, []).append(n)
    groups = sorted(groups.values(), key=len, reverse=True)
    main, small, iso = groups[0], [g for g in groups[1:] if len(g) > 1], [g[0] for g in groups if len(g) == 1]

    big = p > 80
    lfs, nfs = (6.5, 8.0) if big else (8.0, 8.5)          # label size: ordinary / hub
    ns = (18, 150) if big else (22, 180)                  # node size range
    fig = plt.figure(figsize=(7.1, 9.4) if big else (DOUBLE, 6.6))
    hs = [6.0, 0.9 if small else 0.001, 0.45 if iso else 0.001]
    gs = fig.add_gridspec(3, 1, height_ratios=hs, hspace=0.12, top=0.94, bottom=0.05, left=0.02, right=0.98)

    def draw(ax, nodes, pos, scale=1.0):
        for u, v, d in G.subgraph(nodes).edges(data=True):
            w = d['weight']; col = _edge_colour(res, u, v, w)
            ax.annotate('', xy=pos[v], xytext=pos[u], zorder=1,
                        arrowprops=dict(arrowstyle='-|>', color=col, alpha=0.5, lw=0.3 + 1.2 * abs(w) / wmax,
                                        shrinkA=3, shrinkB=3, mutation_scale=5, connectionstyle='arc3,rad=0.12'))
        for n in nodes:
            sz = (ns[0] + (ns[1] - ns[0]) * np.sqrt(strength[n] / smax)) * scale
            ax.scatter(*pos[n], s=sz, color=YELLOW if n in hubs else '#e3d9b8', ec=INK if n in hubs else INK2,
                       lw=0.6 if n in hubs else 0.35, zorder=3)
        for n in nodes:
            ax.annotate(n, pos[n], xytext=(0, 3.2 + 2.5 * np.sqrt(strength[n] / smax) * scale), textcoords='offset points',
                        ha='center', va='bottom', fontsize=nfs if n in hubs else lfs,
                        fontweight='bold' if n in hubs else 'normal', color=INK if n in hubs else INK2, zorder=4,
                        bbox=dict(boxstyle='round,pad=0.08', fc=PAPER, ec='none', alpha=0.7))

    # ---- main component
    ax = fig.add_subplot(gs[0]); ax.axis('off')
    sub = G.subgraph(main)
    pos = nx.kamada_kawai_layout(sub.to_undirected(), weight=None)
    pos = _spread(pos, min_dist=0.13 if big else 0.17, box=np.array([1.9, 1.85]))
    draw(ax, main, pos)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_title(f'main component: {len(main)} of {p} proteins, {sub.number_of_edges()} signed edges',
                 fontsize=8, loc='left', color=INK2)

    # ---- small components in a strip
    if small:
        ax = fig.add_subplot(gs[1]); ax.axis('off')
        nb = len(small); pos = {}
        for i, g in enumerate(small):
            sg = G.subgraph(g)
            pp = ({g[0]: np.array([-1.0, 0.0]), g[1]: np.array([1.0, 0.0])} if len(g) == 2
                  else _spread(nx.spring_layout(sg, seed=seed, k=0.9, iterations=200), min_dist=0.9, box=np.array([1.0, 0.6])))
            a_ = np.array(list(pp.values())); span = max((a_.max(0) - a_.min(0)).max(), 1e-9)
            cx = -1 + (2 * i + 1) / nb
            for n, xy in pp.items(): pos[n] = (xy - a_.mean(0)) / span * np.array([1.2 / nb, 0.5]) + np.array([cx, 0])
        draw(ax, [n for g in small for n in g], pos, scale=0.85)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.75, 0.75)
        ax.set_title(f'{nb} small components ({sum(len(g) for g in small)} proteins)', fontsize=8, loc='left', color=INK2)

    # ---- isolated proteins
    if iso:
        ax = fig.add_subplot(gs[2]); ax.axis('off')
        per_row = min(len(iso), 20); rows = int(np.ceil(len(iso) / per_row))
        for i, n in enumerate(iso):
            x = -1 + (2 * (i % per_row) + 1) / per_row; y = 0.5 - (i // per_row) / max(rows - 1, 1) * 1.0 if rows > 1 else 0.0
            ax.scatter(x, y, s=ns[0] * 0.85, color='#e3d9b8', ec=INK2, lw=0.35, zorder=3)
            ax.annotate(n, (x, y), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                        fontsize=lfs, color=INK2, zorder=4)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.6, 0.9)
        ax.set_title(f'{len(iso)} proteins with no stable edge', fontsize=8, loc='left', color=INK2)

    h = [plt.Line2D([], [], marker='o', ls='', color=YELLOW, mec=INK, mew=0.6, ms=7, label=f'top {top_hubs} hubs by strength'),
         plt.Line2D([], [], marker='o', ls='', color='#e3d9b8', mec=INK2, mew=0.4, ms=4.5, label='protein (size ~ strength)'),
         plt.Line2D([], [], color=BLUE, lw=1.3, label='positive effect'),
         plt.Line2D([], [], color=ORANGE, lw=1.3, label='negative effect')]
    fig.legend(handles=h, loc='lower center', ncol=4, fontsize=8, handlelength=1.5, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.text(0.01, 0.995, f'idopNetwork on {p} proteins: {len(el)} signed edges from the weak-form ODE on the '
             f'support graph (alpha={res["alpha"]}, k={res["k"]}, threshold={res["threshold"]}); {len(groups)} components',
             ha='left', va='top', fontsize=8, color=MUTED)
    return fig


# ================================================================ Figure 3c: three-layer causal view
def fig_layered(res, exposure, label_others=True):
    """The network as the causal branch sees it, in three bands.

    Top: the exposure. Middle: the proxies, Z (treatment-inducing) on the left,
    W (outcome-inducing) on the right. Bottom: every other protein, ordered by
    the barycentre of its neighbours and staggered over two rows. All signed
    ODE edges are drawn; by construction no edge joins W to the exposure or to
    any Z, which is (P2) made visible."""
    set_style()
    el = res['edgelist']; names = res['names']; est = res['estimates']
    row = est[est.exposure == exposure].iloc[0]
    Z = row.Z.split(';') if isinstance(row.Z, str) and row.Z else []
    W = row.W.split(';') if isinstance(row.W, str) and row.W else []
    others = [n for n in names if n != exposure and n not in Z and n not in W]
    G = nx.DiGraph(); G.add_nodes_from(names)
    for r in el.itertuples(): G.add_edge(r.source, r.target, weight=float(r.weight))
    wmax = max(abs(el.weight).max(), 1e-9)
    und = G.to_undirected()

    # ---- positions: bands at fixed heights, x in [0, 1]
    yA, yP, yO = 3.0, 2.0, 0.55
    pos = {exposure: (0.5, yA)}
    def spread(nodes, lo, hi, y):
        k = len(nodes)
        for i, n in enumerate(nodes): pos[n] = (lo + (hi - lo) * (i + 0.5) / k if k > 1 else (lo + hi) / 2, y)
    spread(Z, 0.04, 0.44, yP); spread(W, 0.56, 0.96, yP)
    # others: barycentre of neighbours already placed, isolated nodes last
    def bary(n):
        nb = [pos[m][0] for m in und.neighbors(n) if m in pos]
        return (np.mean(nb), 0) if nb else (1.5, 1)
    fixed = list(pos)
    order = sorted(others, key=lambda n: bary(n))
    # one refinement pass using the provisional bottom-row positions too
    for i, n in enumerate(order): pos[n] = ((i + 0.5) / len(order), yO)
    order = sorted(others, key=lambda n: np.mean([pos[m][0] for m in und.neighbors(n)]) if und.degree(n) else 1.5)
    for i, n in enumerate(order): pos[n] = ((i + 0.5) / len(order), yO)

    fig, ax = plt.subplots(figsize=(DOUBLE, 4.9)); ax.axis('off')
    ax.set_xlim(-0.14, 1.02); ax.set_ylim(-0.62, 3.5)
    # band backgrounds and labels
    for y0, h, lab, fc in ((yA - 0.3, 0.6, 'exposure $A$', '#fdf3ee'), (yP - 0.3, 0.75, 'proxies', '#f4f6f4'), (yO - 0.2, 0.4, 'other proteins', '#f7f7f5')):
        ax.add_patch(plt.Rectangle((-0.01, y0), 1.03, h, fc=fc, ec='none', zorder=0))
        ax.text(-0.02, y0 + h / 2, lab, ha='right', va='center', fontsize=8, color=INK2)
    ax.text(0.04, yP + 0.5, 'treatment proxy $Z$\nadjacent to $A$', ha='left', va='bottom', fontsize=8, color=BLUE, linespacing=1.1)
    ax.text(0.96, yP + 0.5, 'outcome proxy $W$\nno selected path to $A$ or $Z$', ha='right', va='bottom', fontsize=8, color=AQUA, linespacing=1.1)

    # ---- edges
    for u, v, d in G.edges(data=True):
        w = d['weight']; col = BLUE if w > 0 else ORANGE
        same_band = abs(pos[u][1] - pos[v][1]) < 0.5
        role_edge = (u in Z or u == exposure or v in Z or v == exposure)
        rad = (-0.35 if pos[v][0] > pos[u][0] else 0.35) if same_band else 0.12   # same-band arcs always bow upward
        ax.annotate('', xy=pos[v], xytext=pos[u], zorder=2 if role_edge else 1,
                    arrowprops=dict(arrowstyle='-|>', color=col, alpha=0.75 if role_edge else 0.35,
                                    lw=0.4 + 1.3 * abs(w) / wmax, shrinkA=4, shrinkB=4, mutation_scale=6,
                                    connectionstyle=f'arc3,rad={rad}'))
    # ---- nodes
    for n in names:
        role = 'exposure' if n == exposure else 'Z' if n in Z else 'W' if n in W else 'other'
        big = role != 'other'
        ax.scatter(*pos[n], s=150 if big else 28, color=ROLE[role], ec=INK if big else INK2, lw=0.7 if big else 0.4, zorder=4)
    ax.annotate(exposure, pos[exposure], xytext=(10, 0), textcoords='offset points', ha='left', va='center', fontsize=9, fontweight='bold', zorder=5)
    for grp in (Z, W):
        for i, n in enumerate(grp):
            ax.annotate(n, pos[n], xytext=(0, 8 if i % 2 == 0 else 19), textcoords='offset points', ha='center', va='bottom', fontsize=8, fontweight='bold', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.1', fc=PAPER, ec='none', alpha=0.8))
    if label_others:
        for i, n in enumerate(order):
            ax.annotate(n, pos[n], xytext=(0, -6 if i % 2 == 0 else -34), textcoords='offset points', ha='center', va='top', rotation=90, fontsize=8, color=INK2, zorder=5)
    h = [plt.Line2D([], [], marker='o', ls='', color=ROLE['exposure'], mec=INK, mew=0.6, ms=7, label=f'exposure $A$ = {exposure}'),
         plt.Line2D([], [], marker='o', ls='', color=ROLE['Z'], mec=INK, mew=0.6, ms=7, label=f'$Z$ ({len(Z)})'),
         plt.Line2D([], [], marker='o', ls='', color=ROLE['W'], mec=INK, mew=0.6, ms=7, label=f'$W$ ({len(W)})'),
         plt.Line2D([], [], color=BLUE, lw=1.3, label='positive effect'), plt.Line2D([], [], color=ORANGE, lw=1.3, label='negative effect')]
    fig.legend(handles=h, loc='lower center', ncol=5, fontsize=8, handlelength=1.4, columnspacing=1.2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    ax.text(1.0, yA + 0.42, f'$\\nu$ = {row.nu:.2f};  {len(el)} signed edges', ha='right', va='bottom', fontsize=8, color=MUTED)
    return fig


def fig_dpi_scale(ax):
    """Pixels per point for the axes' figure (labels are placed in points)."""
    return ax.figure.dpi / 72.0


# ================================================================ Figure 3d: three stacked planes
def _draw_planes(ax, res, exposure, seed=7, label_others='connected', w_in=DOUBLE, h_in=6.4, compact=False):
    """Three stacked elliptical planes, read as a tilted multilayer network.

    Top plane: the exposure. Middle plane: the proxies, Z on the left half
    and W on the right half. Bottom plane: every other protein in a spring
    layout confined to the ellipse. All signed ODE edges are drawn; edges
    within a plane stay inside its ellipse, edges between planes cross the
    gap. label_others: 'connected' labels the other proteins that touch the
    exposure or a proxy; 'all' labels every node."""
    from matplotlib.patches import Ellipse
    set_style()
    el = res['edgelist']; names = res['names']; est = res['estimates']
    row = est[est.exposure == exposure].iloc[0]
    Z = row.Z.split(';') if isinstance(row.Z, str) and row.Z else []
    W = row.W.split(';') if isinstance(row.W, str) and row.W else []
    others = [n for n in names if n != exposure and n not in Z and n not in W]
    G = nx.DiGraph(); G.add_nodes_from(names)
    for r in el.itertuples(): G.add_edge(r.source, r.target, weight=float(r.weight))
    wmax = max(abs(el.weight).max(), 1e-9); und = G.to_undirected()
    strength = {n: sum(abs(d['weight']) for _, _, d in G.in_edges(n, data=True)) + sum(abs(d['weight']) for _, _, d in G.out_edges(n, data=True)) for n in names}

    # ---- planes: centre (cx, cy), semi-axes (a, b); flat ellipses read as tilted discs
    A_, B_ = 0.50, 0.27
    planes = {'A': (0.5, 2.85), 'P': (0.5, 1.65), 'O': (0.5, 0.35)}
    def to_plane(key, u, v):                        # (u, v) in the unit disc -> plane coordinates
        cx, cy = planes[key]; return (cx + A_ * u, cy + B_ * v)
    pos = {exposure: to_plane('A', 0, 0)}
    rng = np.random.default_rng(seed)
    SX, SY = 6.5 / 2 * w_in / DOUBLE, 1.7 / 2 * h_in / 6.4   # plane semi-axes in display inches (for spacing)
    def half_scatter(nodes, side):                  # proxies scattered inside the left (-1) or right (+1) half-disc
        k = len(nodes)
        if k == 0: return
        # seed on a loose arc, then jitter and push apart; keep inside the half-ellipse
        t = np.linspace(0.25, 0.75, k) * np.pi if k > 1 else np.array([0.5 * np.pi])
        U = np.column_stack([side * (0.22 + 0.62 * np.abs(np.cos(t)) + rng.uniform(-0.12, 0.12, k)), 0.72 * np.sin(t) * rng.choice([-1, 1], k) + rng.uniform(-0.25, 0.25, k)])
        if k > 1: U[:, 1] = U[:, 1] - U[:, 1].mean() + rng.uniform(-0.1, 0.1)
        Q = U * np.array([SX, SY])
        for _ in range(300):
            d = Q[:, None, :] - Q[None, :, :]; dist = np.linalg.norm(d, axis=-1); np.fill_diagonal(dist, 1e9)
            close = dist < 0.62
            Q = Q + np.where(close[..., None], d / np.maximum(dist, 1e-9)[..., None] * (0.62 - dist)[..., None] / 2, 0).sum(1)
            uv = Q / np.array([SX, SY])
            uv[:, 0] = side * np.clip(side * uv[:, 0], 0.14, 0.92)                 # stay in the correct half
            uv[:, 1] = np.clip(uv[:, 1], -0.62, 0.62)
            rr = np.linalg.norm(uv, axis=1); uv[rr > 0.88] *= (0.88 / rr[rr > 0.88])[:, None]
            Q = uv * np.array([SX, SY])
            if not close.any(): break
        for n, (u, v) in zip(nodes, Q / np.array([SX, SY])): pos[n] = to_plane('P', u, v)
    prow = {}
    def half_grid(nodes, side):                     # compact: two rows per half, labels above / below by row
        k = len(nodes); nrow=4 if k>16 else 2;ncol=int(np.ceil(k/nrow)) if k>1 else 1
        for i, n in enumerate(nodes):
            r_, c_ = (i % nrow, i // nrow) if k > 1 else (0, 0)
            u = side * (0.18 + 0.7 * (c_ + 0.5) / ncol)
            v=np.linspace(.65,-.65,nrow)[r_] if k>16 else (.45 if r_==0 else -.45) if k>1 else 0.
            pos[n] = to_plane('P', u, v); prow[n] = r_
    if compact: half_grid(Z, -1); half_grid(W, +1)
    else: half_scatter(Z, -1); half_scatter(W, +1)
    # others: spring layout of their induced subgraph, fitted to the disc, then spread
    sub = und.subgraph(others)
    p = nx.spring_layout(sub, seed=seed, k=1.5 / np.sqrt(max(len(others), 1)), iterations=400)
    arr = np.array([p[n] for n in others]); arr = arr - arr.mean(0)
    # keep the angular arrangement, equalise the radii so the disc is filled evenly
    rad = np.linalg.norm(arr, axis=1); ang_ = np.arctan2(arr[:, 1], arr[:, 0])
    rank = np.argsort(np.argsort(rad)); r_eq = 0.95 * np.sqrt((rank + 0.5) / len(others))
    arr = np.column_stack([r_eq * np.cos(ang_), r_eq * np.sin(ang_)])
    # push apart in display inches (the plane is ~6.5 in wide and ~1 in tall), then clip to the ellipse
    sx, sy = SX, SY                                # semi-axes in inches
    Q = arr * np.array([sx, sy])
    for _ in range(400):
        d = Q[:, None, :] - Q[None, :, :]; dist = np.linalg.norm(d, axis=-1); np.fill_diagonal(dist, 1e9)
        close = dist < 0.2
        if not close.any(): break
        Q = Q + np.where(close[..., None], d / np.maximum(dist, 1e-9)[..., None] * (0.2 - dist)[..., None] / 2, 0).sum(1)
        uv = Q / np.array([sx, sy]); rr = np.linalg.norm(uv, axis=1); uv[rr > 0.95] *= (0.95 / rr[rr > 0.95])[:, None]; Q = uv * np.array([sx, sy])
    arr = Q / np.array([sx, sy])
    for n, (u, v) in zip(others, arr): pos[n] = to_plane('O', u, v)

    ax.axis('off'); ax.set_aspect('auto')
    ax.set_xlim(-0.34 if compact else -0.16, 1.34 if compact else 1.05); ax.set_ylim(0.35 - B_ - 0.12, 2.85 + B_ + 0.3)
    plane_style = {'A': ('#fdf3ee', ORANGE, 'exposure $A$'), 'P': ('#f2f5f3', INK2, 'proxies'), 'O': ('#f6f6f4', '#c9c8c3', 'other proteins')}
    for key, (cx, cy) in planes.items():
        fc, ec, lab = plane_style[key]
        ax.add_patch(Ellipse((cx, cy - 0.04), 2 * A_ + 0.03, 2 * B_ + 0.03, fc='#dedcd6', ec='none', zorder=0))   # thickness
        ax.add_patch(Ellipse((cx, cy), 2 * A_ + 0.03, 2 * B_ + 0.03, fc=fc, ec=ec, lw=0.7, zorder=0.5))
        if not compact: ax.text(cx - A_ - 0.04, cy, lab, ha='right', va='center', fontsize=8, color=INK2)
    if compact:
        pass                                        # the roles are named in the shared legend
    else:
        ax.text(0.5 - A_ * 0.55, 1.65 + B_ + 0.12, 'treatment proxy $Z$' + chr(10) + 'adjacent to $A$', ha='center', va='bottom', fontsize=8, color=BLUE, linespacing=1.1)
        ax.text(0.5 + A_ * 0.55, 1.65 + B_ + 0.12, 'outcome proxy $W$' + chr(10) + 'no selected path to $A$ or $Z$', ha='center', va='bottom', fontsize=8, color=AQUA, linespacing=1.1)
    ax.plot([0.5, 0.5], [1.65 - B_, 1.65 + B_], color=INK2, lw=0.5, ls=(0, (2, 2)), zorder=0.6)

    plane_of = {exposure: 'A', **{n: 'P' for n in Z + W}, **{n: 'O' for n in others}}
    for u, v, d in G.edges(data=True):
        w = d['weight']; col = _edge_colour(res, u, v, w)
        same = plane_of[u] == plane_of[v]
        role_edge = (u in Z or u == exposure or v in Z or v == exposure)
        ax.annotate('', xy=pos[v], xytext=pos[u], zorder=2 if role_edge else 1,
                    arrowprops=dict(arrowstyle='-|>', color=col, alpha=0.8 if role_edge else 0.4,
                                    lw=0.4 + 1.3 * abs(w) / wmax, shrinkA=3.5, shrinkB=3.5, mutation_scale=6,
                                    connectionstyle=f'arc3,rad={0.12 if same else 0.08}'))
    for n in names:
        role = 'exposure' if n == exposure else 'Z' if n in Z else 'W' if n in W else 'other'
        big = role != 'other'
        density=min(1.,12/len(W)) if compact and role=='W' else 1.
        ax.scatter(*pos[n], s=(150 if big else 30) * (0.7 if compact else 1.0)*density, color=ROLE[role], ec=INK if big else INK2, lw=0.7 if big else 0.4, zorder=4)
    ax.annotate(exposure, pos[exposure], xytext=(10, 0), textcoords='offset points', ha='left', va='center', fontsize=9, fontweight='bold', zorder=5)
    for n in Z + W:                                 # label on the side facing away from the nearest proxy
        if compact: continue                       # compact: labels listed in the margins below
        grp = Z if n in Z else W
        others_ = [m for m in grp if m != n]
        q = np.array([pos[n][0] / A_, pos[n][1] / B_])
        if others_:
            nearest = min(others_, key=lambda m: np.linalg.norm(np.array([pos[m][0] / A_, pos[m][1] / B_]) - q))
            vv = q - np.array([pos[nearest][0] / A_, pos[nearest][1] / B_]); vv = vv / max(np.linalg.norm(vv), 1e-9)
        else:
            vv = np.array([0.0, 1.0])
        dx, dy = 10 * vv[0], 9 * vv[1]
        ax.annotate(n, pos[n], xytext=(dx, dy), textcoords='offset points', ha='left' if dx > 2 else 'right' if dx < -2 else 'center',
                    va='bottom' if dy > 2 else 'top' if dy < -2 else 'center',
                    fontsize=8, fontweight='bold', zorder=5, bbox=dict(boxstyle='round,pad=0.1', fc=PAPER, ec='none', alpha=0.85))
    if compact:                                     # Z names down the left margin, W names down the right, with leaders
        cy = planes['P'][1]
        for grp, side in ((Z, -1), (W, +1)):
            if not grp: continue
            order_ = sorted(grp, key=lambda n: -pos[n][1] * 10 - side * pos[n][0])
            span=max(.84,(len(order_)-1)*.105)
            ys = np.linspace(cy+span/2,cy-span/2,len(order_)) if len(order_)>1 else [cy]
            xl = 0.5 + side * (A_ + 0.06)
            for n, y in zip(order_, ys):
                ax.plot([pos[n][0], xl - side * 0.012], [pos[n][1], y], color=INK2, lw=0.4, alpha=0.7, zorder=3.5)
                ax.text(xl, y, n, ha='left' if side > 0 else 'right', va='center', fontsize=8, fontweight='bold', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.1', fc=PAPER, ec='none', alpha=0.85))
    if label_others:
        touch = {n for n in others if any(m == exposure or m in Z or m in W for m in und.neighbors(n))}
        hubs = set(sorted(others, key=lambda n: -strength[n])[:10])
        lab = set(others) if label_others == 'all' else touch
        order_x = sorted(lab, key=lambda n: pos[n][0])
        placed = []                                 # label boxes in display points, for greedy collision avoidance
        fs = 7 if label_others == 'all' else 8
        for i_, n in enumerate(order_x):
            px, py = ax.transData.transform(pos[n]) / fig_dpi_scale(ax)
            w_, h_ = 0.55 * fs * len(n), 1.1 * fs
            cands = [(0, 5, 'center', 'bottom'), (0, -5, 'center', 'top'), (0, 16, 'center', 'bottom'), (0, -16, 'center', 'top'),
                     (7, 0, 'left', 'center'), (-7, 0, 'right', 'center'), (7, 10, 'left', 'bottom'), (-7, -10, 'right', 'top')] if compact else                     ([(0, 5, 'center', 'bottom')] if pos[n][1] >= planes['O'][1] else [(0, -5, 'center', 'top')])
            for dx, dy, ha, va in cands:
                x0 = px + dx - (w_ if ha == 'right' else w_ / 2 if ha == 'center' else 0); y0 = py + dy - (h_ if va == 'top' else h_ / 2 if va == 'center' else 0)
                box = (x0, y0, x0 + w_, y0 + h_)
                if not any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in placed): break
            placed.append(box)
            ax.annotate(n, pos[n], xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=fs, color=INK2, zorder=5,
                        bbox=dict(boxstyle='round,pad=0.08', fc=PAPER, ec='none', alpha=0.7))
    h = [plt.Line2D([], [], marker='o', ls='', color=ROLE['exposure'], mec=INK, mew=0.6, ms=7, label=f'exposure $A$ = {exposure}'),
         plt.Line2D([], [], marker='o', ls='', color=ROLE['Z'], mec=INK, mew=0.6, ms=7, label=f'$Z$ ({len(Z)})'),
         plt.Line2D([], [], marker='o', ls='', color=ROLE['W'], mec=INK, mew=0.6, ms=7, label=f'$W$ ({len(W)})'),
         plt.Line2D([], [], color=BLUE, lw=1.3, label='positive effect'), plt.Line2D([], [], color=ORANGE, lw=1.3, label='negative effect')]
    information=f'$\\nu$ = {row.nu:.2f}' if np.isfinite(row.nu) else f'ranks {int(row.r_joint)}/{int(row.r_signal)}'
    ax.text(1.04 if not compact else 1.3, 2.85 + B_ + 0.2, (f'$A$ = {exposure}:  ' if compact else '') + information + f';  alpha {res["alpha"]:.2f}; {len(el)} signed edges', ha='right', va='bottom', fontsize=8, color=MUTED)
    return h


def fig_layered3d(res, exposure, seed=7, label_others='connected'):
    """Three stacked planes for one exposure (standalone figure)."""
    set_style()
    fig, ax = plt.subplots(figsize=(DOUBLE, 6.4))
    h = _draw_planes(ax, res, exposure, seed, label_others)
    fig.legend(handles=h, loc='lower center', ncol=5, fontsize=8, handlelength=1.4, columnspacing=1.2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    return fig


def fig_causal(res, exposures, targets, tags=None, seed=7, exposure_resolutions=None):
    """(a, b) the network as three stacked planes for two exposures; (c) effect decompositions
    for the exposures and their treatment-inducing proxies, two rows of four."""
    from idop_core import _solve_ode_decomposition
    from matplotlib.gridspec import GridSpec
    set_style(); tags = tags or {}
    fig = plt.figure(figsize=(DOUBLE, 9.0))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[2.5, 2.5, 1.5, 1.5], hspace=0.6, wspace=0.65)
    # ---- planes, side by side
    hs = None
    for k, e in enumerate(exposures[:2]):
        ax = fig.add_subplot(gs[0:2, 2 * k:2 * k + 2])
        selected=(exposure_resolutions or {}).get(e,res)
        hs = _draw_planes(ax, selected, e, seed, False, w_in=DOUBLE / 2, h_in=4.4, compact=True)
        _panel_label(ax, 'ab'[k], x=-0.02, y=0.99)
    legend_h = [plt.Line2D([], [], marker='o', ls='', color=ROLE['exposure'], mec=INK, mew=0.6, ms=7, label='exposure $A$'),
                plt.Line2D([], [], marker='o', ls='', color=ROLE['Z'], mec=INK, mew=0.6, ms=7, label='treatment proxy $Z$'),
                plt.Line2D([], [], marker='o', ls='', color=ROLE['W'], mec=INK, mew=0.6, ms=7, label='outcome proxy $W$'),
                plt.Line2D([], [], color=BLUE, lw=1.3, label='positive effect'), plt.Line2D([], [], color=ORANGE, lw=1.3, label='negative effect')]
    # ---- decompositions
    def decompose(selected):
        return _solve_ode_decomposition(selected['qd'],selected['samples'],selected['supports'],basis_order=selected.get('basis_order',0),ridge=selected.get('ode_ridge',1e-6))
    default_dec=decompose(res)
    decs={e:decompose(selected) for e,selected in (exposure_resolutions or {}).items()}
    palette = [BLUE, AQUA, YELLOW, '#e87ba4', '#4a3aa7', '#008300', '#d95926']
    first = None
    for k, t in enumerate(targets[:8]):
        e=exposures[min(k//4,len(exposures)-1)];dec=decs.get(e,default_dec)
        tau=np.asarray(dec['sample_tau']);feats=list(dec['features'])
        r_, c_ = divmod(k, 4); ax = fig.add_subplot(gs[2 + r_, c_]); first = first or ax
        j_ = feats.index(t)
        obs = np.asarray(dec['response'])[:, j_]; pred = np.asarray(dec['predicted_states'])[:, j_]
        ax.scatter(tau, obs, s=3.5, color=MUTED, alpha=0.5, lw=0); ax.plot(tau, pred, color=INK, lw=1.4)
        self_eff = dec['intercepts'][j_] + dec['interaction_functions'].get((t, t), np.zeros_like(tau))
        ax.plot(tau, self_eff, color=INK2, lw=1.0, ls=(0, (4, 2)))
        srcs = [src for src in dec['support_sets'].get(t, []) if src != t]; ends = []
        for i_, src in enumerate(srcs):
            eff = dec['interaction_functions'][(t, src)]
            ax.plot(tau, eff, color=palette[i_ % len(palette)], lw=0.9); ends.append((eff[-1], src, palette[i_ % len(palette)]))
        ends.sort(key=lambda e: e[0]); yl = [e[0] for e in ends]; span = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
        for k_ in range(1, len(yl)):
            if yl[k_] - yl[k_ - 1] < 0.1 * span: yl[k_] = yl[k_ - 1] + 0.1 * span
        xr = tau[-1] + 0.03 * (tau[-1] - tau[0])
        for (y0, src, col), yy in zip(ends, yl):
            ax.text(xr, yy, src, va='center', ha='left', fontsize=8, color=col, clip_on=False)
            if abs(yy - y0) > 1e-9: ax.plot([tau[-1], xr - 0.005 * (tau[-1] - tau[0])], [y0, yy], color=col, lw=0.4, clip_on=False)
        ax.axhline(0, color=GRID, lw=0.6)
        ax.set_title(t + (f'  ({tags[t]})' if t in tags else ''), fontsize=8, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.4); ax.set_xlim(tau[0], tau[-1] + 0.2 * (tau[-1] - tau[0])); ax.tick_params(length=2)
        if r_ == 1: ax.set_xlabel('niche index $s$')
        if c_ == 0: ax.set_ylabel('cumulative effect')
    _panel_label(first, 'c', x=-0.5, y=1.12)
    legend_h += [plt.Line2D([], [], marker='o', ls='', color=MUTED, ms=3.5, label='observed'),
                 plt.Line2D([], [], color=INK, lw=1.4, label='reconstructed'),
                 plt.Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 2)), label='self effect (with baseline)'),
                 plt.Line2D([], [], color=BLUE, lw=0.9, label='cross effects, labelled by source')]
    fig.legend(handles=legend_h, loc='lower center', ncol=5, fontsize=8, handlelength=1.4, columnspacing=1.2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(bottom=0.1, top=0.98, left=0.07, right=0.98)
    return fig


# ================================================================ Figure 2: the descriptive network in one figure
def fig_descriptive(res, features, targets, tags=None):
    """(a) power-law fits, three per row; (b) effect decompositions, two per row.
    tags: optional {protein: role text} printed under the protein name in (a)."""
    from idop_core import _solve_ode_decomposition
    from matplotlib.gridspec import GridSpec
    set_style()
    X, s, names, C = res['X'], res['s'], res['names'], res['curves']; tags = tags or {}
    bc = 3 if len(targets) > 4 else 2                # decomposition panels per row
    ra = int(np.ceil(len(features) / 3)); rb = int(np.ceil(len(targets) / bc))
    fig = plt.figure(figsize=(DOUBLE, 1.65 * ra + 2.1 * rb + 0.7))
    gs = GridSpec(ra + rb, 6, figure=fig, height_ratios=[1.0] * ra + [1.3] * rb, hspace=0.55, wspace=1.6)
    # ---- (a)
    axa = []
    for k, f in enumerate(features):
        r_, c_ = divmod(k, 3); ax = fig.add_subplot(gs[r_, 2 * c_:2 * c_ + 2]); axa.append(ax)
        j_ = names.index(f)
        ax.scatter(s, X[:, j_], s=4, color=MUTED, alpha=0.45, lw=0)
        ax.plot(s, C[:, j_], color=ORANGE, lw=1.4)
        b = res['params'].loc[f, 'b']
        role = tags.get(f, '')
        col = ORANGE if role.startswith('exposure') else BLUE if role.startswith('Z') else AQUA if role.startswith('W') else INK
        bb = dict(boxstyle='round,pad=0.12', fc=PAPER, ec='none', alpha=0.85)
        ax.text(0.03, 0.95, f, transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='top', color=col, bbox=bb, zorder=5)
        ax.text(0.03, 0.79, (role + '; ' if role else '') + f'$b$ = {b:.2f}', transform=ax.transAxes, fontsize=8, va='top', color=INK2, bbox=bb, zorder=5)
        ax.grid(axis='y', color=GRID, lw=0.4); ax.tick_params(length=2)
        if r_ == ra - 1: ax.set_xlabel('niche index $s$')
        if c_ == 0: ax.set_ylabel('level (shifted z-score)')
    _panel_label(axa[0], 'a', x=-0.34)
    # ---- (b)
    dec = _solve_ode_decomposition(res['qd'], res['samples'], res['supports'], basis_order=res.get('basis_order', 0), ridge=res.get('ode_ridge', 1e-6))
    tau = np.asarray(dec['sample_tau']); feats = list(dec['features'])
    palette = [BLUE, AQUA, YELLOW, '#e87ba4', '#4a3aa7', '#008300', '#d95926']
    axb = []
    for k, t in enumerate(targets):
        r_, c_ = divmod(k, bc); w_ = 6 // bc; ax = fig.add_subplot(gs[ra + r_, w_ * c_:w_ * c_ + w_]); axb.append(ax)
        j_ = feats.index(t)
        obs = np.asarray(dec['response'])[:, j_]; pred = np.asarray(dec['predicted_states'])[:, j_]
        ax.scatter(tau, obs, s=5, color=MUTED, alpha=0.5, lw=0); ax.plot(tau, pred, color=INK, lw=1.5)
        self_eff = dec['intercepts'][j_] + dec['interaction_functions'].get((t, t), np.zeros_like(tau))
        ax.plot(tau, self_eff, color=INK2, lw=1.1, ls=(0, (4, 2)))
        srcs = [src for src in dec['support_sets'].get(t, []) if src != t]; ends = []
        for i_, src in enumerate(srcs):
            eff = dec['interaction_functions'][(t, src)]
            ax.plot(tau, eff, color=palette[i_ % len(palette)], lw=1.0); ends.append((eff[-1], src, palette[i_ % len(palette)]))
        ends.sort(key=lambda e: e[0]); yl = [e[0] for e in ends]; span = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
        for k_ in range(1, len(yl)):
            if yl[k_] - yl[k_ - 1] < 0.07 * span: yl[k_] = yl[k_ - 1] + 0.07 * span
        xr = tau[-1] + 0.03 * (tau[-1] - tau[0])
        for (y0, src, col), yy in zip(ends, yl):
            ax.text(xr, yy, src, va='center', ha='left', fontsize=8, color=col, clip_on=False)
            if abs(yy - y0) > 1e-9: ax.plot([tau[-1], xr - 0.005 * (tau[-1] - tau[0])], [y0, yy], color=col, lw=0.4, clip_on=False)
        ax.axhline(0, color=GRID, lw=0.6)
        ttl = t + (f'  ({tags[t]})' if t in tags else '')
        ax.set_title(ttl, fontsize=8, loc='left'); ax.grid(axis='y', color=GRID, lw=0.4)
        ax.set_xlim(tau[0], tau[-1] + 0.2 * (tau[-1] - tau[0])); ax.tick_params(length=2)
        if r_ == rb - 1: ax.set_xlabel('niche index $s$')
        if c_ == 0: ax.set_ylabel('cumulative effect')
    _panel_label(axb[0], 'b', x=-0.22 if bc == 2 else -0.34)
    h = [plt.Line2D([], [], marker='o', ls='', color=MUTED, ms=3.5, label='observed'),
         plt.Line2D([], [], color=INK, lw=1.5, label='reconstructed'),
         plt.Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 2)), label='self effect (with baseline)'),
         plt.Line2D([], [], color=BLUE, lw=1.0, label='cross effects, labelled by source')]
    fig.legend(handles=h, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.015), frameon=False, handlelength=1.6, columnspacing=1.4)
    return fig


# ================================================================ Figures 2 and 3: eight fits, eight decompositions
def fig_fits(res, features, tags=None, ncol=4):
    """Power-law fits, `ncol` per row; the role tag and exponent are printed in each panel."""
    set_style()
    X, s, names, C = res['X'], res['s'], res['names'], res['curves']; tags = tags or {}
    nrow = int(np.ceil(len(features) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, 1.75 * nrow + 0.5), sharex=True)
    axes = np.atleast_2d(axes)
    for k, f in enumerate(features):
        r_, c_ = divmod(k, ncol); ax = axes[r_, c_]; j_ = names.index(f)
        ax.scatter(s, X[:, j_], s=3.5, color=MUTED, alpha=0.45, lw=0)
        ax.plot(s, C[:, j_], color=ORANGE, lw=1.4)
        b = res['params'].loc[f, 'b']; role = tags.get(f, '')
        col = ORANGE if role.startswith('exposure') else BLUE if role.startswith('Z') else AQUA if role.startswith('W') else INK
        bb = dict(boxstyle='round,pad=0.12', fc=PAPER, ec='none', alpha=0.85)
        ax.text(0.04, 0.95, f, transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='top', color=col, bbox=bb, zorder=5)
        ax.text(0.04, 0.78, (role + '\n' if role else '') + f'$b$ = {b:.2f}', transform=ax.transAxes, fontsize=8, va='top', color=INK2, bbox=bb, zorder=5, linespacing=1.1)
        ax.grid(axis='y', color=GRID, lw=0.4); ax.tick_params(length=2)
        if r_ == nrow - 1: ax.set_xlabel('niche index $s$')
        if c_ == 0: ax.set_ylabel('level (shifted z-score)')
    for k in range(len(features), nrow * ncol): axes.ravel()[k].axis('off')
    fig.subplots_adjust(hspace=0.3, wspace=0.32)
    return fig


def fig_decomps(res, targets, tags=None, ncol=2):
    """Effect decompositions, `ncol` per row, cross effects labelled at the right-hand end."""
    from idop_core import _solve_ode_decomposition
    set_style(); tags = tags or {}
    dec = _solve_ode_decomposition(res['qd'], res['samples'], res['supports'], basis_order=res.get('basis_order', 0), ridge=res.get('ode_ridge', 1e-6))
    tau = np.asarray(dec['sample_tau']); feats = list(dec['features'])
    palette = [BLUE, AQUA, YELLOW, '#e87ba4', '#4a3aa7', '#008300', '#d95926']
    nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, (1.75 if ncol == 2 else 2.0) * nrow + 0.9))
    axes = np.atleast_2d(axes)
    for k, t in enumerate(targets):
        r_, c_ = divmod(k, ncol); ax = axes[r_, c_]; j_ = feats.index(t)
        obs = np.asarray(dec['response'])[:, j_]; pred = np.asarray(dec['predicted_states'])[:, j_]
        ax.scatter(tau, obs, s=4, color=MUTED, alpha=0.5, lw=0); ax.plot(tau, pred, color=INK, lw=1.5)
        self_eff = dec['intercepts'][j_] + dec['interaction_functions'].get((t, t), np.zeros_like(tau))
        ax.plot(tau, self_eff, color=INK2, lw=1.1, ls=(0, (4, 2)))
        srcs = [src for src in dec['support_sets'].get(t, []) if src != t]; ends = []
        for i_, src in enumerate(srcs):
            eff = dec['interaction_functions'][(t, src)]
            ax.plot(tau, eff, color=palette[i_ % len(palette)], lw=1.0); ends.append((eff[-1], src, palette[i_ % len(palette)]))
        ends.sort(key=lambda e: e[0]); yl = [e[0] for e in ends]; span = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
        for k_ in range(1, len(yl)):
            if yl[k_] - yl[k_ - 1] < 0.09 * span: yl[k_] = yl[k_ - 1] + 0.09 * span
        xr = tau[-1] + 0.03 * (tau[-1] - tau[0])
        for (y0, src, col), yy in zip(ends, yl):
            ax.text(xr, yy, src, va='center', ha='left', fontsize=8, color=col, clip_on=False)
            if abs(yy - y0) > 1e-9: ax.plot([tau[-1], xr - 0.005 * (tau[-1] - tau[0])], [y0, yy], color=col, lw=0.4, clip_on=False)
        ax.axhline(0, color=GRID, lw=0.6)
        ax.set_title(t + (f'  ({tags[t]})' if t in tags else ''), fontsize=8, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.4); ax.set_xlim(tau[0], tau[-1] + 0.2 * (tau[-1] - tau[0])); ax.tick_params(length=2)
        if r_ == nrow - 1: ax.set_xlabel('niche index $s$')
        if c_ == 0: ax.set_ylabel('cumulative effect')
    for k in range(len(targets), nrow * ncol): axes.ravel()[k].axis('off')
    h = [plt.Line2D([], [], marker='o', ls='', color=MUTED, ms=3.5, label='observed'),
         plt.Line2D([], [], color=INK, lw=1.5, label='reconstructed'),
         plt.Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 2)), label='self effect (with baseline)'),
         plt.Line2D([], [], color=BLUE, lw=1.0, label='cross effects, labelled by source')]
    fig.legend(handles=h, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.005), frameon=False, handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(hspace=0.55, wspace=0.42 if ncol == 2 else 0.65, bottom=0.11 if ncol == 2 else 0.2)
    return fig


# ================================================================ Figure 5: where the cohorts fall
def fig_cohorts(csv, analysed=('ov', 'luad'), nu_floor=0.05):
    """Cohort diagnostics: residual fraction against proxy strength, point size by n.
    Cohorts that abstain entirely sit on the axis with a cross."""
    set_style()
    d = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(SINGLE, 3.0))
    for r in d.itertuples():
        abst = r.n_pass == 0; y = 0.0 if abst else r.nu_median
        if r.study in analysed: col, fc, mk = ORANGE, ORANGE, 'o'
        elif abst: col, fc, mk = INK2, 'none', 'x'
        elif r.nu_median >= 0.1: col, fc, mk = ORANGE, PAPER, 'o'
        else: col, fc, mk = YELLOW, YELLOW, 'o'
        sz = 12 + 0.11 * r.n
        if mk == 'x': ax.scatter(r.rho_bar, y, s=sz * 0.9, marker='x', color=col, lw=1.1, zorder=3)
        else: ax.scatter(r.rho_bar, y, s=sz, marker='o', facecolor=fc, edgecolor=col, lw=1.0, zorder=3)
        off = dict(luad=(4, 9), blca=(9, -12), ov=(9, 2), stad=(6, 6), coadread=(9, 2), brca=(9, 2), skcm=(0, 9), ucec=(-9, -14), kirc=(4, -14), lgg=(6, 6))
        dx, dy = off.get(r.study, (5, 4)); ha = 'right' if dx < 0 else 'left' if dx > 0 else 'center'
        ax.annotate(r.study.upper(), (r.rho_bar, y), xytext=(dx, dy), textcoords='offset points', fontsize=8, ha=ha,
                    fontweight='bold' if r.study in analysed else 'normal', color=INK if r.study in analysed else INK2)
    ax.axhline(nu_floor, color=INK2, lw=0.7, ls=(0, (3, 2))); ax.text(0.955, nu_floor + 0.004, r'$\nu$ floor', fontsize=8, color=INK2, ha='right')
    ax.set_xlabel(r'residual fraction $\bar\rho$'); ax.set_ylabel(r'proxy strength $\nu$ (median over exposures)')
    ax.set_xlim(0.78, 0.965); ax.set_ylim(-0.018, 0.185); ax.grid(axis='y', color=GRID, lw=0.4)
    h = [plt.Line2D([], [], marker='o', ls='', color=ORANGE, ms=6, label='analysed'),
         plt.Line2D([], [], marker='o', ls='', mfc=PAPER, mec=ORANGE, ms=6, label='operable'),
         plt.Line2D([], [], marker='o', ls='', color=YELLOW, ms=6, label='weak proxies'),
         plt.Line2D([], [], marker='x', ls='', color=INK2, ms=6, mew=1.1, label='abstains'),
         plt.Line2D([], [], marker='o', ls='', mfc='none', mec=MUTED, ms=4, label='size: $n$')]
    ax.legend(handles=h, loc='upper left', fontsize=8, handlelength=1.2, frameon=False, borderpad=0.2)
    return fig


# ================================================================ Figure 4: effect decomposition
def fig_decomposition(res, targets, ncol=3):
    """Cumulative self and cross effects reconstructing each target's trajectory.

    Cross-source curves are labelled at their right-hand end; one legend
    serves the three shared elements. The self effect is drawn in neutral
    grey so that orange keeps its meaning (exposure) from the other figures."""
    from idop_core import _solve_ode_decomposition
    set_style()
    dec = _solve_ode_decomposition(res['qd'], res['samples'], res['supports'], basis_order=res.get('basis_order', 0), ridge=res.get('ode_ridge', 1e-6))
    tau = np.asarray(dec['sample_tau']); feats = list(dec['features'])
    fig, axes = plt.subplots(1, ncol, figsize=(DOUBLE, 2.7))
    palette = [BLUE, AQUA, YELLOW, '#e87ba4', '#4a3aa7', '#008300', '#d95926']
    for ax, t in zip(np.atleast_1d(axes), targets):
        j_ = feats.index(t)
        obs = np.asarray(dec['response'])[:, j_]; pred = np.asarray(dec['predicted_states'])[:, j_]
        ax.scatter(tau, obs, s=5, color=MUTED, alpha=0.5, lw=0)
        ax.plot(tau, pred, color=INK, lw=1.5)
        self_eff = dec['intercepts'][j_] + dec['interaction_functions'].get((t, t), np.zeros_like(tau))
        ax.plot(tau, self_eff, color=INK2, lw=1.1, ls=(0, (4, 2)))
        srcs = [src for src in dec['support_sets'].get(t, []) if src != t]
        ends = []
        for i_, src in enumerate(srcs):
            eff = dec['interaction_functions'][(t, src)]
            ax.plot(tau, eff, color=palette[i_ % len(palette)], lw=1.0)
            ends.append((eff[-1], src, palette[i_ % len(palette)]))
        # right-end labels, pushed apart vertically
        ends.sort(key=lambda e: e[0])
        yl = [e[0] for e in ends]; span = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
        for k_ in range(1, len(yl)):
            if yl[k_] - yl[k_ - 1] < 0.055 * span: yl[k_] = yl[k_ - 1] + 0.055 * span
        xr = tau[-1] + 0.03 * (tau[-1] - tau[0])
        for (y0, src, col), yy in zip(ends, yl):
            ax.text(xr, yy, src, va='center', ha='left', fontsize=8, color=col, clip_on=False)
            if abs(yy - y0) > 1e-9:
                ax.plot([tau[-1], xr - 0.005 * (tau[-1] - tau[0])], [y0, yy], color=col, lw=0.4, clip_on=False)
        ax.axhline(0, color=GRID, lw=0.6)
        ax.set_title(t, fontsize=8, loc='left'); ax.set_xlabel('niche index $s$'); ax.grid(axis='y', color=GRID, lw=0.4)
        ax.set_xlim(tau[0], tau[-1] + 0.22 * (tau[-1] - tau[0]))
        ax.tick_params(length=2)
    np.atleast_1d(axes)[0].set_ylabel('cumulative effect')
    h = [plt.Line2D([], [], marker='o', ls='', color=MUTED, ms=3.5, label='observed'),
         plt.Line2D([], [], color=INK, lw=1.5, label='reconstructed'),
         plt.Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 2)), label='self effect (with baseline)'),
         plt.Line2D([], [], color=BLUE, lw=1.0, label='cross effects, labelled by source')]
    fig.legend(handles=h, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01), frameon=False,
               handlelength=1.6, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.38, bottom=0.26)
    return fig


# ================================================================ Figure 5: simulation
def fig_simulation(cov_csv, tradeoff_csv, p1p2_csv=None):
    """(a) coverage against the violation magnitude; (b) the resolution-strength trade-off."""
    set_style()
    cov = pd.read_csv(cov_csv); tr = pd.read_csv(tradeoff_csv).sort_values('x_latent_share')
    fig, axes = plt.subplots(1, 3 if p1p2_csv else 2, figsize=(DOUBLE, 2.7)); a, b = axes[0], axes[1]
    series = [('proposed', 'idopNetwork–proximal', ORANGE),
              ('designated', 'designated proxies, all $W$ invalid', BLUE)]
    for key, lab, col in series:
        d = cov[cov.method == key].sort_values('delta')
        a.plot(d.delta, d.coverage, color=col, lw=1.3, marker='o', ms=3.5, mec=PAPER, mew=0.6, label=lab)
        if key == 'proposed':                       # abstention rate, one row above the axis
            a.text(-0.005, 0.455, 'proposed method abstains:', ha='left', va='bottom', fontsize=8, color=col)
            for x_, ab in zip(d.delta, d.abstain):
                a.text(x_, 0.412, f'{100 * ab:.0f}%', ha='center', va='bottom', fontsize=8, color=col)
    a.axhline(0.95, color=INK2, lw=0.7, ls=(0, (3, 2))); a.text(0.2, 0.958, 'nominal 0.95', fontsize=8, color=INK2, ha='right')
    a.set_xlabel('violation magnitude $\\delta$'); a.set_ylabel('coverage of the total effect'); a.set_ylim(0.4, 1.0)
    a.grid(axis='y', color=GRID, lw=0.4); _panel_label(a, 'a')

    x = 1 - tr.x_latent_share.values            # non-latent share of the predictors
    b.plot(x, tr.edge_F1, color=BLUE, lw=1.3, marker='o', ms=3.5, mec=PAPER, mew=0.6, label='edge recovery F1')
    b.plot(x, tr.nu / tr.nu.max(), color=AQUA, lw=1.3, marker='s', ms=3.3, mec=PAPER, mew=0.6,
           label=f'proxy strength $\\nu$ / {tr.nu.max():.2f}')
    b.plot(x, tr.W_false_edges / tr.W_false_edges.max(), color=ORANGE, lw=1.1, ls='--', marker='^', ms=3.3,
           mec=PAPER, mew=0.6, label=f'spurious edges on $W$ / {tr.W_false_edges.max():.0f}')
    b.plot(x, tr.proposed_abstain, color=INK2, lw=1.1, ls=':', marker='D', ms=3.0, mec=PAPER, mew=0.6,
           label='proposed method abstains')
    b.set_xlabel('non-latent share of predictor variation'); b.set_ylabel('relative scale')
    b.set_ylim(0, 1.05); b.grid(axis='y', color=GRID, lw=0.4); _panel_label(b, 'b')
    if p1p2_csv:
        c = axes[2]; pp = pd.read_csv(p1p2_csv)
        base = {m: cov[(cov.method == m) & (cov.delta == 0)].coverage.iloc[0] for m in ('proposed', 'designated')}
        for m, col in (('proposed', ORANGE), ('designated', BLUE)):
            for viol, ls, lab in (('P1 only', '-', '(P1) only'), ('P2 only', (0, (3, 2)), '(P2) only')):
                g = pp[(pp.method == m) & (pp.violation == viol)].sort_values('delta')
                c.plot([0] + list(g.delta), [base[m]] + list(g.coverage), color=col, lw=1.3, ls=ls, marker='o', ms=3.3, mec=PAPER, mew=0.6)
        c.axhline(0.95, color=INK2, lw=0.7, ls=(0, (3, 2)))
        c.set_xlabel('violation magnitude $\\delta$'); c.set_ylabel('coverage of the total effect'); c.set_ylim(0.4, 1.04)
        c.grid(axis='y', color=GRID, lw=0.4); _panel_label(c, 'c')
    fig.subplots_adjust(wspace=0.42, bottom=0.2)
    # one legend under each panel
    def under(ax, handles, ncol=1):
        bb = ax.get_position()
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=((bb.x0 + bb.x1) / 2, -0.03), ncol=ncol, fontsize=8,
                   handlelength=1.6, frameon=False, columnspacing=1.0)
    under(a, [plt.Line2D([], [], color=ORANGE, lw=1.3, marker='o', ms=3.5, label='idopNetwork–proximal'),
              plt.Line2D([], [], color=BLUE, lw=1.3, marker='o', ms=3.5, label='designated proxies, all $W$ invalid')])
    under(b, [plt.Line2D([], [], color=BLUE, lw=1.3, marker='o', ms=3.5, label='edge recovery F1'),
              plt.Line2D([], [], color=AQUA, lw=1.3, marker='s', ms=3.3, label=f'proxy strength $\\nu$ / {tr.nu.max():.2f}'),
              plt.Line2D([], [], color=ORANGE, lw=1.1, ls='--', marker='^', ms=3.3, label=f'spurious edges on $W$ / {tr.W_false_edges.max():.0f}'),
              plt.Line2D([], [], color=INK2, lw=1.1, ls=':', marker='D', ms=3.0, label='proposed method abstains')])
    if p1p2_csv:
        under(axes[2], [plt.Line2D([], [], color=INK2, lw=1.3, label='solid: (P1) violation only'),
                        plt.Line2D([], [], color=INK2, lw=1.3, ls=(0, (3, 2)), label='dashed: (P2) violation only'),
                        plt.Line2D([], [], color=ORANGE, lw=1.3, label='idopNetwork–proximal'), plt.Line2D([], [], color=BLUE, lw=1.3, label='designated')])
    return fig


# ================================================================ Figure 6: application results
def fig_application(est, top=14, stability=None, survival=False):
    """(a) proximal vs naive for the strongest exposures; (b) stability across the support window.
    survival: estimates are log hazard ratios; bootstrap se/q (se_boot, q_boot) are used when present."""
    set_style()
    d = est.copy()
    if 'se_boot' in d: d = d.assign(se=d.se_boot, t_prox=d.t_boot, q_prox=d.q_boot)
    d['abs_t'] = d.t_prox.abs()
    d = d.sort_values('abs_t', ascending=False).head(top).sort_values('proximal')
    ncols = 2 if stability is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(DOUBLE if ncols == 2 else SINGLE, 0.27 * top + 1.0),
                             gridspec_kw=dict(width_ratios=[1.35, 1]) if ncols == 2 else None)
    a = axes[0] if ncols == 2 else axes
    y = np.arange(len(d))
    a.axvline(0, color=INK2, lw=0.6)
    a.errorbar(d.naive, y - 0.17, xerr=1.96 * d.naive_se, fmt='o', ms=3.2, color=MUTED, ecolor=MUTED,
               elinewidth=0.7, capsize=0, label='unadjusted')
    a.errorbar(d.proximal, y + 0.17, xerr=1.96 * d.se, fmt='o', ms=3.6, color=BLUE, ecolor=BLUE,
               elinewidth=0.9, capsize=0, label='proximal')
    x_hi = max((d.proximal + 1.96 * d.se).max(), (d.naive + 1.96 * d.naive_se).max())
    x_lo = min((d.proximal - 1.96 * d.se).min(), (d.naive - 1.96 * d.naive_se).min())
    x_ann = x_hi + 0.06 * (x_hi - x_lo)
    for yi, (nu, q) in enumerate(zip(d.nu, d.q_prox)):
        a.text(x_ann, yi,
               f'$\\nu$={nu:.2f}  q={q:.2f}', fontsize=8, va='center', color=INK2)
    a.set_yticks(y); a.set_yticklabels(d.exposure, fontsize=8)
    a.set_xlabel(('log hazard ratio per unit, 95% CI' + (' (bootstrap)' if 'se_boot' in est else '')) if survival else 'risk difference per unit, 95% CI'); a.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, columnspacing=1.5)
    a.grid(axis='x', color=GRID, lw=0.4); a.set_xlim(right=x_ann + 0.5 * (x_ann - x_lo))
    _panel_label(a, 'a', x=-0.32)
    if stability is not None:
        b = axes[1]
        for i, (name, g) in enumerate(stability.groupby('exposure', sort=False)):
            col = [BLUE, ORANGE, AQUA, YELLOW][i % 4]
            xs = np.arange(len(g)) + i * 0.12
            b.errorbar(xs, g.proximal, yerr=1.96 * g.se, fmt='o-', ms=3.2, lw=0.9,
                       color=col, ecolor=col, elinewidth=0.7, capsize=0, label=name)
            miss = g.proximal.isna().values                   # abstained (screen or nu floor): cross at the axis floor
            if miss.any(): b.scatter(xs[miss], np.full(miss.sum(), 0.02), marker='x', s=22, color=col, lw=0.9,
                                     transform=b.get_xaxis_transform(), clip_on=False, zorder=4)
        b.axhline(0, color=INK2, lw=0.6)
        b.scatter([], [], marker='x', s=22, color=INK2, lw=0.9, label='abstains')
        b.set_xticks(np.arange(len(g))); b.set_xticklabels(g.setting, fontsize=8, rotation=30, ha='right')
        b.set_ylabel('proximal log hazard ratio' if survival else 'proximal risk difference'); b.set_xlabel('support-graph setting ($\\alpha$, $k$)')
        b.legend(fontsize=8, loc='best'); b.grid(axis='y', color=GRID, lw=0.4); _panel_label(b, 'b', x=-0.22)
    fig.subplots_adjust(wspace=0.55)
    return fig
