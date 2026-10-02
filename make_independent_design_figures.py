"""Final independent clinical inference and fresh end-to-end study figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figures as F
from paper_tables import SCENARIOS, ROMAN
from clinical_reporting import STUDIES, CASES, records, point_records, case_records, EXTENSION

ROOT=Path(__file__).resolve().parent
SIM=ROOT/'results/discovery_estimation_validation_20261001'
APP=ROOT/'results/discovery_estimation_application_20261001'


def schematic():
    F.set_style()
    fig,ax=plt.subplots(figsize=(F.DOUBLE,6.1))
    ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis('off')
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.4',fc='white',ec=color,lw=1))
        ax.text(x+w/2,y+h-4,title,ha='center',va='center',fontsize=9,fontweight='bold',color=color)
        ax.text(x+w/2,y+(h-5)/2,body,ha='center',va='center',fontsize=8,linespacing=1.6)
    def arrow(x,y,xx,yy):
        ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=10,color=F.INK2,lw=.9))
    box(29,88,42,10,'Patient cohort','One prespecified 75:25 allocation',F.INK2)
    ax.text(25,81,'DISCOVERY PATIENTS  75%',ha='center',color=F.BLUE,fontweight='bold',fontsize=9)
    ax.text(76,81,'ESTIMATION PATIENTS  25%',ha='center',color=F.ORANGE,fontweight='bold',fontsize=9)
    arrow(38,87,25,85);arrow(64,87,76,85)
    box(3,60,43,18,'Molecular representation','Discovery panel, imputation and scaling\nNiche curves and patient deviations',F.BLUE)
    box(3,36,43,18,'Network and proxy roles','Five windows at widths 20% and 40%\nEight support designs; neighbours Z\nSeparated readout pool W',F.BLUE)
    box(3,10,43,20,'Identifying information','Maximum joint dimension\nFull conditional rank; strongest root\nFreeze roles, dimension, P and pivots',F.BLUE)
    arrow(25,59,25,55);arrow(25,35,25,31)
    box(55,60,42,18,'Observed clinical targets','Apply discovery transformations\n36-month RMST and survival\nCensoring fit in estimation patients',F.ORANGE)
    box(55,36,42,18,'Concentrated bridge moments','Fixed observed readouts R = WP\nRegular nuisance block\nEffect moments N = tau D',F.ORANGE)
    box(55,10,42,20,'Effect confidence set','Resample estimation patients\nRefit moments and censoring\nInvert all remaining Fieller moments',F.ORANGE)
    arrow(76,59,76,55);arrow(76,35,76,31)
    arrow(47,20,54,44)
    ax.text(50,53,'Frozen\ndesign',ha='center',fontsize=8,color=F.INK2)
    ax.text(50,3,'Causal model: structural coverage, exposure boundary, complete mean bridge and censoring positivity',
            ha='center',va='center',fontsize=8,color=F.INK2)
    F.save(fig,'fig1_schematic')


def simulation():
    F.set_style()
    d=pd.read_csv(SIM/'all_summary.csv')
    fig,axes=plt.subplots(2,2,figsize=(F.DOUBLE,5.3))
    for i,target in enumerate(['linear','rmst']):
        v=d[d.outcome.eq(target)].set_index('scenario').loc[SCENARIOS]
        axes[i,0].errorbar(range(7),v.bias,yerr=1.96*v.bias_mcse,fmt='o',color=F.BLUE,capsize=2,ms=4)
        axes[i,0].axhline(0,color=F.INK2,ls='--',lw=.7)
        axes[i,0].set_ylabel(('Linear effect' if i==0 else 'RMST')+'\nBias among point fits')
        axes[i,1].errorbar(np.arange(7)-.06,v.coverage,yerr=1.96*v.coverage_mcse,
                          fmt='o',color=F.ORANGE,capsize=2,ms=4,label='All attempted datasets')
        axes[i,1].errorbar(np.arange(7)+.06,v.fit_coverage,yerr=1.96*v.fit_coverage_mcse,
                          fmt='s',color=F.BLUE,capsize=2,ms=3,label='Available point fits')
        axes[i,1].axhline(.95,color=F.INK2,ls='--',lw=.7)
        axes[i,1].set_ylim(.83,1.045)
        axes[i,1].set_ylabel('Confidence-set coverage')
        for j,ax in enumerate(axes[i]):
            ax.set_xticks(range(7),[ROMAN[s] for s in SCENARIOS])
            ax.grid(axis='y',color=F.GRID,lw=.5)
            F._panel_label(ax,'abcd'[2*i+j],x=-.14)
    fig.legend(*axes[0,1].get_legend_handles_labels(),loc='lower center',ncol=2)
    fig.subplots_adjust(left=.12,right=.97,top=.95,bottom=.14,hspace=.44,wspace=.40)
    F.save(fig,'fig4_simulation')


def clinical_points():
    """Every completed coefficient is displayed, including off-range values."""
    F.set_style()
    data = records()
    points = point_records(data)
    fig, axes = plt.subplots(1, 2, figsize=(F.DOUBLE, 5.5), sharey=True)
    for j, (target, scale, limit, title) in enumerate([
            ('rmst', 1, 36, 'RMST contrast (months)'),
            ('survival', 100, 100, 'Survival contrast (percentage points)')]):
        ax = axes[j]
        shown = 0
        for i, study in enumerate(STUDIES):
            cohort = points[points.study.eq(study)].sort_values('exposure')
            offsets = np.linspace(-.24, .24, len(cohort)) if len(cohort) > 1 else np.zeros(len(cohort))
            for offset, (_, row) in zip(offsets, cohort.iterrows()):
                value = row[target + '_estimate'] * scale
                off_range = abs(value) > limit
                color = F.ORANGE if (study, row.exposure) in CASES else F.BLUE
                marker = '<' if value < -limit else '>' if value > limit else 'o'
                displayed = np.clip(value, -.97 * limit, .97 * limit) if off_range else value
                ax.plot(displayed, i + offset,
                        marker=marker, ms=5 if off_range else 4, linestyle='none',
                        color=color, markerfacecolor='none' if off_range else color,
                        markeredgewidth=.8, alpha=.9)
                shown += 1
        assert shown == len(points)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(len(STUDIES) - .5, -.5)
        ax.set_yticks(range(len(STUDIES)),
                      [f'{study.upper()}  {int(points.study.eq(study).sum())}/60' for study in STUDIES])
        ax.axvline(0, color=F.INK2, lw=.7, ls='--')
        ax.grid(axis='x', color=F.GRID, lw=.4)
        ax.set_xlabel(title)
        F._panel_label(ax, 'ab'[j], x=-.15)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', color=F.BLUE, ls='none', ms=4, label='Completed estimate'),
               Line2D([], [], marker='o', color=F.ORANGE, ls='none', ms=4, label='Worked contrast'),
               Line2D([], [], marker='>', color=F.BLUE, markerfacecolor='none', ls='none', ms=5,
                      label='Off-range estimate')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(.53, .035))
    fig.suptitle('Independent point estimates across ten TCGA cohorts', fontsize=10, y=.975)
    fig.subplots_adjust(left=.20, right=.98, top=.91, bottom=.18, wspace=.30)
    F.save(fig, 'fig6_application')
    return points


def clinical_cases():
    F.set_style()
    d=pd.DataFrame(case_records()).reset_index(drop=True)
    fig,axes=plt.subplots(1,2,figsize=(F.DOUBLE,3.5),sharey=True)
    for j,(target,scale,limit,title) in enumerate([('rmst',1,36,'RMST contrast (months)'),
                                                ('survival',100,100,'Survival contrast (percentage points)')]):
        ax=axes[j]
        ax.set_xlim(-limit,limit)
        for i,row in d.iterrows():
            available=row[target+'_status']=='estimated'
            color=F.ORANGE if available else F.MUTED
            for lo,hi in json.loads(row[target+'_intervals']):
                lo,hi=lo*scale,hi*scale
                left,right=max(lo,-limit),min(hi,limit)
                if left<=right:
                    ax.plot([left,right],[i,i],color=color,lw=.9,alpha=.85)
                    if lo<=-limit:
                        ax.annotate('',xy=(-limit,i),xytext=(-limit+.1*limit,i),arrowprops=dict(arrowstyle='->',color=color,lw=.9))
                    else:ax.plot([lo,lo],[i-.11,i+.11],color=color,lw=.7)
                    if hi>=limit:
                        ax.annotate('',xy=(limit,i),xytext=(limit-.1*limit,i),arrowprops=dict(arrowstyle='->',color=color,lw=.9))
                    else:ax.plot([hi,hi],[i-.11,i+.11],color=color,lw=.7)
            point=row[target+'_estimate']*scale
            if np.isfinite(point):
                marker='o' if -limit<=point<=limit else '<' if point<-limit else '>'
                ax.plot(np.clip(point,-.97*limit,.97*limit),i,marker,color=F.BLUE,ms=3)
            elif j==1:
                ax.text(0,i,'Moment resampling unavailable',ha='center',va='center',fontsize=7,
                        color=F.INK2,bbox=dict(fc='white',ec='none',pad=.5))
        ax.axvline(0,color=F.INK2,lw=.6,ls='--')
        ax.set_xlabel(title);ax.grid(axis='x',color=F.GRID,lw=.4)
        ax.set_yticks(range(len(d)),[f'{r.study.upper()} {r.exposure}  r={int(r.r_grid)}' for r in d.itertuples()])
        ax.set_ylim(len(d)-.6,-.6)
        F._panel_label(ax,'ab'[j],x=-.13)
    fig.suptitle('Worked molecular contrasts with 95% confidence sets', fontsize=10,y=.975)
    fig.text(.55,.035,'Dots: point estimates. Lines: 95% sets.\nArrows: confidence set continues beyond the displayed range.',
             ha='center',fontsize=7.5,linespacing=1.4)
    fig.subplots_adjust(left=.23,right=.98,top=.85,bottom=.28,wspace=.3)
    F.save(fig,'fig7_clinical_cases')


def main():
    schematic();simulation()
    points=clinical_points();clinical_cases()
    provenance=dict(cohorts=10,attempted_exposures=600,point_estimates_per_panel=len(points),
                    worked_cases=[dict(study=study,exposure=protein) for study,protein in CASES],
                    rmst_off_range=int(points.rmst_outside_target_range.sum()),
                    survival_off_range=int(points.survival_outside_target_range.sum()),
                    point_selection='Every estimated coefficient; no filtering by direction or interval exclusion',
                    confidence_sets='Retained in source records, worked-case table and Figure 7')
    (EXTENSION/'clinical_figures_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')


if __name__=='__main__':main()
