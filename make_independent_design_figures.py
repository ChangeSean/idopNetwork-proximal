"""Final independent clinical inference and fresh end-to-end study figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figures as F
from paper_tables import SCENARIOS, ROMAN

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


def forest(study,name):
    F.set_style()
    d=pd.read_csv(APP/f'{study}_all_exposures.csv')
    d=d[d.design_ready].sort_values('exposure',ascending=False).reset_index(drop=True)
    fig,axes=plt.subplots(1,2,figsize=(F.DOUBLE,max(4.5,.25*len(d)+1.9)),sharey=True)
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
        ax.set_yticks(range(len(d)),[f'{r.exposure}  r={int(r.r_grid)}' for r in d.itertuples()])
        F._panel_label(ax,'ab'[j],x=-.13)
    meta=json.loads((APP/f'{study}_manifest.json').read_text())
    fig.suptitle(f'{study.upper()}: discovery n={meta["n_discovery"]}; estimation n={meta["n_estimation"]}',
                 fontsize=9,y=.985)
    fig.text(.52,.035,'Arrows: infinite or off-scale endpoints. Triangles: off-scale point estimates.\nGrey: unavailable calculation. Orange: estimated confidence set.',
             ha='center',fontsize=7.5,linespacing=1.4)
    fig.subplots_adjust(left=.18,right=.98,top=.90,bottom=.23 if len(d)<=14 else .16,wspace=.2)
    F.save(fig,name)


def main():
    schematic();simulation();forest('ov','fig6_application');forest('luad','fig7_luad')


if __name__=='__main__':main()
