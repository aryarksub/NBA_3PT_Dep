"""Plots that separate boundary, search extent, resolution and metric validation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK=Path('results/search_sensitivity');OUT=Path('plots/search_sensitivity');OUT.mkdir(exist_ok=True)
INK='#174d70';ORANGE='#c55b29';GREEN='#34846b';GREY='#929ca5'
RADII=[5.,6.5,8.];STEPS=[1.,.5,.25,.1]
FOCUS=['near_k100','best_q095_l025','best_q090_l000']
PRETTY={'naive':'Naive reference','near_k100':'Nearest: equal openness','best_q095_l025':'Best: q=.95, closing=.25','best_q090_l000':'Best: q=.90, no closing'}

def pretty(s):
    if s in PRETTY:return PRETTY[s]
    if s.startswith('near'):return f'Nearest k={int(s[6:])/100:g}'
    return f'Best q={int(s[6:9])/100:g}, closing={int(s[11:])/100:g}'

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
def save(fig,name,footer=None):
    fig.tight_layout(rect=[0,.055 if footer else 0,1,.95])
    if footer:fig.text(.025,.012,footer,fontsize=8,color='#485662')
    fig.savefig(OUT/(name+'.png'),dpi=180,facecolor='white')
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    plt.close(fig)

def geometry():
    data=pd.read_csv(WORK/'geometry_comparison.csv')
    strategies=data[data.version=='corrected_geometry'].strategy.tolist()
    fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True)
    for ax,value,xlabel,scale in zip(axes,['median_dep_share','rho_attempt_rate'],['Median player dependence share (%)','Spearman correlation with attempt rate'],[100,1]):
        for vi,(version,color,marker,label) in enumerate([('as_saved',GREY,'x','Original saved results'),('old_geometry_same_model',ORANGE,'o','Old boundary, matched model'),('corrected_geometry',INK,'o','Correct boundary, matched model')]):
            sub=data[data.version==version].set_index('strategy').loc[strategies]
            y=np.arange(len(strategies))+(vi-1)*.16
            ax.scatter(sub[value]*scale,y,color=color,marker=marker,s=28,label=label,zorder=3)
        ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(np.arange(len(strategies)),[pretty(s) for s in strategies]);axes[0].invert_yaxis()
    axes[1].legend(loc='lower right',fontsize=9)
    fig.suptitle('Separate the boundary effect from probability-model drift',fontsize=16)
    save(fig,'geometry_controlled_comparison','All alternatives use radius 15 ft / grid 1 ft. Each version uses 50 shots / 10 threes for qualification; paired-player changes are also saved as CSV.')

def tradeoffs(summary):
    fig,axes=plt.subplots(2,3,figsize=(13,7.5))
    naive=summary[summary.strategy=='naive'].iloc[0].rho_attempt_rate
    for j,strategy in enumerate(FOCUS):
        sub=summary[(summary.strategy==strategy)&summary.radius.isin(RADII)]
        for i,(metric,vmin,vmax,fmt) in enumerate([('rho_attempt_rate',0,1,'.3f'),('fallback_rate',0,.6,'.1%')]):
            table=sub.pivot(index='radius',columns='grid',values=metric).reindex(index=RADII,columns=STEPS)
            ax=axes[i,j];im=ax.imshow(table,vmin=vmin,vmax=vmax,cmap='Blues' if i==0 else 'Oranges',aspect='auto')
            for r in range(3):
                for c in range(4):ax.text(c,r,format(table.iloc[r,c],fmt),ha='center',va='center',color='white' if table.iloc[r,c]>(vmax*.65) else '#172c3d',fontsize=10)
            ax.set_xticks(range(4),[str(s) for s in STEPS]);ax.set_yticks(range(3),[str(r) for r in RADII]);ax.set_xlabel('Grid spacing (ft)')
            if j==0:ax.set_ylabel(('Correlation\n' if i==0 else 'Naive fallback fraction\n')+'Search radius (ft)')
            if i==0:ax.set_title(pretty(strategy),fontsize=11)
    fig.suptitle(f'Radius changes both the metric and its reliance on fallback (naive rho = {naive:.3f})',fontsize=14)
    save(fig,'radius_resolution_tradeoffs','Full shot sample, fixed probability models. Lower correlation is not automatically better; fallback substitutes the naive value when no candidate qualifies.')

def convergence():
    data=pd.read_csv(WORK/'convergence.csv')
    fig,axes=plt.subplots(3,2,figsize=(11,10),sharex=True)
    for i,s in enumerate(FOCUS):
        for r,color in zip(RADII,[ORANGE,GREEN,INK]):
            sub=data[(data.strategy==s)&(data.radius==r)].sort_values('grid',ascending=False)
            for j,(col,title) in enumerate([('median_abs_change','Median player change'),('p95_abs_change','95th percentile player change')]):
                axes[i,j].plot(sub.grid,100*sub[col],'o-',color=color,label=f'{r:g} ft radius')
                axes[i,j].set_xscale('log');axes[i,j].invert_xaxis() if not axes[i,j].xaxis_inverted() else None
                axes[i,j].set_xticks([1,.5,.25],[1,.5,.25]);axes[i,j].minorticks_off();axes[i,j].grid(alpha=.15)
                axes[i,j].set_title(pretty(s)+'\n'+title,fontsize=10)
                axes[i,j].set_ylabel('Difference from 0.1-ft result (pp)')
    axes[0,1].legend(fontsize=9)
    for ax in axes[-1]:ax.set_xlabel('Grid spacing (ft; finer to the right)')
    fig.suptitle('Does finer spatial resolution materially change player dependence?',fontsize=15)
    save(fig,'resolution_convergence','Matched qualified players at each radius. pp = percentage points of dependence share. The 0.1-ft run is a numerical reference, not ground truth.')

def separation(summary,players):
    fig,axes=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True)
    for ax,s in zip(axes.ravel(),['naive']+FOCUS):
        p=players[(players.strategy==s)&players.qualified]
        m=summary[summary.strategy==s]
        if s!='naive':p=p[(p.radius==8)&(p.grid==.25)];m=m[(m.radius==8)&(m.grid==.25)]
        m=m.iloc[0]
        ax.scatter(p['3pa_rate'],100*p.dep_share,c=p.fallback_fraction if s!='naive' else np.zeros(len(p)),vmin=0,vmax=.5,cmap='cividis',s=16,alpha=.7)
        x=np.linspace(p['3pa_rate'].min(),p['3pa_rate'].max(),200)
        design=np.column_stack([p['3pa_rate']**i for i in range(4)])
        beta=np.linalg.lstsq(design,p.dep_share,rcond=None)[0]
        ax.plot(x,100*sum(beta[i]*x**i for i in range(4)),color=ORANGE,lw=1.6)
        ax.axhline(0,color=GREY,lw=.8)
        ax.set_title(f'{pretty(s)}\nrho={m.rho_attempt_rate:.3f}; cubic R²={m.cubic_r2:.2f}',fontsize=11)
        ax.set_xlabel('Three-point attempt rate');ax.set_ylabel('Dependence share (%)')
    fig.suptitle('Variation beyond volume must survive more than a linear comparison',fontsize=15)
    save(fig,'dependence_and_volume','Alternatives: 8-ft radius / 0.25-ft grid. Orange curves fit cubic attempt-rate trends. Dark-to-yellow points indicate increasing naive fallback (0–50%).')

def repeatability(summary,players):
    fig,axes=plt.subplots(2,2,figsize=(11,9))
    for ax,s in zip(axes.ravel(),['naive']+FOCUS):
        p=players[(players.strategy==s)&players.half_qualified]
        m=summary[summary.strategy==s]
        if s!='naive':p=p[(p.radius==8)&(p.grid==.25)];m=m[(m.radius==8)&(m.grid==.25)]
        m=m.iloc[0]
        ax.scatter(100*p.half0_residual,100*p.half1_residual,s=15,alpha=.6,color=INK)
        ax.axhline(0,color=GREY,lw=.8);ax.axvline(0,color=GREY,lw=.8)
        ax.set_title(f'{pretty(s)}\nRaw half rho={m.half_rho:.3f}; residual r={m.half_residual_r:.3f}',fontsize=11)
        ax.set_xlabel('Half A: dependence beyond cubic volume trend (pp)',fontsize=9)
        ax.set_ylabel('Half B: dependence beyond cubic volume trend (pp)',fontsize=9)
    fig.suptitle('Does the part beyond attempt rate repeat across game halves?',fontsize=15)
    save(fig,'beyond_volume_repeatability','8-ft radius / 0.25-ft grid. Baselines, including fallbacks, are fitted within each half. Shot models remain shared; this is not independent external validation.')

def travel(summary):
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    for s,color in zip(FOCUS,[ORANGE,INK,GREEN]):
        sub=summary[(summary.strategy==s)&(summary.grid==.25)&summary.radius.isin(RADII)].sort_values('radius')
        for ax,col in zip(axes,['median_travel','fallback_rate','rho_attempt_rate']):
            ax.plot(sub.radius,sub[col],'o-',color=color,label=pretty(s))
            ax.set_xticks(RADII);ax.set_xlabel('Search radius (ft)');ax.grid(alpha=.15)
    axes[0].set_ylabel('Median displacement among substitutes (ft)')
    axes[1].set_ylabel('Fraction of threes using naive fallback')
    axes[2].set_ylabel('Spearman correlation with attempt rate')
    axes[1].legend(fontsize=8)
    fig.suptitle('Search radius changes the question, not just numerical precision',fontsize=15)
    save(fig,'radius_tradeoffs','All points use a 0.25-ft grid. Travel excludes fallback shots; fallback and correlations use the full relevant sample.')

def all_strategies(summary):
    sub=summary[((summary.radius==8)&(summary.grid==.25))|(summary.strategy=='naive')].copy()
    fig,axes=plt.subplots(1,3,figsize=(14,8),sharey=True)
    y=np.arange(len(sub))
    for ax,col,label in zip(axes,['rho_attempt_rate','half_residual_r','fallback_rate'],['Correlation with attempt rate','Residual correlation across halves','Naive fallback fraction']):
        ax.barh(y,sub[col],color=[ORANGE if s=='naive' else INK for s in sub.strategy],height=.65)
        ax.axvline(0,color=GREY,lw=.8);ax.set_xlabel(label);ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(y,[pretty(s) for s in sub.strategy]);axes[0].invert_yaxis()
    fig.suptitle('All implemented variants: numerical distinction and repeatability',fontsize=15)
    save(fig,'all_strategy_diagnostics','8-ft radius / 0.25-ft grid. Residuals remove a cubic attempt-rate trend in each half. Low volume correlation alone does not establish a useful metric.')

if __name__=='__main__':
    s=pd.read_csv(WORK/'summary.csv');p=pd.read_csv(WORK/'players.csv')
    geometry();tradeoffs(s);convergence();separation(s,p);repeatability(s,p);travel(s);all_strategies(s)
    print(f'Wrote 7 comparison figures (PNG and SVG) to {OUT}')
