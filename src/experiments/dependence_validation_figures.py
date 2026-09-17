"""Render information-beyond-volume figures using versioned result tables only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA=Path('results/metric_validation');OUT=Path('plots/metric_validation')
OUT.mkdir(parents=True,exist_ok=True)
FOCUS=['naive','near_k100','best_q100_l000','best_q095_l025']
BLUE='#245d7c';ORANGE='#c46235';GREY='#98a6ad'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False})


def label(s):
    if s=='naive':return 'Naive'
    if s.startswith('near'):return f'Nearest k={int(s[6:])/100:g}'
    return f'Best q={int(s[6:9])/100:g}, closing={int(s[11:])/100:g}'


def save(fig,name,footer):
    fig.tight_layout(rect=[0,.055,1,.94])
    fig.text(.025,.014,footer,fontsize=8,color='#44545e')
    fig.savefig(OUT/f'{name}.png',dpi=170,facecolor='white')
    fig.savefig(OUT/f'{name}.svg',facecolor='white')
    plt.close(fig)


def bands():
    data=pd.read_csv(DATA/'attempt_bands.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    for ax,s in zip(axes.ravel(),FOCUS):
        d=data[(data.strategy==s)&(data.n>=5)]
        x=(d.band_lower+.025)*100
        ax.fill_between(x,d.q25*100,d.q75*100,color=BLUE,alpha=.18,label='Player interquartile range')
        ax.plot(x,d['median']*100,'o-',color=BLUE,label='Median player')
        ax.set_title(label(s));ax.set_xlabel('Three-point attempt rate (%)')
        ax.set_ylabel('Dependence share (%)');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Variation among players in the same 5-percentage-point attempt-rate bands',fontsize=14)
    save(fig,'attempt_rate_bands','Full-sample estimates; bands with at least five players. Shading is player variation, not uncertainty. Efficiency is controlled in the separate matched-pair analysis.')


def pairs():
    data=pd.read_csv(DATA/'matched_pairs.csv')
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for ax,s in zip(axes.ravel(),FOCUS):
        d=data[(data.strategy==s)&(data.pair_order<10)].sort_values('pair_order')
        y=np.arange(len(d))
        ax.hlines(y,100*d.ci_lower,100*d.ci_upper,color=GREY,lw=2)
        ax.scatter(100*d.share_gap,y,color=BLUE,zorder=3)
        ax.axvline(0,color=ORANGE,ls='--');ax.invert_yaxis()
        ax.set_yticks(y,[f'{int(a)} / {int(b)}' for a,b in zip(d.player_a,d.player_b)],fontsize=8)
        ax.set_title(label(s));ax.set_xlabel('Player A minus B: dependence share (percentage points)')
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('Ten closest control-matched pairs, selected without looking at dependence',fontsize=14)
    save(fig,'matched_player_pairs','Rate difference <=2 pp; mean expected points/shot difference <=0.05. Bars: conditional 95% intervals from 500 global game resamples; no multiple-testing adjustment.')


def match_summary():
    data=pd.read_csv(DATA/'matched_summary.csv');y=np.arange(len(data))
    fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True)
    axes[0].barh(y,data.median_abs_gap*100,color=BLUE)
    axes[1].barh(y,data.fraction_intervals_excluding_zero*100,color=ORANGE)
    axes[0].set_yticks(y,[label(s) for s in data.strategy]);axes[0].invert_yaxis()
    axes[0].set_xlabel('Median absolute within-pair share difference (pp)')
    axes[1].set_xlabel('Pairs whose conditional interval excludes zero (%)')
    fig.suptitle(f'All {int(data.pairs.iloc[0])} disjoint pairs matched on attempt rate and expected efficiency',fontsize=14)
    save(fig,'matched_strategy_summary','Same control-selected pairs for every strategy. Interval exclusions are descriptive and unadjusted; they do not establish causal or practically useful differences.')


def prediction():
    data=pd.read_csv(DATA/'prediction_summary.csv');order=data.strategy.unique()
    fig,axes=plt.subplots(1,2,figsize=(13,8),sharey=True)
    for ax,feature,title in zip(axes,['share','edge'],['Add dependence share','Add advantage per three']):
        d=data[data.feature==feature].set_index('strategy').loc[order];y=np.arange(len(d))
        ax.hlines(y,d.ci_lower,d.ci_upper,color=GREY,lw=2)
        ax.scatter(d.rmse_improvement,y,color=BLUE,zorder=3)
        ax.axvline(0,color=ORANGE,ls='--');ax.set_title(title)
        ax.set_xlabel('Held-out RMSE improvement (positive = better)');ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(np.arange(len(order)),[label(s) for s in order]);axes[0].invert_yaxis()
    fig.suptitle('Does dependence improve prediction of future realized points per tracked FGA?',fontsize=14)
    save(fig,'forward_prediction',f'Baseline includes cubic attempt rate, expected efficiency, realized efficiency, and shot count. Baseline RMSE={data.baseline_rmse.iloc[0]:.4f}; n={int(data.test_players.iloc[0])}. Bars: paired player-bootstrap 95% intervals.')


def residuals():
    data=pd.read_csv(DATA/'temporal_residuals.csv');summary=pd.read_csv(DATA/'placebo_summary.csv').set_index('strategy')
    fig,axes=plt.subplots(2,2,figsize=(11,9))
    for ax,s in zip(axes.ravel(),FOCUS):
        d=data[data.strategy==s]
        ax.scatter(d.middle_residual*100,d.late_residual*100,s=13,alpha=.5,color=BLUE)
        ax.axhline(0,color=GREY,lw=.7);ax.axvline(0,color=GREY,lw=.7)
        ax.set_title(f'{label(s)}: residual r={summary.loc[s,"observed_residual_r"]:.3f}')
        ax.set_xlabel('Middle-block residual (pp)');ax.set_ylabel('Final-block residual (pp)')
    fig.suptitle('Does variation beyond attempt rate and efficiency persist forward in time?',fontsize=14)
    save(fig,'forward_residual_repeatability','The dependence trend is fitted only in the early block and applied unchanged later. Shot models train only on early games. Repeatability is not predictive or causal validity.')


def placebo():
    d=pd.read_csv(DATA/'placebo_summary.csv');y=np.arange(len(d))
    fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True)
    for ax,actual,median,lo,hi,title in zip(axes,
        ['observed_residual_r','observed_rmse_improvement'],['null_r_median','null_gain_median'],
        ['null_r_lower','null_gain_lower'],['null_r_upper','null_gain_upper'],
        ['Residual repeatability','Future prediction RMSE improvement']):
        ax.hlines(y,d[lo],d[hi],color=GREY,lw=3,label='Middle 95% of shuffled results')
        ax.scatter(d[median],y,color=GREY,s=22,label='Shuffled median')
        ax.scatter(d[actual],y,color=ORANGE,marker='D',s=35,label='Actual matching',zorder=4)
        ax.set_xlabel(title);ax.axvline(0,color=BLUE,lw=.7);ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(y,[label(s) for s in d.strategy]);axes[0].invert_yaxis()
    axes[1].legend(fontsize=8,loc='best')
    fig.suptitle('Does matching each shot to its actual alternative outperform matched-stratum shuffling?',fontsize=14)
    save(fig,'alternative_placebo',f'{int(d.draws.iloc[0])} shuffles per strategy, within time/context strata; fallback entries stay fixed. Grey ranges are placebo distributions, not confidence intervals for the actual metric.')


if __name__=='__main__':
    bands();pairs();match_summary();prediction();residuals();placebo()
    print('Wrote six validation figures as PNG and SVG.')
