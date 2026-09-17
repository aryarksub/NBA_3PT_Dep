"""Verify current 3PT geometry and draw an audit without changing pipeline inputs."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

sys.path.insert(0, str(ROOT / 'src'))
from counterfactual import is_three, is_three_vec

OUT = ROOT / 'plots' / 'geometry_audit'
OUT.mkdir(exist_ok=True)
RESULT = ROOT / 'results' / 'geometry_audit'
RESULT.mkdir(exist_ok=True)


def correct_three(pts):
    """Idealized court coordinates; points on the boundary are treated as twos.

    Sideline boundaries are y=3 and y=47, joined to the 23.75-ft arc.
    This audits coordinates, not the shooter's feet or the recorded scoring decision.
    """
    x, y = np.asarray(pts, dtype=float).T
    hoop_x = np.where(x < 47, 5.25, 88.75)
    return (np.abs(y - 25) > 22) | (np.hypot(x - hoop_x, y - 25) > 23.75)


def fold(pts):
    x, y = np.asarray(pts, dtype=float).T
    right = x >= 47
    return np.column_stack([np.where(right, 94-x, x), np.where(right, 50-y, y)])


# Reproduce the archived boundary; verify corrected production against current inputs.
def legacy_three(pts):
    x,y=np.asarray(pts,dtype=float).T
    hx=np.where(x<47,5.25,88.75)
    return np.hypot(x-hx,y-25)>=23.75

# Compact, versioned historical results make this comparison reproducible without raw data.
summary=json.loads((RESULT/'geometry_verification.json').read_text())
counts=pd.read_csv(RESULT/'geometry_substitute_counts.csv')
affected=pd.read_csv(RESULT/'affected_shot_locations.csv').to_numpy()
assert correct_three(affected).all() and not legacy_three(affected).any()
assert len(affected)==summary['two_to_three']

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(19,8.6),gridspec_kw={'width_ratios':[1,1,1.05]})
fig.subplots_adjust(left=.045,right=.98,bottom=.23,top=.80,wspace=.55)
navy='#174d70'; orange='#cb4b16'; muted='#7e8b94'

def court(ax):
    # Plot width horizontally and distance from baseline vertically; equal feet scaling.
    ax.plot([0,50,50,0,0],[0,0,47,47,0],color='#66717a',lw=1)
    ax.plot([17,17,33,33],[0,19,19,0],color='#d3d9de',lw=1)
    ax.add_patch(Circle((25,5.25),.75,fill=False,ec='#66717a',lw=1.4))
    ax.plot([22,28],[4,4],color='#66717a',lw=1.4)
    yy=np.linspace(0,50,600); xx=np.linspace(0,47,600)
    Y,X=np.meshgrid(yy,xx)
    xy=np.column_stack([X.ravel(),Y.ravel()])
    error=(correct_three(xy)&~legacy_three(xy)).reshape(X.shape)
    ax.contourf(Y,X,error,levels=[.5,1.5],colors=['#f7d9ca'],alpha=.85,zorder=0)
    angle=np.linspace(0,2*np.pi,1500)
    w=25+23.75*np.cos(angle); d=5.25+23.75*np.sin(angle)
    use=(w>=0)&(w<=50)&(d>=0)&(d<=47)
    ax.plot(np.where(use,w,np.nan),np.where(use,d,np.nan),'--',color=muted,lw=2,zorder=4)
    a=np.arccos(22/23.75)
    theta=np.linspace(a,np.pi-a,500)
    ax.plot(25+23.75*np.cos(theta),5.25+23.75*np.sin(theta),color=navy,lw=2.4,zorder=5)
    join=5.25+np.sqrt(23.75**2-22**2)
    ax.plot([3,3],[0,join],color=navy,lw=2.4,zorder=5)
    ax.plot([47,47],[0,join],color=navy,lw=2.4,zorder=5)
    ax.set(xlim=(-1,51),ylim=(-1,48),xlabel='Court width y (feet)',ylabel='Distance from baseline (feet)')
    ax.set_xticks([0,3,25,47,50]); ax.set_xticklabels(['0','3','25','47','50'],fontsize=9)
    ax.set_aspect('equal')

court(axes[0]); court(axes[1])
axes[0].set_title('A  The old corner lines were missing',loc='left',pad=16,fontsize=13)
axes[0].text(25,37,'The archived rule reduces to a circle.\nOrange regions are outside the NBA line\nbut classified as two-point territory.',ha='center',va='center',fontsize=10)
axes[0].scatter([2],[5.25],marker='*',s=120,color=orange,zorder=7)
axes[0].annotate('Example: x = 5.25, y = 2\nArchived code: TWO\nCorrect geometry: THREE',xy=(2,5.25),xytext=(11,24),fontsize=9,color=orange,arrowprops={'arrowstyle':'->','color':orange},bbox={'facecolor':'white','edgecolor':'none','alpha':.9},zorder=8)
folded=fold(affected)
axes[1].scatter(folded[:,1],folded[:,0],s=4,alpha=.18,color=orange,rasterized=True,zorder=2)
axes[1].text(25,37,f"{summary['two_to_three']:,} of {summary['saved_shots']:,} shots ({summary['fraction_affected']:.2%})\nchange from two to three geometrically.\nBoth court ends are folded together.",ha='center',va='center',fontsize=10)
axes[1].set_title('B  Affected recorded shot locations',loc='left',pad=16,fontsize=13)
axes[1].set_ylabel('')

labels=[]
for name in counts.strategy:
    if name.startswith('near'): labels.append(f'Nearest | k = {int(name[6:])/100:g}')
    else: labels.append(f'Best | q = {int(name[6:9])/100:g}, L = {int(name[11:])/100:g}')
yy=np.arange(len(counts))
axes[2].barh(yy,100*counts.fraction,color=[orange if s.startswith('near') else navy for s in counts.strategy],height=.65)
axes[2].set_yticks(yy,labels,fontsize=9)
for i,row in counts.iterrows():
    axes[2].text(100*row.fraction+.09,i,f'{row.fraction:.2%}',va='center',fontsize=9)
axes[2].invert_yaxis(); axes[2].set_xlim(0,8)
axes[2].set_xlabel('Substitutes strictly beyond correct line (%)',fontsize=10)
axes[2].set_title('C  “Two” substitutes that are threes',loc='left',pad=16,fontsize=13)
axes[2].grid(axis='x',alpha=.15); axes[2].set_axisbelow(True)
fig.suptitle('Verified geometry error: old boundary missed corner threes',x=.045,ha='left',fontsize=21,fontweight='bold',y=.965)
fig.text(.045,.905,'Archived code uses distance along the court for the corner test; NBA corner lines are 3 feet from each sideline.',fontsize=12,color='#425563')
fig.legend(handles=[Line2D([0],[0],color=navy,lw=2.4,label='Correct NBA boundary'),Line2D([0],[0],color=muted,lw=2,ls='--',label='Archived code boundary'),Line2D([0],[0],color=orange,marker='s',lw=0,label='Missed corner-three region / affected locations')],loc='lower left',bbox_to_anchor=(.04,.12),ncol=3,frameon=False,fontsize=10)
fig.text(.045,.085,'Scope: naive baseline, nearest feasible, and best feasible alternatives. Candidate percentages exclude fallback shots.',fontsize=10)
fig.text(.045,.057,'Geometry audit only: tracked locations are not foot positions. Points exactly on the idealized line count as twos; observed scoring labels still require reconciliation.',fontsize=9,color='#53626e')
fig.text(.045,.029,'Source: NBA Rule 1, Section I(d)  |  Data: saved 2015–16 sample  |  k = openness multiplier; q = prediction quantile; L = closing penalty',fontsize=9,color='#53626e')
fig.savefig(OUT/'three_point_geometry_verification.png',dpi=180,facecolor='white')
fig.savefig(OUT/'three_point_geometry_verification.svg',facecolor='white')
plt.close(fig)
print(json.dumps(summary,indent=2))
print(OUT/'three_point_geometry_verification.png')

