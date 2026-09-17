"""Evaluate geometry, radius and resolution effects on dependence beyond volume."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,rankdata

WORK=Path('results/search_sensitivity')
ARCHIVE=Path('results/geometry_before_20260912')
STAT='dep_share'

def corr(x,y,rank=True):
    x=np.asarray(x);y=np.asarray(y);keep=np.isfinite(x)&np.isfinite(y)
    if keep.sum()<3 or np.std(x[keep])==0 or np.std(y[keep])==0:return np.nan
    return float(spearmanr(x[keep],y[keep]).statistic if rank else np.corrcoef(x[keep],y[keep])[0,1])

def residual(y,x):
    # A nonlinear attempt-rate comparison avoids equating linear residuals with novelty.
    design=np.column_stack([x**i for i in range(4)])
    return y-design@np.linalg.lstsq(design,y,rcond=None)[0]

class Population:
    def __init__(self,shots):
        self.s=shots
        self.players,self.group=np.unique(shots.player_id,return_inverse=True)
        self.n=len(self.players)
        self.pos=np.flatnonzero(shots['3pt'].to_numpy()==1)
        self.g3=self.group[self.pos]
        self.ep=shots.expected_points.to_numpy();self.ep3=self.ep[self.pos]
        self.naive=shots.exp_pts_naive.to_numpy()[self.pos]
        self.nshots=np.bincount(self.group,minlength=self.n)
        self.n3=np.bincount(self.g3,minlength=self.n)
        self.total=np.bincount(self.group,weights=self.ep,minlength=self.n)
        self.qual=(self.nshots>=50)&(self.n3>=10)
        games=np.sort(shots.game_id.unique());halves={g:i%2 for i,g in enumerate(games)}
        self.h=shots.game_id.map(halves).to_numpy();self.h3=self.h[self.pos]
        self.hgroup=self.group*2+self.h;self.hgroup3=self.g3*2+self.h3
        self.hn=np.bincount(self.hgroup,minlength=self.n*2).reshape(-1,2)
        self.hn3=np.bincount(self.hgroup3,minlength=self.n*2).reshape(-1,2)
        self.hep=np.bincount(self.hgroup,weights=self.ep,minlength=self.n*2).reshape(-1,2)
        two=shots['3pt'].to_numpy()==0
        bn=np.bincount(self.hgroup[two],minlength=self.n*2)
        be=np.bincount(self.hgroup[two],weights=self.ep[two],minlength=self.n*2)
        league=np.array([self.ep[two&(self.h==h)].mean() for h in [0,1]])
        self.hbase=np.divide(be,bn,out=np.tile(league,self.n),where=bn>0)[self.hgroup3]
        self.hqual=((self.hn>=25)&(self.hn3>=5)).all(axis=1)

    def evaluate(self,prob=None):
        available=np.zeros(len(self.pos),dtype=bool) if prob is None else np.isfinite(prob)
        cf=self.naive.copy() if prob is None else np.where(available,2*prob,self.naive)
        delta=self.ep3-cf
        dep=np.bincount(self.g3,weights=delta,minlength=self.n)/self.total
        rate=self.n3/self.nshots
        fallback=np.divide(np.bincount(self.g3,weights=~available,minlength=self.n),self.n3,out=np.zeros(self.n),where=self.n3>0)
        halfcf=self.hbase.copy() if prob is None else np.where(available,2*prob,self.hbase)
        hd=np.bincount(self.hgroup3,weights=self.ep3-halfcf,minlength=self.n*2).reshape(-1,2)
        hd=np.divide(hd,self.hep,out=np.full_like(hd,np.nan),where=self.hep>0)
        hr=np.divide(self.hn3,self.hn,out=np.zeros_like(hd),where=self.hn>0)
        hkeep=self.hqual
        e0=residual(hd[hkeep,0],hr[hkeep,0]);e1=residual(hd[hkeep,1],hr[hkeep,1])
        keep=self.qual;y=dep[keep];x=rate[keep]
        res=residual(y,x)
        metrics={'n_qualified':int(keep.sum()),'median_dep_share':float(np.median(y)),'league_dep_share':float(delta.sum()/self.ep.sum()),'rho_attempt_rate':corr(y,x),'linear_r2':corr(y,x,False)**2,'cubic_r2':float(1-np.var(res)/np.var(y)),'residual_sd':float(np.std(res,ddof=1)),'fallback_rate':float((~available).mean()) if prob is not None else np.nan,'n_half_players':int(hkeep.sum()),'half_rho':corr(hd[hkeep,0],hd[hkeep,1]),'half_residual_r':corr(e0,e1,False),'half_attempt_rate_rho':corr(hr[hkeep,0],hr[hkeep,1]),'fraction_negative':float((y<0).mean())}
        low=keep&(fallback<=.05)
        metrics['n_low_fallback']=int(low.sum())
        metrics['rho_low_fallback']=corr(dep[low],rate[low]) if low.sum()>=10 else np.nan
        # Conditional across-player uncertainty for association, not total metric uncertainty.
        rng=np.random.default_rng(42);ix=rng.integers(0,len(y),(400,len(y)))
        a=rankdata(y[ix],axis=1);b=rankdata(x[ix],axis=1)
        a-=a.mean(axis=1,keepdims=True);b-=b.mean(axis=1,keepdims=True)
        draws=(a*b).sum(axis=1)/np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
        metrics['rho_ci_lower'],metrics['rho_ci_upper']=np.quantile(draws,[.025,.975])
        detail=pd.DataFrame({'player_id':self.players,'n_shots':self.nshots,'n_3pa':self.n3,'qualified':keep,'3pa_rate':rate,'dep_share':dep,'fallback_fraction':fallback,'half0':hd[:,0],'half1':hd[:,1],'half0_rate':hr[:,0],'half1_rate':hr[:,1],'half_qualified':hkeep})
        detail['half0_residual']=np.nan;detail['half1_residual']=np.nan
        detail.loc[hkeep,'half0_residual']=e0;detail.loc[hkeep,'half1_residual']=e1
        return metrics,detail

def evaluate_all(include_geometry=False):
    shots=pd.read_csv('data/alt_exp_pts.csv');pop=Population(shots)
    stats=[];details=[];geometry=[]
    m,d=pop.evaluate();m.update(grid=np.nan,radius=np.nan,strategy='naive');stats.append(m)
    d['grid']=np.nan;d['radius']=np.nan;d['strategy']='naive';details.append(d)
    for step in [1.,.5,.25,.1]:
        path=WORK/f'grid_{step:g}.npz'
        with np.load(path) as cache:
            assert cache['done'].all(),f'{step}: unfinished sweep'
            for ri,r in enumerate(cache['radii']):
                for si,s in enumerate(cache['strategies']):
                    p=cache['prob'][:,ri,si]
                    m,d=pop.evaluate(p)
                    tr=cache['travel'][:,ri,si]
                    m.update(grid=step,radius=r,strategy=str(s),median_travel=float(np.nanmedian(tr)),p90_travel=float(np.nanpercentile(tr,90)),fraction_near_radius=float(np.mean(tr[np.isfinite(tr)]>=r-.5)),mean_candidate_count=float(cache['candidate_count'][:,ri].mean()))
                    stats.append(m)
                    d['grid']=step;d['radius']=r;d['strategy']=str(s);details.append(d)
        print(f'Evaluated {step:g}ft grid',flush=True)
    summary=pd.DataFrame(stats);players=pd.concat(details,ignore_index=True)
    summary.to_csv(WORK/'summary.csv',index=False);players.to_csv(WORK/'players.csv',index=False)
    # Matched-player convergence to the requested finest grid at each radius.
    convergence=[]
    for r in [5.,6.5,8.]:
        for s in summary.strategy.unique():
            if s=='naive':continue
            finest=players[(players.radius==r)&(players.grid==.1)&(players.strategy==s)&players.qualified].set_index('player_id')
            for step in [1.,.5,.25]:
                cur=players[(players.radius==r)&(players.grid==step)&(players.strategy==s)&players.qualified].set_index('player_id').loc[finest.index]
                diff=(cur.dep_share-finest.dep_share).abs()
                top=set(cur.nlargest(20,'dep_share').index);ref=set(finest.nlargest(20,'dep_share').index)
                convergence.append({'radius':r,'grid':step,'strategy':s,'median_abs_change':diff.median(),'p95_abs_change':diff.quantile(.95),'max_abs_change':diff.max(),'rank_rho':corr(cur.dep_share,finest.dep_share),'top20_overlap':len(top&ref)/20})
    pd.DataFrame(convergence).to_csv(WORK/'convergence.csv',index=False)
    if not include_geometry:
        print("Updated current sweep tables; retained published historical geometry tables.")
        return
    # Isolate geometry with identical refitted probabilities; retain archive comparison too.
    controls=[('as_saved',pd.read_csv(ARCHIVE/'data'/'alt_exp_pts.csv'),None),('old_geometry_same_model',pd.read_csv(WORK/'legacy'/'alt_exp_pts.csv'),WORK/'legacy'/'grid_1.npz'),('corrected_geometry',shots,WORK/'grid_1.npz')]
    geom_players=[]
    for name,frame,path in controls:
        pp=Population(frame);m,d=pp.evaluate();m.update(version=name,strategy='naive');geometry.append(m)
        d['version']=name;d['strategy']='naive';geom_players.append(d)
        if path:
            with np.load(path) as cache:
                assert cache['done'].all()
                ri=list(cache['radii']).index(15.)
                for si,s in enumerate(cache['strategies']):
                    m,d=pp.evaluate(cache['prob'][:,ri,si]);m.update(version=name,strategy=str(s));geometry.append(m)
                    d['version']=name;d['strategy']=str(s);geom_players.append(d)
        else:
            oldcf=pd.read_csv(ARCHIVE/'data'/'cf_alternatives_shots.csv')
            oldloc=pd.read_csv(ARCHIVE/'data'/'cf_alternatives_locations.csv')
            assert np.array_equal(oldloc.pos.to_numpy(),pp.pos)
            for s in summary.strategy.unique():
                if s=='naive':continue
                p=oldcf['exp_pts_'+s].to_numpy()[pp.pos]/2
                p[oldloc[s+'_x'].isna().to_numpy()]=np.nan
                m,d=pp.evaluate(p);m.update(version=name,strategy=s);geometry.append(m)
                d['version']=name;d['strategy']=s;geom_players.append(d)
    pd.DataFrame(geometry).to_csv(WORK/'geometry_comparison.csv',index=False)
    gp=pd.concat(geom_players,ignore_index=True);gp.to_csv(WORK/'geometry_players.csv',index=False)
    paired=[]
    for strategy in summary.strategy.unique():
        old=gp[(gp.version=='old_geometry_same_model')&(gp.strategy==strategy)&gp.qualified].set_index('player_id')
        new=gp[(gp.version=='corrected_geometry')&(gp.strategy==strategy)&gp.qualified].set_index('player_id')
        ids=old.index.intersection(new.index)
        diff=new.loc[ids,'dep_share']-old.loc[ids,'dep_share']
        paired.append({'strategy':strategy,'common_players':len(ids),'median_signed_change':diff.median(),'median_abs_change':diff.abs().median(),'p95_abs_change':diff.abs().quantile(.95),'rank_agreement':corr(old.loc[ids,'dep_share'],new.loc[ids,'dep_share'])})
    pd.DataFrame(paired).to_csv(WORK/'geometry_paired_changes.csv',index=False)
    print(summary[['grid','radius','strategy','rho_attempt_rate','fallback_rate','half_residual_r']].to_string(index=False))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--geometry',action='store_true',help='Recompute historical comparisons; requires local archived inputs.')
    evaluate_all(parser.parse_args().geometry)
