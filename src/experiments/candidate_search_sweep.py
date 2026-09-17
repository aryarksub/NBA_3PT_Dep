"""Full-sample radius/resolution sweep with bounded batches and resumable checkpoints.

Every candidate is scored; no shot subsampling. A spatial index accelerates exact
grid lookup. Larger-radius predictions are reused for nested smaller radii.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree
from counterfactual import half_court_twos, defender_positions, defender_features_vec, feature_matrix
from nba_geometry import is_three_vec

OUT=Path('results/search_sensitivity')
KAPPAS=[0.,.5,.75,1.]
LAMBDAS=[0.,.25,.5]
QUANTILES=[1.,.95,.9]
STRATEGIES=[f'near_k{int(k*100):03d}' for k in KAPPAS]+[f'best_q{int(q*100):03d}_l{int(l*100):03d}' for l in LAMBDAS for q in QUANTILES]
LEGACY=False

def legacy_three(pts):
    x,y=np.asarray(pts).T;dx=np.abs(x-np.where(x<47,5.25,88.75))
    return ((dx>=22)&(y<=14)) | (np.hypot(dx,y-25)>=23.75)

def search_grid(side,step):
    if not LEGACY:return half_court_twos(side,step)
    xs=np.arange(47 if side else 0,(94 if side else 47)+step,step)
    ys=np.arange(0,50+step,step)
    x,y=np.meshgrid(xs,ys);pts=np.column_stack([x.ravel(),y.ravel()])
    return pts[~legacy_three(pts)]

def checkpoint(path, **arrays):
    temp=path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**arrays)
    temp.replace(path)

def input_fingerprint(shots, models, folds, step):
    """Bind a checkpoint to its exact input, models, code, and search settings."""
    digest=hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(shots,index=True).to_numpy().tobytes())
    digest.update(json.dumps(list(shots.columns)).encode())
    digest.update(np.asarray(folds).tobytes())
    digest.update(repr((step,LEGACY,KAPPAS,LAMBDAS,QUANTILES)).encode())
    for filename in [__file__,Path(__file__).parents[1]/'counterfactual.py',Path(__file__).parents[1]/'nba_geometry.py']:
        digest.update(Path(filename).read_bytes())
    for model in models:
        booster=model.get_booster()
        if hasattr(booster,'save_raw'):
            digest.update(bytes(booster.save_raw()))
        else:  # Lightweight deterministic test models.
            digest.update(type(model).__qualname__.encode())
    return digest.hexdigest()


def run_grid(shots, models, folds, step):
    radii=np.array([5.,6.5,8.,15.] if step==1 else [5.,6.5,8.])
    positions=np.flatnonzero(shots['3pt'].to_numpy()==1)
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/f'grid_{step:g}.npz'
    shape=(len(positions),len(radii),len(STRATEGIES))
    fingerprint=input_fingerprint(shots,models,folds,step)
    if path.exists():
        cache=np.load(path)
        if 'fingerprint' not in cache or str(cache['fingerprint'])!=fingerprint:
            cache.close()
            raise ValueError(f'Stale search checkpoint: {path}. Archive or remove it before rerunning.')
        assert np.array_equal(cache['positions'],positions)
        prob,travel,xx,yy=[cache[k] for k in ['prob','travel','x','y']]
        count=cache['candidate_count'];done=cache['done']
        cache.close()
        if done.all():print(f'Grid {step:g}: complete cache',flush=True);return
    else:
        prob=np.full(shape,np.nan);travel=np.full(shape,np.nan)
        xx=np.full(shape,np.nan);yy=np.full(shape,np.nan)
        count=np.zeros((len(positions),len(radii)),dtype=np.int32)
        done=np.zeros(len(positions),dtype=bool)
    def save():checkpoint(path,fingerprint=np.array(fingerprint),positions=positions,radii=radii,strategies=np.array(STRATEGIES),prob=prob,travel=travel,x=xx,y=yy,candidate_count=count,done=done)
    grids=[search_grid(side,step) for side in [False,True]]
    trees=[cKDTree(g) for g in grids]
    rows=shots.iloc[positions].to_dict('records')
    started=time.time();before=int(done.sum());last_save=before
    for fold in range(len(models)):
        ids=np.flatnonzero((folds[positions]==fold)&~done)
        booster=models[fold].get_booster()
        booster.set_param({'nthread':4})
        batch_size=32 if step<=.1 else (64 if step<=.25 else 128)
        for start in range(0,len(ids),batch_size):
            batch=ids[start:start+batch_size]
            blocks=[];meta=[]
            for i in batch:
                row=rows[i];side=int(row['shooter_x']>=47)
                origin=np.array([row['shooter_x'],row['shooter_y']])
                indices=np.sort(np.asarray(trees[side].query_ball_point(origin,radii.max()),dtype=np.intp))
                pts=grids[side][indices]
                dist=np.linalg.norm(pts-origin,axis=1)
                masks=[dist<=r for r in radii]
                count[i]=[m.sum() for m in masks]
                if not len(pts):meta.append((i,None));continue
                assert not (legacy_three(pts) if LEGACY else is_three_vec(pts)).any()
                close,avg=defender_features_vec(defender_positions(row),pts)
                raw=feature_matrix(row,pts,close,avg).astype(np.float32)
                blocks.append(raw)
                meta.append((i,(pts,dist,close,masks,len(raw))))
            if blocks:
                matrix=np.concatenate(blocks)
                predictions=[]
                offsets=[];off=0
                for i,info in meta:
                    if info is None:continue
                    pts,dist,close,masks,n=info
                    offsets.append((i,info,off));off+=n
                for lam in LAMBDAS:
                    adjusted=matrix.copy()
                    for i,info,off in offsets:
                        pts,dist,close,masks,n=info
                        adjusted[off:off+n,4]=np.maximum(0.,close-lam*dist)
                    predictions.append(booster.inplace_predict(adjusted,validate_features=False))
                for i,info,off in offsets:
                    pts,dist,close,masks,n=info
                    # Nearest choice is made by distance; q is not used.
                    for ki,k in enumerate(KAPPAS):
                        allowed=np.flatnonzero(close>=k*rows[i]['close_def_dist'])
                        if not len(allowed):continue
                        j=allowed[np.argmin(dist[allowed])]
                        for ri,r in enumerate(radii):
                            if dist[j]<=r:
                                prob[i,ri,ki]=predictions[0][off+j]
                                travel[i,ri,ki]=dist[j];xx[i,ri,ki],yy[i,ri,ki]=pts[j]
                    for li,lam in enumerate(LAMBDAS):
                        p=predictions[li][off:off+n]
                        for ri,mask in enumerate(masks):
                            ix=np.flatnonzero(mask)
                            if not len(ix):continue
                            for qi,q in enumerate(QUANTILES):
                                vals=p[ix];target=np.quantile(vals,q)
                                j=ix[np.argmin(np.abs(vals-target))]
                                si=4+li*3+qi
                                prob[i,ri,si]=p[j];travel[i,ri,si]=dist[j]
                                xx[i,ri,si],yy[i,ri,si]=pts[j]
            done[batch]=True
            n_done=int(done.sum())
            if n_done-last_save>=512 or n_done==len(done):
                save();last_save=n_done
                rate=(n_done-before)/max(time.time()-started,1)
                print(f'grid={step:g}ft {n_done}/{len(done)} shots; {rate:.1f}/sec; remaining {(len(done)-n_done)/max(rate,.01)/60:.1f}min',flush=True)
    save()

def publish_reference(shots, step=1., radius=15.):
    cache=np.load(OUT/f'grid_{step:g}.npz')
    models=[]
    for i in range(5):
        model=xgb.XGBClassifier();model.load_model(OUT/f'fold_{i}.ubj');models.append(model)
    fingerprint=input_fingerprint(shots,models,np.load(OUT/'fold_of_row.npy'),step)
    if 'fingerprint' not in cache or str(cache['fingerprint'])!=fingerprint:
        cache.close()
        raise ValueError('Stale search checkpoint; regenerate it before publication.')
    assert cache['done'].all()
    positions=cache['positions'];ri=list(cache['radii']).index(radius)
    loc=pd.DataFrame({'pos':positions});diag=[]
    for si,strategy in enumerate(STRATEGIES):
        p=cache['prob'][:,ri,si];ok=np.isfinite(p)
        cf=shots.expected_points.to_numpy().copy()
        cf[positions]=np.where(ok,2*p,shots.exp_pts_naive.to_numpy()[positions])
        shots['exp_pts_'+strategy]=cf
        loc[strategy+'_x']=cache['x'][:,ri,si];loc[strategy+'_y']=cache['y'][:,ri,si]
        d=cache['travel'][:,ri,si]
        diag.append({'strategy':strategy,'n_3pa':len(positions),'n_substituted':int(ok.sum()),'fallback_rate':float((~ok).mean()),'median_displacement':float(np.nanmedian(d)),'p90_displacement':float(np.nanpercentile(d,90)),'support_rejection_rate':0.,'no_candidate_rate':float((cache['candidate_count'][:,ri]==0).mean()),'mean_candidates_per_shot':float(cache['candidate_count'][:,ri].mean())})
    keep=['player_id','game_id','team_id','3pt','shooter_x','shooter_y','expected_points','exp_pts_naive']+['exp_pts_'+s for s in STRATEGIES]
    shots[keep].to_csv('data/cf_alternatives_shots.csv',index=False)
    loc.to_csv('data/cf_alternatives_locations.csv',index=False)
    pd.DataFrame(diag).to_csv('results/cf_alternatives_diagnostics.csv',index=False)
    from cf_alternatives import cache_signature
    signature=cache_signature();signature.update(grid_ft=step,radius_ft=radius)
    Path('data/cf_alternatives_shots.csv.metadata.json').write_text(json.dumps(signature,indent=2))
    cache.close()
    print(f'Published corrected {step:g}ft grid / {radius:g}ft radius alternative shots.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--steps',nargs='+',type=float,default=[1.,.5,.25,.1]);ap.add_argument('--legacy',action='store_true');args=ap.parse_args()
    model_dir=OUT
    if args.legacy:LEGACY=True;OUT=OUT/'legacy'
    shots=pd.read_csv(OUT/'alt_exp_pts.csv' if LEGACY else 'data/alt_exp_pts.csv');folds=np.load(model_dir/'fold_of_row.npy')
    models=[]
    for i in range(5):
        m=xgb.XGBClassifier();m.load_model(model_dir/f'fold_{i}.ubj');models.append(m)
    for step in args.steps:
        run_grid(shots,models,folds,step)
