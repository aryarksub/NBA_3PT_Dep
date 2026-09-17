import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'experiments'))
import candidate_search_sweep as sweep
from counterfactual import nearest_feasible_two,best_feasible_two,recompute_features,MODEL_FEATURES

class Model:
    def get_booster(self):return self
    def set_param(self,params):pass
    def inplace_predict(self,X,**kwargs):
        X=np.asarray(X,dtype=np.float32)
        return 1/(1+np.exp(.08*X[:,2]-.12*X[:,4]))

def test_batched_search_matches_scalar_reference(tmp_path,monkeypatch):
    monkeypatch.setattr(sweep,'OUT',tmp_path)
    row={'shooter_x':5.25,'shooter_y':2.,'3pt':1,'seconds_rem':300.,'streak':1.,'shot_clock':10.,'home':1.,'def_hull_area':120.,'close_def_dist':3.}
    for i,(x,y) in enumerate([(5,6),(12,7),(14,20),(8,30),(20,40)],1):
        row[f'def{i}_x']=x;row[f'def{i}_y']=y
    empty=dict(row,shooter_x=47.,shooter_y=0.)
    shots=pd.DataFrame([row,empty]);model=Model()
    sweep.run_grid(shots,[model],np.array([0,0]),.5)
    data=np.load(tmp_path/'grid_0.5.npz')
    assert np.isnan(data['prob'][1]).all()
    assert not data['candidate_count'][1].any()
    def score(feats):return model.inplace_predict([[f[k] for k in MODEL_FEATURES] for f in feats])
    for ri,r in enumerate(data['radii']):
        for ki,k in enumerate(sweep.KAPPAS):
            result=nearest_feasible_two(row,kappa=k,cell_size=.5,max_radius=r)
            if result is None:assert np.isnan(data['prob'][0,ri,ki]);continue
            assert np.isclose(data['travel'][0,ri,ki],result['displacement'])
            assert np.isclose(data['prob'][0,ri,ki],score([recompute_features(row,result['x'],result['y'])])[0])
        for li,lam in enumerate(sweep.LAMBDAS):
            for qi,q in enumerate(sweep.QUANTILES):
                result=best_feasible_two(row,score,quantile=q,lam=lam,cell_size=.5,max_radius=r)
                si=4+li*3+qi
                assert np.isclose(data['prob'][0,ri,si],result['p_hat'])
                assert np.isclose(data['x'][0,ri,si],result['x'])
                assert np.isclose(data['y'][0,ri,si],result['y'])
    data.close()
    sweep.run_grid(shots, [model], np.array([0, 0]), .5)
    # A same-shape data update must not silently reuse old candidate predictions.
    shots.loc[0, 'shot_clock'] = 3.
    with pytest.raises(ValueError, match='Stale search checkpoint'):
        sweep.run_grid(shots, [model], np.array([0, 0]), .5)
