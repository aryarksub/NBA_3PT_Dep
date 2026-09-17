import sys
from pathlib import Path
import numpy as np
import pandas as pd
from metric_validation import match_controls, shuffled_values, fit_predict, holm_adjust

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/experiments'))
from prepare_temporal_validation import chronological_blocks
from dependence_validation import Panel, ForwardTest


def test_matching_is_disjoint_and_respects_both_calipers():
    rate=np.array([.20,.205,.50,.51,.80])
    ep=np.array([1.,1.01,1.1,1.11,.5])
    pairs=match_controls(rate,ep)
    selected=[i for a,b,_ in pairs for i in [a,b]]
    assert len(selected)==len(set(selected))==4
    for a,b,_ in pairs:
        assert abs(rate[a]-rate[b])<=.02
        assert abs(ep[a]-ep[b])<=.05


def test_shuffle_preserves_strata_multisets_and_fallback_entries():
    x=np.arange(8.)
    y=shuffled_values(x,[np.array([0,2,4]),np.array([1,3])],np.random.default_rng(4))
    assert sorted(y[[0,2,4]])==[0,2,4]
    assert sorted(y[[1,3]])==[1,3]
    assert np.array_equal(y[[5,6,7]],x[[5,6,7]])


def test_dates_do_not_cross_blocks_and_future_rows_do_not_fit_scaling():
    dates=pd.Series(pd.date_range('2020-01-01',periods=20).repeat(2))
    blocks=chronological_blocks(dates)
    assert np.all(np.diff(blocks)>=0)
    assert np.array_equal(blocks[::2],blocks[1::2])
    x=np.arange(10.)[:,None];y=2*x[:,0]
    alone=fit_predict(x,y,np.array([[2.]]))
    batch=fit_predict(x,y,np.array([[2.],[1e10]]))
    assert np.allclose(alone,batch[:1])


def test_period_baselines_do_not_use_later_twos_and_identity_holds():
    shots=pd.DataFrame({'player_id':[1]*6,'3pt':[0,1,0,1,0,1]})
    pbp=pd.DataFrame({'fgm':[1]*6,'3pt':shots['3pt']})
    blocks=np.repeat(np.arange(3),2)
    p=np.array([.4,.4,.5,.5,.9,.9])
    panel=Panel(shots,pbp,blocks,p)
    assert np.allclose(panel.naive,[.8,1.,1.8])
    stats=panel.aggregate(panel.naive,np.ones(3,bool))
    assert np.allclose(stats['share'],stats['rate']*stats['edge']/stats['mean_ep'])
    other=Panel(shots,pbp,blocks,np.array([.4,.4,.5,.5,.1,.1]))
    assert np.array_equal(panel.naive[:2],other.naive[:2])


def test_holm_adjusts_the_whole_family_and_preserves_order():
    assert np.allclose(holm_adjust([.01,.04,.03]),[.03,.06,.06])


def test_final_outcomes_cannot_change_fitted_forecasts():
    rng=np.random.default_rng(10)
    initial={'n':np.full((12,3),50.),'n3':np.full((12,3),20.),
        'rate':rng.uniform(.2,.6,(12,3)),'mean_ep':rng.uniform(.8,1.2,(12,3)),
        'real_ppfga':rng.uniform(.8,1.2,(12,3)),'share':rng.uniform(.02,.1,(12,3))}
    before=ForwardTest(None,initial)
    prediction=before.evaluate(initial)[0]
    initial['real_ppfga'][:,2]=100.
    after=ForwardTest(None,initial)
    assert np.array_equal(before.baseline,after.baseline)
    assert np.array_equal(prediction,after.evaluate(initial)[0])
