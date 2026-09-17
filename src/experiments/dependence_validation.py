"""Three tests of information beyond attempt rate; see docs/dependence_validation.md.

Run prepare_temporal_validation.py first. No outcome-driven tuning or strategy selection.
"""
from pathlib import Path
import sys
import json
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from metric_validation import match_controls, controls, fit_predict, rmse, shuffled_values, holm_adjust
from prepare_temporal_validation import load_inputs

OUT = Path('results/metric_validation')
STRATEGIES = ['naive'] + [f'near_k{k:03d}' for k in [0,50,75,100]] + [
    f'best_q{q:03d}_l{l:03d}' for l in [0,25,50] for q in [100,95,90]]


def correlation(a, b):
    return float(np.corrcoef(a, b)[0,1]) if min(np.std(a),np.std(b)) > 0 else np.nan


class Panel:
    def __init__(self, shots, pbp, blocks, probability):
        self.ids, self.player = np.unique(shots.player_id, return_inverse=True)
        self.m = len(self.ids); self.blocks = np.asarray(blocks)
        self.index = self.player*3+self.blocks
        self.three = shots['3pt'].to_numpy().astype(bool)
        self.pos = np.flatnonzero(self.three); self.i3 = self.index[self.pos]
        self.ep = probability*(2+self.three)
        self.n = self.sum(np.ones(len(shots)))
        self.n3 = self.sum(self.three)
        self.total = self.sum(self.ep)
        self.real = self.sum(pbp.fgm.to_numpy()*(2+pbp['3pt'].to_numpy()))
        two_count = self.sum(~self.three)
        two_ep = self.sum(self.ep*(~self.three))
        self.base = np.divide(two_ep,two_count,out=np.zeros_like(two_ep),where=two_count>0)
        for b in range(3):
            keep=(self.blocks==b)&~self.three
            if keep.any(): self.base[two_count[:,b]==0,b] = self.ep[keep].mean()
        self.naive = self.base.ravel()[self.i3]

    def sum(self, values):
        return np.bincount(self.index, weights=values, minlength=self.m*3).reshape(self.m,3)

    def aggregate(self, cf, available):
        delta = np.bincount(self.i3,weights=self.ep[self.pos]-cf,minlength=self.m*3).reshape(self.m,3)
        cf_total = np.bincount(self.i3,weights=cf,minlength=self.m*3).reshape(self.m,3)
        fb = np.bincount(self.i3,weights=~available,minlength=self.m*3).reshape(self.m,3)
        def divide(a,b): return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b>0)
        return {'n':self.n,'n3':self.n3,'rate':divide(self.n3,self.n),
            'mean_ep':divide(self.total,self.n),'real_ppfga':divide(self.real,self.n),
            'share':divide(delta,self.total),'edge':divide(delta,self.n3),
            'mean_cf':divide(cf_total,self.n3),'fallback':divide(fb,self.n3)}


def matched_analysis(shots, pbp):
    panel = Panel(shots,pbp,np.zeros(len(shots),int),shots.shot_prob.to_numpy())
    cf_frame = pd.read_csv('data/cf_alternatives_shots.csv')
    locations = pd.read_csv('data/cf_alternatives_locations.csv')
    assert np.array_equal(locations.pos,panel.pos)
    assert np.array_equal(cf_frame.player_id,shots.player_id)
    qualified=(panel.n[:,0]>=50)&(panel.n3[:,0]>=10)
    ids=np.flatnonzero(qualified)
    raw=panel.aggregate(panel.naive,np.ones(len(panel.pos),bool))
    pairs=match_controls(raw['rate'][ids,0],raw['mean_ep'][ids,0])
    games,g=np.unique(shots.game_id,return_inverse=True)
    gi=g*panel.m+panel.player
    totals=np.bincount(gi,weights=panel.ep,minlength=len(games)*panel.m).reshape(len(games),panel.m)
    rng=np.random.default_rng(107)
    weights=rng.multinomial(len(games),np.full(len(games),1/len(games)),size=500)
    denominator=weights@totals
    rows=[];summary=[];players=[];bands=[]
    for strategy in STRATEGIES:
        cf=cf_frame['exp_pts_'+strategy].to_numpy()[panel.pos]
        ok=np.ones(len(cf),bool) if strategy=='naive' else locations[strategy+'_x'].notna().to_numpy()
        stats=panel.aggregate(cf,ok)
        values=np.zeros(len(shots));values[panel.pos]=panel.ep[panel.pos]-cf
        game_delta=np.bincount(gi,weights=values,minlength=len(games)*panel.m).reshape(len(games),panel.m)
        numerator=weights@game_delta
        draws=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
        gaps=[];excluded=[]
        for rank,(ii,jj,distance) in enumerate(pairs):
            i,j=ids[ii],ids[jj]
            gap=stats['share'][i,0]-stats['share'][j,0]
            lo,hi=np.quantile(draws[:,i]-draws[:,j],[.025,.975])
            row={'strategy':strategy,'pair_order':rank,'player_a':panel.ids[i],'player_b':panel.ids[j],
                'control_distance':distance,'rate_a':stats['rate'][i,0],'rate_b':stats['rate'][j,0],
                'efficiency_a':stats['mean_ep'][i,0],'efficiency_b':stats['mean_ep'][j,0],
                'share_gap':gap,'ci_lower':lo,'ci_upper':hi,'edge_a':stats['edge'][i,0],
                'edge_b':stats['edge'][j,0],'replacement_a':stats['mean_cf'][i,0],
                'replacement_b':stats['mean_cf'][j,0],'fallback_a':stats['fallback'][i,0],
                'fallback_b':stats['fallback'][j,0]}
            rows.append(row);gaps.append(abs(gap));excluded.append(lo>0 or hi<0)
        summary.append({'strategy':strategy,'pairs':len(pairs),'median_abs_gap':np.median(gaps),
            'p90_abs_gap':np.quantile(gaps,.9),'fraction_intervals_excluding_zero':np.mean(excluded)})
        for i in ids:
            players.append({'strategy':strategy,'player_id':panel.ids[i],**{k:v[i,0] for k,v in stats.items()}})
        for start in np.arange(0,1,.05):
            ix=ids[(stats['rate'][ids,0]>=start)&(stats['rate'][ids,0]<start+.05)]
            if len(ix): bands.append({'strategy':strategy,'band_lower':start,'n':len(ix),
                'q25':np.quantile(stats['share'][ix,0],.25),'median':np.median(stats['share'][ix,0]),
                'q75':np.quantile(stats['share'][ix,0],.75)})
    for name,data in [('matched_pairs',rows),('matched_summary',summary),('matched_players',players),('attempt_bands',bands)]:
        pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
    print(f'Matched comparison: {len(pairs)} disjoint pairs.',flush=True)


class ForwardTest:
    def __init__(self, panel, initial):
        self.panel=panel
        self.x=[controls({k:v[:,b] for k,v in initial.items()}) for b in range(3)]
        q=(initial['n']>=25)&(initial['n3']>=5)
        self.train=q[:,0]&(initial['n'][:,1]>=25)
        self.test=q[:,1]&(initial['n'][:,2]>=25)
        self.repeat=q[:,1]&q[:,2]
        self.early=q[:,0]
        self.y_train=initial['real_ppfga'][self.train,1]
        self.y_test=initial['real_ppfga'][self.test,2]
        self.baseline=fit_predict(self.x[0][self.train],self.y_train,self.x[1][self.test])

    def evaluate(self, stats, feature='share'):
        x0=np.column_stack([self.x[0],stats[feature][:,0]])
        x1=np.column_stack([self.x[1],stats[feature][:,1]])
        prediction=fit_predict(x0[self.train],self.y_train,x1[self.test])
        # Fit the dependence trend on early players only; apply unchanged to both later blocks.
        trend=[]
        for b in range(3):
            r=stats['rate'][:,b]
            trend.append(np.column_stack([r,r*r,r*r*r,stats['mean_ep'][:,b]]))
        predicted=fit_predict(trend[0][self.early],stats['share'][self.early,0],
                              np.concatenate([trend[1][self.repeat],trend[2][self.repeat]]))
        a,b=np.split(predicted,2)
        ra=stats['share'][self.repeat,1]-a;rb=stats['share'][self.repeat,2]-b
        return prediction,ra,rb


def permutation_groups(shots, blocks, prob, available, player):
    pos=np.flatnonzero(shots['3pt'].to_numpy()==1)
    # Fixed, predeclared bins; no outcome-dependent cutoffs or player identity strata.
    values=np.column_stack([blocks[pos],np.digitize(prob[pos],[.35,.45,.55,.65]),
        np.digitize(shots.shot_dist.to_numpy()[pos],[23.75,26,30]),
        np.digitize(shots.close_def_dist.to_numpy()[pos],[4,8]),
        np.digitize(shots.shot_clock.to_numpy()[pos],[4,10]),
        (np.abs(shots.shooter_y.to_numpy()[pos]-25)>22).astype(int)])
    _,codes=np.unique(values,axis=0,return_inverse=True)
    groups=[]
    for c in np.unique(codes):
        ix=np.flatnonzero((codes==c)&available)
        if len(ix)>=2 and len(np.unique(player[pos[ix]]))>=2:groups.append(ix)
    return groups


def temporal_analysis(shots,pbp,blocks,n_placebo):
    prob=np.load(OUT/'cache/prob.npy')
    panel=Panel(shots,pbp,blocks,prob)
    with np.load(OUT/'cache/grid_0.25.npz') as cache:
        assert cache['done'].all() and np.array_equal(cache['positions'],panel.pos)
        ri=list(cache['radii']).index(8.)
        candidates=cache['prob'][:,ri,:].copy()
        names=list(cache['strategies'])
    initial=panel.aggregate(panel.naive,np.ones(len(panel.pos),bool))
    test=ForwardTest(panel,initial)
    rng=np.random.default_rng(902)
    bootstrap=rng.integers(0,len(test.y_test),(1000,len(test.y_test)))
    outcome=[];predictions=[];residuals=[];period_players=[];nulls=[];summaries=[]
    base_error=rmse(test.y_test,test.baseline)
    pd.DataFrame([{'baseline_rmse':base_error,
        'constant_training_mean_rmse':rmse(test.y_test,np.full(len(test.y_test),test.y_train.mean())),
        'train_players':int(test.train.sum()),'test_players':int(test.test.sum())}]).to_csv(OUT/'forecast_benchmark.csv',index=False)
    for strategy in STRATEGIES:
        if strategy=='naive':cf=panel.naive.copy();available=np.ones(len(cf),bool)
        else:
            p=candidates[:,names.index(strategy)];available=np.isfinite(p)
            cf=np.where(available,2*p,panel.naive)
        stats=panel.aggregate(cf,available)
        for b in range(3):
            for i,player in enumerate(panel.ids):
                period_players.append({'strategy':strategy,'block':b,'player_id':player,
                                       **{k:v[i,b] for k,v in stats.items()}})
        for feature in ['share','edge']:
            pred,ra,rb=test.evaluate(stats,feature)
            gain=base_error-rmse(test.y_test,pred)
            draws=np.sqrt(np.mean((test.y_test[bootstrap]-test.baseline[bootstrap])**2,axis=1))-np.sqrt(np.mean((test.y_test[bootstrap]-pred[bootstrap])**2,axis=1))
            lo,hi=np.quantile(draws,[.025,.975])
            outcome.append({'strategy':strategy,'feature':feature,'train_players':int(test.train.sum()),
                'test_players':int(test.test.sum()),'baseline_rmse':base_error,'augmented_rmse':rmse(test.y_test,pred),
                'rmse_improvement':gain,'ci_lower':lo,'ci_upper':hi})
            for i,player in enumerate(panel.ids[test.test]):
                predictions.append({'strategy':strategy,'feature':feature,'player_id':player,
                    'outcome':test.y_test[i],'baseline':test.baseline[i],'augmented':pred[i]})
            if feature=='share':
                observed_r=correlation(ra,rb);observed_gain=gain
                for i,player in enumerate(panel.ids[test.repeat]):
                    residuals.append({'strategy':strategy,'player_id':player,'middle_residual':ra[i],'late_residual':rb[i]})
        groups=permutation_groups(shots,blocks,prob,available,panel.player)
        perm_rng=np.random.default_rng(711)
        nr=[];ng=[]
        for draw in range(n_placebo):
            shuffled=shuffled_values(cf,groups,perm_rng)
            st=panel.aggregate(shuffled,available)
            pred,a,b=test.evaluate(st)
            r=correlation(a,b);gain=base_error-rmse(test.y_test,pred)
            nr.append(r);ng.append(gain)
            nulls.append({'strategy':strategy,'draw':draw,'residual_r':r,'rmse_improvement':gain})
        summaries.append({'strategy':strategy,'repeat_players':int(test.repeat.sum()),
            'observed_residual_r':observed_r,'observed_rmse_improvement':observed_gain,
            'null_r_median':np.median(nr),'null_r_lower':np.quantile(nr,.025),'null_r_upper':np.quantile(nr,.975),
            'null_gain_median':np.median(ng),'null_gain_lower':np.quantile(ng,.025),'null_gain_upper':np.quantile(ng,.975),
            'permutation_tail_r':(1+np.sum(np.array(nr)>=observed_r))/(n_placebo+1),
            'permutation_tail_gain':(1+np.sum(np.array(ng)>=observed_gain))/(n_placebo+1),
            'eligible_shuffle_fraction':sum(map(len,groups))/len(cf),'strata':len(groups),'draws':n_placebo})
        print(f'{strategy}: future RMSE gain={observed_gain:.5f}; residual r={observed_r:.3f}; placebo done.',flush=True)
    adjusted_r=holm_adjust([s['permutation_tail_r'] for s in summaries])
    adjusted_gain=holm_adjust([s['permutation_tail_gain'] for s in summaries])
    for i,s in enumerate(summaries):
        s['holm_tail_r']=adjusted_r[i];s['holm_tail_gain']=adjusted_gain[i]
    for name,data in [('prediction_summary',outcome),('predictions',predictions),('temporal_residuals',residuals),
                      ('period_players',period_players),('placebo_draws',nulls),('placebo_summary',summaries)]:
        pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--placebos',type=int,default=500)
    args=parser.parse_args()
    if args.placebos<20:parser.error('Use at least 20 placebo draws')
    OUT.mkdir(parents=True,exist_ok=True)
    shots,pbp,blocks=load_inputs()
    matched_analysis(shots,pbp)
    temporal_analysis(shots,pbp,blocks,args.placebos)
    (OUT/'design.json').write_text(json.dumps({'date_fractions':[.4,.3,.3],
        'matching_rate_caliper':.02,'matching_ep_caliper':.05,'matched_game_bootstraps':500,
        'future_player_bootstraps':1000,'placebo_draws':args.placebos,'ridge_alpha':1,
        'random_seeds':{'matched_bootstrap':107,'forecast_bootstrap':902,'placebo':711},
        'placebo_cutoffs':{'make_probability':[.35,.45,.55,.65],
            'shot_distance_ft':[23.75,26,30],'closest_defender_ft':[4,8],'shot_clock_seconds':[4,10]},
        'holm_family':'14 strategies, separately for residual repeatability and forecast gain',
        'primary_outcome':'future realized points per tracked FGA, source PBP shot values',
        'shot_model_training':'early block only; early predictions cross-fitted by game',
        'predictor_fit':'early features -> middle outcome; evaluated middle features -> late outcome'},indent=2))


if __name__=='__main__':main()
