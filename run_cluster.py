"""Run the final 12-cluster similar-shot probability model.

Court coordinates define 12 hard location clusters. Within the same cluster,
nearest shots are found using standardized shot distance and closest-defender
distance. Their exponentially weighted Train outcomes form the probability.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Shared reporting utilities and location-cluster configuration
# ---------------------------------------------------------------------------
HERE=Path(__file__).resolve().parent
# Import shared metric/plot writers without executing HGB training.
spec=importlib.util.spec_from_file_location("reporting",HERE/"run_hgb.py");reporting=importlib.util.module_from_spec(spec);spec.loader.exec_module(reporting)
SEED=42
# Three rim/paint + four non-rim 2PT + five 3PT clusters = 12 total.
GROUPS={"rim_paint_2pt": (3,"RP"),"non_rim_2pt":(4,"MR"),"three_point":(5,"3P")}


def enrich(d):
    """Fold both baskets onto one half court and assign a broad shot group."""
    # Reflect shots taken toward the right basket so both directions align.
    d=d.copy();right=d.shooter_x>47;d["folded_x"]=np.where(right,94-d.shooter_x,d.shooter_x);d["folded_y"]=np.where(right,50-d.shooter_y,d.shooter_y)
    # Separate 3PT shots, rim/paint 2PT shots, and other 2PT shots.
    d["location_group"]=np.select([d["3pt"].eq(1),d.shot_dist.le(8)],["three_point","rim_paint_2pt"],default="non_rim_2pt");return d


def clusters(train,test):
    """Fit group-specific KMeans models on Train and assign both splits."""
    models={}
    for group,(n,prefix) in GROUPS.items():
        # Test coordinates are transformed only by the Train-fitted KMeans model.
        a=train.location_group.eq(group);b=test.location_group.eq(group);m=KMeans(n_clusters=n,n_init=30,random_state=SEED).fit(train.loc[a,["folded_x","folded_y"]])
        train.loc[a,"location_cluster"]=[f"{prefix}_{i}" for i in m.labels_];test.loc[b,"location_cluster"]=[f"{prefix}_{i}" for i in m.predict(test.loc[b,["folded_x","folded_y"]])];models[group]=m
    return models


def probability(reference,target):
    """Estimate target probabilities from up to 500 same-cluster neighbors."""
    # Missing values, means, and standard deviations are learned from reference.
    cols=["shot_dist","close_def_dist"];median=reference[cols].median();a=reference[cols].fillna(median);b=target[cols].fillna(median);mean=a.mean();std=a.std(ddof=0)
    # The squared-distance weights are 0.40 for shot distance and 0.60 for
    # defender distance; multiplying z-scores by sqrt(weight) implements this.
    za=((a-mean)/std).to_numpy()*np.sqrt([.4,.6]);zb=((b-mean)/std).to_numpy()*np.sqrt([.4,.6]);out=np.empty(len(target));positions=pd.Series(np.arange(len(reference)),index=reference.index);tpos=pd.Series(np.arange(len(target)),index=target.index)
    for cluster,g in target.groupby("location_cluster"):
        # Candidate neighbors must share the exact hard location cluster.
        ref=reference[reference.location_cluster.eq(cluster)];k=min(500,len(ref));finder=NearestNeighbors(n_neighbors=k).fit(za[positions.loc[ref.index]]);distance,idx=finder.kneighbors(zb[tpos.loc[g.index]])
        outcomes=ref.fgm.to_numpy()[idx];weight=np.exp(-(distance**2));out[tpos.loc[g.index]]=(weight*outcomes).sum(1)/weight.sum(1)
    return out


def main():
    """Fit clusters, calculate Train/Test probabilities, and write outputs."""
    # The fixed split is read directly; this script never resamples rows.
    train,test=enrich(pd.read_csv(HERE/"train.csv")),enrich(pd.read_csv(HERE/"test.csv"));models=clusters(train,test)
    # Train probabilities use leave-one-out neighbors; Test probabilities use Train only.
    ptest=probability(train,test);ptrain=np.empty(len(train))
    for cluster,g in train.groupby("location_cluster"):
        # Request one extra neighbor, then remove the shot itself at distance zero.
        cols=["shot_dist","close_def_dist"];a=g[cols].fillna(g[cols].median());z=((a-a.mean())/a.std(ddof=0)).to_numpy()*np.sqrt([.4,.6]);k=min(501,len(g));dist,idx=NearestNeighbors(n_neighbors=k).fit(z).kneighbors(z);dist,idx=dist[:,1:],idx[:,1:];w=np.exp(-(dist**2));ptrain[g.index.to_numpy()]=(w*g.fgm.to_numpy()[idx]).sum(1)/w.sum(1)
    # Update shared metric/prediction files and all comparison figures.
    reporting.update_outputs("cluster",train,test,ptrain,ptest)
    # Visualize the Train-fitted cluster assignment on the folded court.
    fig,ax=plt.subplots(figsize=(8,7));colors=plt.cm.tab20(np.linspace(0,1,12))
    for color,(label,g) in zip(colors,train.groupby("location_cluster")): ax.scatter(g.folded_x,g.folded_y,s=2,alpha=.18,color=color,label=label)
    ax.set(title="12 location clusters (folded to one half court)",xlabel="Folded court x",ylabel="Folded court y");ax.legend(ncol=3,fontsize=8);ax.set_aspect("equal");fig.tight_layout();fig.savefig(HERE/"cluster_visualization.png",dpi=180);plt.close(fig)


if __name__ == "__main__":
    main()
