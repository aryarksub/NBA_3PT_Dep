"""Train and evaluate the final 18-bucket HistGradientBoosting model.

The script reads the fixed ``train.csv`` and ``test.csv`` files in this
directory. It trains one HGB classifier per shot-zone/defensive-pressure
bucket, writes probabilities for both splits, updates the shared comparison
tables and figures, and creates a permutation-importance plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# ---------------------------------------------------------------------------
# Reproducibility, paths, and model inputs
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SEED = 42
# These are the final no-shot-history HGB inputs. The two player percentages
# are retrospective full-season statistics included in the prepared CSVs.
FEATURES = ["period", "seconds_rem", "3pt", "shot_number", "shot_dist", "shot_clock",
            "close_def_dist", "avg_def_dist", "def_hull_area", "shooter_x", "shooter_y",
            "home", "tmate_near1_dx", "tmate_near1_dy", "tmate_near1_dist",
            "def_near1_dx", "def_near1_dy", "def_near1_dist", "def_near2_dx",
            "def_near2_dy", "def_near2_dist", "player_full_season_2p_pct",
            "player_full_season_3p_pct"]


def metrics(y, p):
    """Return classification, ranking, and calibration metrics at threshold 0.5."""
    # Convert probabilities to binary predictions and unpack the confusion matrix.
    c = p >= .5; tn, fp, fn, tp = confusion_matrix(y, c, labels=[0, 1]).ravel()
    # Expected calibration error uses ten equal-width probability bins.
    bins = np.digitize(p, np.linspace(0, 1, 11)[1:-1], right=True)
    ece = sum(abs(p[bins == i].mean() - y[bins == i].mean()) * (bins == i).mean()
              for i in range(10) if (bins == i).any())
    return {"accuracy": accuracy_score(y,c), "balanced_accuracy": balanced_accuracy_score(y,c),
            "precision": precision_score(y,c,zero_division=0), "recall": recall_score(y,c,zero_division=0),
            "f1": f1_score(y,c,zero_division=0), "roc_auc": roc_auc_score(y,p),
            "log_loss": log_loss(y,p), "brier_score": brier_score_loss(y,p), "ece_10bin": ece,
            "true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp}


def update_outputs(model_name, train, test, ptrain, ptest):
    """Merge one model's Train/Test metrics and predictions into shared files."""
    rows=[]
    for split, frame, prob in [("train",train,ptrain),("test",test,ptest)]:
        # Store one comparable metric row for each data split.
        row={"model":model_name,"split":split,"rows":len(frame),"make_rate":frame.fgm.mean(),"threshold":.5}
        row.update(metrics(frame.fgm.to_numpy(),prob)); rows.append(row)
        # Reuse an existing prediction table so scripts may be run in any order.
        path=HERE/f"{split}_predictions.csv"; base=(pd.read_csv(path).set_index("source_row_index") if path.exists()
             else frame[["source_row_index","game_id","game_date","event_id","player_id","3pt","fgm"]].set_index("source_row_index"))
        values=pd.DataFrame({f"{model_name}_probability":prob,f"{model_name}_prediction":(prob>=.5).astype(int)},index=frame.source_row_index)
        for col in values: base.loc[values.index,col]=values[col]
        base.reset_index().to_csv(path,index=False)
    # Replace only this model's rows; never erase results from the other models.
    mp=HERE/"model_metrics.csv"; old=pd.read_csv(mp) if mp.exists() else pd.DataFrame()
    if not old.empty: old=old[old.model.ne(model_name)]
    pd.concat([old,pd.DataFrame(rows)],ignore_index=True).to_csv(mp,index=False)
    comparison_plots()


def comparison_plots():
    """Regenerate all cross-model figures from the shared prediction CSVs."""
    colors={"hgb":"#2563eb","nn":"#f97316","cluster":"#16a34a"}
    # Discover whichever models have already been run.
    available=[]
    for split in ["train","test"]:
        path=HERE/f"{split}_predictions.csv"
        if path.exists(): available.extend(c[:-12] for c in pd.read_csv(path,nrows=1) if c.endswith("_probability"))
    # Plot one confusion matrix per model for both Train and Test.
    names=list(dict.fromkeys(available));fig,axes=plt.subplots(2,max(1,len(names)),figsize=(4*max(1,len(names)),7),squeeze=False)
    for r,split in enumerate(["train","test"]):
        path=HERE/f"{split}_predictions.csv"
        if not path.exists(): continue
        d=pd.read_csv(path)
        for col,name in enumerate(names):
            if f"{name}_probability" not in d: continue
            p=d[f"{name}_probability"].dropna(); y=d.loc[p.index,"fgm"].to_numpy(); pr=p.to_numpy()
            cm=confusion_matrix(y,pr>=.5,labels=[0,1]);ax=axes[r,col];ax.imshow(cm,cmap="Blues")
            for i in range(2):
                for j in range(2): ax.text(j,i,f"{cm[i,j]:,}",ha="center",va="center")
            ax.set(title=f"{name.upper()} — {split.title()}",xlabel="Predicted",ylabel="Actual",xticks=[0,1],yticks=[0,1])
    fig.tight_layout();fig.savefig(HERE/"confusion_matrix_comparison.png",dpi=180);plt.close(fig)
    # Overlay ROC curves to compare ranking performance.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,split in zip(axes,["train","test"]):
        path=HERE/f"{split}_predictions.csv"
        if path.exists():
            d=pd.read_csv(path)
            for name in names:
                if f"{name}_probability" not in d: continue
                p=d[f"{name}_probability"].dropna();y=d.loc[p.index,"fgm"];fpr,tpr,_=roc_curve(y,p);ax.plot(fpr,tpr,label=f"{name} ({roc_auc_score(y,p):.3f})",color=colors.get(name))
        ax.plot([0,1],[0,1],"--",color="gray");ax.set(title=f"{split.title()} ROC curves",xlabel="False-positive rate",ylabel="True-positive rate");ax.legend()
    fig.tight_layout();fig.savefig(HERE/"roc_curve_comparison.png",dpi=180);plt.close(fig)
    # Overlay quantile-binned calibration curves against perfect calibration.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,split in zip(axes,["train","test"]):
        path=HERE/f"{split}_predictions.csv"
        if path.exists():
            d=pd.read_csv(path)
            for name in names:
                if f"{name}_probability" not in d: continue
                p=d[f"{name}_probability"].dropna();y=d.loc[p.index,"fgm"];obs,pred=calibration_curve(y,p,n_bins=10,strategy="quantile");ax.plot(pred,obs,"o-",label=name,color=colors.get(name))
        ax.plot([0,1],[0,1],"--",color="gray");ax.set(title=f"{split.title()} calibration",xlabel="Mean predicted probability",ylabel="Observed make rate");ax.legend()
    fig.tight_layout();fig.savefig(HERE/"calibration_curve_comparison.png",dpi=180);plt.close(fig)
    # Compare the marginal distribution of predicted probabilities.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,split in zip(axes,["train","test"]):
        path=HERE/f"{split}_predictions.csv"
        if path.exists():
            d=pd.read_csv(path)
            for c in [x for x in d if x.endswith("_probability")]: ax.hist(d[c].dropna(),bins=30,alpha=.4,label=c[:-12])
        ax.set(title=f"{split.title()} probability distributions",xlabel="Predicted probability",ylabel="Shots"); ax.legend()
    fig.tight_layout();fig.savefig(HERE/"probability_distribution_comparison.png",dpi=180);plt.close(fig)


def main():
    """Tune, fit, evaluate, and report all 18 bucket-specific HGB models."""
    # The split is already fixed on disk; no resampling occurs here.
    train,test=pd.read_csv(HERE/"train.csv"),pd.read_csv(HERE/"test.csv")
    models={}; ptrain=np.empty(len(train)); ptest=np.empty(len(test)); importances=[]
    # Six shot zones crossed with three pressure levels produce 18 buckets.
    order=[f"{z}_{p}" for z in ["at_rim","paint_non_rim","short_midrange","long_midrange","corner_3","above_break_3"] for p in ["tight","moderate","open"]]
    for i,bucket in enumerate(order):
        tr=train.bucket18.eq(bucket); te=test.bucket18.eq(bucket)
        # Tune within the current Train bucket only. Brier score favors useful
        # probabilities rather than optimizing classification accuracy alone.
        estimator=HistGradientBoostingClassifier(random_state=SEED+i,early_stopping=True,validation_fraction=.1)
        search=RandomizedSearchCV(estimator,{"learning_rate":np.linspace(.04,.18,4),"max_depth":[None,4,6],
              "max_leaf_nodes":[15,31,63],"min_samples_leaf":[20,60,120],"l2_regularization":[0,.01,.1,1],
              "max_bins":[64,128,255]},n_iter=2,scoring="neg_brier_score",cv=2,n_jobs=1,random_state=SEED+i,refit=True)
        # Cap the tuning sample for speed, then refit the selected estimator on
        # every Train row belonging to this bucket.
        xt,yt=train.loc[tr,FEATURES],train.loc[tr,"fgm"]
        if len(xt)>7000: xt,_,yt,_=train_test_split(xt,yt,train_size=7000,stratify=yt,random_state=SEED+i)
        search.fit(xt,yt);model=clone(search.best_estimator_).fit(train.loc[tr,FEATURES],train.loc[tr,"fgm"])
        # Fill the original row positions so all bucket predictions recombine cleanly.
        ptrain[tr]=model.predict_proba(train.loc[tr,FEATURES])[:,1]; ptest[te]=model.predict_proba(test.loc[te,FEATURES])[:,1];models[bucket]=model
        # Estimate feature strength on a small, reproducible Train subset.
        sample=train.loc[tr].sample(min(500,tr.sum()),random_state=SEED)
        imp=permutation_importance(model,sample[FEATURES],sample.fgm,n_repeats=2,random_state=SEED,scoring="neg_log_loss")
        importances.append(imp.importances_mean)
    # Write common outputs first, followed by the HGB-specific importance chart.
    update_outputs("hgb",train,test,ptrain,ptest)
    strength=pd.Series(np.mean(importances,axis=0),index=FEATURES).sort_values().tail(15)
    ax=strength.plot.barh(figsize=(8,6),color="#2563eb");ax.set(title="HGB permutation importance",xlabel="Decrease in negative log loss")
    ax.figure.tight_layout();ax.figure.savefig(HERE/"hgb_feature_importance.png",dpi=180);plt.close(ax.figure)
    print(json.dumps({"model":"hgb","train":metrics(train.fgm.to_numpy(),ptrain),"test":metrics(test.fgm.to_numpy(),ptest)},indent=2,default=float))


if __name__ == "__main__":
    main()
