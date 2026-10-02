"""Train the final unified player-embedding neural network for 100 epochs.

The model combines standardized numeric inputs with embeddings for shot zone,
defensive pressure, and player identity. It reads the fixed Train/Test CSVs,
records the full learning curve, and updates the shared comparison outputs.
"""

from __future__ import annotations

import random
from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

# ---------------------------------------------------------------------------
# Shared reporting utilities and fixed hyperparameters
# ---------------------------------------------------------------------------
HERE=Path(__file__).resolve().parent
# Import reporting functions without running the HGB training entry point.
spec=importlib.util.spec_from_file_location("reporting",HERE/"run_hgb.py");reporting=importlib.util.module_from_spec(spec);spec.loader.exec_module(reporting)
SEED,EPOCHS,BATCH_SIZE,LR=42,100,128,3e-3
# BASE contains the original shot/context inputs; HISTORY contains the 14
# leakage-controlled player-history features already materialized in the CSVs.
BASE=["period","seconds_rem","3pt","shot_number","fgm1","tot1","fgm2","tot2","fgm3","tot3","streak","shot_dist","close_def_dist","avg_def_dist","def_hull_area","shooter_x","shooter_y","home","tmate_near1_dx","tmate_near1_dy","tmate_near1_dist","def_near1_dx","def_near1_dy","def_near1_dist","def_near2_dx","def_near2_dy","def_near2_dist","has_prev_1","has_prev_2","has_prev_3"]
HISTORY=["player_prior_fga_log1p","player_prior_fg_pct_shrunk","player_prior_2pa_log1p","player_prior_2p_pct_shrunk","player_prior_3pa_log1p","player_prior_3p_pct_shrunk","player_prior_same_type_attempts_log1p","player_prior_same_type_pct_shrunk","player_prior_zone_attempts_log1p","player_prior_zone_pct_shrunk","player_prior_type_pressure_attempts_log1p","player_prior_type_pressure_pct_shrunk","player_recent_20_fg_pct_shrunk","player_recent_20_same_type_pct_shrunk"]
BINARY=["3pt","home","has_prev_1","has_prev_2","has_prev_3"]


class Model(nn.Module):
    """MLP with 4D zone, 2D pressure, and 8D player embeddings."""
    def __init__(self,numeric,nplayers):
        super().__init__();self.zone=nn.Embedding(6,4);self.pressure=nn.Embedding(3,2);self.player=nn.Embedding(nplayers,8,padding_idx=0)
        # Numeric inputs plus 14 embedding dimensions feed the dense network.
        self.net=nn.Sequential(nn.Linear(numeric+14,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,x,z,p,i):
        """Concatenate numeric features and embeddings, then return one logit."""
        return self.net(torch.cat([x,self.zone(z),self.pressure(p),self.player(i)],1)).squeeze(1)


def prepare(train,test):
    """Impute, transform, and Train-fit standardize numeric feature blocks."""
    a,b=train[BASE].copy(),test[BASE].copy()
    # Missing historical values mean no available previous shot and become zero.
    for d in [a,b]:
        d[["fgm1","tot1","fgm2","tot2","fgm3","tot3"]]=d[["fgm1","tot1","fgm2","tot2","fgm3","tot3"]].fillna(0)
        for c in ["tot1","tot2","tot3","def_hull_area"]: d[c]=np.log1p(d[c].clip(lower=0))
    # Fit both scalers on Train only and apply their parameters to Test.
    continuous=[c for c in BASE if c not in BINARY];s=StandardScaler();a[continuous]=s.fit_transform(a[continuous]);b[continuous]=s.transform(b[continuous])
    hs=StandardScaler();ah=hs.fit_transform(train[HISTORY]);bh=hs.transform(test[HISTORY])
    return np.c_[a.to_numpy(np.float32),ah.astype(np.float32)],np.c_[b.to_numpy(np.float32),bh.astype(np.float32)]


def ids(train,test):
    """Convert bucket labels and player IDs into embedding indices."""
    zone_order=["at_rim","paint_non_rim","short_midrange","long_midrange","corner_3","above_break_3"]
    pressure_order=["tight","moderate","open"]
    z=lambda d:d.bucket18.str.rsplit("_",n=1).str[0].map({v:i for i,v in enumerate(zone_order)}).to_numpy(np.int64)
    p=lambda d:d.bucket18.str.rsplit("_",n=1).str[1].map({v:i for i,v in enumerate(pressure_order)}).to_numpy(np.int64)
    # Index zero is reserved for a player not observed in Train.
    mapping={str(v):i+1 for i,v in enumerate(sorted(train.player_id.unique()))}
    i=lambda d:d.player_id.astype(str).map(mapping).fillna(0).to_numpy(np.int64)
    return z(train),p(train),i(train),z(test),p(test),i(test),len(mapping)+1


def loader(x,z,p,i,y,shuffle):
    """Create a deterministic mini-batch loader for training or evaluation."""
    g=torch.Generator().manual_seed(SEED);return DataLoader(TensorDataset(*map(torch.from_numpy,[x,z,p,i,y.astype(np.float32)])),batch_size=BATCH_SIZE,shuffle=shuffle,generator=g if shuffle else None)


@torch.no_grad()
def predict(model,data):
    """Run inference without gradients and convert logits to probabilities."""
    model.eval();out=[]
    for x,z,p,i,_ in data: out.append(torch.sigmoid(model(x,z,p,i)).numpy())
    return np.concatenate(out)


def main():
    """Train for 100 epochs and save shared predictions and learning curves."""
    # Fix all random generators so rerunning the script reproduces the result.
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    # Read the existing split and build numeric and categorical tensors.
    train,test=pd.read_csv(HERE/"train.csv"),pd.read_csv(HERE/"test.csv");xtr,xte=prepare(train,test);ztr,ptr,itr,zte,pte,ite,nplayers=ids(train,test)
    ytr=train.fgm.to_numpy(np.float32);yte=test.fgm.to_numpy(np.float32);dl=loader(xtr,ztr,ptr,itr,ytr,True);etr=loader(xtr,ztr,ptr,itr,ytr,False);ete=loader(xte,zte,pte,ite,yte,False)
    # Adam optimizes binary cross-entropy on logits; no dropout is used.
    model=Model(xtr.shape[1],nplayers);opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4);lossfn=nn.BCEWithLogitsLoss();rows=[]
    for epoch in range(1,EPOCHS+1):
        # One optimization pass over all Train mini-batches.
        model.train();total=0
        for x,z,p,i,y in dl: opt.zero_grad(set_to_none=True);loss=lossfn(model(x,z,p,i),y);loss.backward();opt.step();total+=float(loss)*len(y)
        # Evaluate both splits each epoch to produce the requested loss curve.
        a,b=predict(model,etr),predict(model,ete);rows.append({"epoch":epoch,"train_loss":reporting.log_loss(ytr,a),"test_loss":reporting.log_loss(yte,b),"train_accuracy":reporting.accuracy_score(ytr,a>=.5),"test_accuracy":reporting.accuracy_score(yte,b>=.5)})
        print(f"epoch={epoch:03d} train_loss={rows[-1]['train_loss']:.5f} test_loss={rows[-1]['test_loss']:.5f}",flush=True)
    # Save final-epoch predictions and the complete epoch-by-epoch history.
    ptrain,ptest=predict(model,etr),predict(model,ete);reporting.update_outputs("nn",train,test,ptrain,ptest);hist=pd.DataFrame(rows);hist.to_csv(HERE/"training_history.csv",index=False)
    # Create the NN-specific Train/Test log-loss chart.
    ax=hist.plot(x="epoch",y=["train_loss","test_loss"],figsize=(8,5),linewidth=2);ax.set(title="NN train and test loss (100 epochs)",ylabel="Log loss");ax.grid(alpha=.2);ax.figure.tight_layout();ax.figure.savefig(HERE/"nn_loss_curve.png",dpi=180);plt.close(ax.figure)


if __name__ == "__main__":
    main()
