"""Edge-feature GNN on the calibrated synchronisation graph, followed by Leiden.

Graph: candidate account pairs from all layers (obs >= min_obs), not only significant ones, so every node has
context. Edge features: -log10 p per layer (0 when the pair is absent from a layer). Node features: passed in by
the caller and limited to those that survive the trivial-artifact audit (T07).
Training labels: campaign account = 1; organic and benign-LLM accounts = 0. Trained on exploratory data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GINEConv

from cib import fuse


def build_graph(edges_wide: pd.DataFrame, layer_cols: list[str], node_feats: pd.DataFrame) -> tuple[Data, pd.Index]:
    """edges_wide: a, b, nl_<layer>...; node_feats indexed by account_id (missing accounts get zeros)."""
    names = pd.Index(pd.unique(np.r_[edges_wide.a.to_numpy(), edges_wide.b.to_numpy(),
                                     node_feats.index.to_numpy()]))
    ia, ib = names.get_indexer(edges_wide.a), names.get_indexer(edges_wide.b)
    ea = np.nan_to_num(edges_wide.reindex(columns=layer_cols).to_numpy(dtype=np.float32), nan=0.0)
    ea = np.log1p(ea)
    x = node_feats.reindex(names).fillna(0.0).to_numpy(dtype=np.float32)
    ei = np.stack([np.r_[ia, ib], np.r_[ib, ia]])
    return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(ei).long(),
                edge_attr=torch.from_numpy(np.vstack([ea, ea]))), names


class EdgeGNN(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2,
                 conv: str = "gine"):
        super().__init__()
        self.inp = nn.Linear(max(in_dim, 1), hidden)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            if conv == "gine":
                mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            else:
                self.convs.append(GATv2Conv(hidden, hidden, edge_dim=edge_dim, heads=1))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, d: Data) -> torch.Tensor:
        h = torch.relu(self.inp(d.x if d.x.shape[1] else torch.ones(d.num_nodes, 1)))
        for conv in self.convs:
            h = self.drop(torch.relu(conv(h, d.edge_index, d.edge_attr))) + h
        return self.out(h).squeeze(-1)


def train(d: Data, y: np.ndarray, mask: np.ndarray, *, val_mask: np.ndarray | None = None, epochs: int = 300,
          lr: float = 5e-3, seed: int = 0, patience: int = 30, **kw) -> EdgeGNN:
    torch.manual_seed(seed)
    model = EdgeGNN(d.x.shape[1], d.edge_attr.shape[1], **kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    yt = torch.from_numpy(y.astype(np.float32))
    m = torch.from_numpy(mask)
    pos_w = torch.tensor(max((~y[mask].astype(bool)).sum() / max(y[mask].sum(), 1), 1.0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(d)[m], yt[m])
        loss.backward()
        opt.step()
        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                vl = float(loss_fn(model(d)[torch.from_numpy(val_mask)], yt[torch.from_numpy(val_mask)]))
            if vl < best - 1e-4:
                best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


def predict(model: EdgeGNN, d: Data, names: pd.Index) -> pd.Series:
    with torch.no_grad():
        return pd.Series(torch.sigmoid(model(d)).numpy(), index=names)


def groups(edges_wide: pd.DataFrame, prob: pd.Series, theta: float, resolution: float = 1.0,
           min_cluster: int = 3, seed: int = 0) -> pd.DataFrame:
    """Leiden on edges among accounts with prob >= theta, weight = w * s_u * s_v."""
    keep = prob[prob >= theta].index
    e = edges_wide[edges_wide.a.isin(keep) & edges_wide.b.isin(keep) & edges_wide.w.notna()].copy()
    e["gw"] = e.w * prob.reindex(e.a).to_numpy() * prob.reindex(e.b).to_numpy()
    cl = fuse.leiden(e, weight="gw", resolution=resolution, min_cluster=min_cluster, seed=seed)
    return cl.assign(score=prob.reindex(cl.account_id).to_numpy())
