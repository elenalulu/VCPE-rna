"""VCPE-rna P2.3-B: GEARS-style per-gene deviation head.

Replaces the drowned additive pert-token pathway (P2/P2.1/P2.2 all showed
r(real, zeroPert)=1.000) with a direct per-gene deviation predictor:

    pred_dev[i] = MLP([g_i, p, g_i*p, rna_p, ctrl_expr_i, is_neighbor_i, is_target_i])

where g_i is the HVG gene's ESM2 embedding, p the target gene's ESM2 embedding,
rna_p the target transcript's RNA-encoder embedding, ctrl_expr_i the dataset-level
control expression of gene i, and the last two features mark whether gene i is a
STRING neighbor of / the target itself. No SE backbone involved - the perturbation
information reaches every gene token directly, so conditioning cannot be drowned.

Trains on residual targets (full fc minus train-only common core, the P2.1 protocol).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
for _sib in ("maprna_p1", "maprna_p2"):
    _p = _HERE.parent / _sib
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def load_esm_matrix(esm_table_path):
    """dict symbol->tensor (row order = insertion order, matches sym2row) -> [V, D] tensor."""
    tab = torch.load(esm_table_path, map_location="cpu", weights_only=False)
    if isinstance(tab, dict):
        return torch.stack([v.float() for v in tab.values()])
    return tab.float()


# V2-1a: self-response expression gating (structural prior, NOT learnable).
# CRISPRi training data teaches "is_target -> strong self down-regulation" (guide
# directly represses the promoter), which transfers wrongly to low-expression
# contexts ("can't knock down RNA that isn't there"). This multiplicative gate
# on the target's own output row caps the self response by the target's raw
# ctrl expression (log1p CP10K space; tau aligned with the v2-0a cache gate).
# The gate is applied AFTER the MLP and is NOT fed back as a feature, so the
# head cannot learn to pre-compensate around it.
GATE_TAU = 0.7      # log1p CP10K; sigmoid transition point
GATE_SLOPE = 6.0    # sharpness of the ramp


class DeviationModel(nn.Module):
    """Per-gene deviation head over the HVG universe.

    forward(pert_rows, ds_idx, rna_emb, pert_esm_override=None) -> pred_dev [Bp, n_hvg]
    """

    def __init__(self, esm_matrix, hvg_rows, n_ds, d_model=256, hidden=512,
                 rna_encoder=None, rna_d_model=256, dropout=0.1):
        super().__init__()
        esm_matrix = esm_matrix.float()
        self.register_buffer("esm_table", esm_matrix)          # [V, esm_dim] frozen
        self.register_buffer("hvg_rows", torch.as_tensor(hvg_rows).long())
        self.esm_dim = esm_matrix.shape[1]
        self.n_hvg = len(hvg_rows)

        self.proj_g = nn.Linear(self.esm_dim, d_model)          # HVG gene embedding
        self.proj_p = nn.Linear(self.esm_dim, d_model)          # target gene embedding
        self.rna_encoder = rna_encoder                          # may be None (no seq axis)
        rna_d = rna_d_model if rna_encoder is not None else 0
        self.proj_r = nn.Linear(rna_d, d_model) if rna_encoder is not None else None

        DS_EMB = 8
        feat_dim = 4 * d_model + 3 + DS_EMB                    # g,p,g*p,rna + ctrl,nb,tgt + ds emb
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # ds one-hot goes through a small learned mix instead of raw concat
        self.ds_emb = nn.Embedding(n_ds, DS_EMB)

    def _rna_embed(self, rna_tokens, rna_mask):
        if self.rna_encoder is None:
            return None
        return self.rna_encoder(rna_tokens, rna_mask)           # [Bp, rna_d]

    def forward(self, pert_rows, ds_idx, ctrl_feat, rna_emb=None,
                pert_esm_override=None, pert_ctrl_expr=None):
        """pert_rows [Bp] long; ds_idx [Bp] long; ctrl_feat [Bp, n_hvg] float.

        pert_esm_override: [Bp, esm_dim] replacement for the target ESM2 vector
        (ablation entry: zeros / shuffled rows).
        pert_ctrl_expr: [Bp] float, RAW log1p ctrl expression of the target gene
        (v2-1a self-response gate; None = gate off, backward compatible).
        """
        Bp = pert_rows.shape[0]
        dev = self.esm_table.device
        pert_rows = pert_rows.to(dev)
        ds_idx = ds_idx.to(dev)
        ctrl_feat = ctrl_feat.to(dev)

        G = self.proj_g(self.esm_table[self.hvg_rows])                    # [n_hvg, d]
        p_esm = (self.esm_table[pert_rows] if pert_esm_override is None
                 else pert_esm_override.to(dev))                          # [Bp, esm_dim]
        P = self.proj_p(p_esm)                                            # [Bp, d]
        inter = G.unsqueeze(0) * P.unsqueeze(1)                           # [Bp, n_hvg, d]

        parts = [G.unsqueeze(0).expand(Bp, -1, -1), P.unsqueeze(1).expand(-1, self.n_hvg, -1), inter]
        if self.proj_r is not None and rna_emb is not None:
            R = self.proj_r(rna_emb.to(dev))                              # [Bp, d]
            parts.append(R.unsqueeze(1).expand(-1, self.n_hvg, -1))
        else:
            zero = torch.zeros(Bp, self.n_hvg, G.shape[-1], device=dev)
            parts.append(zero)

        # neighbor / target indicator features
        nb = self.neighbor_table[pert_rows] if getattr(self, "neighbor_table", None) is not None \
            else torch.full((Bp, 0), -1, device=dev, dtype=torch.long)
        if nb.shape[1] > 0:
            is_nb = (nb.unsqueeze(-1) == self.hvg_rows.view(1, 1, -1)).any(dim=1)  # [Bp, n_hvg]
        else:
            is_nb = torch.zeros(Bp, self.n_hvg, dtype=torch.bool, device=dev)
        is_tgt = (pert_rows.unsqueeze(-1) == self.hvg_rows.view(1, -1))           # [Bp, n_hvg]
        ds_onehot = self.ds_emb(ds_idx)                                           # [Bp, DS_EMB]

        feats = [ctrl_feat.unsqueeze(-1),
                 is_nb.float().unsqueeze(-1),
                 is_tgt.float().unsqueeze(-1),
                 ds_onehot.unsqueeze(1).expand(-1, self.n_hvg, -1)]
        x = torch.cat(parts + feats, dim=-1)                              # [Bp, n_hvg, F]
        out = self.mlp(x).squeeze(-1)                                    # [Bp, n_hvg]

        # V2-1a: multiplicative expression gate on the target's own row.
        # expr <= 0.1 (log1p) -> gate ~0.05; expr >= 1.3 -> gate ~0.97.
        if pert_ctrl_expr is not None:
            gate = torch.sigmoid((pert_ctrl_expr.to(dev) - GATE_TAU) * GATE_SLOPE)  # [Bp]
            out = out * (1.0 - is_tgt.float() * (1.0 - gate).unsqueeze(-1))
        return out

    def set_neighbor_table(self, table_np, device):
        import numpy as np
        self.neighbor_table = torch.from_numpy(np.asarray(table_np)).long().to(device)
