"""VCPE-rna efficacy head v1: ASO sequence + chemistry + context -> target inhibition %.

Trains on ASO Atlas (190,927 gapmer ASOs, 343 genes, patent-derived). This is the
"efficacy half" of the VCPE engine: for a given ASO (sequence + sugar/backbone
chemistry + dose + cell line), predict how deeply the target mRNA is knocked down.
Combined with the response half (CRISPR-trained deviation head), this completes the
Tier-2 chain: ASO -> off-targets -> response x depth.

Architecture (v1, ~3M trainable):
  - per-position embedding: base(5) + sugar(5) + backbone(2), concat 16-dim
  - 2-layer transformer encoder over positions (d=128) -> mean+max pool
  - target-gene context: frozen ESM2 table (5120) -> proj 128  [mechanism axis reuse]
  - cell_line embedding (16) + dosage log10 + missing flag
  - MLP head -> inhibition %
"""
import torch
import torch.nn as nn

BASES = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": 4}
SUGARS = {"DNA": 0, "MOE": 1, "cEt": 2, "F": 3, "other": 4}
BACKBONES = {"PO": 0, "PS": 1}
MAX_LEN = 30


class EfficacyHead(nn.Module):
    def __init__(self, esm_matrix, n_cell_lines=128, d=128, hidden=256, dropout=0.1):
        super().__init__()
        esm_matrix = esm_matrix.float()
        self.register_buffer("esm_table", esm_matrix)
        self.esm_dim = esm_matrix.shape[1]

        self.base_emb = nn.Embedding(len(BASES), 16, padding_idx=4)
        self.sugar_emb = nn.Embedding(len(SUGARS), 16, padding_idx=0)
        self.bb_emb = nn.Embedding(len(BACKBONES), 16, padding_idx=0)
        self.in_proj = nn.Linear(48, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=2 * d,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.esm_proj = nn.Linear(self.esm_dim, d)
        self.cell_emb = nn.Embedding(n_cell_lines, 16)
        self.head = nn.Sequential(
            nn.Linear(2 * d + d + 16 + 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, base_ids, sugar_ids, bb_ids, pad_mask, gene_rows,
                cell_ids, dose_log, dose_missing):
        """base/sugar/bb [B, L] long; pad_mask [B, L] bool (True=pad);
        gene_rows [B] long; cell_ids [B] long; dose_log/missing [B] float."""
        x = self.in_proj(torch.cat([self.base_emb(base_ids),
                                    self.sugar_emb(sugar_ids),
                                    self.bb_emb(bb_ids)], dim=-1))  # [B, L, 48] -> d
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = torch.cat([(x * valid).sum(1) / valid.sum(1).clamp(min=1.0),
                            x.masked_fill(pad_mask.unsqueeze(-1), -1e4).max(1).values],
                           dim=-1)                                 # [B, 2d]
        g = self.esm_proj(self.esm_table[gene_rows])               # [B, d]
        f = torch.cat([pooled, g, self.cell_emb(cell_ids),
                       dose_log.unsqueeze(-1), dose_missing.unsqueeze(-1)], dim=-1)
        return self.head(f).squeeze(-1)
