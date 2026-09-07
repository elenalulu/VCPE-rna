"""VCPE-rna P2: dual-axis conditioning model.

Base = P1's MAPmodelKD (mechanism axis: target-gene ESM2+KG embedding). Adds
the RNA-sequence axis: the aligned RNAEncoder (Stage A) embeds the target
gene's transcript, a linear projector maps it into dim_emb, and the two are
SUMMED into the perturbation token of `[CLS | Pert | Genes]`.

Checkpoint contract: start from a P1 ckpt (ckpt_best_cosine.pt /
ckpt_best_mse.pt); rna_encoder / rna_projector are new (loaded from the
Stage-A alignment output, then fine-tuned).
"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model_kd import MAPmodelKD  # noqa: E402
from rna_encoder import RNAEncoder  # noqa: E402


class MAPmodelKD2(MAPmodelKD):
    """MAPmodelKD + RNA-sequence conditioning branch (dual-axis)."""

    def __init__(self, se_ckpt, se_cfg, esm_table_path, rna_encoder_state=None,
                 rna_d_model=256, hvg_info=None, freeze_se=True):
        super().__init__(se_ckpt=se_ckpt, se_cfg=se_cfg,
                         esm_table_path=esm_table_path,
                         hvg_info=hvg_info, freeze_se=freeze_se)
        self.rna_encoder = RNAEncoder(d_model=rna_d_model, max_len=600)
        if rna_encoder_state is not None:
            self.rna_encoder.load_state_dict(rna_encoder_state)
        self.rna_projector = nn.Linear(rna_d_model, self.pert_model.base.dim_emb)
        self.pert_token_scale = 1.0  # P2.1: set from train script
        # P2.2: STRING 网络轴。neighbor_table [V, K] long（ESM2 行索引，-1 pad），
        # 由 build_string_neighbors.py 产出、train/diag 用 set_neighbor_table 注入。
        # 注意：普通 attribute（不注册 buffer），不进 state_dict、需手动 .to(device)。
        self.neighbor_table = None
        esm_dim = self.pert_model.esm_table.shape[1]
        self.net_proj = nn.Linear(2 * esm_dim, self.pert_model.base.dim_emb)

    def set_neighbor_table(self, table_np, device):
        import numpy as np
        self.neighbor_table = torch.from_numpy(np.asarray(table_np)).long().to(device)

    def compute_pert_vec(self, pert_rows, rna_tokens, rna_mask):
        """P2.2 三轴条件化：机制轴(ESM2) ⊕ 网络轴(STRING 邻居均值) ⊕ 序列轴(RNA 编码器)。
        返回 [Bp, dim_emb]。"""
        esm_vec = self.pert_model.esm_table[pert_rows]                      # [Bp, 5120]
        if self.neighbor_table is not None:
            nb = self.neighbor_table[pert_rows]                             # [Bp, K]
            valid = (nb >= 0)
            nb_vec = self.pert_model.esm_table[nb.clamp(min=0)]             # [Bp, K, esm_dim]
            w = valid.float().unsqueeze(-1)
            nb_mean = (nb_vec * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
            mech = self.net_proj(torch.cat([esm_vec, nb_mean], dim=-1))     # [Bp, dim_emb]
        else:
            mech = self.pert_model.kd_projector(esm_vec)
        rna_emb = self.rna_encoder(rna_tokens, rna_mask)                    # [Bp, 256]
        return (mech + self.rna_projector(rna_emb)) * self.pert_token_scale

    def forward(self, ctrl_src, ctrl_counts, pert_rows, rna_tokens, rna_mask=None,
                pert_vec_override=None):
        B, S, L = ctrl_src.shape
        gene_ids = ctrl_src.reshape(B * S, L)
        expressions = ctrl_counts.reshape(B * S, L)

        src = self.se.pe_embedding(gene_ids)
        esm_tokens = src[:, 1:, :].clone().detach()

        src = torch.nn.functional.normalize(src, dim=2)
        cls_tokens = self.se.cls_token.expand(src.size(0), 1, -1)
        src = torch.cat([cls_tokens, src[:, 1:, :]], dim=1)
        if self.se.dataset_token is not None:
            src = torch.cat((src, self.se.dataset_token.expand(src.size(0), 1, -1)), dim=1)

        if self.freeze_se:
            with torch.no_grad():
                gene_output, embedding, _ = self.se(
                    src=src, counts=expressions, dataset_nums=None, profile=False)
        else:
            gene_output, embedding, _ = self.se(
                src=src, counts=expressions, dataset_nums=None, profile=False)
        gene_output = gene_output[:, 1:-1, :]

        # P2.2 三轴 pert token；pert_vec_override: ablation 入口（置零/打乱）
        if pert_vec_override is not None:
            pert_vec = pert_vec_override                                        # [Bp, dim_emb]
        else:
            pert_vec = self.compute_pert_vec(pert_rows, rna_tokens, rna_mask)

        Bp = pert_rows.shape[0]
        BS = cls_tokens.shape[0]
        S_ = BS // Bp

        cell_tokens = self.pert_model.base.encode_cells(embedding)          # [B*S, dim_emb]
        pert_tokens = pert_vec.unsqueeze(1).expand(Bp, S_, -1).reshape(BS, -1)
        gene_tokens = self.pert_model.base.encode_genes(gene_output, esm_tokens)

        combined = torch.cat([
            cell_tokens.unsqueeze(1),
            pert_tokens.unsqueeze(1),
            gene_tokens,
        ], dim=1)

        res_pred = self.pert_model.base.transformer_backbone(inputs_embeds=combined).last_hidden_state
        res_pred = res_pred[:, 0, :]
        pred_tokens = self.pert_model.base.project_out(cell_tokens + res_pred)
        pred_hvgs = self.pert_model.base.gene_decoder(pred_tokens)

        pred_tokens = pred_tokens.reshape(B, S, -1)
        pred_hvgs = pred_hvgs.reshape(B, S, -1)
        return pred_tokens, pred_hvgs
