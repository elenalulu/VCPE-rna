"""Knockdown-conditioned MAPmodel (P1).

Replace the drug-SMILES conditioning token with a target-gene conditioning
token: pe_embedding(pert_row) [5120] -> kd_projector -> [1024] token, inserted
as `[CLS | Pert | Genes]` in the pert transformer backbone. Everything else
(SE frozen cell encoding, gene token projector, transformer, HVG decoder) is
inherited from upstream MAPmodel — zero architecture surgery beyond one
projector, exactly as planned in PLAN.md §P1.
"""
import os
import sys

import torch
import torch.nn as nn

_MAP_ROOT = os.environ.get("MAP_REPO_ROOT", "")
if _MAP_ROOT and _MAP_ROOT not in sys.path:
    sys.path.insert(0, _MAP_ROOT)

try:
    from model.model import MAPmodel            # noqa: E402
except ImportError as _e:
    raise ImportError(
        "model package not found. Either export MAP_REPO_ROOT=<patched MAP repo>/MAP, "
        "or run train_p1.py (it inserts --map_repo into sys.path before importing)."
    ) from _e
from model.components import ResidualProjector  # noqa: E402


class PerturbationEncoderKD(nn.Module):
    """Wraps the upstream PerturbationEncoder; adds the knockdown projector.

    We keep the upstream encoder as a submodule so all pretrained weights
    (cell projector, gene token projector, transformer backbone, HVG decoder)
    load unchanged; the KG-SMILES encoder stays frozen and unused.
    """

    def __init__(self, base_encoder, esm_table_weight):
        super().__init__()
        self.base = base_encoder
        self.kd_projector = ResidualProjector(
            input_dim=esm_table_weight.shape[1],   # 5120
            output_dim=self.base.dim_emb,          # 1024
            hidden_dim=2048,
        )
        self.register_buffer("esm_table", esm_table_weight, persistent=False)

    def encode_perturbation(self, pert_rows):
        """pert_rows: [B] long -> [B, dim_emb] conditioning token."""
        vec = self.esm_table[pert_rows]                            # [B, 5120]
        return self.kd_projector(vec)

    def forward(self, gene_tokens, cls_tokens, esm_tokens, pert_rows):
        B = pert_rows.shape[0]
        BS = cls_tokens.shape[0]
        S = BS // B

        cell_tokens = self.base.encode_cells(cls_tokens)          # [B*S, dim_emb]
        pert_tokens = self.encode_perturbation(pert_rows)          # [B, dim_emb]
        gene_tokens = self.base.encode_genes(gene_tokens, esm_tokens)  # [B*S, L-1, dim_emb]

        pert_tokens = pert_tokens.unsqueeze(1).expand(B, S, -1).reshape(BS, -1)

        combined_tokens = torch.cat([
            cell_tokens.unsqueeze(1),
            pert_tokens.unsqueeze(1),
            gene_tokens,
        ], dim=1)                                                  # [B*S, 2049, dim_emb]

        res_pred = self.base.transformer_backbone(inputs_embeds=combined_tokens).last_hidden_state
        res_pred = res_pred[:, 0, :]
        pred_tokens = self.base.project_out(cell_tokens + res_pred)
        pert_cell_counts_preds = self.base.gene_decoder(pred_tokens)

        pred_tokens = pred_tokens.reshape(B, S, -1)
        pert_cell_counts_preds = pert_cell_counts_preds.reshape(B, S, -1)
        return pred_tokens, pert_cell_counts_preds


class MAPmodelKD(MAPmodel):
    """MAPmodel with knockdown conditioning. forward(ctrl_src, ctrl_counts, pert_rows)."""

    def __init__(self, se_ckpt, se_cfg, esm_table_path, hvg_info=None, freeze_se=True):
        super().__init__(se_ckpt=se_ckpt, se_cfg=se_cfg,
                         smile_encoder="MAP-KG", hvg_info=hvg_info, freeze_se=freeze_se)
        tab = torch.load(esm_table_path, map_location="cpu", weights_only=False)
        weight = torch.vstack(list(tab.values())).float()
        self.pert_model = PerturbationEncoderKD(self.pert_model, weight)

    def forward(self, ctrl_src, ctrl_counts, pert_rows, concs=None):
        B, S, L = ctrl_src.shape
        gene_ids = ctrl_src.reshape(B * S, L)
        expressions = ctrl_counts.reshape(B * S, L)

        src = self.se.pe_embedding(gene_ids)
        esm_tokens = src[:, 1:, :].clone().detach()

        src = torch.nn.functional.normalize(src, dim=2)
        cls_tokens = self.se.cls_token.expand(src.size(0), 1, -1)
        src = torch.cat([cls_tokens, src[:, 1:, :]], dim=1)
        if self.se.dataset_token is not None:
            dataset_token = self.se.dataset_token.expand(src.size(0), 1, -1)
            src = torch.cat((src, dataset_token), dim=1)

        if self.freeze_se:
            with torch.no_grad():
                gene_output, embedding, _ = self.se(
                    src=src, counts=expressions, dataset_nums=None, profile=False)
        else:
            gene_output, embedding, _ = self.se(
                src=src, counts=expressions, dataset_nums=None, profile=False)

        gene_output = gene_output[:, 1:-1, :]
        pred_embs, pred_hvgs = self.pert_model(gene_output, embedding, esm_tokens, pert_rows)
        return pred_embs, pred_hvgs


def trainable_parameter_report(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
