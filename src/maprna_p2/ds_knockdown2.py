"""P2 dataset: extends P1's KnockdownDataset with target-gene transcript tokens."""
import numpy as np
import torch

from rna_encoder import encode_seq, lookup_seq


class KD2Dataset:
    """Wraps P1's KnockdownDataset and adds pre-tokenized transcript tokens.

    rna_tokens: [n_items, max_len] long — one sequence per perturbation
    (first gene of the condition; '<sym>1' HGNC fallback; zeros if absent).
    """

    def __init__(self, base_ds, symbol_map, ensembl_map, rna_max_len=600):
        self.base = base_ds
        self.items = base_ds.items
        self.rna_max_len = rna_max_len
        self.tokens = np.zeros((len(self.items), rna_max_len), dtype=np.int64)
        self.mask = np.zeros((len(self.items), rna_max_len), dtype=np.int64)
        n_hit = 0
        for i, (di, cond) in enumerate(self.items):
            gene = cond.split("+")[0]
            seq = lookup_seq(gene, symbol_map, ensembl_map)
            if seq:
                ids, m = encode_seq(seq, rna_max_len)
                self.tokens[i], self.mask[i] = ids, m
                n_hit += 1
            else:
                # no transcript available: fall back to [CLS] + PAD. A row whose
                # mask is all-False would fully mask attention -> NaN output;
                # a lone CLS keeps the encoder well-defined (uninformative but
                # valid — those samples ride on the mechanism axis).
                from rna_encoder import RNAVocab
                self.tokens[i, 0] = RNAVocab.CLS
                self.mask[i, 0] = 1
        print(f"[KD2] transcript tokens: {n_hit}/{len(self.items)} perts with sequence "
              f"({len(self.items) - n_hit} fallback to CLS-only)", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        out = self.base[i]
        out["rna_tokens"] = torch.from_numpy(self.tokens[i])
        out["rna_mask"] = torch.from_numpy(self.mask[i])
        return out


def collate_kd2(batch):
    from ds_knockdown import collate_kd

    out = collate_kd(batch)
    out["rna_tokens"] = torch.stack([b["rna_tokens"] for b in batch])
    out["rna_mask"] = torch.stack([b["rna_mask"] for b in batch]).bool()
    return out
