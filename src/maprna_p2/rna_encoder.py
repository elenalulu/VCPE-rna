"""VCPE-rna P2: self-trained RNA sequence encoder (license-clean, PLAN D2-A).

Small transformer over single-nucleotide tokens. Stage A trains it by
contrastive alignment against ESM2 gene embeddings (align_rna_encoder.py);
Stage B uses it as the RNA-sequence conditioning branch of the response model
(dual-axis: RNA sequence + ESM2/KG mechanism).

Sequence source: gene_transcripts.fa (212,794 records, RNA alphabet with U,
capped at 600 nt, symbol / Ensembl headers).
"""
import re

import torch
import torch.nn as nn


class RNAVocab:
    PAD = 0
    A = 1
    C = 2
    G = 3
    U = 4
    N = 5
    CLS = 6
    SIZE = 7
    _MAP = {"A": 1, "C": 2, "G": 3, "U": 4, "T": 4, "N": 5}


def encode_seq(seq: str, max_len: int):
    """RNA string -> [CLS] + tokens, padded to max_len. Returns (ids, mask)."""
    ids = [RNAVocab.CLS]
    for ch in seq.upper()[: max_len - 1]:
        ids.append(RNAVocab._MAP.get(ch, RNAVocab.N))
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(RNAVocab.PAD)
        mask.append(0)
    return ids, mask


def load_fasta_symbol_seqs(fasta_path):
    """Parse gene_transcripts.fa -> {gene_symbol: seq}.

    Headers are mixed: gene symbols (preferred) and Ensembl IDs (with/without
    version). Returns (symbol_map, ensembl_map); symbol_map wins at lookup.
    """
    import re

    symbol_map, ensembl_map = {}, {}
    name = None
    chunks = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    _store(name, "".join(chunks), symbol_map, ensembl_map)
                name, chunks = line[1:].strip(), []
            elif name is not None:
                chunks.append(line)
    if name is not None:
        _store(name, "".join(chunks), symbol_map, ensembl_map)
    return symbol_map, ensembl_map


def _store(header, seq, symbol_map, ensembl_map):
    """同符号多转录本时保留最长序列（确定性，信息量最大）。"""
    def put(d, k):
        if k not in d or len(seq) > len(d[k]):
            d[k] = seq
    if re.fullmatch(r"ENSG\d+(\.\d+)?", header):
        put(ensembl_map, header.split(".")[0])
    else:
        put(symbol_map, header)


def lookup_seq(symbol, symbol_map, ensembl_map):
    """Symbol -> sequence; tries '<sym>1' HGNC-rename fallback, then Ensembl."""
    if symbol in symbol_map:
        return symbol_map[symbol]
    if (symbol + "1") in symbol_map:
        return symbol_map[symbol + "1"]
    if symbol in ensembl_map:
        return ensembl_map[symbol]
    return None


class RNAEncoder(nn.Module):
    """4-layer transformer over nucleotide tokens, CLS-pooled. ~2.5M params."""

    def __init__(self, d_model=256, nhead=4, num_layers=4, dim_ff=512,
                 max_len=600, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(RNAVocab.SIZE, d_model, padding_idx=RNAVocab.PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, pad_mask=None):
        """tokens [B, T] long; pad_mask [B, T] bool (True = real token)."""
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(tokens) + self.pos_emb(pos)
        key_pad = ~pad_mask if pad_mask is not None else None  # True = ignore
        h = self.encoder(h, src_key_padding_mask=key_pad)
        return self.norm(h[:, 0])  # CLS pooling -> [B, d_model]
