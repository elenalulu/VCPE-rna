"""Stage A: align the RNA sequence encoder to the ESM2 gene-embedding space.

For every gene present in BOTH gene_transcripts.fa (symbol-keyed) and the
ESM2 embedding table, train (RNAEncoder, esm_proj) with symmetric InfoNCE so
that a transcript's 256-d embedding lands next to its gene's ESM2 embedding
(projected to 256-d). After Stage A the encoder's outputs live in a space
compatible with the response model's conditioning branch.

Run (GPU box):
  python align_rna_encoder.py --fasta <AIDO>/vertical_slice/data/gene_transcripts.fa \
    --esm_table <weights>/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
    --out_dir ./p2_align --epochs 8
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rna_encoder import RNAEncoder, RNAVocab, encode_seq, load_fasta_symbol_seqs, lookup_seq  # noqa: E402
from ds_knockdown import build_symbol2row  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", type=str, required=True)
    p.add_argument("--esm_table", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./p2_align")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=600)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"Device: {device}", flush=True)

    # pairs: (gene symbol, transcript seq, esm vec)
    sym2row, _ = build_symbol2row(args.esm_table)
    esm_tab = torch.load(args.esm_table, map_location="cpu", weights_only=False)
    symbol_map, ensembl_map = load_fasta_symbol_seqs(args.fasta)
    print(f"fasta: {len(symbol_map)} symbol / {len(ensembl_map)} ensembl records", flush=True)

    tokens, vecs, genes = [], [], []
    keys = list(esm_tab.keys())
    row2vec = {i: esm_tab[k] for i, k in enumerate(keys)}
    for sym in symbol_map:
        row = sym2row.get(sym, -1)
        if row < 0:
            row = sym2row.get(sym + "1", -1)  # HGNC rename fallback
        if row < 0:
            continue
        ids, _ = encode_seq(symbol_map[sym], args.max_len)
        tokens.append(ids)
        vecs.append(row2vec[row].numpy())
        genes.append(sym)
    tokens = torch.tensor(tokens, dtype=torch.long)
    vecs = torch.tensor(np.stack(vecs), dtype=torch.float32)
    print(f"aligned pairs: {len(genes)}", flush=True)

    n = tokens.shape[0]
    idx = np.random.default_rng(args.seed).permutation(n)
    n_val = max(1, int(round(n * args.val_frac)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    encoder = RNAEncoder(max_len=args.max_len).to(device)
    esm_proj = nn.Linear(vecs.shape[-1], 256).to(device)

    ds = TensorDataset(tokens[train_idx], vecs[train_idx])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(esm_proj.parameters()), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=args.lr * 0.05)

    def infonce(a, b, temp):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = a @ b.t() / temp
        labels = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    best_val = None
    for epoch in range(args.epochs):
        encoder.train(); esm_proj.train()
        run = []
        for tok, vec in loader:
            tok, vec = tok.to(device), vec.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                r = F.normalize(encoder(tok), dim=-1)
                e = F.normalize(esm_proj(vec.float()), dim=-1)
                loss = infonce(r, e, args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            run.append(float(loss.item()))
        # val
        encoder.eval(); esm_proj.eval()
        with torch.no_grad():
            vt = tokens[val_idx].to(device)
            vv = vecs[val_idx].to(device)
            r = F.normalize(encoder(vt), dim=-1)
            e = F.normalize(esm_proj(vv.float()), dim=-1)
            sim = r @ e.t()
            val_loss = float(infonce(r, e, args.temperature).item())
            val_acc1 = float((sim.argmax(dim=1) == torch.arange(sim.shape[0], device=device)).float().mean())
        msg = dict(epoch=epoch + 1, train_loss=float(np.mean(run)),
                   val_loss=val_loss, val_top1=val_acc1)
        print(json.dumps(msg), flush=True)
        with open(os.path.join(args.out_dir, "align_log.jsonl"), "a") as f:
            f.write(json.dumps(msg) + "\n")
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save({"rna_encoder": encoder.state_dict(),
                        "esm_proj": esm_proj.state_dict(),
                        "val_loss": val_loss, "val_top1": val_acc1,
                        "max_len": args.max_len},
                       os.path.join(args.out_dir, "rna_encoder_aligned.pt"))
            print(f"💾 saved aligned encoder (val_loss={val_loss:.4f})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
