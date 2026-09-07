"""L2-b: align the pretrained L2 RNA encoder (85.5M, 768-d) to the ESM2 space.

Methodology identical to L1's Stage A (align_rna_encoder.py): 20,162 gene pairs
(transcript sequence <-> ESM2 embedding), symmetric InfoNCE, val top-1 as the gate.

Gate L2-b: final val_top1 >= 6.0% (>=15% relative over L1's 5.2%) -> proceed to L2-3.
Additionally reports a FROZEN-encoder linear-probe top-1 (trains esm_proj only for
probe_epochs) = intrinsic alignment quality of the pretrained features.

Tokenization matches pretraining exactly: rna_encoder.encode_seq (CLS-prefixed,
T->U by caller, 512 cap), vocab 8 (PAD/A/C/G/U/N/CLS/MASK).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE),
           os.path.join(os.path.dirname(HERE), "maprna_p2"),
           os.path.join(os.path.dirname(HERE), "maprna_p1")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from rna_encoder import encode_seq, load_fasta_symbol_seqs  # noqa: E402
from ds_knockdown import build_symbol2row  # noqa: E402
from pretrain_l2 import L2Encoder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", type=str, required=True)
    p.add_argument("--esm_table", type=str, required=True)
    p.add_argument("--l2_ckpt", type=str, required=True,
                   help="l2_pretrain/l2_ckpt_latest.pt (or l2_encoder_final.pt)")
    p.add_argument("--out_dir", type=str, default="./l2_align")
    p.add_argument("--probe_epochs", type=int, default=2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_proj", type=float, default=3e-4)
    p.add_argument("--lr_enc", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_l2(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("model", ck))
    enc = L2Encoder()
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    head_keys = [k for k in unexpected if k.startswith("head")]
    assert not [k for k in unexpected if not k.startswith("head")], f"unexpected: {unexpected[:5]}"
    print(f"loaded L2 ({path.split('/')[-1]}): missing={len(missing)} "
          f"head-skipped={len(head_keys)} step={ck.get('step', '?')}", flush=True)
    return enc.to(device)


def l2_embed(enc, tokens, pad_mask):
    """CLS-position hidden state (768-d), bypassing the MLM head."""
    B, T = tokens.shape
    pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
    h = enc.token_emb(tokens) + enc.pos_emb(pos)
    key_pad = ~pad_mask if pad_mask is not None else None
    h = enc.encoder(h, src_key_padding_mask=key_pad)
    return enc.norm(h)[:, 0, :]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"Device: {device}", flush=True)

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
            row = sym2row.get(sym + "1", -1)
        if row < 0:
            continue
        seq = symbol_map[sym].upper().replace("T", "U")
        ids, _ = encode_seq(seq, args.max_len)
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

    encoder = load_l2(args.l2_ckpt, device)
    esm_proj = nn.Linear(vecs.shape[-1], 768).to(device)

    ds = TensorDataset(tokens[train_idx], vecs[train_idx])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    def infonce(a, b, temp):
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        logits = a @ b.t() / temp
        labels = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def eval_top1():
        encoder.eval(); esm_proj.eval()
        chunks = 64
        rs, es = [], []
        with torch.no_grad():
            for lo in range(0, len(val_idx), chunks):
                sel = val_idx[lo:lo + chunks]
                vt = tokens[sel].to(device)
                vm = (vt != 0)
                vv = vecs[sel].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    r = F.normalize(l2_embed(encoder, vt, vm).float(), dim=-1)
                    e = F.normalize(esm_proj(vv.float()), dim=-1)
                rs.append(r); es.append(e)
            r = torch.cat(rs); e = torch.cat(es)
            loss = float(infonce(r, e, args.temperature).item())
            acc1 = float(((r @ e.t()).argmax(dim=1) ==
                          torch.arange(r.shape[0], device=device)).float().mean())
        return loss, acc1

    def train_phase(epochs, train_enc, tag):
        params = (list(encoder.parameters()) + list(esm_proj.parameters()) if train_enc
                  else list(esm_proj.parameters()))
        opt = torch.optim.AdamW(params,
                                lr=args.lr_enc if train_enc else args.lr_proj,
                                weight_decay=0.01)
        total = epochs * len(loader)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total,
                                                           eta_min=(args.lr_enc if train_enc
                                                                    else args.lr_proj) * 0.05)
        out = {}
        for epoch in range(epochs):
            encoder.train(); esm_proj.train()
            if not train_enc:
                encoder.eval()
            run = []
            for tok, vec in loader:
                tok, vec = tok.to(device), vec.to(device)
                pm = (tok != 0)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    r = F.normalize(l2_embed(encoder, tok, pm), dim=-1)
                    e = F.normalize(esm_proj(vec.float()), dim=-1)
                    loss = infonce(r, e, args.temperature)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                run.append(float(loss.item()))
            vl, acc = eval_top1()
            out = dict(phase=tag, epoch=epoch + 1, train_loss=float(np.mean(run)),
                       val_loss=vl, val_top1=acc)
            print(json.dumps(out), flush=True)
            with open(os.path.join(args.out_dir, "align_log.jsonl"), "a") as f:
                f.write(json.dumps(out) + "\n")
        return out

    # Phase 1: frozen-encoder linear probe (intrinsic alignment of pretrained features)
    probe = train_phase(args.probe_epochs, train_enc=False, tag="probe")
    # Phase 2: full fine-tune (comparable to L1 Stage A protocol)
    final = train_phase(args.epochs, train_enc=True, tag="finetune")

    torch.save({"l2_encoder": encoder.state_dict(),
                "esm_proj": esm_proj.state_dict(),
                "probe_top1": probe["val_top1"],
                "val_top1": final["val_top1"],
                "max_len": args.max_len},
               os.path.join(args.out_dir, "l2_encoder_aligned.pt"))
    verdict = "PASS" if final["val_top1"] >= 0.060 else "FAIL"
    print(f"Gate L2-b: probe_top1={probe['val_top1']:.4f} (intrinsic) | "
          f"final val_top1={final['val_top1']:.4f} (L1 reference 0.052, gate >=0.060) "
          f"-> {verdict}", flush=True)
    with open(os.path.join(args.out_dir, "gate_l2b.json"), "w") as f:
        json.dump({"probe_top1": probe["val_top1"], "final_top1": final["val_top1"],
                   "l1_reference": 0.052, "gate": 0.060, "verdict": verdict}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
