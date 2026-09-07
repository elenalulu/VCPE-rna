"""L2-1: MLM pretraining of the ~100M RNA encoder (VCPE-rna L2 专项).

Architecture: encoder-only, d768 x 12 layers x 12 heads, FFN 3072 (~100M),
single-nucleotide vocab 8 (PAD/A/C/G/U/N/CLS + MASK), max_len 512.
Objective: BERT-style MLM, 15% corruption (80/10/10), loss on masked positions.

Engineering (per L2 立项书):
  - 断点续训: ckpt + optimizer + scheduler + step every --save_steps (2000),
    auto-resume from l2_ckpt_latest.pt
  - 长度分桶: rows sorted by length, batch = similar-length rows, dynamic pad
  - bf16 autocast + grad accumulation
  - stop: epochs exhausted OR --max_hours reached OR val plateau
    (3 consecutive evals with relative improvement < 2%)

Data: token store from tokenize_store.py
  (l2_tokens.u16 flat + l2_offsets.npy + l2_train_idx.npy + l2_val_idx.npy)

Run (GPU box, nohup):
  cd $BASE/MAP-KG-main/MAP
  nohup python $BASE/src/l2/pretrain_l2.py \
    --store_dir $BASE/data/l2 \
    --out_dir $BASE/l2_pretrain \
    --epochs 3 --max_hours 40 \
  > $BASE/l2_pretrain.log 2>&1 &
  tail -f $BASE/l2_pretrain.log
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# vocab (L1 tokens + MASK)
PAD, A, C, G, U, N, CLS, MASK = 0, 1, 2, 3, 4, 5, 6, 7
VOCAB = 8
RANDOM_SET = (A, C, G, U)


# ---------------- model ----------------
class L2Encoder(nn.Module):
    """Encoder-only transformer for MLM; token outputs (no pooling head)."""

    def __init__(self, d_model=768, nhead=12, num_layers=12, dim_ff=3072,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(VOCAB, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB)

    def forward(self, tokens, pad_mask=None):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(tokens) + self.pos_emb(pos)
        key_pad = ~pad_mask if pad_mask is not None else None
        h = self.encoder(h, src_key_padding_mask=key_pad)
        return self.head(self.norm(h))  # [B, T, VOCAB]


# ---------------- data ----------------
class TokenStore:
    def __init__(self, store_dir):
        # l2_tokens.u16 是裸二进制（tokenize_store 直接 tobytes 写出），用 memmap 裸读
        self.tokens = np.memmap(store_dir + "/l2_tokens.u16", dtype=np.uint16, mode="r")
        self.offsets = np.load(store_dir + "/l2_offsets.npy")
        self.n = len(self.offsets) - 1
        self.lengths = (self.offsets[1:] - self.offsets[:-1]).astype(np.int64)

    def get(self, i):
        return np.asarray(self.tokens[self.offsets[i]:self.offsets[i + 1]],
                          dtype=np.int64)


def build_length_buckets(lengths, rows, tokens_per_batch, rng):
    """Sort rows by length, greedy-pack into similar-length batches."""
    order = np.argsort(lengths[rows], kind="stable")
    sorted_rows = rows[order]
    sorted_lens = lengths[sorted_rows]
    batches, cur, cur_max = [], [], 0
    for r, L in zip(sorted_rows, sorted_lens):
        new_max = max(cur_max, L)
        if cur and (len(cur) + 1) * new_max > tokens_per_batch:
            batches.append(cur)
            cur, cur_max = [r], L
        else:
            cur.append(int(r))
            cur_max = new_max
    if cur:
        batches.append(cur)
    perm = rng.permutation(len(batches))
    return [batches[i] for i in perm], sorted_rows


def collate(store, batch_rows, device):
    maxL = max(store.lengths[r] for r in batch_rows)
    B = len(batch_rows)
    tok = np.full((B, maxL), PAD, dtype=np.int64)
    msk = np.zeros((B, maxL), dtype=np.bool_)
    for bi, r in enumerate(batch_rows):
        t = store.get(r)
        tok[bi, :len(t)] = t
        msk[bi, :len(t)] = True
    return (torch.from_numpy(tok).to(device),
            torch.from_numpy(msk).to(device))


def apply_mlm(tokens, pad_mask, rng_device, mask_id=MASK, p=0.15):
    """BERT-style corruption. Returns (corrupted, labels) — labels has -100 on
    non-masked positions; loss must ignore -100."""
    labels = tokens.clone()
    prob = torch.rand(tokens.shape, device=tokens.device)
    cand = (prob < p) & pad_mask                      # candidate positions
    labels[~cand] = -100
    r = torch.rand(tokens.shape, device=tokens.device)
    mask_pos = cand & (r < 0.8)
    rand_pos = cand & (r >= 0.8) & (r < 0.9)
    keep_pos = cand & (r >= 0.9)
    tokens = tokens.clone()
    tokens[mask_pos] = mask_id
    if rand_pos.any():
        rnd = torch.randint(0, len(RANDOM_SET), (int(rand_pos.sum()),),
                            device=tokens.device)
        tokens[rand_pos] = torch.tensor(RANDOM_SET, device=tokens.device)[rnd]
    tokens[keep_pos] = labels[keep_pos]               # unchanged
    tokens[~pad_mask] = PAD                           # padding stays PAD
    return tokens, labels


# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./l2_pretrain")
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--nhead", type=int, default=12)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--dim_ff", type=int, default=3072)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--tokens_per_batch", type=int, default=8192)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--val_batches", type=int, default=300)
    p.add_argument("--max_hours", type=float, default=40.0)
    p.add_argument("--plateau_rtol", type=float, default=0.02)
    p.add_argument("--plateau_patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    t_start = time.time()
    log_path = args.out_dir + "/l2_pretrain_log.jsonl"
    print(f"Device: {device}", flush=True)

    store = TokenStore(args.store_dir)
    print(f"store: {store.n} rows", flush=True)
    with open(args.store_dir + "/l2_train_idx.npy", "rb") as f:
        train_idx = np.load(args.store_dir + "/l2_train_idx.npy")
    val_idx = np.load(args.store_dir + "/l2_val_idx.npy")

    model = L2Encoder(args.d_model, args.nhead, args.num_layers, args.dim_ff,
                      args.max_len).to(device)
    n_par = sum(p_.numel() for p_ in model.parameters())
    print(f"L2 encoder params: {n_par/1e6:.1f}M", flush=True)

    # resume
    step = 0
    val_losses = []
    latest = sorted(glob.glob(args.out_dir + "/l2_ckpt_latest.pt"))
    if latest:
        ck = torch.load(latest[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        step = ck["step"]
        val_losses = ck.get("val_losses", [])
        print(f"🔁 resumed from step {step}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.98), eps=1e-8)
    if latest:
        opt.load_state_dict(ck["opt"])

    # length-bucketed batches per epoch (deterministic per epoch seed)
    per_epoch = {}
    for ep in range(args.epochs):
        rng = np.random.default_rng(args.seed + ep)
        per_epoch[ep], _ = build_length_buckets(
            store.lengths, train_idx, args.tokens_per_batch, rng)
    steps_per_epoch = math.ceil(len(per_epoch[0]) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    print(f"batches/epoch={len(per_epoch[0])} steps/epoch={steps_per_epoch} "
          f"total_steps={total_steps}", flush=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: s / args.warmup if s < args.warmup
        else max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, (s - args.warmup)
                                                        / max(1, total_steps - args.warmup))))),
    )
    if latest and "sched" in ck:
        sched.load_state_dict(ck["sched"])

    def save_ckpt():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "val_losses": val_losses},
                   args.out_dir + "/l2_ckpt_latest.pt")

    def run_val():
        model.eval()
        rng_v = np.random.default_rng(args.seed + 999)
        vb, _ = build_length_buckets(store.lengths, val_idx,
                                     args.tokens_per_batch, rng_v)
        vb = vb[: args.val_batches]
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for rows in vb:
                tok, pm = collate(store, rows, device)
                corrupted, labels = apply_mlm(tok, pm, device)
                logits = model(corrupted, pm)
                loss = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1),
                                       ignore_index=-100)
                tot += float(loss.item()); cnt += 1
        model.train()
        return tot / max(cnt, 1)

    model.train()
    t0 = time.time()
    done = False
    for ep in range(args.epochs):
        batches = per_epoch[ep]
        # resume: skip batches already consumed in earlier epochs
        consumed_before = ep * steps_per_epoch * args.grad_accum
        start_b = max(0, step * args.grad_accum - consumed_before)
        if start_b >= len(batches):
            continue
        if start_b > 0:
            print(f"[resume] ep{ep + 1}: skipping first {start_b} batches",
                  flush=True)
        micro = 0
        for bi in range(start_b, len(batches), args.grad_accum):
            group = batches[bi: bi + args.grad_accum]
            loss_acc = 0.0
            for rows in group:
                tok, pm = collate(store, rows, device)
                corrupted, labels = apply_mlm(tok, pm, device)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    logits = model(corrupted, pm)
                    loss = F.cross_entropy(logits.view(-1, VOCAB),
                                           labels.view(-1), ignore_index=-100) \
                           / args.grad_accum
                loss.backward()
                loss_acc += float(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip
                                           if hasattr(args, "grad_clip") else 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            micro += 1

            if step % 500 == 0:
                el_h = (time.time() - t0) / 3600
                msg = dict(step=step, epoch=ep + 1,
                           train_loss=round(loss_acc, 4),
                           lr=round(sched.get_last_lr()[0], 8),
                           hours=round(el_h, 2))
                print(json.dumps(msg), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(msg) + "\n")

            if step % args.eval_steps == 0:
                vl = run_val()
                val_losses.append(vl)
                msg = dict(step=step, val_loss=vl,
                           val_ppl=round(math.exp(min(vl, 20)), 2))
                print("[VAL]", json.dumps(msg), flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(msg) + "\n")
                # plateau: 3 consecutive evals with <2% relative improvement
                if len(val_losses) >= args.plateau_patience + 1:
                    recent = val_losses[-(args.plateau_patience + 1):]
                    rel_impr = [(recent[i] - recent[i + 1]) / recent[i]
                                for i in range(len(recent) - 1)]
                    if all(r < args.plateau_rtol for r in rel_impr):
                        print(f"⛔ val plateau ({args.plateau_patience} evals "
                              f"<{args.plateau_rtol:.0%} improvement) — stopping",
                              flush=True)
                        done = True
                        break

            if step % args.save_steps == 0 or step == total_steps:
                save_ckpt()
                print(f"💾 ckpt @ step {step}", flush=True)

            if (time.time() - t_start) / 3600 >= args.max_hours:
                print(f"⛔ max_hours={args.max_hours} reached — stopping",
                      flush=True)
                done = True
                break
            if done:
                break
        if done:
            break

    save_ckpt()
    # also save a named final artifact for Stage A'
    torch.save({"model": model.state_dict(), "step": step,
                "val_losses": val_losses, "n_par": n_par},
               args.out_dir + "/l2_encoder_final.pt")
    print(f"DONE at step {step} ({(time.time() - t_start) / 3600:.1f}h)", flush=True)


if __name__ == "__main__":
    main()
