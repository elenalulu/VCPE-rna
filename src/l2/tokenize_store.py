"""L2 Phase 0-3: tokenize the filtered corpus into a flat token store.

Input : l2_filtered.fasta (streaming)
Output:
  l2_tokens.u16   flat uint16 token ids (CLS-prefixed, T-capped at 512; NO padding)
  l2_offsets.i64  int64 offsets, length n+1 (row i = tokens[offsets[i]:offsets[i+1]])
  l2_train_idx.npy / l2_val_idx.npy   row-index splits (99.5/0.5)
  tokenize_stats.json

Storage: n=10-15M rows x ~250 avg tokens x 2B ≈ 5-8 GB. The pretraining script
does length-bucketed batching over (flat tokens, offsets) — no padding waste.

Run (local machine, after filter_dedup):
  python tokenize_store.py --fasta ./data/l2/l2_filtered.fasta --out-dir ./data/l2
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "maprna_p2"))
from rna_encoder import encode_seq  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--val_frac", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tokens_f = open(args.out_dir + "/l2_tokens.u16", "wb")
    offsets = [0]
    n = 0
    total_tokens = 0
    name, chunks = None, []

    def flush_record():
        nonlocal n, total_tokens
        if name is None:
            return
        seq = "".join(chunks).upper().replace("T", "U")
        ids, _ = encode_seq(seq, min(args.max_len, len(seq) + 1))
        arr = np.array(ids, dtype=np.uint16)
        tokens_f.write(arr.tobytes())
        n += 1
        total_tokens += len(ids)
        offsets.append(offsets[-1] + len(ids))

    with open(args.fasta) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                flush_record()
                name = line[1:]
                chunks = []
            elif name is not None:
                chunks.append(line)
            if n - (n // 1_000_000) * 1_000_000 == 999_999 and n > 0:
                print(f"  tokenized {n} rows ({total_tokens/1e9:.2f}B tokens)", flush=True)
    flush_record()
    tokens_f.close()
    offsets = np.array(offsets, dtype=np.int64)
    np.save(args.out_dir + "/l2_offsets.npy", offsets)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * args.val_frac)))
    np.save(args.out_dir + "/l2_val_idx.npy", idx[:n_val])
    np.save(args.out_dir + "/l2_train_idx.npy", idx[n_val:])

    stats = {"n_rows": n, "total_tokens": total_tokens,
             "val_rows": n_val, "train_rows": n - n_val,
             "avg_tokens": round(total_tokens / max(n, 1), 1)}
    with open(args.out_dir + "/tokenize_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ rows={n} tokens={total_tokens/1e9:.2f}B avg={stats['avg_tokens']} "
          f"(train {n - n_val} / val {n_val})", flush=True)


if __name__ == "__main__":
    main()
