"""L2 Phase 0-2: filter + exact dedup of the RNAcentral corpus.

Input : rnacentral_active.fasta.gz (streaming)
Output: l2_filtered.fasta (plain text, streaming) + filter_stats.json

Filters (per L2 立项书):
  - length 20-500 nt
  - N fraction <= 10%
  - exact dedup (blake2b hash of normalized sequence; 45M entries ~ 1GB RAM)
Alphabet normalization: uppercase, U kept, T -> U (MLM alphabet is RNA).
Species: all kept (cross-species pretraining, AIDO.RNA same strategy).
Type filter (rRNA/tRNA): deferred — length cap removes most rRNA; tRNA unique
seqs are ~1% of corpus (noise-level for MLM). Optional via id_mapping later.

Streaming both ends: constant RAM (~1 GB for the hash set).

Run (local machine, after download):
  python filter_dedup.py --fasta ./data/l2/rnacentral_active.fasta.gz \
    --out ./data/l2/l2_filtered.fasta
"""
import argparse
import gzip
import hashlib
import json

MAX_LEN = 500
MIN_LEN = 20
N_MAX_FRAC = 0.10


def norm(seq):
    """Uppercase, T->U, strip whitespace."""
    return seq.upper().replace("T", "U").replace(" ", "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--exclude-types", type=str, default="rRNA,tRNA,scRNA,SRP_RNA,RNase_MRP_RNA,RNase_P_RNA",
                   help="comma-separated RNA types to drop (matched against header text; "
                        "the data_msa copy of rnacentral_active has 'rRNA from N species' style headers)")
    args = p.parse_args()
    excl = [t.strip() for t in args.exclude_types.split(",") if t.strip()]

    n_raw = n_pass = n_dup = 0
    seen = set()
    stats = {"n_raw": 0, "n_pass": 0, "n_dup": 0, "n_short": 0, "n_long": 0,
             "n_ncontent": 0, "n_type_excluded": 0}
    out = open(args.out, "w")
    name, desc, chunks = None, "", []
    last_print = 0

    def flush_record():
        nonlocal n_raw, n_pass, n_dup
        if name is None:
            return
        # type exclusion (header text match, e.g. 'rRNA from 1 species')
        for t in excl:
            if t.lower() in desc.lower():
                stats["n_type_excluded"] += 1
                return
        seq = norm("".join(chunks))
        n_raw += 1
        L = len(seq)
        if L < MIN_LEN:
            stats["n_short"] += 1
            return
        if L > MAX_LEN:
            stats["n_long"] += 1
            return
        n_count = seq.count("N")
        if n_count / max(L, 1) > N_MAX_FRAC:
            stats["n_ncontent"] += 1
            return
        h = hashlib.blake2b(seq.encode(), digest_size=12).digest()
        if h in seen:
            n_dup += 1
            stats["n_dup"] += 1
            return
        seen.add(h)
        out.write(f">{name}\n{seq}\n")
        n_pass += 1
        stats["n_pass"] += 1

    with gzip.open(args.fasta, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                flush_record()
                parts = line[1:].split()
                name = parts[0] if parts else line[1:]
                desc = line[1:]
                chunks = []
            elif name is not None:
                chunks.append(line)
            if n_raw - last_print >= 1_000_000:
                last_print = n_raw
                print(f"  scanned {n_raw} raw / {n_pass} kept "
                      f"({n_dup} dup, {stats['n_short']} short, {stats['n_long']} long)",
                      flush=True)
    flush_record()
    out.close()
    seen.clear()

    stats["n_raw"] = n_raw
    stats["n_pass"] = n_pass
    stats["n_dup"] = n_dup
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ {args.out}: {n_pass} kept / {n_raw} raw "
          f"(dup {n_dup}, short {stats['n_short']}, long {stats['n_long']}, "
          f"N-rich {stats['n_ncontent']})", flush=True)


if __name__ == "__main__":
    main()
