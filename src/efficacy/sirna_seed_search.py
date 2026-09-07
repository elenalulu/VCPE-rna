"""siRNA seed-region off-target search (Tier-1 chain entry for siRNA tasks).

Mechanism: the guide (antisense) strand's positions 2-8 (the 'seed') pair with the
target mRNA via reverse-complement matching. A 7-mer exact match on any transcript
of a gene flags that gene as a potential off-target (standard seed model).

Input: guide strand (21nt, RNA alphabet U).
Transcript source: gene_transcripts.fa (per-symbol transcripts; NOTE — the copy in
AIDO/vertical_slice/data is 600nt-truncated, an approximation for 3'UTR coverage).

Usage:
  python sirna_seed_search.py --guide "CUAAUAUGUUAAUUGAUUUAU" \
      --fasta <gene_transcripts.fa> --min_seed 7
Returns JSON: {gene: {hits, positions}} sorted by hit count.
"""
import argparse
import json

COMP = str.maketrans("ACGU", "UGCA")


def revcomp(seq):
    return seq.translate(COMP)[::-1]


def search_seed(guide, fasta, min_seed=7):
    guide = guide.upper().replace("T", "U")
    seed = guide[1:1 + min_seed]
    seed_rc = revcomp(seed)                       # what appears in the mRNA
    hits = {}
    name, chunks = None, []
    with open(fasta) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    seq = "".join(chunks).upper().replace("T", "U")
                    n = seq.count(seed_rc)
                    if n:
                        hits.setdefault(name, []).append((n, seq.find(seed_rc)))
                name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        seq = "".join(chunks).upper().replace("T", "U")
        n = seq.count(seed_rc)
        if n:
            hits.setdefault(name, []).append((n, seq.find(seed_rc)))
    out = {g: {"hits": sum(x[0] for x in v), "sites": len(v)} for g, v in hits.items()}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["hits"])), seed_rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--guide", required=True, help="siRNA antisense/guide strand (21nt, U)")
    p.add_argument("--fasta", default="data/gene_transcripts.fa",
                   help="gene transcripts FASTA (see data/README.md)")
    p.add_argument("--min_seed", type=int, default=7)
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()
    hits, seed_rc = search_seed(args.guide, args.fasta, args.min_seed)
    print(f"seed({args.min_seed}nt) = {args.guide[1:1+args.min_seed]} -> mRNA motif {seed_rc}")
    print(f"off-target genes: {len(hits)}")
    for g, v in list(hits.items())[:args.top]:
        print(f"  {g:12} hits={v['hits']:3d} sites={v['sites']}")
    json.dump(hits, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "last_seed_search.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
