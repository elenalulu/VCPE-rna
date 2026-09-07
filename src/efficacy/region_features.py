"""v3 region features: locate each ASO on its target gene's transcripts.

Pipeline:
  1. Parse GENCODE v47 GTF -> per-ENST exon/CDS (genomic), gene_name -> ENSTs.
  2. Load transcript sequences (pc + lncRNA) for target genes only.
  3. For each ASO (190,927 rows): motif = reverse-complement of the ASO (DNA),
     search in every ENST of the target gene; classify matched cDNA position as
     5UTR / CDS / 3UTR via the ENST's CDS interval (strand-aware mapping).
  4. Output features parquet aligned 1:1 with in_vitro_clean.parquet rows.

Features: match_any, n_enst_matched/n_enst_total, region, rel_pos, n_occurrences.
"""
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
GEN = DATA / "gencode"
ATLAS = DATA / "aso_atlas_2"

COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp_motif(aso_seq):
    """ASO (5'->3', DNA-ish letters) -> RNA target motif (U-world), uppercase."""
    dna = re.sub(r"[^ACGTN]", "N", aso_seq.upper())
    return dna.translate(COMP)[::-1].replace("T", "U")


def parse_gtf(gtf_gz, gene_names):
    """-> {enstag: dict(exons=[(s,e)], cds=[(s,e)], strand, gene_name)} for target genes."""
    want = {g.upper() for g in gene_names}
    tx = {}
    pat_attr = re.compile(r'gene_id "([^"]+)"|transcript_id "([^"]+)"|gene_name "([^"]+)"')
    with gzip.open(gtf_gz, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            feat, start, end, strand = parts[2], int(parts[3]), int(parts[4]), parts[6]
            attrs = pat_attr.findall(parts[8])
            gid = next((a for a in attrs[0:1] for a in [a[0]]), None)
            gid = attrs[0][0] if attrs else None
            tid = next((a[1] for a in attrs if a[1]), None)
            gname = next((a[2] for a in attrs if a[2]), None)
            if gname is None or gname.upper() not in want:
                continue
            if feat == "transcript":
                tx[tid.split(".")[0]] = dict(exons=[], cds=[], strand=strand, gene=gname)
            elif feat in ("exon", "CDS") and tid.split(".")[0] in tx:
                tx[tid.split(".")[0]][("exons" if feat == "exon" else "cds")].append((start, end))
    return tx


def tx_cds_interval(enst):
    """CDS interval in SPLICED transcript coordinates (1-based, inclusive)."""
    d = ENST[enst]
    exons = sorted(d["exons"], reverse=(d["strand"] == "-"))
    cds = sorted(d["cds"], reverse=(d["strand"] == "-"))
    if not cds:
        return None
    cum, cds_tx = 0, []
    for (es, ee) in exons:
        elen = ee - es + 1
        # overlap of this exon with any CDS block
        for (cs, ce) in d["cds"]:
            ov_s, ov_e = max(es, cs), min(ee, ce)
            if ov_s <= ov_e:
                if d["strand"] == "+":
                    cds_tx.append(cum + (ov_s - es) + 1)
                    cds_tx.append(cum + (ov_e - es + 1))
                else:
                    cds_tx.append(cum + (ee - ov_e) + 1)
                    cds_tx.append(cum + (ee - ov_s + 1))
        cum += elen
    return (min(cds_tx), max(cds_tx)) if cds_tx else None


def region_of(tx_pos, cds_iv, tx_len):
    if cds_iv is None:
        return "nc"          # non-coding transcript
    if tx_pos < cds_iv[0]:
        return "5UTR"
    if tx_pos > cds_iv[1]:
        return "3UTR"
    return "CDS"


def main():
    # 1. target genes from the clean atlas
    df = pd.read_parquet(ATLAS / "in_vitro_clean.parquet")
    df["aso_dna"] = df["base_seq"].str.split(",").str.join("") \
        .str.replace("5meC", "C", regex=False)
    df["aso_dna"] = df["aso_dna"].str.replace(r"[^ACGTN]", "N", regex=True)
    gene_names = df["target_RNA"].dropna().unique().tolist()
    print(f"rows: {len(df)} | target genes: {len(gene_names)}", flush=True)

    # 2. GTF -> target transcripts
    global ENST
    ENST = parse_gtf(GEN / "gencode.v47.annotation.gtf.gz", gene_names)
    print(f"GTF: {len(ENST)} target transcripts", flush=True)

    # 3. sequences for those ENSTs (pc + lncRNA fasta)
    want_tx = set(ENST)
    seqs = {}
    for fn in ["gencode.v47.pc_transcripts.fa.gz", "gencode.v47.lncRNA_transcripts.fa.gz"]:
        with gzip.open(GEN / fn, "rt") as f:
            tid, chunks = None, []
            for line in f:
                if line[0] == ">":
                    if tid in want_tx:
                        seqs[tid] = "".join(chunks).upper()
                    tid = line[1:].split("|")[0].split(".")[0]
                    chunks = []
                else:
                    chunks.append(line.strip())
            if tid in want_tx:
                seqs[tid] = "".join(chunks).upper()
    print(f"sequences loaded: {len(seqs)}/{len(want_tx)} ENSTs", flush=True)
    for tid in list(ENST):
        if tid not in seqs:
            del ENST[tid]

    # gene -> ENSTs
    gene_txs = {}
    for tid, d in ENST.items():
        gene_txs.setdefault(d["gene"].upper(), []).append(tid)

    # 4. match each ASO
    out_rows = []
    cache_seqs = {}
    for r_i, (aso, gene) in enumerate(zip(df["aso_dna"], df["target_RNA"])):
        motif = revcomp_motif(aso)
        tids = gene_txs.get(str(gene).upper(), [])
        n_match, first = 0, None
        occ = 0
        for tid in tids:
            if tid not in cache_seqs:
                cache_seqs[tid] = seqs[tid].replace("T", "U")  # U-world search
            s = cache_seqs[tid]
            cnt = s.count(motif)
            if cnt:
                n_match += 1
                occ += cnt
                if first is None:
                    p = s.find(motif) + 1                     # 1-based tx pos
                    iv = tx_cds_interval(tid)
                    first = (tid, p, region_of(p, iv, len(s)), len(s))
        if first:
            tid, p, reg, slen = first
            rel = p / slen
        else:
            reg, rel = "no_match", np.nan
        out_rows.append((int(n_match), len(tids), reg, rel, int(occ)))
        if (r_i + 1) % 20000 == 0:
            print(f"[match] {r_i+1}/{len(df)}", flush=True)

    feat = pd.DataFrame(out_rows, columns=["n_enst_matched", "n_enst_total",
                                           "region", "rel_pos", "n_occurrences"])
    feat.to_parquet(ATLAS / "region_features.parquet", index=False)
    print("✅ region_features.parquet:", feat.shape)
    print(feat["region"].value_counts().to_string())
    print(f"match_any rate: {(feat['n_enst_matched'] > 0).mean():.3f}")


if __name__ == "__main__":
    main()
