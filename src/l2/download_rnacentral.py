"""L2 Phase 0-1: resumable downloader for RNAcentral release files.

Verified URLs (RNAcentral current_release, EBI FTP, Range-resumable):
  sequences/rnacentral_active.fasta.gz   ~10.92 GB  (main corpus, 45M seqs)
  id_mapping/id_mapping.tsv.gz           ~2.66 GB   (optional: rna_type filter source)

Usage:
  python download_rnacentral.py --out-dir ./data/l2 --list-only
  python download_rnacentral.py --out-dir ./data/l2            # active fasta
  python download_rnacentral.py --out-dir ./data/l2 --with-id-mapping

Resume: partial downloads are kept as <file>.part and continued via HTTP Range
on re-run. Safe to re-launch after disconnects.
"""
import argparse
import os
import time

import requests

BASE = "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/"
FILES = {
    "active": "sequences/rnacentral_active.fasta.gz",
    "id_mapping": "id_mapping/id_mapping.tsv.gz",
}


def head_size(url):
    r = requests.head(url, timeout=60, allow_redirects=True)
    return r.status_code, int(r.headers.get("Content-Length", 0))


def download(url, out_path, chunk_mb=8):
    part = out_path + ".part"
    done = 0
    headers = {}
    if os.path.exists(part):
        done = os.path.getsize(part)
        headers["Range"] = f"bytes={done}-"
        print(f"  resume from {done/1e9:.2f} GB", flush=True)
    t0 = time.time()
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code == 416:  # already complete
            os.replace(part, out_path)
            print(f"  already complete: {out_path}", flush=True)
            return
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + done
        mode = "ab" if done else "wb"
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                el = time.time() - t0
                spd = (done - (headers["Range"] and 0 or 0)) / 1e6 / max(el, 1)
                print(f"\r  {done/1e9:6.2f}/{total/1e9:.2f} GB  {spd:5.1f} MB/s",
                      end="", flush=True)
    os.replace(part, out_path)
    print(f"\n  ✅ {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--with-id-mapping", action="store_true",
                   help="also download id_mapping.tsv.gz (2.66 GB, optional rna-type filter)")
    p.add_argument("--chunk-mb", type=int, default=8)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    wanted = ["active"] + (["id_mapping"] if args.with_id_mapping else [])
    print("== manifest ==")
    for key in wanted:
        rel = FILES[key]
        st, size = head_size(BASE + rel)
        print(f"  [{key}] {rel}  {size/1e9:.2f} GB  (HTTP {st})")
    if args.list_only:
        return

    for key in wanted:
        rel = FILES[key]
        out_path = os.path.join(args.out_dir, os.path.basename(rel))
        if os.path.exists(out_path):
            print(f"== {key}: already downloaded, skip ({out_path})", flush=True)
            continue
        print(f"== downloading [{key}] {rel}", flush=True)
        for attempt in range(1, 6):
            try:
                download(BASE + rel, out_path, args.chunk_mb)
                break
            except Exception as e:
                print(f"\n  attempt {attempt} failed: {str(e)[:120]}", flush=True)
                if attempt == 5:
                    raise
                time.sleep(10 * attempt)


if __name__ == "__main__":
    main()
