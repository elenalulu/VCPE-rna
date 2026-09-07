"""Download the scPerturb priority pool (~22GB, CC-BY 4.0) from Zenodo record 13350497.

Resumable: skips existing complete files; partial .part files are re-downloaded.
Retries: 5 attempts per file, alternating direct/proxy. Verify: size check vs API.
Usage: python download_scperturb.py            # download priority pool
       python download_scperturb.py --all      # all 54 files (43GB)
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("VCPE_DATA_DIR", str(REPO_ROOT / "data" / "scperturb")))
PROXY = os.environ.get("VCPE_PROXY")  # optional, e.g. http://127.0.0.1:7892
PROXIES = {"https": PROXY, "http": PROXY} if PROXY else None
RECORD = "13350497"
API = f"https://zenodo.org/api/records/{RECORD}"
BASE = f"https://zenodo.org/records/{RECORD}/files"

PRIORITY = [
    "ReplogleWeissman2022_K562_gwps.h5ad",
    "JoungZhang2023_atlas.h5ad",
    "SrivatsanTrapnell2020_sciplex3.h5ad",
    "GasperiniShendure2019_atscale.h5ad",
    "NadigOConner2024_jurkat.h5ad",
    "NadigOConner2024_hepg2.h5ad",
]


def list_files():
    r = requests.get(API, timeout=30)
    r.raise_for_status()
    rec = r.json()
    sizes = {f["key"]: f["size"] for f in rec["files"]}
    lic = rec["metadata"].get("license", {}).get("id", "?")
    print(f"record license: {lic} | {len(sizes)} files | {sum(sizes.values())/1e9:.1f} GB")
    return sizes


def fetch(key, size):
    dest = OUT / key
    if dest.exists() and dest.stat().st_size == size:
        print(f"skip {key} (done)", flush=True)
        return True
    tmp = dest.with_suffix(".part")
    url = f"{BASE}/{key}?download=1"
    attempts = [("direct", None)] + [("proxy", PROXIES)] * 4
    for name, proxies in attempts:
        try:
            start = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={start}-"} if start else {}
            r = requests.get(url, timeout=(15, 600), proxies=proxies, stream=True,
                             headers=headers)
            if r.status_code == 416:  # already complete
                tmp.rename(dest)
                return True
            r.raise_for_status()
            mode = "ab" if start and r.status_code == 206 else "wb"
            with open(tmp, mode) as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
            if tmp.stat().st_size == size:
                tmp.rename(dest)
                print(f"✅ {key}: {size/1e9:.2f} GB ({name})", flush=True)
                return True
            raise IOError(f"size mismatch {tmp.stat().st_size} != {size}")
        except Exception as e:
            print(f"  attempt[{name}] {key}: {str(e)[:90]}", flush=True)
            time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="download all 54 files")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = list_files()
    keys = sorted(sizes) if args.all else PRIORITY
    failed = []
    for k in keys:
        if k not in sizes:
            failed.append(k)
            continue
        if not fetch(k, sizes[k]):
            failed.append(k)
    if failed:
        print(f"FAILED ({len(failed)}): {failed}", flush=True)
        sys_exit = 1
    else:
        print("ALL DONE", flush=True)
        sys_exit = 0
    raise SystemExit(sys_exit)


if __name__ == "__main__":
    main()
