"""Download the OligoGym dataset family (12 csv.gz, CC BY 4.0) from HuggingFace.

CollageBio/oligo-datasets — Roche/CollageBio, NeurIPS 2025 D&B.
Files: https://huggingface.co/datasets/CollageBio/oligo-datasets/resolve/main/<key>.csv.gz
Retry logic: direct -> proxy(127.0.0.1:7892) -> 3 attempts each. Resumable (skips existing).
"""
import sys
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[2] / "data" / "oligogym"
BASE = "https://huggingface.co/datasets/CollageBio/oligo-datasets/resolve/main"
KEYS = ["ichihara_2007_1", "ichihara_2007_2", "alharbi_2020_1", "alharbi_2020_2",
        "hagedorn_2022_1", "hwang_2024_1", "knott_2014_1", "moe_neurotox_1",
        "martinelli_2023_1", "mcquisten_2007_1", "papargyri_2020_1",
        "shmushkovich_2018_1"]
PROXIES = {"https": "http://127.0.0.1:7892", "http": "http://127.0.0.1:7892"}


def fetch(key):
    dest = OUT / f"{key}.csv.gz"
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"skip {key} (exists {dest.stat().st_size/1e6:.1f} MB)", flush=True)
        return True
    url = f"{BASE}/{key}.csv.gz"
    attempts = [("direct", None)] + [("proxy", PROXIES)] * 2
    for name, proxies in attempts:
        try:
            r = requests.get(url, timeout=(15, 300), proxies=proxies, stream=True)
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            if total and tmp.stat().st_size != total:
                raise IOError(f"size mismatch {tmp.stat().st_size} != {total}")
            tmp.rename(dest)
            print(f"✅ {key}: {dest.stat().st_size/1e6:.1f} MB ({name})", flush=True)
            return True
        except Exception as e:
            print(f"  attempt[{name}] {key}: {str(e)[:90]}", flush=True)
            time.sleep(2)
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    failed = [k for k in KEYS if not fetch(k)]
    if failed:
        print(f"FAILED: {failed}", flush=True)
        sys.exit(1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
