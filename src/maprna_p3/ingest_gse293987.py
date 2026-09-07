#!/usr/bin/env python
"""V2-1c: GSE293987 (Ionis CR-DGE) 摄入 —— ASO 域真实扰动数据。

数据集（Liang et al., "A Workflow for Transcriptome-wide Assessment of
Antisense Oligonucleotide Selectivity"）：
  - A431 细胞（epidermoid carcinoma，全新细胞语境，非 K562）
  - ACTN1 gapmer ASO + 非靶向 CONTROL ASO，10 个浓度（0.0052~20 uM，4 倍稀释）× 2 重复
  - 8 个 UTC（未处理对照）；3'Tag-Seq / salmonDGE quant（53 个 quant.sf.gz）

产出（scPerturb 同构，可直接进 load_kd_datasets / train_p3）：
  gse293987_proc.h5ad —— X = log1p(CP10K)，obs:
      condition = "ACTN1"（浓度 >= --min_uM，默认 1.28）| "ctrl"（UTC + CONTROL 全浓度）
      control   = 1 (ctrl 行)
      concentration_uM / sample_id / aso_class / cell_line = A431 / perturbation_type = ASO
  gse293987_full.h5ad —— 全部 53 样本带完整元数据（供后续剂量建模，不进训练）

用法：
  python ingest_gse293987.py --list-only      # 只列设计表（元数据来自本地/远程 miniml）
  python ingest_gse293987.py                  # 下载 RAW.tar(136MB) + 聚合 + 出 h5ad
"""
import argparse
import gzip
import io
import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data"
DIR = DATA / "gse293987"
GTF = DATA / "gencode" / "gencode.v47.annotation.gtf.gz"
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE293nnn/GSE293987"
MINIML_TGZ = f"{BASE_URL}/miniml/GSE293987_family.xml.tgz"
RAW_TAR = f"{BASE_URL}/suppl/GSE293987_RAW.tar"
NS = {"m": "http://www.ncbi.nlm.nih.gov/geo/info/MINiML"}
MIN_CONC_UM = 1.28      # 纳入 ACTN1 扰动条件的最低浓度（低于此 KD 信号被稀释）


def load_metadata():
    """GSM/标题/浓度/类别 → DataFrame（本地 family.xml 优先，缺失则下载）。"""
    xml_fp = DIR / "GSE293987_family.xml"
    if not xml_fp.exists():
        print("[meta] downloading miniml ...", flush=True)
        with urllib.request.urlopen(MINIML_TGZ, timeout=120) as r:
            tf = tarfile.open(fileobj=io.BytesIO(r.read()))
            xml_fp.write_bytes(tf.extractfile(tf.getnames()[0]).read())
    root = ET.fromstring(xml_fp.read_text(encoding="utf-8"))
    rows = []
    for s in root.findall(".//m:Sample", NS):
        gsm = s.get("iid") or s.get("accession") or ""
        title = (s.findtext("m:Title", "", NS) or "").strip()
        chars = [c.text.strip() for c in s.findall(".//m:Characteristics", NS)]
        conc = next((c for c in chars if c.endswith("uM")), "0 uM")
        conc_um = float(conc.replace("uM", "").strip())
        # 子串匹配（EXP1/EXP2 两批的 UTC 都要识别，不能只匹配前缀）
        aso_class = ("UTC" if "UTC" in title else
                     "ACTN1" if "ACTN" in title else
                     "CONTROL" if "CONTROL" in title else "OTHER")
        rows.append(dict(sample_id=title, gsm=gsm,
                         aso_class=aso_class, concentration_uM=conc_um))
    df = pd.DataFrame(rows)
    df["suppl_file"] = df["gsm"].map(
        lambda g: f"{g}_*.quant.sf.gz" if g else "")
    return df


def load_tx2gene():
    """GENCODE v47 transcript_id -> gene_name（strip 版本号）。"""
    tx2gene = {}
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if '\ttranscript\t' not in line:
                continue
            attrs = dict(kv.split(" ", 1) for kv in
                         line.split("\t")[8].rstrip("\n").split("; ") if " " in kv)
            tx = attrs.get("transcript_id", "").strip('"').split(".")[0]
            gn = attrs.get("gene_name", "").strip('"')
            if tx and gn:
                tx2gene[tx] = gn
    print(f"[gtf] transcript->gene: {len(tx2gene)}", flush=True)
    return tx2gene


def quant_to_gene(quant_fp, tx2gene):
    """单个 quant.sf(.gz) -> (gene -> NumReads, gene -> TPM) 聚合（求和）。"""
    opener = gzip.open if str(quant_fp).endswith(".gz") else open
    reads, tpm = defaultdict(float), defaultdict(float)
    with opener(quant_fp, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_name = header.index("Name")
        i_tpm = header.index("TPM")
        i_reads = header.index("NumReads")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            tx = parts[i_name].split(".")[0]
            g = tx2gene.get(tx)
            if g is None:
                continue
            tpm[g] += float(parts[i_tpm])
            reads[g] += float(parts[i_reads])
    return reads, tpm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--min_uM", type=float, default=MIN_CONC_UM,
                    help="ACTN1 纳入扰动条件的最低浓度（默认 1.28）")
    ap.add_argument("--out_prefix", default="gse293987")
    args = ap.parse_args()

    DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    print(f"[meta] {len(meta)} samples | "
          f"ACTN1 {sum(meta.aso_class == 'ACTN1')} | "
          f"CONTROL {sum(meta.aso_class == 'CONTROL')} | "
          f"UTC {sum(meta.aso_class == 'UTC')} | "
          f"浓度 {sorted(set(meta.concentration_uM))}", flush=True)
    if args.list_only:
        print(meta.to_string())
        return

    # ---- 下载 RAW.tar：断点续传（Range）+ Content-Length 校验 ----
    # NCBI 经代理限速且会中途掐断连接（实测 61MB 处 EOF），必须续传+验长。
    tar_fp = DIR / "GSE293987_RAW.tar"
    part_fp = DIR / "GSE293987_RAW.tar.part"
    EXPECTED = 136570880  # filelist.txt 报的大小（bytes）
    if not tar_fp.exists():
        print(f"[dl] {RAW_TAR} (EXPECTED {EXPECTED/1e6:.1f} MB) ...", flush=True)
        for attempt in range(1, 21):   # 最多 20 段续传
            have = part_fp.stat().st_size if part_fp.exists() else 0
            if have >= EXPECTED:
                break
            req = urllib.request.Request(RAW_TAR)
            if have > 0:
                req.add_header("Range", f"bytes={have}-")
                print(f"[dl] resume from {have/1e6:.1f} MB (attempt {attempt})",
                      flush=True)
            try:
                with urllib.request.urlopen(req, timeout=1800) as r, \
                        open(part_fp, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            except Exception as e:
                print(f"[dl] connection dropped ({e}); will resume", flush=True)
            have = part_fp.stat().st_size if part_fp.exists() else 0
            if have >= EXPECTED:
                break
            print(f"[dl] have {have/1e6:.1f}/{EXPECTED/1e6:.1f} MB, retrying...",
                  flush=True)
        if part_fp.stat().st_size != EXPECTED:
            raise RuntimeError(
                f"download incomplete: {part_fp.stat().st_size}/{EXPECTED} "
                f"(20 次续传仍未完成，请手动下载 {RAW_TAR} 放到 {DIR})")
        os.replace(part_fp, tar_fp)      # 原子改名：只有完整 tar 才会落位
    print(f"[dl] tar ready: {tar_fp.stat().st_size/1e6:.1f} MB", flush=True)

    # ---- GSM -> tar 内文件名映射 ----
    with tarfile.open(tar_fp) as tf:
        members = {m.name.split("_")[0]: m.name for m in tf.getmembers()}
    tx2gene = load_tx2gene()

    # ---- 逐样本聚合 ----
    gene_set = set()
    samples, matrices_reads = [], {}
    with tarfile.open(tar_fp) as tf:
        for _, row in meta.iterrows():
            gsm = row["gsm"]
            fname = members.get(gsm)
            if not fname:
                print(f"[warn] no suppl for {row.sample_id} ({gsm})", flush=True)
                continue
            member = tf.getmember(fname)
            with tf.extractfile(member) as fh:
                raw = gzip.decompress(fh.read()).decode()
            # 解析 quant.sf：Name / TPM / NumReads，转录本聚合到基因（求和）
            header = raw.split("\n", 1)[0].split("\t")
            i_name, i_tpm, i_reads = header.index("Name"), header.index("TPM"), header.index("NumReads")
            r_agg, t_agg = defaultdict(float), defaultdict(float)
            for line in raw.split("\n")[1:]:
                if not line:
                    continue
                parts = line.split("\t")
                g = tx2gene.get(parts[i_name].split(".")[0])
                if g is None:
                    continue
                t_agg[g] += float(parts[i_tpm])
                r_agg[g] += float(parts[i_reads])
            samples.append(row["sample_id"])
            matrices_reads[row["sample_id"]] = r_agg
            gene_set.update(r_agg)
            print(f"[quant] {row['sample_id']}: {len(r_agg)} genes", flush=True)

    genes = sorted(gene_set)
    gi = {g: i for i, g in enumerate(genes)}
    n = len(samples)
    R = np.zeros((n, len(genes)), dtype=np.float32)
    for si, s in enumerate(samples):
        for g, v in matrices_reads[s].items():
            R[si, gi[g]] = v
    # CP10K + log1p（scPerturb 同构）
    tot = R.sum(axis=1, keepdims=True) + 1e-6
    X = np.log1p(R / tot * 1e4)

    obs = meta.set_index("sample_id").loc[samples].copy()
    # ctrl 口径（论文同款 selectivity 对照）：只用 CONTROL ASO（浓度匹配、同转染流程）。
    # 实测 CONTROL vs UTC 有大应激偏移（NDRG1 -3.0 / PSAP +2.5）——非靶向 ASO 转染
    # 本身引起转录响应；UTC 混进 ctrl 会把转染应激泄进基线。治疗语义（in vivo 无
    # 转染）下，扣除转染效应后的剩余响应才是序列特异敲低。
    # UTC/OTHER -> condition="utc_ref"：不进 ctrl 池；load 后 resolve_pert_row 失败
    # 会被 train_p3 安全跳过，不产生伪扰动。
    is_actn1_high = (obs.aso_class == "ACTN1") & (obs.concentration_uM >= args.min_uM)
    obs["condition"] = np.where(is_actn1_high, "ACTN1",
                       np.where(obs.aso_class == "CONTROL", "ctrl", "utc_ref"))
    obs["control"] = (obs.aso_class == "CONTROL").astype(int)
    obs["perturbation_raw"] = np.where(
        is_actn1_high,
        obs.apply(lambda r: f"ACTN1@{r.concentration_uM}uM", axis=1),
        np.where(obs.aso_class == "CONTROL", "ctrl", "utc_ref"))
    obs["cell_line"] = "A431"
    obs["perturbation_type"] = "ASO"
    obs["tissue_type"] = "epithelial"

    import anndata as ad
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    full_fp = DIR / f"{args.out_prefix}_full.h5ad"
    adata.write_h5ad(full_fp)
    # proc = full 的行子集（ACTN1 高浓度 + 全部 ctrl），无需拼接
    keep = (adata.obs.condition != "ctrl") | (adata.obs.control == 1)
    proc = adata[keep.values].copy()
    proc_fp = DIR / f"{args.out_prefix}_proc.h5ad"
    proc.write_h5ad(proc_fp)
    print(f"[done] full: {full_fp} {adata.shape} | proc: {proc_fp} {proc.shape} "
          f"(ACTN1 扰动 {int(is_actn1_high.sum())} 样本, ctrl=CONTROL "
          f"{int((obs.aso_class == 'CONTROL').sum())}, utc_ref "
          f"{int((obs.condition == 'utc_ref').sum())})", flush=True)


if __name__ == "__main__":
    main()
