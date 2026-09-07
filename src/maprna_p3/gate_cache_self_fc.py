#!/usr/bin/env python
"""VCPE V2-0a: 平台 cache 靶基因自响应表达门控（后处理，无需 GPU 重推理）。

背景（2026-09-04 诊断）：VCPE P2.3-B 的 is_target 特征从 CRISPRi 训练数据学得
"被扰动基因自身强下调"先验，迁移到低表达语境时失真（如 APOC3 在 K562 近零
表达仍预测 self_fc=-0.60，ALB/TTR/SERPINA1 同类失真）。CRISPRi 里 guide 直接
压启动子、靶基因必然沉默，这是模态真实差异——敲不掉不存在的 RNA。

门控规则（只作用于靶基因自身条目 self-entry，不动 trans 响应）：
  1) panel 基因（语境 h5ad ctrl 有测量）：gate = min(1, expr_log1p / FLOOR)
     expr 为 ctrl 细胞 pseudobulk 均值（CP10K+log1p 空间），FLOOR 默认 0.7。
  2) panel 外基因（Perturb-seq panel 不含，恰是组织特异基因聚集区）：
     用本地 Protein Atlas tissue specificity——若类别为 "Tissue enriched" 且
     富集组织谱系与查询语境谱系不匹配（如 liver-enriched 基因 × K562 血系），
     gate = 0.15；其余（Group enriched / Tissue enhanced / 低特异性）不门控。
  最终 self_fc *= gate，并重排 top_up/top_down。

用法：
  python gate_cache_self_fc.py --cache <vcpe_cache_v3_k562a.json.gz> \
      --expr_h5ad <proc h5ad> --context k562a [--apply]
默认 dry-run 只打印报告；--apply 时先备份到同目录 backups_v2_0a/ 再写回。

验收判据（V2-0a go/no-go）：
  - k562a: APOC3/ALB/TTR/SERPINA1 self|fc| 0.20~0.60 → ≤0.15；MYC/RPL13A 不变
  - hepg2: ALB（panel 内、肝语境高表达）self_fc 不变（差分验证）
"""
import argparse
import gzip
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np

FLOOR = 0.7                 # panel 表达门控阈值（log1p CP10K）
FLAG_NEAR_ZERO = 0.1        # UI 警示阈值：仅近零表达才亮"低表达语境"徽章（分布校准后）
MISMATCH_GATE = 0.15        # Tissue-enriched × 谱系不匹配时的压制系数
HPA_TSV = (Path(__file__).resolve().parents[4]
           / "rna_platform" / "bio_literature" / "bundles"
           / "protein_atlas" / "proteinatlas.tsv")

# 语境谱系关键词（HPA 组织名子串匹配）
CONTEXT_LINEAGE = {
    "k562a": ("bone marrow", "blood", "lymphoid", "spleen", "thymus"),
    "k562g": ("bone marrow", "blood", "lymphoid", "spleen", "thymus"),
    "jurkat": ("bone marrow", "blood", "lymphoid", "spleen", "thymus"),
    "hepg2": ("liver",),
}


def load_panel_expr(h5ad_fp):
    """语境 ctrl 细胞 pseudobulk 表达（基因 -> log1p(CP10K) 均值）。"""
    a = ad.read_h5ad(h5ad_fp, backed="r")
    obs = a.obs
    if "control" in obs.columns:
        mask = obs["control"].astype(int) == 1
    else:
        mask = obs["condition"].astype(str).str.lower() == "ctrl"
    idx = np.where(mask.values)[0]
    sub = a[idx].to_memory()
    X = sub.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    expr = X.mean(axis=0)
    genes = list(sub.var_names)
    sub.file.close()
    a.file.close()
    return dict(zip(genes, map(float, expr)))


def load_hpa_tissue_enriched():
    """Protein Atlas：Tissue enriched 基因 -> 富集组织列表（仅不匹配时需要）。"""
    out = {}
    with open(HPA_TSV, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_gene = header.index("Gene")
        i_spec = header.index('"RNA tissue specificity"')
        i_ntpm = header.index('"RNA tissue specific nTPM"')
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= i_ntpm:
                continue
            spec = parts[i_spec].strip('"')
            if spec != "Tissue enriched":
                continue
            ntpm = parts[i_ntpm].strip('"')
            tissues = [seg.split(":")[0].strip().lower()
                       for seg in ntpm.split(";") if ":" in seg]
            out[parts[i_gene]] = tissues
    return out


def lineage_gate(gene, hpa, lineage_kw):
    """panel 外基因：Tissue enriched 且富集组织谱系全不匹配 → MISMATCH_GATE。"""
    tissues = hpa.get(gene)
    if not tissues:
        return 1.0, "off-panel/no-tissue-enriched"
    if any(any(k in t for k in lineage_kw) for t in tissues):
        return 1.0, "off-panel/lineage-match"
    return MISMATCH_GATE, "off-panel/lineage-mismatch"


def gate_cache(cache_fp, expr_h5ad, context, apply=False):
    cache_fp = Path(cache_fp)
    with gzip.open(cache_fp, "rt", encoding="utf-8") as f:
        cache = json.load(f)
    genes = cache["genes"]
    lineage_kw = CONTEXT_LINEAGE.get(context, CONTEXT_LINEAGE["k562a"])

    print(f"[expr] loading ctrl pseudobulk from {Path(expr_h5ad).name} ...")
    panel_expr = load_panel_expr(expr_h5ad)
    print(f"[expr] panel genes with ctrl mean: {len(panel_expr)}")
    hpa = load_hpa_tissue_enriched()
    print(f"[hpa]  tissue-enriched genes: {len(hpa)}")

    report = {"panel_gated": [], "lineage_gated": [], "unchanged_self": 0}
    expr_flags = {}
    for gene, entry in genes.items():
        # 表达标志（不论有无 self-entry 都记录，用于平台查询时"低表达语境"警示）
        e = panel_expr.get(gene)
        if e is not None:
            if e < FLAG_NEAR_ZERO:
                expr_flags[gene] = {
                    "level": "near_zero",
                    "detail": f"ctrl 表达 log1p={e:.2f}（本语境近零表达）",
                    "gate": round(min(1.0, e / FLOOR), 3),
                }
        else:
            lg, _src = lineage_gate(gene, hpa, lineage_kw)
            if lg < 0.999:
                expr_flags[gene] = {
                    "level": "lineage_mismatch",
                    "detail": "组织富集基因（Protein Atlas Tissue enriched）与本语境谱系不匹配",
                    "gate": lg,
                }
        tu, td = entry.get("top_up", []), entry.get("top_down", [])
        if not tu and not td:
            continue
        # self-entry：值列表中基因名 == 靶基因自身
        e = panel_expr.get(gene)
        if e is not None:
            gate = min(1.0, e / FLOOR)
            src = f"panel expr={e:.3f}"
        else:
            gate, src = lineage_gate(gene, hpa, lineage_kw)
        if gate >= 0.999:
            report["unchanged_self"] += 1
            continue
        changed = False
        for lst in (tu, td):
            for i, (gn, v) in enumerate(lst):
                if gn == gene:
                    lst[i] = [gn, round(v * gate, 6)]
                    changed = True
        if changed:
            # 重排（top_up 降序 / top_down 升序）保持榜单有序
            entry["top_up"] = sorted(tu, key=lambda x: -x[1])
            entry["top_down"] = sorted(td, key=lambda x: x[1])
            item = (gene, src, gate)
            (report["panel_gated"] if e is not None else report["lineage_gated"]).append(item)

    n_p, n_l = len(report["panel_gated"]), len(report["lineage_gated"])
    print(f"[gate] panel-gated self entries: {n_p} | lineage-gated: {n_l} "
          f"| unchanged/no-self: {report['unchanged_self']}")

    # 验收集
    print("\n[verify] 关键基因自响应（门控后）：")
    for g in ("APOC3", "ALB", "TTR", "SERPINA1", "PCSK9", "MYC", "RPL13A", "HBZ"):
        if g not in genes:
            print(f"  {g:10s} (不在该语境 cache)")
            continue
        entry = genes[g]
        for lst, tag in ((entry.get("top_up", []), "up"), (entry.get("top_down", []), "down")):
            for gn, v in lst:
                if gn == g:
                    e = panel_expr.get(g)
                    print(f"  {g:10s} self_fc={v:+.4f} ({tag})  "
                          f"panel_expr={e if e is not None else 'off-panel'}")

    if not apply:
        print("\n[dry-run] 未写入。加 --apply 落盘（自动备份到 backups_v2_0a/）。")
        return

    cache["self_fc_gating"] = {
        "version": "v2-0a",
        "panel_floor_log1p": FLOOR,
        "mismatch_gate": MISMATCH_GATE,
        "expr_source": str(Path(expr_h5ad).name),
        "lineage": list(lineage_kw),
        "n_panel_gated": n_p,
        "n_lineage_gated": n_l,
        "n_expr_flags": len(expr_flags),
        "note": ("self-response expression gating (post-process); panel genes linear "
                 "ramp vs ctrl pseudobulk, off-panel tissue-enriched lineage mismatch "
                 "suppressed; trans responses untouched"),
    }
    cache["expr_flags"] = expr_flags
    bdir = cache_fp.parent / "backups_v2_0a"
    bdir.mkdir(exist_ok=True)
    bak = bdir / cache_fp.name
    if not bak.exists():
        shutil.copy2(cache_fp, bak)
        print(f"[backup] {bak}")
    with gzip.open(cache_fp, "wt", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[write] {cache_fp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--expr_h5ad", required=True)
    p.add_argument("--context", required=True, choices=list(CONTEXT_LINEAGE))
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    gate_cache(args.cache, args.expr_h5ad, args.context, apply=args.apply)
